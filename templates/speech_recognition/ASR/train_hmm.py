#!/usr/bin/env/python3
"""This minimal example trains an HMM-based aligner with the Viterbi algorithm.
The encoder is based on a combination of convolutional, recurrent, and
feed-forward networks (CRDNN) that predict phoneme states.
Given the tiny dataset, the expected behavior is to overfit the training data
(with a validation performance that stays high).
"""

import pathlib
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class AlignBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        # Compute sequence loss against targets with EOS
        tokens, tokens_lens = self.prepare_tokens(
            stage, batch.tokens
        )
        tokens_orig = sb.utils.data_utils.undo_padding(tokens, tokens_lens)
        tokens = self.hparams.aligner.expand_phns_by_states_per_phoneme(
            tokens, tokens_lens
        )
        # print('tokens', tokens)
        # print('tokens_lens', tokens_lens)
        # print('tokens_orig', tokens_orig)
        "Given the network predictions and targets computed the forward loss."
        predictions, lens = predictions
        # print('predictions', predictions)
        # print('predictions.shape', predictions.shape)
        # print('lens', lens)
        # phns, phn_lens = batch.phn_encoded
        prev_alignments = self.hparams.aligner.get_prev_alignments(
            batch.id, predictions, lens, tokens, tokens_lens
        )
        # print('prev_alignments', prev_alignments)
        loss = self.hparams.compute_cost(predictions, prev_alignments)

        # if stage != sb.Stage.TEST:
        viterbi_scores, alignments = self.hparams.aligner(
            predictions, lens, tokens, tokens_lens, "viterbi"
        )
        # print('alignments', alignments)
        self.hparams.aligner.store_alignments(batch.id, alignments)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss},
            )

            print("Epoch %d complete" % epoch)
            print("Train loss: %.3f" % self.train_loss)
            print("Valid loss: %.3f" % stage_loss)
            print("Recalculating and recording alignments...")
            self.evaluate(self.hparams.train_data)

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

# def data_prep(data_folder, hparams):
#     "Creates the datasets and their data processing pipelines."

#     # 1. Declarations:
#     train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
#         json_path=data_folder / "../annotation/ASR_train.json",
#         replacements={"data_root": data_folder},
#     )
#     valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
#         json_path=data_folder / "../annotation/ASR_dev.json",
#         replacements={"data_root": data_folder},
#     )

#     # The evaluate method of the brain class, needs to align over training data
#     hparams["train_data"] = train_data

#     datasets = [train_data, valid_data]
#     label_encoder = sb.dataio.encoder.CTCTextEncoder()
#     label_encoder.expect_len(hparams["num_labels"])

#     # 2. Define audio pipeline:
#     @sb.utils.data_pipeline.takes("wav")
#     @sb.utils.data_pipeline.provides("sig")
#     def audio_pipeline(wav):
#         sig = sb.dataio.dataio.read_audio(wav)
#         return sig

#     sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

#     # 3. Define text pipeline:
#     @sb.utils.data_pipeline.takes("phn")
#     @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
#     def text_pipeline(phn):
#         phn_list = phn.strip().split()
#         yield phn_list
#         phn_encoded = label_encoder.encode_sequence_torch(phn_list)
#         yield phn_encoded

#     sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

#     # 3. Fit encoder:
#     # NOTE: In this minimal example, also update from valid data
#     label_encoder.update_from_didataset(train_data, output_key="phn_list")
#     label_encoder.update_from_didataset(valid_data, output_key="phn_list")

#     # 4. Set output:
#     sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

#     return train_data, valid_data


def data_prep(data_folder, hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys that correspond
        to the DynamicItemDataset objects.
    """
    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        normalizer = sb.dataio.preprocess.AudioNormalizer(sample_rate=16000, mix='avg-to-mono')
        sig = sb.dataio.dataio.read_audio(wav)
        sig = normalizer(sig, sample_rate=16000)
        return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(words):
        """Processes the transcriptions to generate proper labels"""
        yield words
        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "words",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )

    return datasets['train'], datasets['valid']

def main(device="cpu"):
    
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = experiment_dir / "hyperparams.yaml"

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
        
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file
    )

    data_folder = hparams["data_folder"]
    data_folder = (experiment_dir / data_folder).resolve()

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    hparams["train_data"] = train_data
    hparams["valid_data"] = valid_data

    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()
    
    # Trainer initialization
    ali_brain = AlignBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )

    # Training/validation loop
    torch.autograd.set_detect_anomaly(True)
    ali_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Evaluation is run separately (now just evaluating on valid data)
    test_stats = ali_brain.evaluate(
        valid_data,
        min_key="WER"
    )

    # Save final checkpoint (fixed name)
    ali_brain.checkpointer.save_checkpoint(name="latest")


if __name__ == "__main__":
    main()


def test_error(device):
    main(device)
