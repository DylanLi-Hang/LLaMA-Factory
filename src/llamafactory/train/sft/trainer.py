# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], training_weight_ratio: Optional[float] = 1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.training_weight_ratio = training_weight_ratio
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation to apply different loss weights based on dataset type.
        """
        labels = inputs.get("labels")
        if "dataset_label" not in inputs:
            print("Warning: 'dataset_label' not found in inputs. Defaulting to equal weights.")
            dataset_ids = torch.zeros(labels.size(0), device=labels.device, dtype=torch.long)
        else:
            dataset_ids = inputs.pop("dataset_label")  # Assuming 'dataset_label' is passed in inputs as a tensor

        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Loss calculation
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            # Compute token-level loss
            loss_fct_per_sample = CrossEntropyLoss(reduction='none')
            loss_per_sample = loss_fct_per_sample(shift_logits, shift_labels)

            # Handle padding tokens
            non_ignored = shift_labels != -100
            loss_per_sample = loss_per_sample * non_ignored

            # Expand dataset_ids to token level
            seq_len = shift_labels.size(0) // dataset_ids.size(0)
            if shift_labels.size(0) % dataset_ids.size(0) != 0:
                raise ValueError("shift_labels size must be divisible by dataset_ids size.")
            token_level_dataset_ids = dataset_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)

            # Compute weights
            weight_ratio = torch.tensor(self.training_weight_ratio, device=shift_labels.device)
            sample_weights = torch.where(token_level_dataset_ids == 1, weight_ratio, torch.tensor(1.0, device=shift_labels.device))

            # Apply weights to loss
            weighted_loss_per_sample = loss_per_sample * sample_weights

            # **Print summarized debugging information**
            # print("\nDebugging Information:")
            # print(f"Weight ratio applied: {self.training_weight_ratio}")
            # print(f"Dataset IDs (label == 1): {dataset_ids[dataset_ids == 1].size(0)} samples")
            # print(f"Dataset IDs (label == 0): {dataset_ids[dataset_ids == 0].size(0)} samples")
            # if dataset_ids[dataset_ids == 1].size(0) > 0:
            #     print(f"Loss per sample (label == 1): Mean={loss_per_sample[token_level_dataset_ids == 1].mean().item():.4f}, Std={loss_per_sample[token_level_dataset_ids == 1].std().item():.4f}")
            # else:
            #     print("Loss per sample (label == 1): No samples")
            # print(f"Loss per sample (label == 0): Mean={loss_per_sample[token_level_dataset_ids == 0].mean().item():.4f}, Std={loss_per_sample[token_level_dataset_ids == 0].std().item():.4f}")

            # Compute custom mean loss
            custom_mean_loss = weighted_loss_per_sample.sum() / non_ignored.sum()
            print(f"Custom Mean Loss: {custom_mean_loss.item():.4f}")
            loss = custom_mean_loss
        else:
            if self.training_weight_ratio != 1.0:
                print("Weight Ratio is Set but no labels provided, using original loss")
            loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
