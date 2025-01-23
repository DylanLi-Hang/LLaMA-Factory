# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

import copy
import json
import time 
import torch
from llama import Dialog, Llama
from datasets import DatasetDict

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if data_args.dynamic_eval:
        training_args.dataloader_pin_memory = False
    print(f'model_args: {model_args}')
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # for key in dataset_module['eval_dataset'].column_names:
    #     print(f"Column: {key}")
    #     for i, value in enumerate(dataset_module['eval_dataset'][key]):  # 检查前 3 行
    #         if isinstance(value, torch.Tensor):
    #             print(f"  Row {i}: {value.device}")
    #         else:
    #             print(f"  Row {i}: CPU (Non-Tensor)")

    temp_model_args = copy.deepcopy(model_args)
    if training_args.do_predict and data_args.dynamic_eval: # 
        print('Dynamic Prediction Mode Start')
        print(f'Start to load the Base model')
        temp_model_args.adapter_name_or_path = None
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    #print("model_args:", model_args)

    # print('\n')
    #print(f'fine_tuning_args: {finetuning_args}')
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        training_weight_ratio=data_args.weight_ratio,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    print("training_weight_ratio:", trainer.training_weight_ratio)

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        if data_args.dynamic_eval:
            all_predictions = []
            #selector_model = _load_modelevaluate_adapter_index()
            print(f'len(model_args.adapter_name_or_path): {len(model_args.adapter_name_or_path)}')
            for expert_index in range(len(model_args.adapter_name_or_path)+1):
                lora_model_args = copy.deepcopy(model_args)
                adapter_relates_data = get_adapter_index_dataset(dataset_module['eval_dataset'],expert_index)
                print(f'Adapter_relates_data: {adapter_relates_data}')
                if adapter_relates_data.num_rows == 0: # Nothing we just jump to next one
                    continue
                # Set the correct LoRA adapter path for the current group
                if model_args.adapter_name_or_path is  None or expert_index == 0: # Reuse the model that we create at the beginning
                    lora_model_args.adapter_name_or_path = None
                    print(f'Base Model evaluation Start')
                elif model_args.adapter_name_or_path is not None:
                    lora_model_args.adapter_name_or_path = model_args.adapter_name_or_path[:expert_index]
                    print(f"Adapter name: {expert_index}")
                    print(f'Adapter name or path equipped : {lora_model_args.adapter_name_or_path}')
                    # Load model with the appropriate LoRA adapter
                    model = load_model(tokenizer, lora_model_args, finetuning_args, is_trainable=False)
                    # Initialize the custom trainer
                    trainer = CustomSeq2SeqTrainer(
                        model=model,
                        args=training_args,
                        finetuning_args=finetuning_args,
                        training_weight_ratio=data_args.weight_ratio,
                        data_collator=data_collator,
                        callbacks=callbacks,
                        **dataset_module,
                        **tokenizer_module,
                        **metric_module,
                    )
                # Perform the prediction
                predict_results = trainer.predict(adapter_relates_data, metric_key_prefix=f"predict_{expert_index}", **gen_kwargs)
                #trainer.print_result(adapter_relates_data,predict_results)
                trainer.log_metrics("predict", predict_results.metrics)
                trainer.save_metrics("predict", predict_results.metrics)
                trainer.save_predictions(adapter_relates_data, predict_results)
                print(f"Predictions for expert {expert_index} completed")
                all_predictions.append({
                    "expert_index": expert_index,
                    "predict_results":predict_results,
                })
        else:
            print(f'Normal Prediction Model Start')
            predict_results = trainer.predict(dataset_module['eval_dataset'], metric_key_prefix="predict", **gen_kwargs)
            if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
                predict_results.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(dataset_module["eval_dataset"], predict_results)


    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)



from datasets import Dataset

def get_adapter_index_dataset(original_dataset: Dataset, adapter_index: int) -> Dataset:
    """
    Extracts specific rows from an existing Dataset and creates a new Dataset.
    
    Args:
        original_dataset (Dataset): The input dataset to extract rows from.
        row_indices (list): A list of indices specifying the rows to extract.

    Returns:
        Dataset: A new dataset containing only the specified rows.
    """
    if "adapter_indices" not in original_dataset.column_names:
        raise ValueError("'adapter_indices' column is missing in the dataset.")

    # 打印数据集基本信息
    print(f"Total Dataset Rows: {original_dataset.num_rows}")
    print(f"Input Adapter Index: {adapter_index}")

    row_indices = []
    for i in range(original_dataset.num_rows):
        if original_dataset['adapter_indices'][i] == adapter_index:
            row_indices.append(i)
    
    print(f'Row Indices: {row_indices}')
    
    # 选择符合条件的行
    new_dataset = original_dataset.select(row_indices)
    
    # 删除 adapter_indices 列
    new_dataset = new_dataset.remove_columns(['adapter_indices'])
    print(f"filtered  Dataset Rows: {new_dataset.num_rows}")
    return new_dataset
