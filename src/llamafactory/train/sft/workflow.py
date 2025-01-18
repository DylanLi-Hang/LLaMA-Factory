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
    
    #print("data_args:", data_args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    #if not training_args.do_predict:  # 在 do predict 的时候不需要加载模型
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
    #if not training_args.do_predict: # 在 do predict 的时候不需要加载模型
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
        # Step 1: 加载基础模型
        print(f' Start to load the Base model')
        base_model_args = copy.deepcopy(model_args)
        base_model_args.adapter_name_or_path = None  # 只load base model
        base_model = load_model(tokenizer, base_model_args, finetuning_args, is_trainable=False)
        print(f' Base Model loaded successfully')
        base_trainer = CustomSeq2SeqTrainer(
            model=base_model,
            args=training_args,
            finetuning_args=finetuning_args,
            training_weight_ratio=data_args.weight_ratio,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
        base_predict_results = base_trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="base_predict", **gen_kwargs)
        base_predictions = base_predict_results.predictions  # 拿到这些index
        print(f' Base model predictions obtained')

        # Step 2: 根据基础模型的预测结果分组输入数据
        grouped_inputs = {}
        for i, input_data in enumerate(dataset_module["eval_dataset"]):
            adapter_index = int(base_predictions[i])  # 根据基础模型的预测结果确定适配器索引
            if adapter_index not in grouped_inputs:
                grouped_inputs[adapter_index] = []
            grouped_inputs[adapter_index].append((i, input_data))  # 保存原始索引和输入数据

        # 打印每个类别的数量和总数量
        total_count = 0
        for adapter_index, inputs in grouped_inputs.items():
            count = len(inputs)
            total_count += count
            print(f'Adapter index {adapter_index} has {count} prompts.')
        print(f'Total number of prompts: {total_count}')

        # Step 3: 对每个分组加载对应的适配器并进行预测
        final_predictions = [None] * len(dataset_module["eval_dataset"])  # 初始化最终预测结果列表
        for adapter_index, inputs in grouped_inputs.items():
            lora_model_args = copy.deepcopy(model_args)
            lora_model_args.adapter_name_or_path = model_args.adapter_name_or_path[:adapter_index]
            print(f'Evaluatuion with adapter_index: {adapter_index}. \n')
            print(f'LoRA Adapeters args: {lora_model_args.adapter_name_or_path}')
            model_with_adapter = load_model(tokenizer, lora_model_args, finetuning_args, is_trainable=False)
            adapter_trainer = CustomSeq2SeqTrainer(
                model=model_with_adapter,
                args=training_args,
                finetuning_args=finetuning_args,
                training_weight_ratio=data_args.weight_ratio,
                data_collator=data_collator,
                callbacks=callbacks,
                **dataset_module,
                **tokenizer_module,
                **metric_module,
            )
            input_dataset = [input_data for _, input_data in inputs]  # 提取输入数据
            predict_results = adapter_trainer.predict(input_dataset, metric_key_prefix=f"predict_{adapter_index}", **gen_kwargs)
            for (original_index, _), prediction in zip(inputs, predict_results.predictions):
                final_predictions[original_index] = prediction  # 按原始顺序保存预测结果

        # Step 4: 保存最终预测结果
        with open(f"{training_args.output_dir}/final_predictions.json", "w") as f:
            json.dump(final_predictions, f)

        # Step 5: 记录和保存最终预测结果的指标
        final_metrics = {"final_predictions": final_predictions}
        trainer.log_metrics("final_predict", final_metrics)
        trainer.save_metrics("final_predict", final_metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], final_predictions)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
