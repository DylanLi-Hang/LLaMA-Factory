# Copyright 2024 the LlamaFactory team.
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

import os
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers.utils.versions import require_version

from ..extras.constants import FILEEXT2TYPE
from ..extras.logging import get_logger
from ..extras.misc import has_tokenized_data
from .aligner import align_dataset
from .data_utils import merge_dataset, split_dataset
from .parser import get_dataset_list
from .preprocess import get_preprocess_and_print_func
from llama import Dialog, Llama

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .template import Template


logger = get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            trust_remote_code=True,
        )
    
    if 'dataset_label' in dataset[0].keys():
        dataset_attr.dataset_label = 'dataset_label'

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args, is_eval)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args, is_eval))

    return merge_dataset(datasets, data_args, seed=training_args.seed)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None
    if training_args.predict_with_generate and is_eval:
        if data_args.dynamic_eval and stage == "sft":
            print(f'Load base selector model')
            selector_model = _load_modelevaluate_adapter_index()
            print(f'Load base selector model Successfully')
            
            def add_adapter_index_to_dataset(dataset, selector_model):
                import time
                adapter_indexs = []
                start_time = time.time()
                for i in range(len(dataset["_prompt"])):
                    s_t = time.time()
                    adapter_index = evaluate_adapter_index(selector_model, dataset["_prompt"][i])
                    adapter_indexs.append(adapter_index)
                    e_t = time.time()
                    print(f"Processing example {i + 1}/{len(dataset)}, Adapter Index: {adapter_index}, Process time is : {e_t - s_t:.2f} seconds")
                    #check_and_print_devices(dataset["_prompt"][i], adapter_index)
                print(f'len of the adapter indexs added: {len(adapter_indexs)}')
                dataset = dataset.add_column("_adapter_index", adapter_indexs)
                end_time = time.time()
                print(f"Dynamic adapter index computation completed in {end_time - start_time:.2f} seconds")
                print(f'Added dataset: {dataset}')
                return dataset

            print("Processing dynamic eval dataset with for loop...")
            dataset = add_adapter_index_to_dataset(dataset, selector_model)

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    column_names = list(next(iter(dataset)).keys())
    print(f'column_names: {column_names}')

    try:
        dataset = dataset.map(
            preprocess_func,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,  
            remove_columns=column_names,
            **kwargs,
        )
    except Exception as e:
        print(f"Error during dataset.map: {e}")
        raise
    
    if training_args.should_log and not data_args.dynamic_eval :
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset



def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]

            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")
    
    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage, is_eval=False)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage, is_eval=True)

        print("dataset: ", dataset)
        print("eval_dataset: ", eval_dataset)
        
    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )
        print('preprocessed_dataset: ', dataset)
        print('preprocessed_eval_dataset: ', eval_dataset)

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module


def _load_modelevaluate_adapter_index(model_path='Meta-Llama-3-8B-Instruct'):
    import os
    tokenizer_path = os.path.join(model_path, 'tokenizer.model')
    
    adapter_selector = Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=8192,
            max_batch_size=8,
        )

    return adapter_selector

def evaluate_adapter_index(model,input_question):
    # Load the prompt
    system_prompt = """
            You have two experts available:
                You have three options:

                1. Expert A: Specializes in scientific concepts and problem-solving across various disciplines.

                2. Expert B: Specializes in basic mathematical reasoning and grade school math word problems.

                0. Neither expert is suitable.

                Please select the option that best fits the situation and output only the corresponding number: 1, 2, or 0.

                Respond with just the number, no additional text or explanation."

                Question:            
                        """ 
    import re
    def extract_first_number(responses):
        for response in responses:
            # Try to extract the first number from the content of the first item in the response
            first_content = response['generation']['content']
            
            # Use regular expression to find the first number in the string
            match = re.search(r'\d+', first_content)
            
            return int(match.group(0)) if match else 0
    
    dialogs: List[Dialog] = []
    if isinstance(input_question, list) and len(input_question) > 0:
        input_question = input_question[0].get('content', '')
    dialogs.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_question},
            ])
    selector_result = model.chat_completion(
                    dialogs,
                    max_gen_len=None,
                    temperature=0.6,
                    top_p=0.9,
                )
    index = extract_first_number(selector_result)
    
    return index

import torch
def check_and_print_devices(prompt, index):
    """
    打印并检查 prompt 和 index 的设备。
    如果设备不一致，打印警告信息。
    """
    # 获取 prompt 的设备
    if isinstance(prompt, torch.Tensor):
        prompt_device = prompt.device
    else:
        prompt_device = "CPU"

    # 获取 index 的设备
    if isinstance(index, torch.Tensor):
        index_device = index.device
    else:
        index_device = "CPU"

    # 打印设备信息
    print(f"Prompt Device: {prompt_device}")
    print(f"Index Device: {index_device}")

    # 检查设备是否一致
    if prompt_device != index_device:
        print(f"WARNING: Devices are inconsistent! Prompt is on {prompt_device}, but Index is on {index_device}.")
    else:
        print("Devices are consistent.")
