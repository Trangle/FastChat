# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json, jsonlines
import math
import pathlib
from multiprocessing import Pool
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


class Preprocessor(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.conv = get_conversation_template("vicuna")
        assert self.conv.sep_style == SeparatorStyle.ADD_COLON_TWO

        self.roles = {"human": self.conv.roles[0], "gpt": self.conv.roles[1]}
        self.turn_sep = self.conv.sep + self.roles.get("gpt") + ": "

    def apply_prompt_template(self, sources, systems=None) -> List[str]:
        conversations = []
        for i, source in enumerate(sources):
            if source[0]["from"] != "human":
                source = source[1:]

            self.conv.messages = []
            for j, sentence in enumerate(source):
                role = self.roles.get(sentence["from"])
                assert role == self.conv.roles[j % 2], f"{i}"
                self.conv.append_message(role, sentence["value"])
            if systems and systems[i]:
                self.conv.system = systems[i]
            prompt = self.conv.get_prompt()
            conversations.append(prompt)
        return conversations

    def tokenize_conversations(self, conversations) -> Tuple[List, List]:
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        return input_ids, targets

    def mask_targets(self, conversations, targets) -> List[int]:
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            turns = conversation.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(self.tokenizer(turn).input_ids)

                parts = turn.split(self.turn_sep)
                if len(parts) != 2:
                    break
                parts[0] += self.turn_sep

                # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not self.tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not self.tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                rank0_print(self.tokenizer.decode(z))
                exit()

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )
        return targets

    def preprocess(self, sources, **kwargs) -> Dict:
        systems = None if not kwargs else kwargs.get("systems", None)

        # If the data volume is small, process it directly in the main thread
        if len(sources) <= 1000:
            conversations = self.apply_prompt_template(sources, systems)
            input_ids, targets = self.tokenize_conversations(conversations)
            targets = self.mask_targets(conversations, targets)
        else:  # If the data volume is large, use multithreading for processing
            with Pool() as p:
                conversations = p.apply_async(
                    self.apply_prompt_template, (sources, systems)
                ).get()
                input_ids, targets = p.apply_async(
                    self.tokenize_conversations, (conversations)
                ).get()
                targets = p.apply_async(
                    self.mask_targets, (conversations, targets)
                ).get()
                p.close()
                p.join()

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, preprocessor: Preprocessor):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        systems = [example.get("system", "") for example in raw_data]
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocessor.preprocess(sources, systems=systems)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, preprocessor: Preprocessor):
        super(LazySupervisedDataset, self).__init__()
        self.preprocessor = preprocessor

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = self.preprocessor.preprocess(
            [self.raw_data[i]["conversations"]],
            systems=[self.raw_data[i].get("system", "")],
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    preprocessor: Preprocessor, data_args, train_ratio=0.98
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_ratio = min(train_ratio, 1.0)
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    data_path = data_args.data_path
    if data_path.endswith(".json"):
        raw_data = json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, mode="r") as reader:
            raw_data = [item for item in reader]

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * train_ratio)
    train_indices = perm[:split]
    if train_ratio < 1:
        eval_indices = perm[split:]
    else:
        # if train_ratio==1, we use 5% of data as eval data, make sure trainer will not throw error when eval data is empty
        eval_indices = perm[-int(len(perm) * 0.05) :]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, preprocessor=preprocessor)
    eval_dataset = dataset_cls(eval_raw_data, preprocessor=preprocessor)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    preprocessor = Preprocessor(tokenizer)
    data_module = make_supervised_data_module(
        preprocessor=preprocessor, data_args=data_args
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
