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

import pathlib
from multiprocessing import Pool
from typing import List

import torch
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    Preprocessor,
    make_supervised_data_module,
    rank0_print,
    safe_save_model_for_hf_trainer,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None


class BaichuanPreprocessor(Preprocessor):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(BaichuanPreprocessor, self).__init__(tokenizer)

    def mask_targets(self, conversations, targets) -> List[int]:
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            turns = conversation.split(self.conv.sep2)
            cur_len = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(self.tokenizer(turn + self.conv.sep2).input_ids)

                parts = turn.split(self.turn_sep)
                if len(parts) != 2:
                    break
                parts[0] += self.turn_sep

                # "-1" is hardcoded for the Baichuan tokenizer to make the offset correct.
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                rank0_print(self.tokenizer.decode(z))

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
        return targets


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    # Tie the weights
    model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # NOTE: if the token_id exceed the vocab_size will cause failing in training process! we need add special config and resize the embedding size!
    tokenizer.pad_token = tokenizer.unk_token
    print(f"tokens len: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    preprocessor = Preprocessor(tokenizer)
    data_module = make_supervised_data_module(
        preprocessor=preprocessor, train_ratio=0.98, data_args=data_args
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
