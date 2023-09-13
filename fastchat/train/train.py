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
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# IGNORE_TOKEN_ID: -100


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


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(trainer, training_args, zero_stage_3):
    os.makedirs(training_args.output_dir, exist_ok=True)
    model_ema = trainer.model
    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema

    if not zero_stage_3:
        if training_args.local_rank == 0:
            state_dict = model_to_save.state_dict()
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            trainer._save(training_args.output_dir, state_dict=cpu_state_dict)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if training_args.local_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p

        if training_args.local_rank == 0:
            trainer._save(training_args.output_dir, state_dict=output_state_dict)
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
        del output_state_dict


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # We adjust the training loss to account for multi-round conversations and
    # compute the fine-tuning loss solely on the chatbot's output.

    '''
    sources:
    [[
        {'from': 'human', 'value': 'Who are you?'},
        {'from': 'gpt', 'value': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).'},
        {'from': 'human', 'value': 'What can you do?'},
        {'from': 'gpt', 'value': 'I can chat with you.'}
    ]]
    '''
    # 注意, 这里使用的是vicuna的对话模板
    conv_name = "vicuna_v1.1"
    conv = get_conversation_template(conv_name)
    if conv_name == "llama-2":
        conv.system_message = "You are a helpful assistant. 你是一个乐于助人的助手。"
        conv.sep2 = "</s>"
    # conv: Conversation(name='vicuna_v1.1', ...)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # roles: {"human": 'USER', "gpt": 'ASSISTANT'}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # source:
        # [
        #     {'from': 'human', 'value': 'Who are you?'},
        #     {'from': 'gpt', 'value': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).'},
        #     {'from': 'human', 'value': 'What can you do?'},
        #     {'from': 'gpt', 'value': 'I can chat with you.'}
        # ]
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        # conv.messages:
        # [
        #     ['USER', 'Who are you?'],
        #     ['ASSISTANT', 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).'],
        #     ['USER', 'What can you do?'],
        #     ['ASSISTANT', 'I can chat with you.']
        # ]

        # conversations:
        # [
        #     "A chat between a curious user and an artificial intelligence assistant.
        #     The assistant gives helpful, detailed, and polite answers to the user's questions.
        #     USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).</s>
        #     USER: What can you do? ASSISTANT: I can chat with you.</s>"
        # ]

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    # input_ids: (演示用的max length是128)
    # input_ids的第一位是额外添加的句子开始符的id
    # tensor([[1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116,
    #          21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892,
    #          322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155,
    #          29889, 3148, 1001, 29901, 11644, 526, 366, 29973, 319, 1799,
    #          9047, 13566, 29901, 306, 626, 13423, 4347, 29892, 263, 4086,
    #          1904, 16370, 491, 5925, 414, 515, 8218, 479, 8125, 23985,
    #          9205, 2133, 313, 29931, 4345, 21554, 467, 2, 3148, 1001,
    #          29901, 1724, 508, 366, 437, 29973, 319, 1799, 9047, 13566,
    #          29901, 306, 508, 13563, 411, 366, 29889, 2, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]])
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    if conv_name == 'vicuna_v1.1':
        sep = conv.sep + conv.roles[1] + ": "
    elif conv_name == 'llama-2':
        sep = f' {conv.roles[1]} '
    # sep: ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        # conversation: 这次会话按对话模板处理后的文本
        # "A chat between a curious user and an artificial intelligence assistant.
        # The assistant gives helpful, detailed, and polite answers to the user's questions.
        # USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).</s>
        # USER: What can you do? ASSISTANT: I can chat with you.</s>"

        # target: conversation tokenize 同时 padding/truncate
        # tensor([1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116,
        #          21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892,
        #          322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155,
        #          29889, 3148, 1001, 29901, 11644, 526, 366, 29973, 319, 1799,
        #          9047, 13566, 29901, 306, 626, 13423, 4347, 29892, 263, 4086,
        #          1904, 16370, 491, 5925, 414, 515, 8218, 479, 8125, 23985,
        #          9205, 2133, 313, 29931, 4345, 21554, 467, 2, 3148, 1001,
        #          29901, 1724, 508, 366, 437, 29973, 319, 1799, 9047, 13566,
        #          29901, 306, 508, 13563, 411, 366, 29889, 2, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0])

        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # tokenizer.pad_token_id: 0
        # target.ne(tokenizer.pad_token_id):
        # tensor([True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False, False, False,
        #         False, False, False, False, False, False, False, False])
        # total_len: 88

        turns = conversation.split(conv.sep2)
        # conv.sep2: '</s>'
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        # IGNORE_TOKEN_ID: -100
        for i, turn in enumerate(turns):
            # i=0  turn:
            # "A chat between a curious user and an artificial intelligence assistant.
            # The assistant gives helpful, detailed, and polite answers to the user's questions.
            # USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."

            # i=1  turn: "USER: What can you do? ASSISTANT: I can chat with you."

            # i=2  turn: ""

            if turn == "":
                break
            # -1 是为了去除tokenizer默认加在开头的'<s>'
            turn_len = len(tokenizer(turn+conv.sep2).input_ids) - 1

            parts = turn.split(sep)
            # sep: ' ASSISTANT: '
            if len(parts) != 2:
                break
            parts[0] += sep
            # parts[0]+parts[1]==turn
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            # conv_name=="vicuna_v1.1" 时 sep = ' ASSISTANT: '
            # conv_name=="llama-2" 时 sep = ' [/INST] '
            # 不管conv_name等于哪个, 其sep的末尾都是一个' ', 在sep后面不接字符时, ' '会被tokenizer单独看待; 如果sep后面接了字符比如'I', 则空格会被算成'_I'
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            # i=0  instruction_len: 42

            # - 2 的由来, 用于去除首尾的token
            #     0    id:        1   token: <s>
            #     1    id:      319   token: ▁A
            #     2    id:    13563   token: ▁chat
            #     3    id:     1546   token: ▁between
            #     4    id:      263   token: ▁a
            #     5    id:    12758   token: ▁curious
            #     6    id:     1404   token: ▁user
            #     7    id:      322   token: ▁and
            #     8    id:      385   token: ▁an
            #     9    id:    23116   token: ▁artificial
            #    10    id:    21082   token: ▁intelligence
            #    11    id:    20255   token: ▁assistant
            #    12    id:    29889   token: .
            #    13    id:      450   token: ▁The
            #    14    id:    20255   token: ▁assistant
            #    15    id:     4076   token: ▁gives
            #    16    id:     8444   token: ▁helpful
            #    17    id:    29892   token: ,
            #    18    id:    13173   token: ▁detailed
            #    19    id:    29892   token: ,
            #    20    id:      322   token: ▁and
            #    21    id:     1248   token: ▁pol
            #    22    id:      568   token: ite
            #    23    id:     6089   token: ▁answers
            #    24    id:      304   token: ▁to
            #    25    id:      278   token: ▁the
            #    26    id:     1404   token: ▁user
            #    27    id:    29915   token: '
            #    28    id:    29879   token: s
            #    29    id:     5155   token: ▁questions
            #    30    id:    29889   token: .
            #    31    id:     3148   token: ▁US
            #    32    id:     1001   token: ER
            #    33    id:    29901   token: :
            #    34    id:    11644   token: ▁Who
            #    35    id:      526   token: ▁are
            #    36    id:      366   token: ▁you
            #    37    id:    29973   token: ?
            #    38    id:      319   token: ▁A
            #    39    id:     1799   token: SS
            #    40    id:     9047   token: IST
            #    41    id:    13566   token: ANT
            #    42    id:    29901   token: :
            #    43    id:    29871   token: ▁

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 此时按理说cur_len应该落在非padding内容的末尾
                # 如果没落在末尾就会把整个target里的内容全部忽略
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

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

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        # ret['input_ids']:
        # tensor([[1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116,
        #          21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892,
        #          322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155,
        #          29889, 3148, 1001, 29901, 11644, 526, 366, 29973, 319, 1799,
        #          9047, 13566, 29901, 306, 626, 13423, 4347, 29892, 263, 4086,
        #          1904, 16370, 491, 5925, 414, 515, 8218, 479, 8125, 23985,
        #          9205, 2133, 313, 29931, 4345, 21554, 467, 2, 3148, 1001,
        #          29901, 1724, 508, 366, 437, 29973, 319, 1799, 9047, 13566,
        #          29901, 306, 508, 13563, 411, 366, 29889, 2, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0]])

        # ret['labels']:
        # tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, 306, 626, 13423, 4347, 29892, 263, 4086,
        #          1904, 16370, 491, 5925, 414, 515, 8218, 479, 8125, 23985,
        #          9205, 2133, 313, 29931, 4345, 21554, 467, 2, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, 306, 508, 13563, 411, 366, 29889, 2, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #          -100, -100, -100, -100, -100, -100, -100, -100]])

        # ret['attention_mask']:
        # tensor([[True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, True, True,
        #          True, True, True, True, True, True, True, True, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False]])

        # for i, (id, token, label, attn) in enumerate(zip(ret['input_ids'], tokenizer.convert_ids_to_tokens(ret['input_ids']), ret['labels'], ret['attention_mask'])):
        #     print(f"{i:5d}    id: {id:8d}   token: {token:15s}   label: {label:8d}  attn: {attn}")
        #     0    id:        1   token: <s>               label:     -100  attn: True
        #     1    id:      319   token: ▁A                label:     -100  attn: True
        #     2    id:    13563   token: ▁chat             label:     -100  attn: True
        #     3    id:     1546   token: ▁between          label:     -100  attn: True
        #     4    id:      263   token: ▁a                label:     -100  attn: True
        #     5    id:    12758   token: ▁curious          label:     -100  attn: True
        #     6    id:     1404   token: ▁user             label:     -100  attn: True
        #     7    id:      322   token: ▁and              label:     -100  attn: True
        #     8    id:      385   token: ▁an               label:     -100  attn: True
        #     9    id:    23116   token: ▁artificial       label:     -100  attn: True
        #    10    id:    21082   token: ▁intelligence     label:     -100  attn: True
        #    11    id:    20255   token: ▁assistant        label:     -100  attn: True
        #    12    id:    29889   token: .                 label:     -100  attn: True
        #    13    id:      450   token: ▁The              label:     -100  attn: True
        #    14    id:    20255   token: ▁assistant        label:     -100  attn: True
        #    15    id:     4076   token: ▁gives            label:     -100  attn: True
        #    16    id:     8444   token: ▁helpful          label:     -100  attn: True
        #    17    id:    29892   token: ,                 label:     -100  attn: True
        #    18    id:    13173   token: ▁detailed         label:     -100  attn: True
        #    19    id:    29892   token: ,                 label:     -100  attn: True
        #    20    id:      322   token: ▁and              label:     -100  attn: True
        #    21    id:     1248   token: ▁pol              label:     -100  attn: True
        #    22    id:      568   token: ite               label:     -100  attn: True
        #    23    id:     6089   token: ▁answers          label:     -100  attn: True
        #    24    id:      304   token: ▁to               label:     -100  attn: True
        #    25    id:      278   token: ▁the              label:     -100  attn: True
        #    26    id:     1404   token: ▁user             label:     -100  attn: True
        #    27    id:    29915   token: '                 label:     -100  attn: True
        #    28    id:    29879   token: s                 label:     -100  attn: True
        #    29    id:     5155   token: ▁questions        label:     -100  attn: True
        #    30    id:    29889   token: .                 label:     -100  attn: True
        #    31    id:     3148   token: ▁US               label:     -100  attn: True
        #    32    id:     1001   token: ER                label:     -100  attn: True
        #    33    id:    29901   token: :                 label:     -100  attn: True
        #    34    id:    11644   token: ▁Who              label:     -100  attn: True
        #    35    id:      526   token: ▁are              label:     -100  attn: True
        #    36    id:      366   token: ▁you              label:     -100  attn: True
        #    37    id:    29973   token: ?                 label:     -100  attn: True
        #    38    id:      319   token: ▁A                label:     -100  attn: True
        #    39    id:     1799   token: SS                label:     -100  attn: True
        #    40    id:     9047   token: IST               label:     -100  attn: True
        #    41    id:    13566   token: ANT               label:     -100  attn: True
        #    42    id:    29901   token: :                 label:     -100  attn: True
        #    43    id:      306   token: ▁I                label:      306  attn: True
        #    44    id:      626   token: ▁am               label:      626  attn: True
        #    45    id:    13423   token: ▁Vic              label:    13423  attn: True
        #    46    id:     4347   token: una               label:     4347  attn: True
        #    47    id:    29892   token: ,                 label:    29892  attn: True
        #    48    id:      263   token: ▁a                label:      263  attn: True
        #    49    id:     4086   token: ▁language         label:     4086  attn: True
        #    50    id:     1904   token: ▁model            label:     1904  attn: True
        #    51    id:    16370   token: ▁trained          label:    16370  attn: True
        #    52    id:      491   token: ▁by               label:      491  attn: True
        #    53    id:     5925   token: ▁research         label:     5925  attn: True
        #    54    id:      414   token: ers               label:      414  attn: True
        #    55    id:      515   token: ▁from             label:      515  attn: True
        #    56    id:     8218   token: ▁Lar              label:     8218  attn: True
        #    57    id:      479   token: ge                label:      479  attn: True
        #    58    id:     8125   token: ▁Model            label:     8125  attn: True
        #    59    id:    23985   token: ▁Systems          label:    23985  attn: True
        #    60    id:     9205   token: ▁Organ            label:     9205  attn: True
        #    61    id:     2133   token: ization           label:     2133  attn: True
        #    62    id:      313   token: ▁(                label:      313  attn: True
        #    63    id:    29931   token: L                 label:    29931  attn: True
        #    64    id:     4345   token: MS                label:     4345  attn: True
        #    65    id:    21554   token: YS                label:    21554  attn: True
        #    66    id:      467   token: ).                label:      467  attn: True
        #    67    id:        2   token: </s>              label:        2  attn: True
        #    68    id:     3148   token: ▁US               label:     -100  attn: True
        #    69    id:     1001   token: ER                label:     -100  attn: True
        #    70    id:    29901   token: :                 label:     -100  attn: True
        #    71    id:     1724   token: ▁What             label:     -100  attn: True
        #    72    id:      508   token: ▁can              label:     -100  attn: True
        #    73    id:      366   token: ▁you              label:     -100  attn: True
        #    74    id:      437   token: ▁do               label:     -100  attn: True
        #    75    id:    29973   token: ?                 label:     -100  attn: True
        #    76    id:      319   token: ▁A                label:     -100  attn: True
        #    77    id:     1799   token: SS                label:     -100  attn: True
        #    78    id:     9047   token: IST               label:     -100  attn: True
        #    79    id:    13566   token: ANT               label:     -100  attn: True
        #    80    id:    29901   token: :                 label:     -100  attn: True
        #    81    id:      306   token: ▁I                label:      306  attn: True
        #    82    id:      508   token: ▁can              label:      508  attn: True
        #    83    id:    13563   token: ▁chat             label:    13563  attn: True
        #    84    id:      411   token: ▁with             label:      411  attn: True
        #    85    id:      366   token: ▁you              label:      366  attn: True
        #    86    id:    29889   token: .                 label:    29889  attn: True
        #    87    id:        2   token: </s>              label:        2  attn: True
        #    88    id:        0   token: <unk>             label:     -100  attn: False
        #    89    id:        0   token: <unk>             label:     -100  attn: False
        #    90    id:        0   token: <unk>             label:     -100  attn: False
        #    91    id:        0   token: <unk>             label:     -100  attn: False
        #    92    id:        0   token: <unk>             label:     -100  attn: False
        #    93    id:        0   token: <unk>             label:     -100  attn: False
        #    94    id:        0   token: <unk>             label:     -100  attn: False
        #    95    id:        0   token: <unk>             label:     -100  attn: False
        #    96    id:        0   token: <unk>             label:     -100  attn: False
        #    97    id:        0   token: <unk>             label:     -100  attn: False
        #    98    id:        0   token: <unk>             label:     -100  attn: False
        #    99    id:        0   token: <unk>             label:     -100  attn: False
        #   100    id:        0   token: <unk>             label:     -100  attn: False
        #   101    id:        0   token: <unk>             label:     -100  attn: False
        #   102    id:        0   token: <unk>             label:     -100  attn: False
        #   103    id:        0   token: <unk>             label:     -100  attn: False
        #   104    id:        0   token: <unk>             label:     -100  attn: False
        #   105    id:        0   token: <unk>             label:     -100  attn: False
        #   106    id:        0   token: <unk>             label:     -100  attn: False
        #   107    id:        0   token: <unk>             label:     -100  attn: False
        #   108    id:        0   token: <unk>             label:     -100  attn: False
        #   109    id:        0   token: <unk>             label:     -100  attn: False
        #   110    id:        0   token: <unk>             label:     -100  attn: False
        #   111    id:        0   token: <unk>             label:     -100  attn: False
        #   112    id:        0   token: <unk>             label:     -100  attn: False
        #   113    id:        0   token: <unk>             label:     -100  attn: False
        #   114    id:        0   token: <unk>             label:     -100  attn: False
        #   115    id:        0   token: <unk>             label:     -100  attn: False
        #   116    id:        0   token: <unk>             label:     -100  attn: False
        #   117    id:        0   token: <unk>             label:     -100  attn: False
        #   118    id:        0   token: <unk>             label:     -100  attn: False
        #   119    id:        0   token: <unk>             label:     -100  attn: False
        #   120    id:        0   token: <unk>             label:     -100  attn: False
        #   121    id:        0   token: <unk>             label:     -100  attn: False
        #   122    id:        0   token: <unk>             label:     -100  attn: False
        #   123    id:        0   token: <unk>             label:     -100  attn: False
        #   124    id:        0   token: <unk>             label:     -100  attn: False
        #   125    id:        0   token: <unk>             label:     -100  attn: False
        #   126    id:        0   token: <unk>             label:     -100  attn: False
        #   127    id:        0   token: <unk>             label:     -100  attn: False
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    # 注意用该代码训练其他模型时, 需要自定义Model/Tokenizer/Conversation以及训练代码, 这和推断一样
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
        # cache_dir: Path to a directory in which a downloaded pretrained model configuration
        # should be cached if the standard cache should not be used.
        model_max_length=training_args.model_max_length,
        # model_max_length: Will be passed to the Tokenizer __init__() method.
        padding_side="right",
        # padding_side: Will be passed to the Tokenizer __init__() method.
        use_fast=False,
        # use_fast: Use a fast Rust-based tokenizer if it is supported for a given model.
        # If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
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
