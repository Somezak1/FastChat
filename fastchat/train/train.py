# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
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

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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
    conv = get_conversation_template("vicuna")  # Conversation(name='vicuna_v1.1', ...)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # {"human": 'USER', "gpt": 'ASSISTANT'}

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

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  # sep: ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        # conversation:
        # "A chat between a curious user and an artificial intelligence assistant.
        # The assistant gives helpful, detailed, and polite answers to the user's questions.
        # USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).</s>
        # USER: What can you do? ASSISTANT: I can chat with you.</s>"

        # target:
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

        rounds = conversation.split(conv.sep2)  # conv.sep2: '</s>'
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # IGNORE_TOKEN_ID: -100
        for i, rou in enumerate(rounds):
            # i=0  rou:
            # "A chat between a curious user and an artificial intelligence assistant.
            # The assistant gives helpful, detailed, and polite answers to the user's questions.
            # USER: Who are you? ASSISTANT: I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."

            # i=1  rou: "USER: What can you do? ASSISTANT: I can chat with you."

            # i=2  rou: ""
            if rou == "":
                break

            parts = rou.split(sep)  # sep: ' ASSISTANT: '
            if len(parts) != 2:
                break
            parts[0] += sep  # parts[0]+parts[1]==rou
            round_len = len(tokenizer(rou).input_ids)  # i=0  round_len: 67
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # i=0  instruction_len: 42

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            # i=0
            # temp = tokenizer.tokenize(rou)
            # for i, j in zip(temp, target[1:len(temp) + 1]):
            #     print(i, '  ', j.item())
            #
            # ▁A - 100
            # ▁chat - 100
            # ▁between - 100
            # ......
            # ?    -100
            # ▁A - 100
            # SS - 100
            # IST - 100
            # ANT - 100
            # :    -100
            # ▁I 306
            # ▁am 626
            # ▁Vic 13423
            # una 4347
            # , 29892
            # ▁a 263
            # ▁language 4086
            # ▁model 1904
            # ▁trained 16370
            # ▁by 491
            # ▁research 5925
            # ers 414
            # ▁from 515
            # ▁Lar 8218
            # ge 479
            # ▁Model 8125
            # ▁Systems 23985
            # ▁Organ 9205
            # ization 2133
            # ▁( 313
            # L  29931
            # MS 4345
            # YS 21554
            # ). 467

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
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
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]  # [dict, dict, dict]
    eval_raw_data = [raw_data[i] for i in eval_indices]  # [dict, dict, dict]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir, # Path to a directory in which a downloaded pretrained model configuration
        # should be cached if the standard cache should not be used.
        model_max_length=training_args.model_max_length,  # Will be passed to the Tokenizer __init__() method.
        padding_side="right",  # Will be passed to the Tokenizer __init__() method.
        use_fast=False, # Use a fast Rust-based tokenizer if it is supported for a given model.
        # If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # two ways of saving weight and other things under deepspeed mode
    # way1:
    if trainer.hf_deepspeed_config_orig.is_zero3():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
            trainer._save(training_args.output_dir, state_dict=state_dict)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # way2:
    # save_zero_three_model(trainer, training_args, trainer.hf_deepspeed_config_orig.is_zero3())


if __name__ == "__main__":
    train()
