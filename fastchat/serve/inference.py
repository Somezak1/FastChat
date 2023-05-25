"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Iterable, Optional
import sys
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.model.chatglm_model import chatglm_generate_stream


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    # temperature: 默认是0.7, 但此处为了复现改为0
    # repetition_penalty: 1.0
    # top_p: 1.0
    # top_k: -1

    processor_list = LogitsProcessorList()

    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    # temperature: The value used to module the logits distribution.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))

    # repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    # top_p: If set to < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))

    # top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    '''
    推断的主要过程:
    多轮对话时, 将初始prompt, 过往的多轮提问及回答拼接在一起输入模型, 从输出的概率向量中随机采样一个作为下个输出字符
    连续不停地生成下个字符, 同时每两步保存一下当前的生成信息, 并判断是否终止. 最后将生成信息返回, 用于在命令行中输出生成信息
    '''

    # prompt:
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    "### Human: What are the key differences between renewable and non-renewable energy sources?"
    "### Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:"
    "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out."
    "2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects."
    "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources."
    "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources."
    "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible."
    "6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability."
    "### Human: Who are you"
    "### Assistant:"
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))  # 默认是0.7, 但此处为了复现改为0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # repetition_penalty: 1.0
    top_p = float(params.get("top_p", 1.0))  # top_p: 1.0
    top_k = int(params.get("top_k", -1))  # top_k: -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))  # max_new_tokens: 512
    stop_str = params.get("stop", None)  # stop_str: '###'
    echo = bool(params.get("echo", True))  # echo: False
    stop_token_ids = params.get("stop_token_ids", None) or []  # stop_token_ids: [2]
    stop_token_ids.append(tokenizer.eos_token_id)  # tokenizer.eos_token_id: 2

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)  # input_echo_len: 404
    output_ids = list(input_ids)

    # model.config (模型权重路径下的config.json文件):
    # LlamaConfig {
    #     "_name_or_path": "llama-7b-hf",
    #     "architectures": [
    #         "LlamaForCausalLM"
    #     ],
    #     "bos_token_id": 1,
    #     "eos_token_id": 2,
    #     "hidden_act": "silu",
    #     "hidden_size": 4096,
    #     "initializer_range": 0.02,
    #     "intermediate_size": 11008,
    #     "max_position_embeddings": 2048,
    #     "model_type": "llama",
    #     "num_attention_heads": 32,
    #     "num_hidden_layers": 32,
    #     "pad_token_id": 0,
    #     "rms_norm_eps": 1e-06,
    #     "tie_word_embeddings": false,
    #     "torch_dtype": "float16",
    #     "transformers_version": "4.28.1",
    #     "use_cache": true,
    #     "vocab_size": 32000
    # }

    if model.config.is_encoder_decoder:  # model.config.is_encoder_decoder: False
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8  # max_src_len: 1528  模型能够接受的最长输入序列的长度

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    for i in range(max_new_tokens):  # 模型在一次回答中能够输出的最长回答是512个token
        if i == 0:  # 首次进入
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)  # input shape: (1, 404)
                logits = out.logits  # logits shape: (1, 404, 32000), 404是当前prompt的token数量
            past_key_values = out.past_key_values  # ((tensor, tensor)_1, ..., (tensor, tensor)_32)  第一次循环时tensor shape: (1, 32, 404, 128)
        else:  # 非首次进入
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits  # logits shape: (1, 1, 32000)
            past_key_values = out.past_key_values  # ((tensor, tensor)_1, ..., (tensor, tensor)_32)  第二次循环时tensor shape: (1, 32, 405, 128)

        if logits_processor:
            if repetition_penalty > 1.0:  # repetition_penalty: 1.0
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]  # last_token_logits shape: (32000, )
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:  # stop_token_ids: [2]
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:  # stream_interval: 2
            if echo:  # False
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if stop_str:  # stop_str: '###'
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)  # rfind_start: 0
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,  # 'cuda'
    num_gpus: int,  # 1
    max_gpu_memory: str,  # None
    load_8bit: bool,  # False
    cpu_offloading: bool,  # False
    conv_template: Optional[str],  # None
    temperature: float,  # 默认是0.7, 但此处为了复现改为0
    max_new_tokens: int,  # 512
    chatio: ChatIO,
    debug: bool,  # False
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat, 根据模型权重路径的名称, 获取一个与之匹配的Conversation模板
    if conv_template:
        conv = get_conv_template(conv_template)
    else:
        conv = get_conversation_template(model_path)
    # 由于调试时使用的权重路径名称是llama, 所以匹配到如下的Conversation对象
    # Conversation(
    #     name="one_shot",
    #     system="A chat between a curious human and an artificial intelligence assistant. "
    #     "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    #     roles=("Human", "Assistant"),
    #     messages=(
    #         (
    #             "Human",
    #             "What are the key differences between renewable and non-renewable energy sources?",
    #         ),
    #         (
    #             "Assistant",
    #             "Renewable energy sources are those that can be replenished naturally in a relatively "
    #             "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
    #             "Non-renewable energy sources, on the other hand, are finite and will eventually be "
    #             "depleted, such as coal, oil, and natural gas. Here are some key differences between "
    #             "renewable and non-renewable energy sources:\n"
    #             "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
    #             "energy sources are finite and will eventually run out.\n"
    #             "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
    #             "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
    #             "and other negative effects.\n"
    #             "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
    #             "have lower operational costs than non-renewable sources.\n"
    #             "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
    #             "locations than non-renewable sources.\n"
    #             "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
    #             "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
    #             "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
    #             "non-renewable sources are not, and their depletion can lead to economic and social instability.",
    #         ),
    #     ),
    #     offset=2,
    #     sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    #     sep="\n### ",
    #     stop_str="###",
    # )

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])  # roles: ("Human", "Assistant")  # 手动输入: 'Who are you'
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
            prompt = conv.messages[conv.offset :]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        # 首次循环时prompt如下
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        "### Human: What are the key differences between renewable and non-renewable energy sources?"
        "### Assistant: Renewable energy sources are those that can be replenished naturally in a relatively short amount of time, such as solar, wind, hydro, geothermal, and biomass. Non-renewable energy sources, on the other hand, are finite and will eventually be depleted, such as coal, oil, and natural gas. Here are some key differences between renewable and non-renewable energy sources:"
        "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable energy sources are finite and will eventually run out."
        "2. Environmental impact: Renewable energy sources have a much lower environmental impact than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, and other negative effects."
        "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically have lower operational costs than non-renewable sources."
        "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote locations than non-renewable sources."
        "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different situations and needs, while non-renewable sources are more rigid and inflexible."
        "6. Sustainability: Renewable energy sources are more sustainable over the long term, while non-renewable sources are not, and their depletion can lead to economic and social instability."
        "### Human: Who are you"
        "### Assistant:"
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,  # stop_str: "###",
            "stop_token_ids": conv.stop_token_ids,  # stop_token_ids: None
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
