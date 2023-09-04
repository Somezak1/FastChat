"""Inference for FastChat models."""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
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
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


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
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
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

    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    # stop_str也部分扮演了停止符的作用
    stop_token_ids = params.get("stop_token_ids", None) or []
    # stop_token_ids: [2]
    stop_token_ids.append(tokenizer.eos_token_id)
    # tokenizer.eos_token_id: 2
    # tokenizer.decode(tokenizer.eos_token_id): '</s>'

    # ①
    # curl http://localhost:8002/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    # 	  "model": "OriginOne",
    #     "messages": [{"content":"你了解冰鉴吗？", "role":"user"}]
    # }'
    # params:
    # {
    #             'model': ...,
    #             'prompt': ...,
    #             'temperature': 0.7,
    #             'top_p': 1.0,
    #             'max_new_tokens': 512,
    #             'stream': False,
    #             'stop': '###',
    #             'stop_token_ids': None
    #             'echo': False,
    # }
    # 补充:
    # repetition_penalty: 1.0
    # top_k: -1

    # ②
    # curl http://localhost:8002/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    # 	  "model": "vicuna-7b-v1.3",
    #     "messages": [{"content":"你了解冰鉴吗？", "role":"user"}]
    # }'
    # params:
    # {
    #             'model': ...,
    #             'prompt': ...,
    #             'temperature': 0.7,
    #             'top_p': 1.0,
    #             'max_new_tokens': 512,
    #             'stream': False,
    #             'stop': None,
    #             'stop_token_ids': None
    #             'echo': False,
    # }
    # 补充:
    # repetition_penalty: 1.0
    # top_k: -1

    # ③
    # python3 -m fastchat.serve.cli --model-path /data1/csw_model_weights/OriginOne/ --temperature 0
    # params:
    # {
    #             "model": ...,
    #             "prompt": ...,
    #             "temperature": 0.0,
    #             "repetition_penalty": 1.0,
    #             "max_new_tokens": 512,
    #             "stop": "###",
    #             "stop_token_ids": None,
    #             "echo": False,
    # }
    # 补充:
    # top_p: 1.0
    # top_k: -1

    # ④
    # python3 -m fastchat.serve.cli --model-path /data1/csw_model_weights/vicuna-7b-v1.3/ --temperature 0
    # params:
    # {
    #             "model": ...,
    #             "prompt": ...,
    #             "temperature": 0.0,
    #             "repetition_penalty": 1.0,
    #             "max_new_tokens": 512,
    #             "stop": None,
    #             "stop_token_ids": None,
    #             "echo": False,
    # }
    # 补充:
    # top_p: 1.0
    # top_k: -1

    # 从传入generate_stream的params可见
    # 上面几种调用方式的差别就在温度和"stop"
    # ①附带温度为0的调用结果与③的结果一致
    # ②附带温度为0的调用结果与④的结果一致

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids
    # tokenizer会在原始句子开头加个'<s>', token_id为1

    # model.config.is_encoder_decoder: False
    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1
        # max_src_len: 1535  模型能够接受的最长输入序列的长度

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

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
    sent_interrupt = False
    for i in range(max_new_tokens):
        # 模型在一次回答中能够输出的最长回答是max_new_tokens个token
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                # input shape: (1, 404)
                logits = out.logits
                # logits shape: (1, 404, 32000), 404是当前prompt的token数量
            past_key_values = out.past_key_values
            # past_key_values: ((tensor, tensor)_1, ..., (tensor, tensor)_32)  第一次循环时tensor shape: (1, 32, 404, 128)
        else:  # decoding
            # 非首次进入
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
                # logits shape: (1, 1, 32000)
            past_key_values = out.past_key_values
            # past_key_values： ((tensor, tensor)_1, ..., (tensor, tensor)_32)  第二次循环时tensor shape: (1, 32, 405, 128)

        if logits_processor:
            if repetition_penalty > 1.0:
                # repetition_penalty: 1.0
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            # last_token_logits shape: (32000, )
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            # stop_token_ids: [2]
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            # stream_interval: 2
            if echo:
                # echo: False
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                # 判断当前的输出中是否有'###', 如果有, 则停止输出并去除'###'
                # 如果当前输出的末尾有'#'或者'##', 则这一轮的partially_stopped=True
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    # 返回字符串最后一次出现的位置，如果没有匹配项则返回 -1
                    # rfind_start: 0
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
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

    # Finish stream event, which contains finish reason
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

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()


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

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""


def chat_loop(
    model_path: str,
    # model_path: /data1/csw_model_weights/OriginOne or /data1/csw_model_weights/vicuna-7b-v1.3
    device: str,
    # device: 'cuda'
    num_gpus: int,
    # num_gpus: 1
    max_gpu_memory: str,
    # max_gpu_memory: None
    load_8bit: bool,
    # load_8bit: False
    cpu_offloading: bool,
    # cpu_offloading: False
    conv_template: Optional[str],
    # conv_template: None
    conv_system_msg: Optional[str],
    # conv_system_msg: None
    temperature: float,
    # temperature: 默认是0.7, 但此处为了复现改为0
    repetition_penalty: float,
    # repetition_penalty: 1.0
    max_new_tokens: int,
    # max_new_tokens: 512
    chatio: ChatIO,
    # chatio: SimpleChatIO
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    revision: str = "main",
    # revision: 'main'
    judge_sent_end: bool = True,
    # judge_sent_end: False
    debug: bool = True,
    # debug: False
    history: bool = True,
    # history: True
):
    # !!在推断时, 不同模型的权重加载/模板指定load_model和生成代码get_generate_stream_function都需要定制

    # Model
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        revision=revision,
        debug=debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)
    # 不同的模型可能需要采用不同的生成代码
    # 只要模型不是chatglm/falcon/codet5p, 那么generate_stream_func就是generate_stream
    # if model_path == /data1/csw_model_weights/OriginOne, generate_stream_func: fastchat.serve.inference.generate_stream
    # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, generate_stream_func: fastchat.serve.inference.generate_stream

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)
    # context_len: 2048

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
            # 根据conv_template, 获取指定模板
        else:
            conv = get_conversation_template(model_path)
            # 根据model_path路径名称对应的adapter, 匹配其对应的对话模板
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    while True:
        # 如果history=False, 那么每轮对话都是一轮全新、没有历史的对话
        if not history or not conv:
            conv = new_chat()
            # if model_path == /data1/csw_model_weights/OriginOne, conv:
            # Conversation(
            #     name="one_shot",
            #     system="A chat between a curious human and an artificial intelligence assistant. "
            #     "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            #     roles=("Human", "Assistant"),
            #     messages=(
            #         (
            #             "Human",
            #             "Got any creative ideas for a 10 year old’s birthday?",
            #         ),
            #         (
            #             "Assistant",
            #             """Of course! Here are some creative ideas for a 10-year-old's birthday party:
            # 1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
            # 2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
            # 3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
            # 4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
            # 5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
            # 6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
            # 7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
            # 8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
            # Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            #         ),
            #     ),
            #     offset=2,
            #     sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            #     sep="\n### ",
            #     stop_str="###",
            #     stop_token_ids=None, (类变量默认值)
            # )

            # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, conv:
            # Conversation(
            #     name="vicuna_v1.1",
            #     system="A chat between a curious user and an artificial intelligence assistant. "
            #     "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            #     roles=("USER", "ASSISTANT"),
            #     messages=(),
            #     offset=0,
            #     sep_style=SeparatorStyle.ADD_COLON_TWO,
            #     sep=" ",
            #     sep2="</s>",
            #     stop_str=None, (类变量默认值)
            #     stop_token_ids=None, (类变量默认值)
            # )

        try:
            inp = chatio.prompt_for_input(conv.roles[0])
            # 手动输入: 'Who are you'
        except EOFError:
            inp = ""

        if inp == "!!exit" or not inp:
            print("exit...")
            break
        elif inp == "!!reset":
            print("resetting...")
            conv = new_chat()
            continue
        elif inp == "!!remove":
            print("removing last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()
                reload_conv(conv)
            else:
                print("No messages to remove.")
            continue
        elif inp == "!!regen":
            print("regenerating last message...")
            if len(conv.messages) > conv.offset:
                # Assistant
                if conv.messages[-1][0] == conv.roles[1]:
                    conv.messages.pop()
                # User
                if conv.messages[-1][0] == conv.roles[0]:
                    reload_conv(conv)
                    # Set inp to previous message
                    inp = conv.messages.pop()[1]
                else:
                    # Shouldn't happen in normal circumstances
                    print("No user message to regenerate from.")
                    continue
            else:
                print("No messages to regenerate.")
                continue
        elif inp.startswith("!!save"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!save <filename>")
                continue
            else:
                filename = args[1]

            # Add .json if extension not present
            if not "." in filename:
                filename += ".json"

            print("saving...", filename)
            with open(filename, "w") as outfile:
                json.dump(conv.dict(), outfile)
            continue
        elif inp.startswith("!!load"):
            args = inp.split(" ", 1)

            if len(args) != 2:
                print("usage: !!load <filename>")
                continue
            else:
                filename = args[1]

            # Check if file exists and add .json if needed
            if not os.path.exists(filename):
                if (not filename.endswith(".json")) and os.path.exists(
                    filename + ".json"
                ):
                    filename += ".json"
                else:
                    print("file not found:", filename)
                    continue

            print("loading...", filename)
            with open(filename, "r") as infile:
                new_conv = json.load(infile)

            conv = get_conv_template(new_conv["template_name"])
            conv.set_system_message(new_conv["system_message"])
            conv.messages = new_conv["messages"]
            reload_conv(conv)
            continue

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # if model_path == /data1/csw_model_weights/OriginOne, conv: one_shot
        # Prompt: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        # User: 'Human'
        # Assistant: 'Assistant'
        # Q0: 'Got any creative ideas for a 10 year old’s birthday?'
        # A0: "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"
        # 第 1 次提问输入模型的文本: '{Prompt}\n### {User}: {Q0}\n### {Assistant}: {A0}\n### {User}: {Q1}\n### {Assistant}:'
        # 第 2 次提问输入模型的文本: '{Prompt}\n### {User}: {Q0}\n### {Assistant}: {A0}\n### {User}: {Q1}\n### {Assistant}: {A1}\n### {User}: {Q2}\n### {Assistant}:'
        # 第 3 次提问输入模型的文本: '{Prompt}\n### {User}: {Q0}\n### {Assistant}: {A0}\n### {User}: {Q1}\n### {Assistant}: {A1}\n### {User}: {Q2}\n### {Assistant}: {A2}\n### {User}: {Q3}\n### {Assistant}:'

        # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, conv: vicuna_v1.1
        # Prompt: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        # User: 'USER'
        # Assistant: 'ASSISTANT'
        # 第 1 次提问输入模型的文本: '{Prompt} {User}: {Q1} {Assistant}:'
        # 第 2 次提问输入模型的文本: '{Prompt} {User}: {Q1} {Assistant}: {A1}</s>{User}: {Q2} {Assistant}:'
        # 第 3 次提问输入模型的文本: '{Prompt} {User}: {Q1} {Assistant}: {A1}</s>{User}: {Q2} {Assistant}: {A2}</s>{User}: {Q3} {Assistant}:'

        if is_codet5p:  # codet5p is a code completion model.
            prompt = inp

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            # repetition_penalty: 1.0
            "max_new_tokens": max_new_tokens,
            # max_new_tokens: 512
            "stop": conv.stop_str,
            # if conv.name == 'one_shot', conv.stop_str: "###"
            # if conv.name == 'vicuna_v1.1', conv.stop_str: None
            "stop_token_ids": conv.stop_token_ids.copy(),
            # stop_token_ids是Conversation的类变量, 对其append会直接改变类变量
            # if conv.name == 'one_shot', conv.stop_token_ids: None
            # if conv.name == 'vicuna_v1.1', conv.stop_token_ids: None
            "echo": False,
        }
        cp_gen_params = gen_params.copy()
        del cp_gen_params['prompt']
        cp_gen_params['conv'] = conv.name

        try:
            chatio.prompt_for_output(conv.roles[1])
            t = time.time()
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            outputs = chatio.stream_output(output_stream)
            duration = time.time() - t
            conv.update_last_message(outputs.strip())

            if debug:
                prompt_tokens = len(tokenizer.encode(prompt))
                num_tokens = len(tokenizer.encode(outputs))
                msg = {
                    "conv_template": conv.name,
                    "prompt": prompt,
                    "outputs": outputs,
                    "speed (token/s)": round(num_tokens / duration, 2),
                    "total_tokens": prompt_tokens+num_tokens,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": num_tokens,
                }
                print(f"\n{msg}\n{cp_gen_params}\n")

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()

                reload_conv(conv)
