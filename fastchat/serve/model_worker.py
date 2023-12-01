"""
A model worker that executes the model.
"""
import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid

import torch
import torch.nn.functional as F
from transformers import set_seed
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.base_model_worker import BaseModelWorker, app
from fastchat.utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)


worker_id = str(uuid.uuid4())[:8]
# 记录worker的日志
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        # controller_addr: "http://{controller_ip}:21001"
        worker_addr: str,
        # worker_addr: "http://{worker_ip}:21002"
        worker_id: str,
        # worker_id: 一个每次运行都不固定的字符串, 这里以'e73850d4'举例
        model_path: str,
        # model_path: '/data1/csw_model_weights/Llama-2-13b-chat-hf'
        model_names: List[str],
        # model_names: None
        limit_worker_concurrency: int,
        # limit_worker_concurrency: 5
        no_register: bool,
        # no_register: False
        device: str,
        # device: 'cuda'
        num_gpus: int,
        # num_gpus: 1
        max_gpu_memory: str,
        # max_gpu_memory: None
        dtype: Optional[torch.dtype] = None,
        # dtype: None
        load_8bit: bool = False,
        # load_8bit: False
        cpu_offloading: bool = False,
        # cpu_offloading: False
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        # exllama_config: None
        xft_config: Optional[XftConfig] = None,
        # xft_config: None
        stream_interval: int = 2,
        # stream_interval: 2
        conv_template: Optional[str] = None,
        # conv_template: 'llama-2'
        embed_in_truncate: bool = False,
        # embed_in_truncate: False
        seed: Optional[int] = None,
        # seed: None
        debug: bool = False,
        # debug: False
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        # 加载模型权重和tokenizer
        self.model, self.tokenizer = load_model(
            model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            debug=debug,
        )
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        # 读取config.json文件里的内容获得模型所能支持的最大上下文长度, Llama-2-13b-chat-hf是4096
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        # 只要模型不是chatglm/falcon/codet5p/exllama/xft, 那么generate_stream_func就是fastchat.serve.inference.generate_stream
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            # 单独开一个线程, 该线程每隔45秒会向controller报备一下, 确保worker与controller之间的连接畅通
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            # self.generate_stream_func()是一个由yield关键字控制返回的生成器
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):

                # curl http://localhost:8001/v1/chat/completions   -H "Content-Type: application/json"   -d '{
                #     "model":"Llama-2-13b-chat-hf",
                #     "max_tokens":500,
                #     "messages":[{"content":"What is the boiling point of water","role":"user"}],
                #     "stream":true,
                #     "temperature":0
                #   }'
                # 则历次的output为:
                """
                {'text': '', 'usage': {'prompt_tokens': 16, 'completion_tokens': 0, 'total_tokens': 16}, 'finish_reason': None}
                {'text': ' The bo', 'usage': {'prompt_tokens': 16, 'completion_tokens': 2, 'total_tokens': 18}, 'finish_reason': None}
                {'text': ' The boiling point', 'usage': {'prompt_tokens': 16, 'completion_tokens': 4, 'total_tokens': 20}, 'finish_reason': None}
                {'text': ' The boiling point of water', 'usage': {'prompt_tokens': 16, 'completion_tokens': 6, 'total_tokens': 22}, 'finish_reason': None}
                {'text': ' The boiling point of water is ', 'usage': {'prompt_tokens': 16, 'completion_tokens': 8, 'total_tokens': 24}, 'finish_reason': None}
                {'text': ' The boiling point of water is 10', 'usage': {'prompt_tokens': 16, 'completion_tokens': 10, 'total_tokens': 26}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees', 'usage': {'prompt_tokens': 16, 'completion_tokens': 12, 'total_tokens': 28}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsi', 'usage': {'prompt_tokens': 16, 'completion_tokens': 14, 'total_tokens': 30}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (', 'usage': {'prompt_tokens': 16, 'completion_tokens': 16, 'total_tokens': 32}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C', 'usage': {'prompt_tokens': 16, 'completion_tokens': 18, 'total_tokens': 34}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or', 'usage': {'prompt_tokens': 16, 'completion_tokens': 20, 'total_tokens': 36}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 2', 'usage': {'prompt_tokens': 16, 'completion_tokens': 22, 'total_tokens': 38}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212', 'usage': {'prompt_tokens': 16, 'completion_tokens': 24, 'total_tokens': 40}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees F', 'usage': {'prompt_tokens': 16, 'completion_tokens': 26, 'total_tokens': 42}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit', 'usage': {'prompt_tokens': 16, 'completion_tokens': 28, 'total_tokens': 44}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°', 'usage': {'prompt_tokens': 16, 'completion_tokens': 30, 'total_tokens': 46}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F)', 'usage': {'prompt_tokens': 16, 'completion_tokens': 32, 'total_tokens': 48}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard', 'usage': {'prompt_tokens': 16, 'completion_tokens': 34, 'total_tokens': 50}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard atmospher', 'usage': {'prompt_tokens': 16, 'completion_tokens': 36, 'total_tokens': 52}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard atmospheric pressure', 'usage': {'prompt_tokens': 16, 'completion_tokens': 38, 'total_tokens': 54}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard atmospheric pressure.', 'usage': {'prompt_tokens': 16, 'completion_tokens': 40, 'total_tokens': 56}, 'finish_reason': None}
                {'text': ' The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard atmospheric pressure.', 'usage': {'prompt_tokens': 16, 'completion_tokens': 40, 'total_tokens': 56}, 'finish_reason': 'stop'}
                """
                # 这里每行output正好可以对应该请求流式返回的结果, 最后一行output的内容也对应非流式请求返回的结果
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
                # with b"\0"
                # b'{"text": "...", "error_code": 0, ...}\x00'
                # without b"\0"
                # b'{"text": "...", "error_code": 0, ...}'

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            # SERVER_ERROR_MSG:
            # "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        # 直接取生成器结果的最后一个返回
        # [:-1]是为了将末尾添加的b"\0"去除
        return json.loads(x[:-1].decode())

    def __process_embed_chunk(self, input_ids, attention_mask, **model_type_dict):
        if model_type_dict.get("is_bert"):
            model_output = self.model(input_ids)
            if model_type_dict.get("is_robert"):
                data = model_output.last_hidden_state
            else:
                data = model_output[0]
        elif model_type_dict.get("is_t5"):
            model_output = self.model(input_ids, decoder_input_ids=input_ids)
            data = model_output.encoder_last_hidden_state
        else:
            model_output = self.model(input_ids, output_hidden_states=True)
            if model_type_dict.get("is_chatglm"):
                data = model_output.hidden_states[-1].transpose(0, 1)
            else:
                data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        token_num = torch.sum(attention_mask).item()

        return sum_embeddings, token_num

    def __encode_base64(self, embeddings: torch.Tensor) -> List[str]:
        embeddings = embeddings.cpu()
        return [
            base64.b64encode(e.numpy().tobytes()).decode("utf-8") for e in embeddings
        ]

    @torch.inference_mode()
    def get_embeddings(self, params):
        self.call_ct += 1

        try:
            tokenizer = self.tokenizer
            ret = {"embedding": [], "token_num": 0}

            model_type_dict = {
                "is_llama": "llama" in str(type(self.model)),
                "is_t5": "t5" in str(type(self.model)),
                "is_chatglm": "chatglm" in str(type(self.model)),
                "is_bert": "bert" in str(type(self.model)),
                "is_robert": "robert" in str(type(self.model)),
            }

            if self.embed_in_truncate:
                encoding = tokenizer.batch_encode_plus(
                    params["input"],
                    padding=True,
                    truncation="longest_first",
                    return_tensors="pt",
                    max_length=self.context_len,
                )
            else:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = input_ids != tokenizer.pad_token_id

            base64_encode = params.get("encoding_format", None)

            if self.embed_in_truncate:
                chunk_embeddings, token_num = self.__process_embed_chunk(
                    input_ids, attention_mask, **model_type_dict
                )
                embedding = chunk_embeddings / token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret["token_num"] = token_num
            else:
                all_embeddings = []
                all_token_num = 0
                for i in range(0, input_ids.size(1), self.context_len):
                    chunk_input_ids = input_ids[:, i : i + self.context_len]
                    chunk_attention_mask = attention_mask[:, i : i + self.context_len]

                    chunk_embeddings, token_num = self.__process_embed_chunk(
                        chunk_input_ids, chunk_attention_mask, **model_type_dict
                    )
                    all_embeddings.append(chunk_embeddings)
                    all_token_num += token_num

                all_embeddings_tensor = torch.stack(all_embeddings)
                embedding = torch.sum(all_embeddings_tensor, dim=0) / all_token_num
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)

                ret["token_num"] = all_token_num

            if base64_encode == "base64":
                out_embeddings = self.__encode_base64(normalized_embeddings)
            else:
                out_embeddings = normalized_embeddings.tolist()
            ret["embedding"] = out_embeddings

            gc.collect()
            torch.cuda.empty_cache()
            if self.device == "xpu":
                torch.xpu.empty_cache()
            if self.device == "npu":
                torch.npu.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    # model_name是worker所加载权重的代号, 如不指定会从权重路径中提取
    # 多个worker可以共享同一个model_name
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    # 对话模板
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    # 并发数量
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Print debugging messages"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # args:
    # Namespace(
    #   host='localhost',
    #   port=21002,
    #   worker_address='http://localhost:21002',
    #   controller_address='http://localhost:21001',
    #   model_path='/data1/csw_model_weights/Llama-2-13b-chat-hf',
    #   revision='main',
    #   device='cuda',
    #   gpus=None,
    #   num_gpus=1,
    #   max_gpu_memory=None,
    #   dtype=None,
    #   load_8bit=False,
    #   cpu_offloading=False,
    #   gptq_ckpt=None,
    #   gptq_wbits=16,
    #   gptq_groupsize=-1,
    #   gptq_act_order=False,
    #   awq_ckpt=None,
    #   awq_wbits=16,
    #   awq_groupsize=-1,
    #   enable_exllama=False,
    #   exllama_max_seq_len=4096,
    #   exllama_gpu_split=None,
    #   enable_xft=False,
    #   xft_max_seq_len=4096,
    #   xft_dtype=None,
    #   model_names=None,
    #   conv_template='llama-2',
    #   embed_in_truncate=False,
    #   limit_worker_concurrency=5,
    #   stream_interval=2,
    #   no_register=False,
    #   seed=None,
    #   debug=False
    # )

    # args.gpus: None
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        # args.gptq_wbits: 16
        groupsize=args.gptq_groupsize,
        # args.gptq_groupsize: -1
        act_order=args.gptq_act_order,
        # args.gptq_act_order: False
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )

    # args.enable_exllama: False
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None

    # args.enable_xft: False
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    worker = ModelWorker(
        args.controller_address,
        # args.controller_address: "http://{controller_ip}:21001"
        args.worker_address,
        # args.worker_address: "http://{worker_ip}:21002"
        worker_id,
        # worker_id: 一个每次运行都不固定的字符串, 这里以'e73850d4'举例
        args.model_path,
        # args.model_path: '/data1/csw_model_weights/Llama-2-13b-chat-hf'
        args.model_names,
        # args.model_names: None
        args.limit_worker_concurrency,
        # args.limit_worker_concurrency: 5
        no_register=args.no_register,
        # args.no_register: False
        device=args.device,
        # args.device: 'cuda'
        num_gpus=args.num_gpus,
        # args.num_gpus: 1
        max_gpu_memory=args.max_gpu_memory,
        # args.max_gpu_memory: None
        dtype=str_to_torch_dtype(args.dtype),
        # str_to_torch_dtype(None): None
        load_8bit=args.load_8bit,
        # args.load_8bit: False
        cpu_offloading=args.cpu_offloading,
        # args.cpu_offloading: False
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        # exllama_config: None
        xft_config=xft_config,
        # xft_config: None
        stream_interval=args.stream_interval,
        # args.stream_interval: 2
        conv_template=args.conv_template,
        # args.conv_template: 'llama-2'
        embed_in_truncate=args.embed_in_truncate,
        # args.embed_in_truncate: False
        seed=args.seed,
        # args.seed: None
        debug=args.debug,
        # args.debug: False
    )
    return args, worker


if __name__ == "__main__":
    # 此代码及base_model_worker.py的注释都是基于如下指令debug获得的
    # ================================================================================================================
    # python3 -m fastchat.serve.model_worker --model-path /data1/csw_model_weights/Llama-2-13b-chat-hf --conv-template llama-2
    # ================================================================================================================
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
