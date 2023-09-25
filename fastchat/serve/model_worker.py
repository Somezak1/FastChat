"""
A model worker that executes the model.
"""
import argparse
import asyncio
import base64
import dataclasses
import gc
import logging
import json
import os
import threading
import time
from typing import List, Optional
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
from transformers import set_seed
import uvicorn

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import (
    build_logger,
    pretty_print_semaphore,
    get_context_length,
    str_to_torch_dtype,
)


worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

app = FastAPI()


def heart_beat_worker(obj):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        # WORKER_HEART_BEAT_INTERVAL: 45
        obj.send_heart_beat()


class BaseModelWorker:
    '''
    BaseModelWorker类主要负责与controller通信, ModelWorker类主要负责推断
    '''
    def __init__(
        self,
        controller_addr: str,
        # controller_addr: "http://{controller_ip}:21001"
        worker_addr: str,
        # worker_addr: "http://{worker_ip}:21002"
        worker_id: str,
        # worker_id: 一个每次运行都不固定的字符串, 这里以'e73850d4'举例
        model_path: str,
        # model_path: '/data1/csw_model_weights/vicuna-7b-v1.3'
        model_names: List[str],
        # model_names: None
        limit_worker_concurrency: int,
        # limit_worker_concurrency: 5
        conv_template: str = None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.limit_worker_concurrency = limit_worker_concurrency

        if conv_template:
            self.conv = get_conv_template(conv_template)
        else:
            # 根据model_path, 即可知道与之匹配的对话模板
            self.conv = get_conversation_template(model_path)
            # 根据model_path路径名称对应的adapter, 匹配其对应的对话模板
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
            # )

        self.conv.sep_style = int(self.conv.sep_style)
        self.tokenizer = None
        self.context_len = None
        # self.call_ct 调用次数计数
        self.call_ct = 0
        self.semaphore = None

        self.heart_beat_thread = None

    def init_heart_beat(self):
        '''
        向controller报备完毕之后, 程序会单独再开一个线程
        该线程每隔45秒会向controller报备一下, 确保worker与controller之间的连接畅通
        如果当次报备失败, 则间隔5秒重新报备, 直至成功
        '''
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            args=(self,),
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        '''
        向controller发送一个包含自身信息的data, 报备一下, 确保通信畅通
        如果报备失败, 则程序会报错
        '''
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        # url: 'http://{controller_ip}:21001/register_worker'
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
            # self.get_status():
            # {
            #   "model_names": 'vicuna-7b-v1.3',
            #   "speed": 1,
            #   "queue_length": 0,
            # }
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )
        # 'Send heart beat. Models: ['vicuna-7b-v1.3']. Semaphore: None. call_ct: 0. worker_id: e73850d4.'

        url = self.controller_addr + "/receive_heart_beat"
        # url: 'http://{controller_ip}:21001/receive_heart_beat'

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        # self.worker_addr: 'http://{worker_ip}:21002'
                        "queue_length": self.get_queue_length(),
                        # self.get_queue_length(): 0
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            self.semaphore is None
            or self.semaphore._value is None
            or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_worker_concurrency
                - self.semaphore._value
                + len(self.semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
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
        )
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        # 读取config.json文件里的内容获得模型所能支持的最大上下文长度
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        # 只要模型不是chatglm/falcon/codet5p, 那么generate_stream_func就是generate_stream
        # if model_path == /data1/csw_model_weights/OriginOne, generate_stream_func: fastchat.serve.inference.generate_stream
        # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, generate_stream_func: fastchat.serve.inference.generate_stream
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    def generate_stream_gate(self, params):
        self.call_ct += 1

        try:
            if self.seed is not None:
                set_seed(self.seed)
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, 则历次的output为:
                '''
                {'text': 'I', 'usage': {'prompt_tokens': 42, 'completion_tokens': 0, 'total_tokens': 42}, 'finish_reason': None}
                {'text': 'I am Vic', 'usage': {'prompt_tokens': 42, 'completion_tokens': 2, 'total_tokens': 44}, 'finish_reason': None}
                {'text': 'I am Vicuna,', 'usage': {'prompt_tokens': 42, 'completion_tokens': 4, 'total_tokens': 46}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language', 'usage': {'prompt_tokens': 42, 'completion_tokens': 6, 'total_tokens': 48}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained', 'usage': {'prompt_tokens': 42, 'completion_tokens': 8, 'total_tokens': 50}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by research', 'usage': {'prompt_tokens': 42, 'completion_tokens': 10, 'total_tokens': 52}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from', 'usage': {'prompt_tokens': 42, 'completion_tokens': 12, 'total_tokens': 54}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large', 'usage': {'prompt_tokens': 42, 'completion_tokens': 14, 'total_tokens': 56}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems', 'usage': {'prompt_tokens': 42, 'completion_tokens': 16, 'total_tokens': 58}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization', 'usage': {'prompt_tokens': 42, 'completion_tokens': 18, 'total_tokens': 60}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (L', 'usage': {'prompt_tokens': 42, 'completion_tokens': 20, 'total_tokens': 62}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS', 'usage': {'prompt_tokens': 42, 'completion_tokens': 22, 'total_tokens': 64}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).', 'usage': {'prompt_tokens': 42, 'completion_tokens': 24, 'total_tokens': 66}, 'finish_reason': None}
                {'text': 'I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).', 'usage': {'prompt_tokens': 42, 'completion_tokens': 24, 'total_tokens': 66}, 'finish_reason': 'stop'}
                '''
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


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = worker.generate_gate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    # 如果controller和worker不在同一台机器上开启, 那么host参数要填worker的ip地址, 同时还得指定worker_address和controller_address
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    # --worker-address的默认参数值得改成"http://{worker_ip}:21002"形式
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    # --controller-address的默认参数值得改成"http://{controller_ip}:21001"形式
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
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
    args = parser.parse_args()
    logger.info(f"args: {args}")
    # args:
    # Namespace(
    # 	host='localhost',
    # 	port=21002,
    # 	worker_address='http://{worker_ip}:21002',
    # 	controller_address='http://{controller_ip}:21001',
    # 	model_path='/data1/csw_model_weights/vicuna-7b-v1.3',
    # 	revision='main',
    # 	device='cuda',
    # 	gpus=None,
    # 	num_gpus=1,
    # 	max_gpu_memory=None,
    # 	load_8bit=False,
    # 	cpu_offloading=False,
    # 	gptq_ckpt=None,
    # 	gptq_wbits=16,
    # 	gptq_groupsize=-1,
    # 	gptq_act_order=False,
    # 	model_names=None,
    # 	limit_worker_concurrency=5,
    # 	stream_interval=2,
    # 	no_register=False
    # )

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

    worker = ModelWorker(
        args.controller_address,
        # args.controller_address: "http://{controller_ip}:21001"
        args.worker_address,
        # args.worker_address: "http://{worker_ip}:21002"
        worker_id,
        # worker_id: 一个每次运行都不固定的字符串, 这里以'e73850d4'举例
        args.model_path,
        # args.model_path: '/data1/csw_model_weights/vicuna-7b-v1.3'
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
        load_8bit=args.load_8bit,
        # args.load_8bit: False
        cpu_offloading=args.cpu_offloading,
        # args.cpu_offloading: False
        gptq_config=gptq_config,
        awq_config=awq_config,
        stream_interval=args.stream_interval,
        # args.stream_interval: 2
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
