import asyncio
import threading
import time
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL
from fastchat.conversation import Conversation
from fastchat.utils import pretty_print_semaphore, build_logger


worker = None
logger = None

app = FastAPI()


def heart_beat_worker(obj):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        # WORKER_HEART_BEAT_INTERVAL: 45
        obj.send_heart_beat()


class BaseModelWorker:
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
        conv_template: str = None,
        # conv_template: 'llama-2'
    ):
        global logger, worker

        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.limit_worker_concurrency = limit_worker_concurrency
        # 获取该模型的对话模板, 如果--conv-template参数未指定, 则会根据传入的--model-path匹配一个对话模板
        self.conv = self.make_conv_template(conv_template, model_path)
        self.conv.sep_style = int(self.conv.sep_style)
        self.tokenizer = None
        self.context_len = None
        self.call_ct = 0
        self.semaphore = None

        self.heart_beat_thread = None

        if logger is None:
            logger = build_logger("model_worker", f"model_worker_{self.worker_id}.log")
        if worker is None:
            worker = self

    def make_conv_template(
        self,
        conv_template: str = None,
        model_path: str = None,
    ) -> Conversation:
        """
        can be overrided to costomize the conversation template for different model workers.
        """
        from fastchat.conversation import get_conv_template
        from fastchat.model.model_adapter import get_conversation_template

        if conv_template:
            conv = get_conv_template(conv_template)
            # 根据conv_template, 获取指定模板
        else:
            conv = get_conversation_template(model_path)
            # 根据model_path路径名称对应的adapter, 匹配其对应的对话模板
        return conv

    def init_heart_beat(self):
        """
        向controller报备完毕之后, 程序会再单独开一个线程
        该线程每隔45秒会向controller报备一下, 确保worker与controller之间的连接畅通
        如果当次报备失败, 则间隔5秒重新报备, 直至成功
        """
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            args=(self,),
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        """
        向controller发送一个包含自身信息的data, 报备一下, 确保通信畅通
        如果报备失败, 则程序会报错
        """
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        # 向controller发送信息, 报备一下
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
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

        try:
            input_ids = self.tokenizer(prompt).input_ids
            input_echo_len = len(input_ids)
        except TypeError:
            input_echo_len = self.tokenizer.num_tokens(prompt)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}

    def generate_stream_gate(self, params):
        raise NotImplementedError

    def generate_gate(self, params):
        raise NotImplementedError

    def get_embeddings(self, params):
        raise NotImplementedError


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
    output = await asyncio.to_thread(worker.generate_gate, params)
    # 在老版本中, FastChat模型部署后, 当前面有流式调用正在输出, 随后新增一个非流式调用时, 流式调用会完全停滞
    # GPU会被非流式调用全部侵占, 这是因为老版model_worker.py使用的是output = worker.generate_gate(params)运行非流式调用
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
