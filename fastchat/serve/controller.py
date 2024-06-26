"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import os
import time
from typing import List, Union
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from fastchat.constants import (
    CONTROLLER_HEART_BEAT_EXPIRATION,
    WORKER_API_TIMEOUT,
    ErrorCode,
    SERVER_ERROR_MSG,
)
from fastchat.utils import build_logger


# 记录controller的日志
logger = build_logger("controller", "controller.log")


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str
    multimodal: bool


def heart_beat_controller(controller):
    """
    每隔90秒, 遍历一次controller.worker_info (controller中所有登记过的worker)
    若该worker在过去90秒内与controller没通信过, 则从controller.worker_info中删除该worker
    """
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        # CONTROLLER_HEART_BEAT_EXPIRATION: 90
        controller.remove_stale_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        # note: self.worker_info中worker的queue_length信息
        # 只有在worker向controller heart_beat 和 api_server向controller get_worker_address 这两种情况下才会得到更新
        # 其中heart_beat 每45秒发送一次, 所以get_worker_address时所依据的queue_length信息可能是过时的
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        # self.dispatch_method: DispatchMethod.SHORTEST_QUEUE
        # 该值用于get_worker_address

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,)
        )
        self.heart_beat_thread.start()
        # 开启一个独立的线程, 专门用于监控controller与各worker之间的通信状态

    def register_worker(
        self,
        worker_name: str,
        check_heart_beat: bool,
        worker_status: dict,
        multimodal: bool,
    ):
        # 该函数用于将worker发送来的身份信息保存在controller的worker_info中

        # worker_name: 'http://{worker_ip}:21002'
        # check_heart_beat: True
        # worker_status: {
        #    "model_names": 'Llama-2-13b-chat-hf',
        #    "speed": 1,
        #    "queue_length": 0,
        # }
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        # 如果worker向controller报备时没自带worker_status, 则controller会依据worker_name向worker发送一个获取其状态的请求
        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"],
            worker_status["speed"],
            worker_status["queue_length"],
            check_heart_beat,
            time.time(),
            multimodal,
        )

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        # 将controller中所有注册过的worker删除, 并重新建立连接
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            # w_name: 'http://{worker_ip}:21002'
            # w_info: WorkerInfo(...) obj
            if not self.register_worker(
                w_name, w_info.check_heart_beat, None, w_info.multimodal
            ):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        # 返回self.worker_info中不重复的所有model_names
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            # w_name: 'http://{worker_ip}:21002'
            # w_info: WorkerInfo(...) obj
            model_names.update(w_info.model_names)

        return list(model_names)

    def list_multimodal_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            if w_info.multimodal:
                model_names.update(w_info.model_names)

        return list(model_names)

    def list_language_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            if not w_info.multimodal:
                model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        # api_server接收到用户对于某个model_name的生成请求, 但api_server不知道这个model_name对应worker的联系方式
        # 于是api_server拜托controller, 让controller告诉api_server一个名为model_name的worker的联系方式
        # controller于是从小本本(worker_info)中查找所有名为model_name的worker, 并随机返回一个该worker的联系方式给api_server
        # 该函数的作用就是从所有名为model_name的worker中按DispatchMethod指定的方式返回一个该worker的地址
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)), p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            # 将self.worker_info中所有名称含model_name的worker挑出来
            for w_name, w_info in self.worker_info.items():
                # w_name: 'http://{worker_ip}:21002'
                # w_info: WorkerInfo(...) obj
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            # 从挑选出来的worker中根据(上次worker向controller报备时该worker的)queue_length最短原则, 随机挑选一个worker
            # 注意: controller中登记的worker, 其queue_length只会在register/receive_heart_beat/get_worker_address三种情况下得到更新
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            # 被选中的worker queue_length加1
            self.worker_info[w_name].queue_length += 1
            logger.info(
                f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}"
            )
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        # 用worker发来的最新信息, 更新controller的小本本中该worker的部分
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stale_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        # CONTROLLER_HEART_BEAT_EXPIRATION: 90
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            # worker_name: 'http://{worker_ip}:21002'
            # w_info: WorkerInfo(...)对象
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                # w_info.check_heart_beat: True
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def handle_no_worker(self, params):
        logger.info(f"no worker: {params['model']}")
        ret = {
            "text": SERVER_ERROR_MSG,
            # SERVER_ERROR_MSG: "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
            "error_code": ErrorCode.CONTROLLER_NO_WORKER,
        }
        return json.dumps(ret).encode() + b"\0"

    def handle_worker_timeout(self, worker_address):
        logger.info(f"worker timeout: {worker_address}")
        ret = {
            "text": SERVER_ERROR_MSG,
            # SERVER_ERROR_MSG: "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
            "error_code": ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        }
        return json.dumps(ret).encode() + b"\0"

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        model_names = sorted(list(model_names))
        return {
            "model_names": model_names,
            "speed": speed,
            "queue_length": queue_length,
        }

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        # worker_addr: 'http://{worker_ip}:21002'
        if not worker_addr:
            yield self.handle_no_worker(params)

        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                json=params,
                stream=True,
                timeout=WORKER_API_TIMEOUT,
                # WORKER_API_TIMEOUT: 100
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            yield self.handle_worker_timeout(worker_addr)


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"],
        data["check_heart_beat"],
        data.get("worker_status", None),
        data.get("multimodal", False),
    )
    # data["worker_name"]: 'http://{worker_ip}:21002'
    # data["check_heart_beat"]: True
    # data.get("worker_status", None): {
    #    "model_names": 'Llama-2-13b-chat-hf',
    #    "speed": 1,
    #    "queue_length": 0,
    # }


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/list_multimodal_models")
async def list_multimodal_models():
    models = controller.list_multimodal_models()
    return {"models": models}


@app.post("/list_language_models")
async def list_language_models():
    models = controller.list_language_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["worker_name"], data["queue_length"])
    # data["worker_name"]: 'http://{worker_ip}:21002'
    # data["queue_length"]: 0
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()


@app.get("/test_connection")
async def worker_api_get_status(request: Request):
    return "success"


def create_controller():
    parser = argparse.ArgumentParser()
    # 如果controller和worker不在同一台机器上开启, 那么host参数要填controller的ip地址
    parser.add_argument("--host", type=str, default="localhost")
    # 端口号
    parser.add_argument("--port", type=int, default=21001)
    # api_server向controller索要worker地址时, controller采用如下参数指定的逻辑返回worker地址给api_server
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
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
    # args: Namespace(host='localhost', port=21001, dispatch_method='shortest_queue')

    controller = Controller(args.dispatch_method)
    return args, controller


if __name__ == "__main__":
    # 此代码的注释都是基于如下指令debug获得的
    # ================================================================================================================
    # python3 -m fastchat.serve.controller
    # ================================================================================================================
    args, controller = create_controller()
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
