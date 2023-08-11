"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server
"""
import asyncio
import argparse
import json
import logging
import os
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi import Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from pydantic import BaseSettings
import shortuuid
import tiktoken
import uvicorn

from fastchat.constants import (
    WORKER_API_TIMEOUT,
    WORKER_API_EMBEDDING_BATCH_SIZE,
    ErrorCode,
)
from fastchat.conversation import Conversation, SeparatorStyle
from fastapi.exceptions import RequestValidationError
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    # ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from fastchat.protocol.api_protocol import (
    APIChatCompletionRequest,
    APITokenCheckRequest,
    APITokenCheckResponse,
    APITokenCheckResponseItem,
)
from typing import Literal, Optional, List, Dict, Any, Union
import time
import shortuuid
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

conv_template_map = {}


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: UsageInfo


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    # 创建一个JSON格式的错误响应
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )
    # class ErrorResponse(BaseModel):
    #     object: str = "error"
    #     message: str
    #     code: int


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    '''
    向controller_address发送get_worker_address请求
    判断controller所登记的model中是否有request.model
    如有则返回None, 如没有则返回一个JSON格式错误响应
    '''
    controller_address = app_settings.controller_address
    ret = None
    async with httpx.AsyncClient() as client:
        models_ret = await client.post(controller_address + "/list_models")
        models = models_ret.json()["models"]
        if request.model not in models:
            ret = create_error_response(
                ErrorCode.INVALID_MODEL,
                f"Only {'&&'.join(models)} allowed now, your model {request.model}",
            )
    return ret


async def check_length(
    request, prompt, max_tokens, worker_addr, client: httpx.AsyncClient
):
    '''
    assert prompt_tokens + max_tokens <= context_len
    '''

    # 向worker_address发送model_details请求
    response = await client.post(
        worker_addr + "/model_details",
        headers=headers,
        json={"model": request.model},
        timeout=WORKER_API_TIMEOUT,
    )
    # response: {"context_length": worker.context_len}
    context_len = response.json()["context_length"]
    # context_len: model worker中加载的模型所能支持的最大上下文长度, 该值由模型权重路径下的config.json文件定义, 默认值是2048

    # 向worker_address发送count_token请求
    response = await client.post(
        worker_addr + "/count_token",
        headers=headers,
        json={"model": request.model, "prompt": prompt},
        timeout=WORKER_API_TIMEOUT,
        # WORKER_API_TIMEOUT: 100s
    )
    # response: {"count": input_echo_len, "error_code": 0}
    token_num = response.json()["count"]
    # token_num: 传入模型的prompt(system_content+Q+A+...+Q)所占用的tokens数量

    # max_tokens: 模型所能支持的最大输出长度(answer的最大tokens), 该值可以人为定义
    if token_num + max_tokens > context_len:
        # prompt+answer所占用的tokens数量需要小于模型所能支持的最大上下文长度, 不然会报错
        # 模型在正常情况下会输出完整的answer, 那么返回的finish_reason='stop'
        # 如果max_tokens过小, 则模型还未输出到停止符就会被迫停止, 且返回的finish_reason='length'
        # 注意: 这里和真正推断的时候还有一点不一样, 真正推断时候prompt会进行如下截断, 优先保证answer的输出空间
        # prompt = prompt[-(context_len-max_tokens-8):]
        return create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
        # max_tokens, integer, Optional, Defaults to inf
        # The maximum number of tokens to generate in the chat completion.
        # The total length of input tokens and generated tokens is limited by the model's context length.
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
        # n, integer, Optional, Defaults to 1
        # How many chat completion choices to generate for each input message.
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
        # temperature, number, Optional, Defaults to 1
        # What sampling temperature to use, between 0 and 2.
        # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        # We generally recommend altering this or top_p but not both.
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
        # top_p, number, Optional, Defaults to 1
        # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
        # So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        # We generally recommend altering this or temperature but not both.
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )
        # stop, string or array, Optional, Defaults to null
        # Up to 4 sequences where the API will stop generating further tokens.

    return None


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(text) for text in inp]

    return inp


async def get_gen_params(
    model_name: str,
    worker_addr: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    echo: Optional[bool],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]],
) -> Dict[str, Any]:
    # 获取推断时所需的model, prompt, 温度, max_new_tokens 等信息
    conv = await get_conv(model_name, worker_addr)
    conv = Conversation(
        name=conv["name"],
        system_template=conv["system_template"],
        system_message=conv["system_message"],
        roles=conv["roles"],
        messages=list(conv["messages"]),  # prevent in-place modification
        offset=conv["offset"],
        sep_style=SeparatorStyle(conv["sep_style"]),
        sep=conv["sep"],
        sep2=conv["sep2"],
        stop_str=conv["stop_str"],
        stop_token_ids=conv["stop_token_ids"],
    )

    if isinstance(messages, str):
        prompt = messages
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                # 若调用api时指定system_content, 则会覆盖conv中原本的system_content
                conv.set_system_message(message["content"])
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # 按conv中定义的方式将system_content+Q+A+...+Q等内容拼接在一起

    if max_tokens is None:
        max_tokens = 512
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_tokens,
        "echo": echo,
        "stream": stream,
    }

    if not stop:
        gen_params.update(
            {"stop": conv.stop_str, "stop_token_ids": conv.stop_token_ids}
        )
    else:
        gen_params.update({"stop": stop})

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def get_worker_address(model_name: str, client: httpx.AsyncClient) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :param client: The httpx client to use
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    # 若controller.dispatch_method==LOTTERY, 则从所有名称为model_name的worker中随机挑一个返回
    # 若controller.dispatch_method==SHORTEST_QUEUE, 则从所有名称为model_name的worker中挑一个负载最小的返回
    controller_address = app_settings.controller_address

    # 向controller_address发送get_worker_address请求
    ret = await client.post(
        controller_address + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")

    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str, worker_addr: str):
    # 向worker_address通信, 得到该worker初始化时的对话模板conv_template
    async with httpx.AsyncClient() as client:
        conv_template = conv_template_map.get((worker_addr, model_name))
        # 先去字典中查询这个worker_address下该model的对话模板
        # 如果字典中没有该信息, 就向worker_address发送worker_get_conv_template的请求, 将获取到的conv_template存储在字典中
        if conv_template is None:
            response = await client.post(
                worker_addr + "/worker_get_conv_template",
                headers=headers,
                json={"model": model_name},
                timeout=WORKER_API_TIMEOUT,
                # WORKER_API_TIMEOUT: 100
            )
            conv_template = response.json()["conv"]
            conv_template_map[(worker_addr, model_name)] = conv_template
        return conv_template


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        ret = await client.post(controller_address + "/refresh_all_workers")
        ret = await client.post(controller_address + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):

    # class ChatCompletionRequest(BaseModel):
    #     model: str
    #     messages: Union[str, List[Dict[str, str]]]
    #     temperature: Optional[float] = 0.7
    #     top_p: Optional[float] = 1.0
    #     n: Optional[int] = 1
    #     max_tokens: Optional[int] = None
    #     stop: Optional[Union[str, List[str]]] = None
    #     stream: Optional[bool] = False
    #     presence_penalty: Optional[float] = 0.0
    #     frequency_penalty: Optional[float] = 0.0
    #     user: Optional[str] = None

    """Creates a completion for the chat message"""

    # request: (调用请求时除model和messages以外, 其他参数未指定, 查看默认参数值)
    # model='vicuna-7b-v1.3' messages=[{'role': 'user', 'content': '你了解冰鉴吗'}] temperature=0.7
    # top_p=1.0 n=1 max_tokens=None stop=None stream=False presence_penalty=0.0 frequency_penalty=0.0 user=None

    # create_chat_completion函数中各函数通信次数统计:
    #    函数-->子函数                                                       调用controller            调用worker
    # ① check_model                                                        .get_worker_address      None
    # ② get_gen_params-->get_conv                                          .get_worker_address      .worker_get_conv_template
    # ③ check_length                                                       .get_worker_address      .model_details, .count_token

    # if request.stream == True
    # ④ chat_completion_stream_generator-->generate_completion_stream      .get_worker_address      .worker_generate_stream
    # else
    # ④ generate_completion                                                .get_worker_address      .worker_generate

    # 每有一个create_chat_completion请求, openai_api_server就会向controller进行4次get_worker_address通信
    # controller会依据同名model之前通信保存的负载, 返回一个dispatch_method方法指定的worker_address
    # 但依据上面的表格所示, ①②③④四个函数所调用的get_worker_address, 其后跟随的调用worker的通信负载是不一样的
    # worker_generate_stream和worker_generate的通信负载要远大于worker_get_conv_template, model_details, count_token
    # 所以controller中依据get_worker_address的负载来返回worker_address的策略经常会出现
    # 一张卡经常干model_details, count_token之类的杂活
    # 另一张卡却经常, 多次被分配到worker_generate_stream这样的重活, 导致卡的利用率不高

    # error_check_ret = await check_model(request)
    # if error_check_ret is not None:
    #     return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)
        # 获取对话模板, 同时根据对话模板及传入的messages将文本拼接成输入模型的prompt
        # 汇总其他模型生成参数, 获得gen_params
        gen_params = await get_gen_params(
            request.model,
            worker_addr,
            request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            echo=False,
            stream=request.stream,
            stop=request.stop,
        )
        error_check_ret = await check_length(
            request,
            gen_params["prompt"],
            gen_params["max_new_tokens"],
            worker_addr,
            client,
        )
        if error_check_ret is not None:
            return error_check_ret

        if request.stream:
            generator = chat_completion_stream_generator(
                request.model, gen_params, request.n, worker_addr
            )
            return StreamingResponse(generator, media_type="text/event-stream")

        choices = []
        chat_completions = []
        for i in range(request.n):
            content = asyncio.create_task(generate_completion(gen_params, worker_addr))
            chat_completions.append(content)
        try:
            all_tasks = await asyncio.gather(*chat_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=content["text"]),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            if "usage" in content:
                task_usage = UsageInfo.parse_obj(content["usage"])
                for usage_key, usage_value in task_usage.dict().items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    s_time = time.time()
    usage = UsageInfo()
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        # class ChatCompletionResponseStreamChoice(BaseModel):
        #     index: int
        #     delta: DeltaMessage
        #     finish_reason: Optional[Literal["stop", "length"]]

        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name, usage=usage
        )
        # class ChatCompletionStreamResponse(BaseModel):
        #     id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
        #     object: str = "chat.completion.chunk"
        #     created: int = Field(default_factory=lambda: int(time.time()))
        #     model: str
        #     choices: List[ChatCompletionResponseStreamChoice]

        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = decoded_unicode

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            # class ChatCompletionResponseStreamChoice(BaseModel):
            #     index: int
            #     delta: DeltaMessage
            #     finish_reason: Optional[Literal["stop", "length"]]

            task_usage = UsageInfo.parse_obj(content["usage"])
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name, usage=task_usage
            )
            # class ChatCompletionStreamResponse(BaseModel):
            #     id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
            #     object: str = "chat.completion.chunk"
            #     created: int = Field(default_factory=lambda: int(time.time()))
            #     model: str
            #     choices: List[ChatCompletionResponseStreamChoice]

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    e_time = time.time()
    speed = finish_chunk.usage.completion_tokens / (e_time - s_time)
    yield f"data: [DONE]  cost: {e_time - s_time:.1f} s  speed: {speed:.1f} tokens/s\n\n"


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)

        for text in request.prompt:
            error_check_ret = await check_length(
                request, text, request.max_tokens, worker_addr, client
            )
            if error_check_ret is not None:
                return error_check_ret

        if request.stream:
            generator = generate_completion_stream_generator(
                request, request.n, worker_addr
            )
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            text_completions = []
            for text in request.prompt:
                gen_params = await get_gen_params(
                    request.model,
                    worker_addr,
                    text,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    echo=request.echo,
                    stream=request.stream,
                    stop=request.stop,
                )
                for i in range(request.n):
                    content = asyncio.create_task(
                        generate_completion(gen_params, worker_addr)
                    )
                    text_completions.append(content)

            try:
                all_tasks = await asyncio.gather(*text_completions)
            except Exception as e:
                return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

            choices = []
            usage = UsageInfo()
            for i, content in enumerate(all_tasks):
                if content["error_code"] != 0:
                    return create_error_response(content["error_code"], content["text"])
                choices.append(
                    CompletionResponseChoice(
                        index=i,
                        text=content["text"],
                        logprobs=content.get("logprobs", None),
                        finish_reason=content.get("finish_reason", "stop"),
                    )
                )
                task_usage = UsageInfo.parse_obj(content["usage"])
                for usage_key, usage_value in task_usage.dict().items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

            return CompletionResponse(
                model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
            )


async def generate_completion_stream_generator(
    request: CompletionRequest, n: int, worker_addr: str
):
    model_name = request.model
    id = f"cmpl-{shortuuid.random()}"
    finish_stream_events = []
    for text in request.prompt:
        for i in range(n):
            previous_text = ""
            gen_params = await get_gen_params(
                request.model,
                worker_addr,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                echo=request.echo,
                stream=request.stream,
                stop=request.stop,
            )
            async for content in generate_completion_stream(gen_params, worker_addr):
                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text) :]
                previous_text = decoded_unicode
                # todo: index is not apparent
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", None),
                )
                chunk = CompletionStreamResponse(
                    id=id,
                    object="text_completion",
                    choices=[choice_data],
                    model=model_name,
                )
                if len(delta_text) == 0:
                    if content.get("finish_reason", None) is not None:
                        finish_stream_events.append(chunk)
                    continue
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion_stream(payload: Dict[str, Any], worker_addr: str):
    controller_address = app_settings.controller_address
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
            # WORKER_API_TIMEOUT: 100
        ) as response:
            # content = await response.aread()
            async for raw_chunk in response.aiter_raw():
                for chunk in raw_chunk.split(delimiter):
                    if not chunk:
                        continue
                    data = json.loads(chunk.decode())
                    yield data


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    async with httpx.AsyncClient() as client:
        # 向worker_address发送worker_generate请求
        response = await client.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
            # WORKER_API_TIMEOUT: 100
        )
        completion = response.json()
        return completion


@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
@app.post("/v1/engines/{model_name}/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    request.input = process_input(request.model, request.input)

    data = []
    token_num = 0
    batch_size = WORKER_API_EMBEDDING_BATCH_SIZE
    batches = [
        request.input[i : min(i + batch_size, len(request.input))]
        for i in range(0, len(request.input), batch_size)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
        }
        embedding = await get_embedding(payload)
        if "error_code" in embedding and embedding["error_code"] != 0:
            return create_error_response(embedding["error_code"], embedding["text"])
        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * batch_size + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


async def get_embedding(payload: Dict[str, Any]):
    controller_address = app_settings.controller_address
    model_name = payload["model"]
    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(model_name, client)

        response = await client.post(
            worker_addr + "/worker_get_embeddings",
            headers=headers,
            json=payload,
            timeout=WORKER_API_TIMEOUT,
        )
        embedding = response.json()
        return embedding


### GENERAL API - NOT OPENAI COMPATIBLE ###


@app.post("/api/v1/token_check")
async def count_tokens(request: APITokenCheckRequest):

    # class APITokenCheckRequestItem(BaseModel):
    #     model: str
    #     prompt: str
    #     max_tokens: int
    #
    # class APITokenCheckRequest(BaseModel):
    #     prompts: List[APITokenCheckRequestItem]

    """
    Checks the token count for each message in your list
    This is not part of the OpenAI API spec.
    """
    checkedList = []
    async with httpx.AsyncClient() as client:
        for item in request.prompts:
            worker_addr = await get_worker_address(item.model, client)

            response = await client.post(
                worker_addr + "/model_details",
                headers=headers,
                json={"model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            context_len = response.json()["context_length"]

            response = await client.post(
                worker_addr + "/count_token",
                headers=headers,
                json={"prompt": item.prompt, "model": item.model},
                timeout=WORKER_API_TIMEOUT,
            )
            token_num = response.json()["count"]

            can_fit = True
            if token_num + item.max_tokens > context_len:
                can_fit = False

            checkedList.append(
                APITokenCheckResponseItem(
                    fits=can_fit, contextLength=context_len, tokenCount=token_num
                )
            )

    return APITokenCheckResponse(prompts=checkedList)


@app.post("/api/v1/chat/completions")
async def create_chat_completion(request: APIChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    async with httpx.AsyncClient() as client:
        worker_addr = await get_worker_address(request.model, client)

        gen_params = await get_gen_params(
            request.model,
            worker_addr,
            request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            echo=False,
            stream=request.stream,
            stop=request.stop,
        )

        if request.repetition_penalty is not None:
            gen_params["repetition_penalty"] = request.repetition_penalty

        error_check_ret = await check_length(
            request,
            gen_params["prompt"],
            gen_params["max_new_tokens"],
            worker_addr,
            client,
        )
        if error_check_ret is not None:
            return error_check_ret

        if request.stream:
            generator = chat_completion_stream_generator(
                request.model, gen_params, request.n, worker_addr
            )
            return StreamingResponse(generator, media_type="text/event-stream")

        choices = []
        chat_completions = []
        for i in range(request.n):
            content = asyncio.create_task(generate_completion(gen_params, worker_addr))
            chat_completions.append(content)
        try:
            all_tasks = await asyncio.gather(*chat_completions)
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=content["text"]),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


### END GENERAL API - NOT OPENAI COMPATIBLE ###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.api_keys = args.api_keys

    logger.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
