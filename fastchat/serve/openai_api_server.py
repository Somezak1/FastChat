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
import os
from typing import Generator, Optional, Union, Dict, List, Any

import aiohttp
import fastapi
from fastapi import Depends, HTTPException
from fastapi.exceptions import RequestValidationError
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
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
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
    LogProbs,
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

from fastchat.utils import build_logger

logger = build_logger("openai_api_server", "openai_api_server.log")

conv_template_map = {}

# ClientTimeout 用于设置整个会话 (session) 的超时时间, 默认情况下是300s超时
fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)


async def fetch_remote(url, pload=None, name=None):
    # 向url发送post请求, pload为传入的json数据
    # 如果调用请求后返回的结果中存在名为name的键, 则将该键对应的值返回
    # 如果调用请求后返回的结果中不存在名为name的键, 则将调用结果返回

    # 开启一个会话, 并在规定时间内 (10800s) 内完成 建立连接、发送请求、读取响应
    # 如果规定时间内没有完成则会报 asyncio.TimeoutError 错误
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "text": f"{response.reason}",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                    # ErrorCode.INTERNAL_ERROR: 50001
                }
                return json.dumps(ret)
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


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
    """
    判断controller所登记的众多models中是否有request.model
    如有则返回None, 如没有则返回一个JSON格式错误响应
    """
    controller_address = app_settings.controller_address
    ret = None

    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if request.model not in models:
        ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {request.model}",
        )
    return ret


async def check_length(request, prompt, max_tokens, worker_addr):
    if (
        not isinstance(max_tokens, int) or max_tokens <= 0
    ):  # model worker not support max_tokens=None
        max_tokens = 1024 * 1024

    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    # context_len: worker中加载的模型所能支持的最大上下文长度, 该值由模型权重路径下的config.json文件定义
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    length = min(max_tokens, context_len - token_num)

    if length <= 0:
        return None, create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. However, your messages resulted in {token_num} tokens. Please reduce the length of the messages.",
        )

    return length, None


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
            f"{request.top_p} is greater than the maximum of 1 - 'top_p'",
        )
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
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


def create_openai_logprobs(logprob_dict):
    """Create OpenAI-style logprobs."""
    return LogProbs(**logprob_dict) if logprob_dict is not None else None


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


async def get_gen_params(
    model_name: str,
    # model_name: API接口调用时传入的"model"值
    worker_addr: str,
    messages: Union[str, List[Dict[str, str]]],
    # messages: API接口调用时传入的"messages"值
    *,
    temperature: float,
    # temperature: API接口调用时传入的"temperature"值, 如不指定则采用默认值0.7
    top_p: float,
    # top_p: API接口调用时传入的"top_p"值, 如不指定则采用默认值1.0
    top_k: Optional[int],
    # top_k: API接口调用时传入的"top_k"值, 如不指定则采用默认值-1
    presence_penalty: Optional[float],
    # presence_penalty: API接口调用时传入的"presence_penalty"值, 如不指定则采用默认值0.0
    frequency_penalty: Optional[float],
    # frequency_penalty: API接口调用时传入的"frequency_penalty"值, 如不指定则采用默认值0.0
    max_tokens: Optional[int],
    # max_tokens: API接口调用时传入的"max_tokens"值, 如不指定则采用默认值None
    echo: Optional[bool],
    # echo: False
    logprobs: Optional[int] = None,
    # logprobs: None
    stop: Optional[Union[str, List[str]]],
    # stop: API接口调用时传入的"stop"值, 如不指定则采用默认值None
    best_of: Optional[int] = None,
    # best_of: None
    use_beam_search: Optional[bool] = None,
    # use_beam_search: None
) -> Dict[str, Any]:
    # 获取推断时所需的model, prompt, 温度, max_new_tokens 等信息

    # 向worker发送"/worker_get_conv_template"请求, 获取worker_addr下指定request.model的对话模板
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

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        # 按对话模板中指定的方式, 将API接口调用时传入的"messages"拼接起来
        "temperature": temperature,
        # temperature: API接口调用时传入的"temperature"值, 如不指定则采用默认值0.7
        "logprobs": logprobs,
        # logprobs: None
        "top_p": top_p,
        # top_p: API接口调用时传入的"top_p"值, 如不指定则采用默认值1.0
        "top_k": top_k,
        # top_k: API接口调用时传入的"top_k"值, 如不指定则采用默认值-1
        "presence_penalty": presence_penalty,
        # presence_penalty: API接口调用时传入的"presence_penalty"值, 如不指定则采用默认值0.0
        "frequency_penalty": frequency_penalty,
        # frequency_penalty: API接口调用时传入的"frequency_penalty"值, 如不指定则采用默认值0.0
        "max_new_tokens": max_tokens,
        # max_tokens: API接口调用时传入的"max_tokens"值, 如不指定则采用默认值None
        "echo": echo,
        # echo: None
        "stop_token_ids": conv.stop_token_ids,
    }

    # best_of: None
    if best_of is not None:
        gen_params.update({"best_of": best_of})
    # use_beam_search: None
    if use_beam_search is not None:
        gen_params.update({"use_beam_search": use_beam_search})

    new_stop = set()
    _add_to_set(stop, new_stop)
    _add_to_set(conv.stop_str, new_stop)

    gen_params["stop"] = list(new_stop)

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


async def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    # 若controller.dispatch_method==LOTTERY, 则从所有名称为model_name的worker中随机挑一个返回
    # 若controller.dispatch_method==SHORTEST_QUEUE, 则从所有名称为model_name的worker中挑一个负载最小的返回
    controller_address = app_settings.controller_address

    # 向controller_address发送get_worker_address请求
    worker_addr = await fetch_remote(
        controller_address + "/get_worker_address", {"model": model_name}, "address"
    )

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    logger.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str, worker_addr: str):
    # 向worker_address通信, 得到该worker初始化时的对话模板conv_template
    conv_template = conv_template_map.get((worker_addr, model_name))
    # 先去字典conv_template_map中查询这个worker_address下该model_name的对话模板
    # 如果字典中没有该信息, 就向worker_address发送worker_get_conv_template的请求, 将获取到的conv_template存储在字典中
    # 下次再调用worker_address下该model_name时候, 就不用反复与worker通讯来获取其对话模板了
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    controller_address = app_settings.controller_address
    ret = await fetch_remote(controller_address + "/refresh_all_workers")
    models = await fetch_remote(controller_address + "/list_models", None, "models")

    models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m in models:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1chat/completions", dependencies=[Depends(check_api_key)])
@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""

    # API调用时未指定的参数使用如下默认值:
    # class ChatCompletionRequest(BaseModel):
    #     model: str
    #     messages: Union[str, List[Dict[str, str]]]
    #     temperature: Optional[float] = 0.7
    #     top_p: Optional[float] = 1.0
    #     top_k: Optional[int] = -1
    #     n: Optional[int] = 1
    #     max_tokens: Optional[int] = None
    #     stop: Optional[Union[str, List[str]]] = None
    #     stream: Optional[bool] = False
    #     presence_penalty: Optional[float] = 0.0
    #     frequency_penalty: Optional[float] = 0.0
    #     user: Optional[str] = None

    # 向controller发送"/list_models"请求, 判断controller所登记的众多models中是否有request.model
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    # 检查一下request中的参数是否符合规范
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    # 向controller发送"/get_worker_address"请求, 从controller中所有名称为request.model的worker_addr中挑一个返回
    worker_addr = await get_worker_address(request.model)

    # 向worker_addr发送"/worker_get_conv_template"请求, 获取worker_addr下指定request.model的对话模板
    # 同时根据对话模板及传入的messages将文本拼接成输入模型的prompt, 汇总模型其他的生成参数得到gen_params
    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=request.max_tokens,
        echo=False,
        stop=request.stop,
    )

    # 向worker_addr发送"/model_details"请求, 获取request.model的最大上下文长度context_len
    # 向worker_addr发送"/count_token"请求, 获取gen_params["prompt"]的token_num
    # 最后返回 min(gen_params["max_new_tokens"], context_len - token_num), 若为负数则报错
    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )
    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

    # 流式调用
    if request.stream:
        generator = chat_completion_stream_generator(
            request.model, gen_params, request.n, worker_addr
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    # 非流式调用
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

    # 调用:
    # curl http://localhost:8001/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    #     "model":"Llama-2-13b-chat-hf",
    #     "max_tokens":500,
    #     "messages":[{"content":"What is the boiling point of water","role":"user"}],
    #     "stream":false,
    #     "temperature":0
    #   }'
    # 打印信息例举:
    # {"id":"chatcmpl-3mZ5bo8Damkzk5HDMywRqU","object":"chat.completion","created":1701348029,"model":"Llama-2-13b-chat-hf","choices":[{"index":0,"message":{"role":"assistant","content":" The boiling point of water is 100 degrees Celsius (°C) or 212 degrees Fahrenheit (°F) at standard atmospheric pressure."},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"total_tokens":56,"completion_tokens":40}}
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
    # class UsageInfo(BaseModel):
    #     prompt_tokens: int = 0
    #     total_tokens: int = 0
    #     completion_tokens: Optional[int] = 0

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
        #     finish_reason: Optional[Literal["stop", "length"]] = None

        # class DeltaMessage(BaseModel):
        #     role: Optional[str] = None
        #     content: Optional[str] = None

        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name, usage=usage
        )
        # class ChatCompletionStreamResponse(BaseModel):
        #     id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
        #     object: str = "chat.completion.chunk"
        #     created: int = Field(default_factory=lambda: int(time.time()))
        #     model: str
        #     choices: List[ChatCompletionResponseStreamChoice]
        #     usage: UsageInfo

        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
        # 调用:
        # curl http://localhost:8001/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        #     "model":"Llama-2-13b-chat-hf",
        #     "max_tokens":500,
        #     "messages":[{"content":"What is the boiling point of water","role":"user"}],
        #     "stream":true,
        #     "temperature":0
        #   }'
        # 打印信息例举:
        # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}], "usage": {}}

        # previous_text储存前面已打印/已返回的字符
        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            # 剔除返回字符中之前已打印的字符
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = (
                decoded_unicode
                if len(decoded_unicode) > len(previous_text)
                else previous_text
            )

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
            #     finish_reason: Optional[Literal["stop", "length"]] = None

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
            #     usage: UsageInfo

            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            # 打印信息例举:
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " The bo"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 18, "completion_tokens": 2}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "iling point"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 20, "completion_tokens": 4}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " of water"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 22, "completion_tokens": 6}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " is "}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 24, "completion_tokens": 8}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "10"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 26, "completion_tokens": 10}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "0 degrees"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 28, "completion_tokens": 12}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " Celsi"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 30, "completion_tokens": 14}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "us ("}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 32, "completion_tokens": 16}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "°C"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 34, "completion_tokens": 18}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": ") or"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 36, "completion_tokens": 20}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " 2"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 38, "completion_tokens": 22}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "12"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 40, "completion_tokens": 24}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " degrees F"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 42, "completion_tokens": 26}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "ahrenheit"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 44, "completion_tokens": 28}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " (°"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 46, "completion_tokens": 30}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "F)"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 48, "completion_tokens": 32}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " at standard"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 50, "completion_tokens": 34}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": " atmospher"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 52, "completion_tokens": 36}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "ic pressure"}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 54, "completion_tokens": 38}}
            #
            # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {"content": "."}, "finish_reason": null}], "usage": {"prompt_tokens": 16, "total_tokens": 56, "completion_tokens": 40}}

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
        # 打印信息例举:
        # data: {"id": "chatcmpl-kQ9woA4Nno8DC6MEbBnFdQ", "object": "chat.completion.chunk", "created": 1701347972, "model": "Llama-2-13b-chat-hf", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 16, "total_tokens": 56, "completion_tokens": 40}}
    e_time = time.time()
    speed = finish_chunk.usage.completion_tokens / (e_time - s_time)
    yield f"data: [DONE]  cost: {e_time - s_time:.1f} s  speed: {speed:.1f} tokens/s\n\n"
    # 打印信息例举:
    # data: [DONE]  cost: 1.5 s  speed: 26.5 tokens/s


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionRequest):
    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    request.prompt = process_input(request.model, request.prompt)

    worker_addr = await get_worker_address(request.model)
    for text in request.prompt:
        max_tokens, error_check_ret = await check_length(
            request, text, request.max_tokens, worker_addr
        )
        if error_check_ret is not None:
            return error_check_ret

        if isinstance(max_tokens, int) and max_tokens < request.max_tokens:
            request.max_tokens = max_tokens

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
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
                best_of=request.best_of,
                use_beam_search=request.use_beam_search,
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
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
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
                top_k=request.top_k,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                max_tokens=request.max_tokens,
                logprobs=request.logprobs,
                echo=request.echo,
                stop=request.stop,
            )
            async for content in generate_completion_stream(gen_params, worker_addr):
                if content["error_code"] != 0:
                    yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                decoded_unicode = content["text"].replace("\ufffd", "")
                delta_text = decoded_unicode[len(previous_text) :]
                previous_text = (
                    decoded_unicode
                    if len(decoded_unicode) > len(previous_text)
                    else previous_text
                )
                # todo: index is not apparent
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text=delta_text,
                    logprobs=create_openai_logprobs(content.get("logprobs", None)),
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
    # 在 Python 中, 访问网络资源最有名的库就是 requests、aiohttp 和 httpx
    # 一般情况下, requests 只能发送同步请求, aiohttp 只能发送异步请求, httpx 既能发送同步请求, 又能发送异步请求
    # 使用 async with httpx.AsyncClient() as client, 打开和关闭 Client
    async with httpx.AsyncClient() as client:
        delimiter = b"\0"
        async with client.stream(
            "POST",
            worker_addr + "/worker_generate_stream",
            headers=headers,
            # headers: {"User-Agent": "FastChat API Server"}
            json=payload,
            timeout=WORKER_API_TIMEOUT,
            # WORKER_API_TIMEOUT: 100
        ) as response:
            # content = await response.aread()
            buffer = b""
            # response.aiter_raw(): 用于流式传输原始响应字节, 无需应用内容解码
            async for raw_chunk in response.aiter_raw():
                # worker那边每次返回的内容是json.dumps(ret).encode() + b"\0", 结尾都是b"\0"
                # 将空字符串与worker单次的返回内容拼接
                buffer += raw_chunk
                # 将历次返回的二进制信息拼接
                while (chunk_end := buffer.find(delimiter)) >= 0:
                    # 使用b"\0"作为分隔符进行二进制字符串的分割
                    # 这就像s="something-"; chunk, buffer = s.split('-')[0], s.split('-')[-1]
                    # chunk 为 "something"; buffer 为 ""
                    chunk, buffer = buffer[:chunk_end], buffer[chunk_end + 1 :]
                    if not chunk:
                        continue
                    yield json.loads(chunk.decode())


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    return await fetch_remote(worker_addr + "/worker_generate", payload, "")


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
            "encoding_format": request.encoding_format,
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
    worker_addr = await get_worker_address(model_name)

    embedding = await fetch_remote(worker_addr + "/worker_get_embeddings", payload)
    return json.loads(embedding)


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
    for item in request.prompts:
        worker_addr = await get_worker_address(item.model)

        context_len = await fetch_remote(
            worker_addr + "/model_details",
            {"prompt": item.prompt, "model": item.model},
            "context_length",
        )

        token_num = await fetch_remote(
            worker_addr + "/count_token",
            {"prompt": item.prompt, "model": item.model},
            "count",
        )

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

    worker_addr = await get_worker_address(request.model)

    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=request.max_tokens,
        echo=False,
        stop=request.stop,
    )

    if request.repetition_penalty is not None:
        gen_params["repetition_penalty"] = request.repetition_penalty

    max_new_tokens, error_check_ret = await check_length(
        request,
        gen_params["prompt"],
        gen_params["max_new_tokens"],
        worker_addr,
    )

    if error_check_ret is not None:
        return error_check_ret

    gen_params["max_new_tokens"] = max_new_tokens

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


def create_openai_api_server():
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
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
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
    # args:
    # Namespace(host='localhost', port=8001, controller_address='http://localhost:21001', allow_credentials=False,
    # allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_keys=None, ssl=False)
    return args


if __name__ == "__main__":
    # 此代码的注释都是基于如下指令debug获得的
    # ================================================================================================================
    # python3 -m fastchat.serve.openai_api_server --host localhost --port 8001
    # ================================================================================================================
    args = create_openai_api_server()
    # args.ssl: False
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
