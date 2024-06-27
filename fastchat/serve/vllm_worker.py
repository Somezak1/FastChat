"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop


app = FastAPI()


class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLMEngine,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.tokenizer = llm_engine.engine.tokenizer
        # This is to support vllm >= 0.2.7 where TokenizerGroup was introduced
        # and llm_engine.engine.tokenizer was no longer a raw tokenizer
        if hasattr(self.tokenizer, "tokenizer"):
            self.tokenizer = llm_engine.engine.tokenizer.tokenizer
        self.context_len = get_context_length(llm_engine.engine.model_config.hf_config)

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)

        request = params.get("request", None)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        # 传入 SamplingParams 的参数 取值一览
        # temperature:       可以通过参数 "temperature" 指定, 默认值 0.7
        # top_p:             可以通过参数 "top_p" 指定, 若不指定则为 1.0, 若指定则为 max(top_p, 1e-5). 且当 temperature < 1e-5 时, top_p 会无视前面规则, 强行为 1.0
        # top_k:             可以通过参数 "top_k" 指定, 默认值 -1
        # presence_penalty:  可以通过参数 "presence_penalty" 指定, 默认值 0.0
        # frequency_penalty: 可以通过参数 "frequency_penalty" 指定, 默认值 0.0
        # max_new_tokens:    可以通过参数 "max_tokens" 指定, 若不指定则为 context_len - prompt_len, 若指定则为 min(max_tokens, context_len - prompt_len). context_len 由 config.json 决定, 一般为 rope_scaling_factor * max_position_embeddings, 如果 config.json 中未定义 rope_scaling_factor, 则 rope_scaling_factor 默认为 1
        # use_beam_search:   只要不修改 fastchat 源码就无法通过参数指定, 值为 False
        # best_of:           只要不修改 fastchat 源码就无法通过参数指定, 值为 None
        # n:                 只要不修改 fastchat 源码就无法通过参数指定, 值为 1
        # stop_token_ids:    只要不修改 fastchat 源码就无法通过参数指定, conv 是检索到的对话模板, 若 conv.stop_token_ids 存在, 则值为 conv.stop_token_ids, 若不存在则为 []. 此外如果 tokenizer.eos_token_id 存在, 就将 tokenizer.eos_token_id 添加到上述结果中去
        # stop:              stop 由 stop_str 决定, 而 stop_str 则受传入 "stop" 参数的值影响, 传入的值既可是字符串, 也可以是字符串组成的列表. 若 conversation.py 中的 conv.stop_str 存在, 则 stop_str = your_stop + [conv.stop_str]; 若 conversation.py 中 conv.stop_str 不存在, 则 stop_str = your_stop, 如果你自己也不指定, 则 stop_str = None
        #                    按下面逻辑形成最终的 stop
        #                    stop = set()
        #                    if isinstance(stop_str, str) and stop_str != "":
        #                        stop.add(stop_str)
        #                    elif isinstance(stop_str, list) and stop_str != []:
        #                        stop.update(stop_str)
        #                    for tid in stop_token_ids:
        #                        if tid is not None:
        #                            s = self.tokenizer.decode(tid)
        #                            if s != "":
        #                                stop.add(s)
        #                    stop = list(stop)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = " ".join(text_outputs)

            partial_stop = any(is_partial_stop(text_outputs, i) for i in stop)
            # prevent yielding partial stop sequence
            if partial_stop:
                continue

            aborted = False
            if request and await request.is_disconnected():
                await engine.abort(request_id)
                request_output.finished = True
                aborted = True
                for output in request_output.outputs:
                    output.finish_reason = "abort"

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [
                    output.cumulative_logprob for output in request_output.outputs
                ],
                "finish_reason": request_output.outputs[0].finish_reason
                if len(request_output.outputs) == 1
                else [output.finish_reason for output in request_output.outputs],
            }
            # Emit twice here to ensure a 'finish_reason' with empty content in the OpenAI API response.
            # This aligns with the behavior of model_worker.
            if request_output.finished:
                yield (json.dumps({**ret, **{"finish_reason": None}}) + "\0").encode()
            yield (json.dumps(ret) + "\0").encode()

            if aborted:
                break

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)

    # AsyncEngineArgs.add_cli_args(parser) 为 parser 添加了以下参数:
    #         # Model arguments
    #         parser.add_argument(
    #             '--model',
    #             type=str,
    #             default='facebook/opt-125m',
    #             help='Name or path of the huggingface model to use.')
    #         parser.add_argument(
    #             '--tokenizer',
    #             type=nullable_str,
    #             default=EngineArgs.tokenizer,
    #             help='Name or path of the huggingface tokenizer to use. '
    #             'If unspecified, model name or path will be used.')
    #         parser.add_argument(
    #             '--skip-tokenizer-init',
    #             action='store_true',
    #             help='Skip initialization of tokenizer and detokenizer')
    #         parser.add_argument(
    #             '--revision',
    #             type=nullable_str,
    #             default=None,
    #             help='The specific model version to use. It can be a branch '
    #             'name, a tag name, or a commit id. If unspecified, will use '
    #             'the default version.')
    #         parser.add_argument(
    #             '--code-revision',
    #             type=nullable_str,
    #             default=None,
    #             help='The specific revision to use for the model code on '
    #             'Hugging Face Hub. It can be a branch name, a tag name, or a '
    #             'commit id. If unspecified, will use the default version.')
    #         parser.add_argument(
    #             '--tokenizer-revision',
    #             type=nullable_str,
    #             default=None,
    #             help='Revision of the huggingface tokenizer to use. '
    #             'It can be a branch name, a tag name, or a commit id. '
    #             'If unspecified, will use the default version.')
    #         parser.add_argument(
    #             '--tokenizer-mode',
    #             type=str,
    #             default=EngineArgs.tokenizer_mode,
    #             choices=['auto', 'slow'],
    #             help='The tokenizer mode.\n\n* "auto" will use the '
    #             'fast tokenizer if available.\n* "slow" will '
    #             'always use the slow tokenizer.')
    #         parser.add_argument('--trust-remote-code',
    #                             action='store_true',
    #                             help='Trust remote code from huggingface.')
    #         parser.add_argument('--download-dir',
    #                             type=nullable_str,
    #                             default=EngineArgs.download_dir,
    #                             help='Directory to download and load the weights, '
    #                             'default to the default cache dir of '
    #                             'huggingface.')
    #         parser.add_argument(
    #             '--load-format',
    #             type=str,
    #             default=EngineArgs.load_format,
    #             choices=[
    #                 'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
    #                 'bitsandbytes'
    #             ],
    #             help='The format of the model weights to load.\n\n'
    #             '* "auto" will try to load the weights in the safetensors format '
    #             'and fall back to the pytorch bin format if safetensors format '
    #             'is not available.\n'
    #             '* "pt" will load the weights in the pytorch bin format.\n'
    #             '* "safetensors" will load the weights in the safetensors format.\n'
    #             '* "npcache" will load the weights in pytorch format and store '
    #             'a numpy cache to speed up the loading.\n'
    #             '* "dummy" will initialize the weights with random values, '
    #             'which is mainly for profiling.\n'
    #             '* "tensorizer" will load the weights using tensorizer from '
    #             'CoreWeave. See the Tensorize vLLM Model script in the Examples'
    #             'section for more information.\n'
    #             '* "bitsandbytes" will load the weights using bitsandbytes '
    #             'quantization.\n')
    #         parser.add_argument(
    #             '--dtype',
    #             type=str,
    #             default=EngineArgs.dtype,
    #             choices=[
    #                 'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
    #             ],
    #             help='Data type for model weights and activations.\n\n'
    #             '* "auto" will use FP16 precision for FP32 and FP16 models, and '
    #             'BF16 precision for BF16 models.\n'
    #             '* "half" for FP16. Recommended for AWQ quantization.\n'
    #             '* "float16" is the same as "half".\n'
    #             '* "bfloat16" for a balance between precision and range.\n'
    #             '* "float" is shorthand for FP32 precision.\n'
    #             '* "float32" for FP32 precision.')
    #         parser.add_argument(
    #             '--kv-cache-dtype',
    #             type=str,
    #             choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
    #             default=EngineArgs.kv_cache_dtype,
    #             help='Data type for kv cache storage. If "auto", will use model '
    #             'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
    #             'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    #         parser.add_argument(
    #             '--quantization-param-path',
    #             type=nullable_str,
    #             default=None,
    #             help='Path to the JSON file containing the KV cache '
    #             'scaling factors. This should generally be supplied, when '
    #             'KV cache dtype is FP8. Otherwise, KV cache scaling factors '
    #             'default to 1.0, which may cause accuracy issues. '
    #             'FP8_E5M2 (without scaling) is only supported on cuda version'
    #             'greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead '
    #             'supported for common inference criteria.')
    #         parser.add_argument('--max-model-len',
    #                             type=int,
    #                             default=EngineArgs.max_model_len,
    #                             help='Model context length. If unspecified, will '
    #                             'be automatically derived from the model config.')
    #         parser.add_argument(
    #             '--guided-decoding-backend',
    #             type=str,
    #             default='outlines',
    #             choices=['outlines', 'lm-format-enforcer'],
    #             help='Which engine will be used for guided decoding'
    #             ' (JSON schema / regex etc) by default. Currently support '
    #             'https://github.com/outlines-dev/outlines and '
    #             'https://github.com/noamgat/lm-format-enforcer.'
    #             ' Can be overridden per request via guided_decoding_backend'
    #             ' parameter.')
    #         # Parallel arguments
    #         parser.add_argument(
    #             '--distributed-executor-backend',
    #             choices=['ray', 'mp'],
    #             default=EngineArgs.distributed_executor_backend,
    #             help='Backend to use for distributed serving. When more than 1 GPU '
    #             'is used, will be automatically set to "ray" if installed '
    #             'or "mp" (multiprocessing) otherwise.')
    #         parser.add_argument(
    #             '--worker-use-ray',
    #             action='store_true',
    #             help='Deprecated, use --distributed-executor-backend=ray.')
    #         parser.add_argument('--pipeline-parallel-size',
    #                             '-pp',
    #                             type=int,
    #                             default=EngineArgs.pipeline_parallel_size,
    #                             help='Number of pipeline stages.')
    #         parser.add_argument('--tensor-parallel-size',
    #                             '-tp',
    #                             type=int,
    #                             default=EngineArgs.tensor_parallel_size,
    #                             help='Number of tensor parallel replicas.')
    #         parser.add_argument(
    #             '--max-parallel-loading-workers',
    #             type=int,
    #             default=EngineArgs.max_parallel_loading_workers,
    #             help='Load model sequentially in multiple batches, '
    #             'to avoid RAM OOM when using tensor '
    #             'parallel and large models.')
    #         parser.add_argument(
    #             '--ray-workers-use-nsight',
    #             action='store_true',
    #             help='If specified, use nsight to profile Ray workers.')
    #         # KV cache arguments
    #         parser.add_argument('--block-size',
    #                             type=int,
    #                             default=EngineArgs.block_size,
    #                             choices=[8, 16, 32],
    #                             help='Token block size for contiguous chunks of '
    #                             'tokens.')
    #
    #         parser.add_argument('--enable-prefix-caching',
    #                             action='store_true',
    #                             help='Enables automatic prefix caching.')
    #         parser.add_argument('--disable-sliding-window',
    #                             action='store_true',
    #                             help='Disables sliding window, '
    #                             'capping to sliding window size')
    #         parser.add_argument('--use-v2-block-manager',
    #                             action='store_true',
    #                             help='Use BlockSpaceMangerV2.')
    #         parser.add_argument(
    #             '--num-lookahead-slots',
    #             type=int,
    #             default=EngineArgs.num_lookahead_slots,
    #             help='Experimental scheduling config necessary for '
    #             'speculative decoding. This will be replaced by '
    #             'speculative config in the future; it is present '
    #             'to enable correctness tests until then.')
    #
    #         parser.add_argument('--seed',
    #                             type=int,
    #                             default=EngineArgs.seed,
    #                             help='Random seed for operations.')
    #         parser.add_argument('--swap-space',
    #                             type=int,
    #                             default=EngineArgs.swap_space,
    #                             help='CPU swap space size (GiB) per GPU.')
    #         parser.add_argument(
    #             '--gpu-memory-utilization',
    #             type=float,
    #             default=EngineArgs.gpu_memory_utilization,
    #             help='The fraction of GPU memory to be used for the model '
    #             'executor, which can range from 0 to 1. For example, a value of '
    #             '0.5 would imply 50%% GPU memory utilization. If unspecified, '
    #             'will use the default value of 0.9.')
    #         parser.add_argument(
    #             '--num-gpu-blocks-override',
    #             type=int,
    #             default=None,
    #             help='If specified, ignore GPU profiling result and use this number'
    #             'of GPU blocks. Used for testing preemption.')
    #         parser.add_argument('--max-num-batched-tokens',
    #                             type=int,
    #                             default=EngineArgs.max_num_batched_tokens,
    #                             help='Maximum number of batched tokens per '
    #                             'iteration.')
    #         parser.add_argument('--max-num-seqs',
    #                             type=int,
    #                             default=EngineArgs.max_num_seqs,
    #                             help='Maximum number of sequences per iteration.')
    #         parser.add_argument(
    #             '--max-logprobs',
    #             type=int,
    #             default=EngineArgs.max_logprobs,
    #             help=('Max number of log probs to return logprobs is specified in'
    #                   ' SamplingParams.'))
    #         parser.add_argument('--disable-log-stats',
    #                             action='store_true',
    #                             help='Disable logging statistics.')
    #         # Quantization settings.
    #         parser.add_argument('--quantization',
    #                             '-q',
    #                             type=nullable_str,
    #                             choices=[*QUANTIZATION_METHODS, None],
    #                             default=EngineArgs.quantization,
    #                             help='Method used to quantize the weights. If '
    #                             'None, we first check the `quantization_config` '
    #                             'attribute in the model config file. If that is '
    #                             'None, we assume the model weights are not '
    #                             'quantized and use `dtype` to determine the data '
    #                             'type of the weights.')
    #         parser.add_argument('--rope-scaling',
    #                             default=None,
    #                             type=json.loads,
    #                             help='RoPE scaling configuration in JSON format. '
    #                             'For example, {"type":"dynamic","factor":2.0}')
    #         parser.add_argument('--rope-theta',
    #                             default=None,
    #                             type=float,
    #                             help='RoPE theta. Use with `rope_scaling`. In '
    #                             'some cases, changing the RoPE theta improves the '
    #                             'performance of the scaled model.')
    #         parser.add_argument('--enforce-eager',
    #                             action='store_true',
    #                             help='Always use eager-mode PyTorch. If False, '
    #                             'will use eager mode and CUDA graph in hybrid '
    #                             'for maximal performance and flexibility.')
    #         parser.add_argument('--max-context-len-to-capture',
    #                             type=int,
    #                             default=EngineArgs.max_context_len_to_capture,
    #                             help='Maximum context length covered by CUDA '
    #                             'graphs. When a sequence has context length '
    #                             'larger than this, we fall back to eager mode. '
    #                             '(DEPRECATED. Use --max-seq-len-to-capture instead'
    #                             ')')
    #         parser.add_argument('--max-seq-len-to-capture',
    #                             type=int,
    #                             default=EngineArgs.max_seq_len_to_capture,
    #                             help='Maximum sequence length covered by CUDA '
    #                             'graphs. When a sequence has context length '
    #                             'larger than this, we fall back to eager mode.')
    #         parser.add_argument('--disable-custom-all-reduce',
    #                             action='store_true',
    #                             default=EngineArgs.disable_custom_all_reduce,
    #                             help='See ParallelConfig.')
    #         parser.add_argument('--tokenizer-pool-size',
    #                             type=int,
    #                             default=EngineArgs.tokenizer_pool_size,
    #                             help='Size of tokenizer pool to use for '
    #                             'asynchronous tokenization. If 0, will '
    #                             'use synchronous tokenization.')
    #         parser.add_argument('--tokenizer-pool-type',
    #                             type=str,
    #                             default=EngineArgs.tokenizer_pool_type,
    #                             help='Type of tokenizer pool to use for '
    #                             'asynchronous tokenization. Ignored '
    #                             'if tokenizer_pool_size is 0.')
    #         parser.add_argument('--tokenizer-pool-extra-config',
    #                             type=nullable_str,
    #                             default=EngineArgs.tokenizer_pool_extra_config,
    #                             help='Extra config for tokenizer pool. '
    #                             'This should be a JSON string that will be '
    #                             'parsed into a dictionary. Ignored if '
    #                             'tokenizer_pool_size is 0.')
    #         # LoRA related configs
    #         parser.add_argument('--enable-lora',
    #                             action='store_true',
    #                             help='If True, enable handling of LoRA adapters.')
    #         parser.add_argument('--max-loras',
    #                             type=int,
    #                             default=EngineArgs.max_loras,
    #                             help='Max number of LoRAs in a single batch.')
    #         parser.add_argument('--max-lora-rank',
    #                             type=int,
    #                             default=EngineArgs.max_lora_rank,
    #                             help='Max LoRA rank.')
    #         parser.add_argument(
    #             '--lora-extra-vocab-size',
    #             type=int,
    #             default=EngineArgs.lora_extra_vocab_size,
    #             help=('Maximum size of extra vocabulary that can be '
    #                   'present in a LoRA adapter (added to the base '
    #                   'model vocabulary).'))
    #         parser.add_argument(
    #             '--lora-dtype',
    #             type=str,
    #             default=EngineArgs.lora_dtype,
    #             choices=['auto', 'float16', 'bfloat16', 'float32'],
    #             help=('Data type for LoRA. If auto, will default to '
    #                   'base model dtype.'))
    #         parser.add_argument(
    #             '--long-lora-scaling-factors',
    #             type=nullable_str,
    #             default=EngineArgs.long_lora_scaling_factors,
    #             help=('Specify multiple scaling factors (which can '
    #                   'be different from base model scaling factor '
    #                   '- see eg. Long LoRA) to allow for multiple '
    #                   'LoRA adapters trained with those scaling '
    #                   'factors to be used at the same time. If not '
    #                   'specified, only adapters trained with the '
    #                   'base model scaling factor are allowed.'))
    #         parser.add_argument(
    #             '--max-cpu-loras',
    #             type=int,
    #             default=EngineArgs.max_cpu_loras,
    #             help=('Maximum number of LoRAs to store in CPU memory. '
    #                   'Must be >= than max_num_seqs. '
    #                   'Defaults to max_num_seqs.'))
    #         parser.add_argument(
    #             '--fully-sharded-loras',
    #             action='store_true',
    #             help=('By default, only half of the LoRA computation is '
    #                   'sharded with tensor parallelism. '
    #                   'Enabling this will use the fully sharded layers. '
    #                   'At high sequence length, max rank or '
    #                   'tensor parallel size, this is likely faster.'))
    #         parser.add_argument("--device",
    #                             type=str,
    #                             default=EngineArgs.device,
    #                             choices=["auto", "cuda", "neuron", "cpu", "tpu"],
    #                             help='Device type for vLLM execution.')
    #
    #         # Related to Vision-language models such as llava
    #         parser = EngineArgs.add_cli_args_for_vlm(parser)
    #
    #         parser.add_argument(
    #             '--scheduler-delay-factor',
    #             type=float,
    #             default=EngineArgs.scheduler_delay_factor,
    #             help='Apply a delay (of delay factor multiplied by previous'
    #             'prompt latency) before scheduling next prompt.')
    #         parser.add_argument(
    #             '--enable-chunked-prefill',
    #             action='store_true',
    #             help='If set, the prefill requests can be chunked based on the '
    #             'max_num_batched_tokens.')
    #
    #         parser.add_argument(
    #             '--speculative-model',
    #             type=nullable_str,
    #             default=EngineArgs.speculative_model,
    #             help=
    #             'The name of the draft model to be used in speculative decoding.')
    #         parser.add_argument(
    #             '--num-speculative-tokens',
    #             type=int,
    #             default=EngineArgs.num_speculative_tokens,
    #             help='The number of speculative tokens to sample from '
    #             'the draft model in speculative decoding.')
    #
    #         parser.add_argument(
    #             '--speculative-max-model-len',
    #             type=int,
    #             default=EngineArgs.speculative_max_model_len,
    #             help='The maximum sequence length supported by the '
    #             'draft model. Sequences over this length will skip '
    #             'speculation.')
    #
    #         parser.add_argument(
    #             '--speculative-disable-by-batch-size',
    #             type=int,
    #             default=EngineArgs.speculative_disable_by_batch_size,
    #             help='Disable speculative decoding for new incoming requests '
    #             'if the number of enqueue requests is larger than this value.')
    #
    #         parser.add_argument(
    #             '--ngram-prompt-lookup-max',
    #             type=int,
    #             default=EngineArgs.ngram_prompt_lookup_max,
    #             help='Max size of window for ngram prompt lookup in speculative '
    #             'decoding.')
    #
    #         parser.add_argument(
    #             '--ngram-prompt-lookup-min',
    #             type=int,
    #             default=EngineArgs.ngram_prompt_lookup_min,
    #             help='Min size of window for ngram prompt lookup in speculative '
    #             'decoding.')
    #
    #         parser.add_argument('--model-loader-extra-config',
    #                             type=nullable_str,
    #                             default=EngineArgs.model_loader_extra_config,
    #                             help='Extra config for model loader. '
    #                             'This will be passed to the model loader '
    #                             'corresponding to the chosen load_format. '
    #                             'This should be a JSON string that will be '
    #                             'parsed into a dictionary.')
    #         parser.add_argument(
    #             '--preemption_mode',
    #             type=str,
    #             default=None,
    #             help='If \'recompute\', the engine performs preemption by block '
    #             'swapping; If \'swap\', the engine performs preemption by block '
    #             'swapping.')
    #
    #         parser.add_argument(
    #             "--served-model-name",
    #             nargs="+",
    #             type=str,
    #             default=None,
    #             help="The model name(s) used in the API. If multiple "
    #             "names are provided, the server will respond to any "
    #             "of the provided names. The model name in the model "
    #             "field of a response will be the first name in this "
    #             "list. If not specified, the model name will be the "
    #             "same as the `--model` argument. Noted that this name(s)"
    #             "will also be used in `model_name` tag content of "
    #             "prometheus metrics, if multiple names provided, metrics"
    #             "tag will take the first one.")
    #         parser.add_argument('--qlora-adapter-name-or-path',
    #                             type=str,
    #                             default=None,
    #                             help='Name or path of the QLoRA adapter.')

    #         parser.add_argument('--engine-use-ray',
    #                             action='store_true',
    #                             help='Use Ray to start the LLM engine in a '
    #                             'separate process as the server process.')
    #         parser.add_argument('--disable-log-requests',
    #                             action='store_true',
    #                             help='Disable logging requests.')
    #         parser.add_argument('--max-log-len',
    #                             type=int,
    #                             default=None,
    #                             help='Max number of prompt characters or prompt '
    #                             'ID numbers being printed in log.'
    #                             '\n\nDefault: Unlimited')

    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
