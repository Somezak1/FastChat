"""Send a test message."""
import argparse
import json

import requests

from fastchat.model.model_adapter import get_conversation_template


def main():
    model_name = args.model_name
    # model_name: 'vicuna-7b-v1.3'

    if args.worker_address:
        # args.worker_address: None
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        # controller_addr: 'http://localhost:21001'
        ret = requests.post(controller_addr + "/refresh_all_workers")
        # ret: <Response [200]>
        ret = requests.post(controller_addr + "/list_models")
        # ret: <Response [200]>
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")
        # models: ['vicuna-7b-v1.3']

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")
        # worker_addr: http://localhost:21002

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    conv = get_conversation_template(model_name)
    # 根据model_path路径名称对应的adapter, 匹配其对应的对话模板
    # if model_path == /data1/csw_model_weights/OriginOne, conv:
    # Conversation(
    #     name="one_shot",
    #     ...
    # )
    # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, conv:
    # Conversation(
    #     name="vicuna_v1.1",
    #     ...
    # )
    conv.append_message(conv.roles[0], args.message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # if model_path == /data1/csw_model_weights/vicuna-7b-v1.3, prompt:
    # 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Who are you ASSISTANT:'

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        # model_name: 'vicuna-7b-v1.3'
        "prompt": prompt,
        "temperature": args.temperature,
        # args.temperature: 0.0
        "max_new_tokens": args.max_new_tokens,
        # args.max_new_tokens: 32
        "stop": conv.stop_str,
        # conv.stop_str: None
        "stop_token_ids": conv.stop_token_ids,
        # conv.stop_token_ids: None
        "echo": False,
    }
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
    )

    print(f"{conv.roles[0]}: {args.message}")
    print(f"{conv.roles[1]}: ", end="")
    prev = 0
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


if __name__ == "__main__":
    # 该测试程序可以在除controller和worker之外的第三台机器上进行
    # 若指定worker_address, 则会直接向worker_address发送推断请求
    # 若未指定worker_address, 则可以向controller询问, 在与其保持通讯的worker_address中, 是否有名称是model_name的worker
    # 然后取得该model_name的worker_address发送推断请求
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    main()
