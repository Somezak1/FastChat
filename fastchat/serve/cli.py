"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.serve.inference import ChatIO, chat_loop


class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            # python3 -m fastchat.serve.cli --model-path /data1/csw_model_weights/vicuna-7b-v1.3/ --temperature 0
            # 历次的outputs:
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
            # output_text: "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."

            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)


class RichChatIO(ChatIO):
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset", "!!remove", "!!regen", "!!save", "!!load"],
            pattern=re.compile("$"),
        )
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text

    def print_output(self, text: str):
        self.stream_output([{"text": text}])


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        contents = ""
        # `end_sequence` signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = " __END_OF_A_MESSAGE_47582648__\n"
        len_end = len(end_sequence)
        while True:
            if len(contents) >= len_end:
                last_chars = contents[-len_end:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        contents = contents[:-len_end]
        print(f"[!OP:{role}]: {contents}", flush=True)
        return contents

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)


def main(args):
    if args.gpus:
        # 如果指定的GPU数量, 与指定可见的GPU数量信息不匹配则报错
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        # args.style: 'simple'
        # args.multiline: False
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        # args.mouse: False
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(
            args.model_path,
            # args.model_path: /data1/csw_model_weights/OriginOne or /data1/csw_model_weights/vicuna-7b-v1.3
            args.device,
            # args.device: 'cuda'
            args.num_gpus,
            # args.num_gpus: 1
            args.max_gpu_memory,
            # args.max_gpu_memory: None
            args.load_8bit,
            # args.load_8bit: False
            args.cpu_offloading,
            # args.cpu_offloading: False
            args.conv_template,
            # args.conv_template: None
            args.conv_system_msg,
            args.temperature,
            # args.temperature: 默认是0.7, 但此处为了复现改为0
            args.repetition_penalty,
            # args.repetition_penalty: 1.0
            args.max_new_tokens,
            # args.max_new_tokens: 512
            chatio,
            gptq_config=GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                # args.gptq_ckpt: None
                wbits=args.gptq_wbits,
                # args.gptq_wbits: 16
                groupsize=args.gptq_groupsize,
                # args.gptq_groupsize: -1
                act_order=args.gptq_act_order,
                # args.gptq_act_order: False
            ),
            awq_config=AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            ),
            revision=args.revision,
            # args.revision: 'main'
            judge_sent_end=args.judge_sent_end,
            # args.judge_sent_end: False
            debug=args.debug,
            # args.debug: False
            history=not args.no_history,
            # args.no_history: False
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    # debug代码
    # python3 -m fastchat.serve.cli --model-path /data1/csw_model_weights/OriginOne/ --temperature 0
    # or
    # python3 -m fastchat.serve.cli --model-path /data1/csw_model_weights/vicuna-7b-v1.3/  --temperature 0

    # 接受必要的参数信息
    parser = argparse.ArgumentParser()
    # 额外添加一些预先设定好的模型相关参数
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
