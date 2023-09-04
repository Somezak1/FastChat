# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

# To enable Vicuna's understanding of long context, we expand the max context length from 512 in alpaca to 2048,
# which substantially increases GPU memory requirements.
# We tackle the memory pressure by utilizing gradient checkpointing and flash attention.
replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    train()
