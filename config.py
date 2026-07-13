"""Shared configuration for the Enhinged V2 conversational model."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_TOKENIZER_NAME = "gpt2"

# HF model repository where V2 weights live.
# A new model repo keeps V2 weights separate from V1's inpersonin/HinGPT.
HF_MODEL_REPO = "inpersonin/HinGPTv2"
HF_MODEL_FILENAME = "rlhf_best_fp16.pt"

# Local path where the downloaded checkpoint is cached
HF_MODEL_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "rlhf_best_fp16.pt")

# fp16 model is baked into the image via Git LFS (primary path)
DEFAULT_CHECKPOINT_PATH = "checkpoints/rlhf_best_fp16.pt"
if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
    DEFAULT_CHECKPOINT_PATH = "checkpoints/rlhf_best.pt"  # fp32 fallback
if not os.path.exists(DEFAULT_CHECKPOINT_PATH) and os.path.exists("best.pt"):
    DEFAULT_CHECKPOINT_PATH = "best.pt"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "checkpoints"
SUPPORTED_PRETRAINED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

# Toggle int8 quantization at inference time (never used in training/RLHF).
# Set ENHINGED_QUANTIZE=1 in environment to enable; default off.
USE_INT8_QUANTIZE = os.getenv("ENHINGED_QUANTIZE", "0").strip() == "1"


@dataclass(slots=True)
class GPTConfig:
    """Configuration for the GPT-2 style Enhinged V2 language model."""

    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
