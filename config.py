from __future__ import annotations
import os
from dataclasses import dataclass
DEFAULT_TOKENIZER_NAME = 'gpt2'
HF_MODEL_REPO = 'inpersonin/HinGPTv2'
HF_MODEL_FILENAME = 'rlhf_best.pt'
HF_MODEL_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'rlhf_best.pt')
ONNX_MODEL_PATH = 'model_quant.onnx'
DEFAULT_CHECKPOINT_PATH = 'checkpoints/best.pt'
if not os.path.exists(DEFAULT_CHECKPOINT_PATH) and os.path.exists('best.pt'):
    DEFAULT_CHECKPOINT_PATH = 'best.pt'
DEFAULT_DATA_DIR = 'data'
DEFAULT_OUTPUT_DIR = 'checkpoints'
SUPPORTED_PRETRAINED_MODELS = ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
USE_INT8_QUANTIZE = os.getenv('ENHINGED_QUANTIZE', '0').strip() == '1'

@dataclass(slots=True)
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True