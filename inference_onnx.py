from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
import tiktoken
import onnxruntime as ort
from config import DEFAULT_TOKENIZER_NAME, HF_MODEL_REPO, HF_MODEL_FILENAME, HF_MODEL_CACHE_PATH, ONNX_MODEL_PATH

@dataclass
class _RuntimeState:
    session: Optional[ort.InferenceSession] = None
    encoding: Optional[tiktoken.Encoding] = None
    model_path: Optional[str] = None
_STATE = _RuntimeState()

def _resolve_onnx_path() -> str:
    if os.path.exists(ONNX_MODEL_PATH) and os.path.getsize(ONNX_MODEL_PATH) > 1024 * 1024:
        print(f'inference_onnx: using local ONNX model at {ONNX_MODEL_PATH}')
        return ONNX_MODEL_PATH
    print('inference_onnx: local ONNX file missing or is an LFS pointer. Downloading from GitHub LFS …')
    try:
        import urllib.request
        url = 'https://github.com/inpersonin/EnhingedV2/raw/main/model_quant.onnx'
        downloaded = ONNX_MODEL_PATH
        print(f'Downloading {url} to {downloaded}...')
        urllib.request.urlretrieve(url, downloaded)
        if os.path.getsize(downloaded) < 1024 * 1024:
            raise RuntimeError('Downloaded file is too small, expected > 1MB.')
        print('inference_onnx: download complete!')
        return downloaded
    except Exception as exc:
        raise FileNotFoundError(f"ONNX model not found locally and download failed: {exc}\nPlease ensure you uploaded 'model.onnx' to {HF_MODEL_REPO} on HuggingFace!") from exc

def load_model(path: Optional[str]=None) -> None:
    onnx_path = _resolve_onnx_path()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    providers = ['CPUExecutionProvider']
    print(f'inference_onnx: loading ONNX session from {onnx_path} …')
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    print('inference_onnx: session ready.')
    _STATE.session = session
    _STATE.encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)
    _STATE.model_path = onnx_path

def unload_model() -> None:
    _STATE.session = None
    _STATE.encoding = None
    _STATE.model_path = None

def is_model_loaded() -> bool:
    return _STATE.session is not None and _STATE.encoding is not None

def get_loaded_checkpoint_path() -> Optional[str]:
    return _STATE.model_path

def _top_k_top_p_filter(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    if top_k and top_k > 0:
        threshold = np.sort(logits)[-top_k]
        logits = np.where(logits < threshold, -10000000000.0, logits)
    if top_p and 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        cumprobs = np.cumsum(np.exp(sorted_logits - sorted_logits.max()) / np.exp(sorted_logits - sorted_logits.max()).sum())
        sorted_logits[cumprobs > top_p] = -10000000000.0
        logits = sorted_logits[np.argsort(sorted_idx)]
    return logits

def _sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float, do_sample: bool) -> int:
    if not do_sample:
        return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-08)
    logits = _top_k_top_p_filter(logits, top_k, top_p)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return int(np.random.choice(len(probs), p=probs))

def _build_history_string(history: list[dict]) -> str:
    lines = []
    for turn in history:
        role = turn.get('role', 'user')
        content = turn.get('content', '').strip()
        if not content:
            continue
        label = 'User' if role == 'user' else 'Assistant'
        lines.append(f'{label}: {content}')
    return '\n'.join(lines)

def generate_response(prompt: str, max_new_tokens: int=110, temperature: float=0.8, top_k: Optional[int]=50, top_p: Optional[float]=0.95, repetition_penalty: float=1.1, do_sample: bool=True, seed: Optional[int]=None, conversation_history: Optional[list[dict]]=None) -> str:
    if not is_model_loaded():
        raise RuntimeError('Model not loaded. Call load_model() first.')
    if seed is not None:
        np.random.seed(seed)
    session = _STATE.session
    encoding = _STATE.encoding
    block_size: int = 256
    if conversation_history:
        history_text = _build_history_string(conversation_history)
        full_prompt = f'{history_text}\nUser: {prompt}\nAssistant:'
    else:
        full_prompt = f'User: {prompt}\nAssistant:'
    token_ids: list[int] = encoding.encode(full_prompt)
    max_prompt_tokens = max(1, block_size - max_new_tokens - 10)
    if len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[-max_prompt_tokens:]
    input_name = session.get_inputs()[0].name
    generated: list[int] = []
    for _ in range(max_new_tokens):
        seq = token_ids + generated
        if len(seq) > block_size:
            seq = seq[-block_size:]
        ids_np = np.array([seq], dtype=np.int64)
        logits = session.run(None, {input_name: ids_np})[0][0]
        if repetition_penalty != 1.0:
            for tok in set(seq):
                logits[tok] /= repetition_penalty
        next_tok = _sample(logits, temperature, top_k or 0, top_p or 1.0, do_sample)
        generated.append(next_tok)
        decoded_so_far = encoding.decode(generated)
        for stop in ('User:', '\nUser:', '<|endoftext|>'):
            if stop in decoded_so_far:
                decoded_so_far = decoded_so_far[:decoded_so_far.index(stop)]
                return decoded_so_far.strip()
    return encoding.decode(generated).strip()