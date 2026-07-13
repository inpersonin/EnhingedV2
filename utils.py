from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from config import DEFAULT_TOKENIZER_NAME
from model import generate_text
_EOT_TOKEN_ID = 50256

def load_dataset_metadata(data_dir: str) -> dict:
    metadata_path = Path(data_dir) / 'meta.json'
    if not metadata_path.exists():
        return {}
    with metadata_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def _load_boundaries(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if len(arr) == 0:
        return None
    return arr.astype(np.int64)

def _mask_eot_crossings(y: torch.Tensor, eot_id: int=_EOT_TOKEN_ID) -> torch.Tensor:
    eot_mask = y[:, :-1] == eot_id
    y = y.clone()
    y[:, 1:][eot_mask] = -100
    return y

class HinglishDataset:

    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: torch.device, token_dtype: Optional[str]=None) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        metadata = load_dataset_metadata(data_dir)
        resolved_dtype = np.dtype(token_dtype or metadata.get('token_dtype', 'uint16'))
        train_path = os.path.join(data_dir, 'train.bin')
        val_path = os.path.join(data_dir, 'val.bin')
        self.train_data = np.memmap(train_path, dtype=resolved_dtype, mode='r')
        self.val_data = np.memmap(val_path, dtype=resolved_dtype, mode='r')
        self.train_boundaries = _load_boundaries(os.path.join(data_dir, 'train_boundaries.npy'))
        self.val_boundaries = _load_boundaries(os.path.join(data_dir, 'val_boundaries.npy'))
        if self.train_boundaries is not None:
            print(f'HinglishDataset: boundary-aligned sampling active ({len(self.train_boundaries):,} train pairs)')
        else:
            print('HinglishDataset: no boundary file found -- using random sampling (legacy mode)')

    def _sample_start_indices(self, pool: np.ndarray, boundaries: Optional[np.ndarray], n: int) -> torch.Tensor:
        if boundaries is not None:
            valid = boundaries[boundaries <= len(pool) - self.block_size - 1]
            if len(valid) == 0:
                return torch.randint(len(pool) - self.block_size, (n,))
            chosen = torch.randint(len(valid), (n,))
            return torch.from_numpy(valid[chosen.numpy()])
        return torch.randint(len(pool) - self.block_size, (n,))

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        boundaries = self.train_boundaries if split == 'train' else self.val_boundaries
        indices = self._sample_start_indices(data, boundaries, self.batch_size)
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy(data[i + 1:i + self.block_size + 1].astype(np.int64)) for i in indices])
        if boundaries is not None:
            y = _mask_eot_crossings(y)
        if self.device.type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return (x, y)

class BilingualDataset:

    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: torch.device, token_dtype: Optional[str]=None) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        metadata = load_dataset_metadata(data_dir)
        resolved_dtype = np.dtype(token_dtype or metadata.get('token_dtype', 'uint16'))

        def _load(pool_name: str):
            train_path = os.path.join(data_dir, f'{pool_name}_train.bin')
            val_path = os.path.join(data_dir, f'{pool_name}_val.bin')
            if not (os.path.exists(train_path) and os.path.exists(val_path)):
                return (None, None, None, None)
            train = np.memmap(train_path, dtype=resolved_dtype, mode='r')
            val = np.memmap(val_path, dtype=resolved_dtype, mode='r')
            if len(train) <= block_size or len(val) <= block_size:
                print(f"WARNING: '{pool_name}' pool is smaller than block_size ({block_size}); ignoring it.")
                return (None, None, None, None)
            train_bounds = _load_boundaries(os.path.join(data_dir, f'{pool_name}_train_boundaries.npy'))
            val_bounds = _load_boundaries(os.path.join(data_dir, f'{pool_name}_val_boundaries.npy'))
            return (train, val, train_bounds, val_bounds)
        self.hinglish_train, self.hinglish_val, self.hinglish_train_bounds, self.hinglish_val_bounds = _load('hinglish')
        self.english_train, self.english_val, self.english_train_bounds, self.english_val_bounds = _load('english')
        if self.hinglish_train is None and self.english_train is None:
            raise FileNotFoundError(f"No usable pools found in '{data_dir}'. Expected hinglish_train.bin/hinglish_val.bin and/or english_train.bin/english_val.bin -- did you run `python prepare_data.py --mode bilingual ...` first?")
        available = []
        if self.hinglish_train is not None:
            b = f', {len(self.hinglish_train_bounds):,} pairs' if self.hinglish_train_bounds is not None else ', no boundaries'
            available.append(f'hinglish ({len(self.hinglish_train):,} train tokens{b})')
        if self.english_train is not None:
            b = f', {len(self.english_train_bounds):,} pairs' if self.english_train_bounds is not None else ', no boundaries'
            available.append(f'english ({len(self.english_train):,} train tokens{b})')
        print(f"BilingualDataset loaded: {', '.join(available)}")
        boundary_mode = 'boundary-aligned' if self.hinglish_train_bounds is not None or self.english_train_bounds is not None else 'random (no boundary files found)'
        print(f'BilingualDataset sampling mode: {boundary_mode}')

    def _sample_start_indices(self, pool: np.ndarray, boundaries: Optional[np.ndarray], n: int) -> torch.Tensor:
        if boundaries is not None:
            valid = boundaries[boundaries <= len(pool) - self.block_size - 1]
            if len(valid) == 0:
                return torch.randint(len(pool) - self.block_size, (n,))
            chosen = torch.randint(len(valid), (n,))
            return torch.from_numpy(valid[chosen.numpy()])
        return torch.randint(len(pool) - self.block_size, (n,))

    def _gather_blocks(self, pool: np.ndarray, boundaries: Optional[np.ndarray], start_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([torch.from_numpy(pool[i:i + self.block_size].astype(np.int64)) for i in start_indices])
        y = torch.stack([torch.from_numpy(pool[i + 1:i + self.block_size + 1].astype(np.int64)) for i in start_indices])
        if boundaries is not None:
            y = _mask_eot_crossings(y)
        return (x, y)

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type == 'cuda':
            return (x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True))
        return (x.to(self.device), y.to(self.device))

    def get_batch(self, split: str, english_ratio: float=0.5) -> tuple[torch.Tensor, torch.Tensor]:
        hinglish_pool = self.hinglish_train if split == 'train' else self.hinglish_val
        hinglish_bounds = self.hinglish_train_bounds if split == 'train' else self.hinglish_val_bounds
        english_pool = self.english_train if split == 'train' else self.english_val
        english_bounds = self.english_train_bounds if split == 'train' else self.english_val_bounds
        if hinglish_pool is None:
            english_ratio = 1.0
        if english_pool is None:
            english_ratio = 0.0
        n_english = int(round(self.batch_size * english_ratio))
        n_hinglish = self.batch_size - n_english
        x_parts, y_parts = ([], [])
        if n_hinglish > 0:
            idx = self._sample_start_indices(hinglish_pool, hinglish_bounds, n_hinglish)
            x, y = self._gather_blocks(hinglish_pool, hinglish_bounds, idx)
            x_parts.append(x)
            y_parts.append(y)
        if n_english > 0:
            idx = self._sample_start_indices(english_pool, english_bounds, n_english)
            x, y = self._gather_blocks(english_pool, english_bounds, idx)
            x_parts.append(x)
            y_parts.append(y)
        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        permutation = torch.randperm(x.size(0))
        x, y = (x[permutation], y[permutation])
        return self._to_device(x, y)

    def get_batch_single_language(self, split: str, language: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pool = getattr(self, f"{language}_{('train' if split == 'train' else 'val')}", None)
        bounds = getattr(self, f"{language}_{('train' if split == 'train' else 'val')}_bounds", None)
        if pool is None:
            return (None, None)
        idx = self._sample_start_indices(pool, bounds, self.batch_size)
        x, y = self._gather_blocks(pool, bounds, idx)
        return self._to_device(x, y)

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataset, device: torch.device, n_batches: int=100) -> dict:
    model.eval()
    is_bilingual = hasattr(dataset, 'get_batch_single_language')
    losses = []
    for _ in range(n_batches):
        x, y = dataset.get_batch('val') if not is_bilingual else dataset.get_batch('val', english_ratio=0.5)
        _, loss = model(x, y)
        losses.append(loss.item())
    val_loss = float(np.mean(losses))
    perplexity = float(np.exp(val_loss))
    results: dict = {'val_loss': val_loss, 'perplexity': perplexity, 'parameters': model.count_params() if hasattr(model, 'count_params') else sum((p.numel() for p in model.parameters()))}
    import tiktoken
    encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)
    if is_bilingual:
        for language in ('hinglish', 'english'):
            lang_losses = []
            for _ in range(n_batches):
                x, y = dataset.get_batch_single_language('val', language)
                if x is None:
                    break
                _, loss = model(x, y)
                lang_losses.append(loss.item())
            if lang_losses:
                lang_val_loss = float(np.mean(lang_losses))
                results[f'val_loss_{language}'] = lang_val_loss
                results[f'perplexity_{language}'] = float(np.exp(lang_val_loss))
        sample_prompts = {'hinglish': 'Yeh kya ho raha hai', 'english': "Hey, how's it going"}
    else:
        sample_prompts = {'default': 'Yeh kya ho raha hai'}
    import time
    started_at = time.time()
    generation_count = 0
    for label, prompt in sample_prompts.items():
        for _ in range(3):
            sample = generate_text(model, encoding, f'User: {prompt}\nAssistant:', device, max_new_tokens=50)
            generation_count += 1
        print(f'  sample [{label}] -> {sample!r}')
    elapsed = time.time() - started_at
    tokens_per_second = generation_count * 50 / max(elapsed, 1e-08)
    results['tokens_per_sec_gen'] = tokens_per_second
    for key, value in results.items():
        print(f'  {key:<25}: {value:,.2f}' if isinstance(value, float) else f'  {key:<25}: {value:,}')
    return results