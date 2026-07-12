"""Dataset loading and evaluation helpers for Enhinged.

=====================================================================
WHAT CHANGED AND WHY
=====================================================================
The original `HinglishDataset` reads ONE memory-mapped token pool
(train.bin/val.bin) and samples random blocks from it. That's kept
below, now extended with boundary-aligned sampling when
`{split}_boundaries.npy` is present alongside the .bin file.

New: `BilingualDataset`. It reads up to TWO pools --
hinglish_{train,val}.bin and english_{train,val}.bin, produced by
prepare_data.py's `--mode bilingual` -- and, critically, lets you
control what *fraction of every training batch* comes from each pool
via `english_ratio`, independent of how much raw text exists in each
pool.

=====================================================================
KEY FIX: BOUNDARY-ALIGNED SAMPLING + EOS-CROSSING LOSS MASK
=====================================================================
The original sampler picked a uniformly random token offset from
anywhere in the pool, so training windows routinely spanned the tail
of one conversation and the head of a completely unrelated one.

With the new prepare_data.py, every .bin file is accompanied by a
`_boundaries.npy` file -- a uint32 array of token start positions for
each (User, Assistant) pair. The sampler now works as follows:

  1. Load the boundaries array at __init__ time.
  2. In _sample_start_indices, pick a random *pair index* from that
     array, then use that pair's start position as the window start.
     This guarantees every training window begins at the start of a
     "User: ..." turn -- never mid-sentence, never mid-pair.
  3. In _gather_blocks, after slicing the token window, scan for any
     EOT token (50256 for GPT-2) inside the window. Set the target
     tokens at positions AFTER each EOT to -100 (PyTorch's
     ignore_index). This means the model is never trained to predict
     the first token of a new conversation given the last token of
     the previous one -- those cross-pair transitions contribute
     zero gradient.

If _boundaries.npy is absent (old data), the sampler falls back to
fully random sampling and no masking -- exactly the old behaviour.
Nothing breaks.

`evaluate_model` is also extended to report Hinglish and English val
loss SEPARATELY when a BilingualDataset is passed in (it auto-detects
via `hasattr(dataset, "get_batch_single_language")`), and to generate
one sample completion in each language.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import DEFAULT_TOKENIZER_NAME
from model import generate_text

# GPT-2 EOT token id -- the boundary mask uses this to detect pair
# transitions inside a training window.
_EOT_TOKEN_ID = 50256


def load_dataset_metadata(data_dir: str) -> dict:
    """Load optional dataset metadata from data/meta.json."""

    metadata_path = Path(data_dir) / "meta.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_boundaries(path: str) -> Optional[np.ndarray]:
    """Load a pair-start boundary array from a .npy file, or return None."""

    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if len(arr) == 0:
        return None
    return arr.astype(np.int64)


def _mask_eot_crossings(y: torch.Tensor, eot_id: int = _EOT_TOKEN_ID) -> torch.Tensor:
    """Set target tokens that follow an EOT in the same window to -100.

    For each sequence in the batch:
      - Find positions where the INPUT token (x[i]) is EOT.
      - Set y[i+1] (the target immediately after the EOT) to -100.
    This prevents the model from being trained to predict the first
    token of a new conversation given the last token of the previous one.

    NOTE: x[i] = tokens[start+i], y[i] = tokens[start+i+1], so
    y[i] is the target for x[i]. The EOT is IN x at position p, and
    y[p] is already the correct prediction (the EOT itself, since
    y[p] = tokens[start+p+1] = eot = x[p]... actually y[p] = tokens[start+p+1]
    which is the FIRST token of the next pair). We want to mask y at
    the position right after EOT, i.e. y[p] where x[p] == EOT.

    Concretely: if x[p] == eot_id, mask y[p] = -100 so the model
    doesn't learn "after seeing EOT, predict the first word of some
    unrelated next conversation."
    """

    # y is shape (batch, block_size), x would be y shifted left but
    # we reconstruct x from y: x[i] = y[i-1] for i>0. Since we only
    # have y here, we detect EOT positions via y itself:
    # x[p] == eot means y[p-1] == eot (y is shifted by 1 from x).
    # So: find positions where y[..., :-1] == eot, then mask y[..., 1:].
    eot_mask = y[:, :-1] == eot_id          # (batch, block_size-1): True where y has EOT
    y = y.clone()
    y[:, 1:][eot_mask] = -100               # mask the token AFTER each EOT in targets
    return y


# =====================================================================
# Original single-pool dataset (extended with boundary-aligned sampling)
# =====================================================================

class HinglishDataset:
    """Memory-mapped dataset for language-model training (single pool).

    Extended: if train_boundaries.npy / val_boundaries.npy are present
    in `data_dir`, sampling is boundary-aligned (every window starts at
    a pair boundary). Falls back to random sampling if absent.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
        token_dtype: Optional[str] = None,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        metadata = load_dataset_metadata(data_dir)
        resolved_dtype = np.dtype(token_dtype or metadata.get("token_dtype", "uint16"))

        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        self.train_data = np.memmap(train_path, dtype=resolved_dtype, mode="r")
        self.val_data = np.memmap(val_path, dtype=resolved_dtype, mode="r")

        self.train_boundaries = _load_boundaries(os.path.join(data_dir, "train_boundaries.npy"))
        self.val_boundaries = _load_boundaries(os.path.join(data_dir, "val_boundaries.npy"))

        if self.train_boundaries is not None:
            print(f"HinglishDataset: boundary-aligned sampling active ({len(self.train_boundaries):,} train pairs)")
        else:
            print("HinglishDataset: no boundary file found -- using random sampling (legacy mode)")

    def _sample_start_indices(self, pool: np.ndarray, boundaries: Optional[np.ndarray], n: int) -> torch.Tensor:
        """Sample n start positions, boundary-aligned if boundaries available."""

        if boundaries is not None:
            # Pick random pair indices, use their token start positions.
            # Only include pairs that have enough room for a full block.
            valid = boundaries[boundaries <= len(pool) - self.block_size - 1]
            if len(valid) == 0:
                # Fallback: something went wrong with boundaries, use random
                return torch.randint(len(pool) - self.block_size, (n,))
            chosen = torch.randint(len(valid), (n,))
            return torch.from_numpy(valid[chosen.numpy()])
        return torch.randint(len(pool) - self.block_size, (n,))

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of training or validation sequences."""

        data = self.train_data if split == "train" else self.val_data
        boundaries = self.train_boundaries if split == "train" else self.val_boundaries
        indices = self._sample_start_indices(data, boundaries, self.batch_size)

        x = torch.stack(
            [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in indices]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + self.block_size + 1].astype(np.int64)) for i in indices]
        )

        if boundaries is not None:
            y = _mask_eot_crossings(y)

        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


# =====================================================================
# NEW: bilingual dataset with controllable per-batch language mix
# =====================================================================

class BilingualDataset:
    """Memory-mapped dataset drawing from separate Hinglish/English pools.

    Looks for hinglish_{train,val}.bin and english_{train,val}.bin in
    `data_dir`. Either pool may be absent (e.g. you haven't attached
    Cornell yet) -- `get_batch` transparently falls back to whichever
    pool(s) actually exist.

    When _boundaries.npy files are present, sampling is pair-aligned
    and targets crossing pair boundaries are masked to -100.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
        token_dtype: Optional[str] = None,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        metadata = load_dataset_metadata(data_dir)
        resolved_dtype = np.dtype(token_dtype or metadata.get("token_dtype", "uint16"))

        def _load(pool_name: str):
            train_path = os.path.join(data_dir, f"{pool_name}_train.bin")
            val_path = os.path.join(data_dir, f"{pool_name}_val.bin")
            if not (os.path.exists(train_path) and os.path.exists(val_path)):
                return None, None, None, None
            train = np.memmap(train_path, dtype=resolved_dtype, mode="r")
            val = np.memmap(val_path, dtype=resolved_dtype, mode="r")
            if len(train) <= block_size or len(val) <= block_size:
                # A pool that's smaller than one block_size can't produce
                # a single training example -- treat it as absent rather
                # than crashing later with a confusing negative-size error.
                print(f"WARNING: '{pool_name}' pool is smaller than block_size ({block_size}); ignoring it.")
                return None, None, None, None
            train_bounds = _load_boundaries(os.path.join(data_dir, f"{pool_name}_train_boundaries.npy"))
            val_bounds = _load_boundaries(os.path.join(data_dir, f"{pool_name}_val_boundaries.npy"))
            return train, val, train_bounds, val_bounds

        self.hinglish_train, self.hinglish_val, self.hinglish_train_bounds, self.hinglish_val_bounds = _load("hinglish")
        self.english_train, self.english_val, self.english_train_bounds, self.english_val_bounds = _load("english")

        if self.hinglish_train is None and self.english_train is None:
            raise FileNotFoundError(
                f"No usable pools found in '{data_dir}'. Expected hinglish_train.bin/"
                f"hinglish_val.bin and/or english_train.bin/english_val.bin -- did you "
                f"run `python prepare_data.py --mode bilingual ...` first?"
            )

        available = []
        if self.hinglish_train is not None:
            b = f", {len(self.hinglish_train_bounds):,} pairs" if self.hinglish_train_bounds is not None else ", no boundaries"
            available.append(f"hinglish ({len(self.hinglish_train):,} train tokens{b})")
        if self.english_train is not None:
            b = f", {len(self.english_train_bounds):,} pairs" if self.english_train_bounds is not None else ", no boundaries"
            available.append(f"english ({len(self.english_train):,} train tokens{b})")
        print(f"BilingualDataset loaded: {', '.join(available)}")

        boundary_mode = "boundary-aligned" if (
            self.hinglish_train_bounds is not None or self.english_train_bounds is not None
        ) else "random (no boundary files found)"
        print(f"BilingualDataset sampling mode: {boundary_mode}")

    def _sample_start_indices(self, pool: np.ndarray, boundaries: Optional[np.ndarray], n: int) -> torch.Tensor:
        """Sample n start positions, boundary-aligned if boundaries available."""

        if boundaries is not None:
            valid = boundaries[boundaries <= len(pool) - self.block_size - 1]
            if len(valid) == 0:
                return torch.randint(len(pool) - self.block_size, (n,))
            chosen = torch.randint(len(valid), (n,))
            return torch.from_numpy(valid[chosen.numpy()])
        return torch.randint(len(pool) - self.block_size, (n,))

    def _gather_blocks(
        self,
        pool: np.ndarray,
        boundaries: Optional[np.ndarray],
        start_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack(
            [torch.from_numpy(pool[i : i + self.block_size].astype(np.int64)) for i in start_indices]
        )
        y = torch.stack(
            [torch.from_numpy(pool[i + 1 : i + self.block_size + 1].astype(np.int64)) for i in start_indices]
        )
        if boundaries is not None:
            y = _mask_eot_crossings(y)
        return x, y

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

    def get_batch(self, split: str, english_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch mixing hinglish/english according to `english_ratio`.

        english_ratio=0.0 -> all Hinglish, 1.0 -> all English, 0.5 -> even
        split. If one pool is missing, the ratio is silently overridden so
        training can still proceed on whichever pool(s) you do have.
        """

        hinglish_pool = self.hinglish_train if split == "train" else self.hinglish_val
        hinglish_bounds = self.hinglish_train_bounds if split == "train" else self.hinglish_val_bounds
        english_pool = self.english_train if split == "train" else self.english_val
        english_bounds = self.english_train_bounds if split == "train" else self.english_val_bounds

        if hinglish_pool is None:
            english_ratio = 1.0
        if english_pool is None:
            english_ratio = 0.0

        n_english = int(round(self.batch_size * english_ratio))
        n_hinglish = self.batch_size - n_english

        x_parts, y_parts = [], []
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

        # Shuffle so hinglish/english examples within the batch aren't
        # ordered in a block (defensive hygiene -- avoids any accidental
        # position-dependent effects, and makes printed batches easier
        # to sanity-check by eye).
        permutation = torch.randperm(x.size(0))
        x, y = x[permutation], y[permutation]

        return self._to_device(x, y)

    def get_batch_single_language(self, split: str, language: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample a batch from ONLY one language pool (for per-language val loss).

        Returns (None, None) if that language's pool isn't available.
        """

        pool = getattr(self, f"{language}_{'train' if split == 'train' else 'val'}", None)
        bounds = getattr(self, f"{language}_{'train' if split == 'train' else 'val'}_bounds", None)
        if pool is None:
            return None, None
        idx = self._sample_start_indices(pool, bounds, self.batch_size)
        x, y = self._gather_blocks(pool, bounds, idx)
        return self._to_device(x, y)


# =====================================================================
# Evaluation (extended to support per-language reporting)
# =====================================================================

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    n_batches: int = 100,
) -> dict:
    """Return validation metrics for the current model.

    Works with both HinglishDataset (single pool) and BilingualDataset
    (two pools). When given a BilingualDataset, also reports Hinglish
    and English val loss/perplexity separately, and generates one
    sample completion per language.
    """

    model.eval()
    is_bilingual = hasattr(dataset, "get_batch_single_language")

    losses = []
    for _ in range(n_batches):
        x, y = dataset.get_batch("val") if not is_bilingual else dataset.get_batch("val", english_ratio=0.5)
        _, loss = model(x, y)
        losses.append(loss.item())

    val_loss = float(np.mean(losses))
    perplexity = float(np.exp(val_loss))

    results: dict = {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "parameters": model.count_params() if hasattr(model, "count_params") else sum(p.numel() for p in model.parameters()),
    }

    import tiktoken

    encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)

    if is_bilingual:
        # Per-language val loss -- this is the number to actually watch.
        for language in ("hinglish", "english"):
            lang_losses = []
            for _ in range(n_batches):
                x, y = dataset.get_batch_single_language("val", language)
                if x is None:
                    break
                _, loss = model(x, y)
                lang_losses.append(loss.item())
            if lang_losses:
                lang_val_loss = float(np.mean(lang_losses))
                results[f"val_loss_{language}"] = lang_val_loss
                results[f"perplexity_{language}"] = float(np.exp(lang_val_loss))

        sample_prompts = {
            "hinglish": "Yeh kya ho raha hai",
            "english": "Hey, how's it going",
        }
    else:
        sample_prompts = {"default": "Yeh kya ho raha hai"}

    import time

    started_at = time.time()
    generation_count = 0
    for label, prompt in sample_prompts.items():
        for _ in range(3):
            sample = generate_text(model, encoding, f"User: {prompt}\nAssistant:", device, max_new_tokens=50)
            generation_count += 1
        print(f"  sample [{label}] -> {sample!r}")
    elapsed = time.time() - started_at

    tokens_per_second = (generation_count * 50) / max(elapsed, 1e-8)
    results["tokens_per_sec_gen"] = tokens_per_second

    for key, value in results.items():
        print(f"  {key:<25}: {value:,.2f}" if isinstance(value, float) else f"  {key:<25}: {value:,}")

    return results
