"""Training and evaluation entry point for Enhinged.

=====================================================================
WHAT CHANGED AND WHY
=====================================================================
1. --init_from defaults to "gpt2" (fine-tune real GPT-2 weights).
   load_model_from_checkpoint in model.py reads model_config from the
   checkpoint dict, so architecture is always restored correctly.

2. Learning rate defaults are tuned for FINE-TUNING, not scratch.
   Default LR is 3e-5 (not 6e-4). warmup_iters defaults to 0 because
   when resuming from a checkpoint the model is already warm -- a
   ramp-up only wastes the first 100 steps.

3. max_iters and lr_decay_iters default to 16000 (not 8000).
   If you're resuming from best.pt at iter 8000, the loop runs from
   iter_num=8000 to max_iters=16000, giving 8000 more training steps.
   The cosine decay schedule re-anchors to the restored iter_num, so
   LR continues smoothly rather than jumping or repeating.

4. Per-pair EOS + boundary-aligned sampling (from prepare_data.py and
   utils.py changes). The sampler now guarantees every training window
   starts at a (User, Assistant) pair boundary. Targets after EOS
   tokens are masked to -100, and model.py's cross_entropy uses
   ignore_index=-100. No Trainer changes needed -- masks flow
   transparently through _get_batch -> model.forward.

5. eos_token_id=50256 is passed to generate() during eval so the model
   can stop cleanly on <|endoftext|> without relying solely on
   string-matching in inference.py.

=====================================================================
A NOTE ON KAGGLE SESSIONS
=====================================================================
Kaggle GPU sessions are capped (currently ~9-12 continuous hours, and
a weekly quota). This script is resumable via --ckpt_path:
`save_checkpoint("latest")` runs on every eval_interval, saving full
model+optimiser+iteration state. When your session is about to end,
make sure checkpoints/latest.pt is saved as part of your Kaggle
notebook's output ("Save Version"), then in your NEXT session, add
that previous output as an input dataset and pass
`--ckpt_path /kaggle/input/<your-previous-output>/latest.pt` to
resume exactly where you left off.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import DEFAULT_OUTPUT_DIR, GPTConfig
from model import HinglishGPT, load_model_from_checkpoint
from utils import BilingualDataset, HinglishDataset, evaluate_model


def _resolve_dataset(data_dir: str, block_size: int, batch_size: int, device: torch.device):
    """Pick BilingualDataset or the legacy HinglishDataset based on what's on disk.

    No new CLI flag needed: if prepare_data.py was run with
    `--mode bilingual`, hinglish_train.bin/english_train.bin will
    exist and we use BilingualDataset. Otherwise we fall back to the
    original single-pool train.bin/val.bin via HinglishDataset, so old
    data directories keep working unmodified.
    """

    bilingual_markers = [
        os.path.join(data_dir, "hinglish_train.bin"),
        os.path.join(data_dir, "english_train.bin"),
    ]
    if any(os.path.exists(path) for path in bilingual_markers):
        print(f"Detected bilingual data pools in {data_dir} -> using BilingualDataset.")
        return BilingualDataset(data_dir=data_dir, block_size=block_size, batch_size=batch_size, device=device)

    print(f"No bilingual pools found in {data_dir} -> falling back to legacy HinglishDataset (train.bin/val.bin).")
    return HinglishDataset(data_dir=data_dir, block_size=block_size, batch_size=batch_size, device=device)


class Trainer:
    """Train an Enhinged checkpoint."""

    def __init__(
        self,
        model: HinglishGPT,
        dataset,
        out_dir: str,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_iters: int = 0,
        lr_decay_iters: int = 16000,
        min_lr: float = 3e-6,
        max_iters: int = 16000,
        batch_size: int = 32,
        gradient_accum_steps: int = 1,
        eval_interval: int = 200,
        eval_iters: int = 50,
        english_ratio: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: str = "bfloat16",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = model
        self.dataset = dataset
        self.is_bilingual = hasattr(dataset, "get_batch_single_language")
        self.english_ratio = english_ratio
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.gradient_accum = gradient_accum_steps
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = self.device.type

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map[dtype]
        self.scaler = torch.cuda.amp.GradScaler(enabled=dtype == "float16" and self.device_type == "cuda")
        self.autocast_ctx = (
            torch.amp.autocast(device_type=self.device_type, dtype=torch_dtype)
            if self.device_type == "cuda"
            else nullcontext()
        )
        self.optimiser = model.configure_optimiser(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            device_type=self.device_type,
        )

        self.iter_num = 0
        self.best_val = float("inf")
        self.best_val_per_language: dict[str, float] = {}

    def _get_batch(self, split: str):
        if self.is_bilingual:
            return self.dataset.get_batch(split, english_ratio=self.english_ratio)
        return self.dataset.get_batch(split)

    def get_lr(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / max(1, self.warmup_iters)
        if iteration >= self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (iteration - self.warmup_iters) / max(1, self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Combined loss/perplexity, PLUS per-language breakdown when bilingual.

        The per-language numbers are the important addition: a single
        combined loss can look fine while one language is quietly
        being neglected (or overfit, if it's the smaller pool). Watch
        val_loss_english and val_loss_hinglish separately, not just
        val_loss.
        """

        self.model.eval()
        losses: dict[str, float] = {}

        for split in ("train", "val"):
            split_losses = torch.zeros(self.eval_iters)
            for index in range(self.eval_iters):
                x, y = self._get_batch(split)
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                split_losses[index] = loss.item()
            losses[split] = split_losses.mean().item()

        if self.is_bilingual:
            for language in ("hinglish", "english"):
                lang_losses = []
                for _ in range(self.eval_iters):
                    x, y = self.dataset.get_batch_single_language("val", language)
                    if x is None:
                        break
                    with self.autocast_ctx:
                        _, loss = self.model(x, y)
                    lang_losses.append(loss.item())
                if lang_losses:
                    losses[f"val_{language}"] = float(np.mean(lang_losses))

        self.model.train()
        return losses

    def save_checkpoint(self, tag: str = "latest") -> str:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "model_config": dataclasses.asdict(self.model.config),
            "iter_num": self.iter_num,
            "best_val": self.best_val,
            "best_val_per_language": self.best_val_per_language,
        }
        path = os.path.join(self.out_dir, f"{tag}.pt")
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val = checkpoint["best_val"]
        self.best_val_per_language = checkpoint.get("best_val_per_language", {})

    def _write_metrics_json(self, losses: dict[str, float]) -> None:
        """Write a metrics.json snapshot -- the same style of file you shared
        before, so progress is easy to track and share across runs."""

        metrics = {
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val,
            "best_val_perplexity": math.exp(self.best_val) if self.best_val != float("inf") else None,
            "current_val_loss": losses.get("val"),
            "model_parameters": f"{self.model.count_params() / 1e6:.2f}M",
        }
        for language in ("hinglish", "english"):
            key = f"val_{language}"
            if key in losses:
                metrics[f"val_loss_{language}"] = losses[key]
                metrics[f"val_perplexity_{language}"] = math.exp(losses[key])
        with open(os.path.join(self.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def train(self) -> None:
        self.model.train()
        self.model.to(self.device)
        tokens = 0
        start_time = time.time()

        while self.iter_num < self.max_iters:
            learning_rate = self.get_lr(self.iter_num)
            for param_group in self.optimiser.param_groups:
                param_group["lr"] = learning_rate

            if self.iter_num % self.eval_interval == 0:
                losses = self.estimate_loss()
                log_parts = [
                    f"iter {self.iter_num:5d}",
                    f"train loss {losses['train']:.4f}",
                    f"val loss {losses['val']:.4f}",
                    f"val ppl {math.exp(losses['val']):.2f}",
                ]
                if "val_hinglish" in losses:
                    log_parts.append(f"val_hi {losses['val_hinglish']:.4f}")
                if "val_english" in losses:
                    log_parts.append(f"val_en {losses['val_english']:.4f}")
                log_parts.append(f"lr {learning_rate:.2e}")
                print(" | ".join(log_parts))

                self.save_checkpoint("latest")
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.best_val_per_language = {
                        k: v for k, v in losses.items() if k.startswith("val_")
                    }
                    self.save_checkpoint("best")
                self._write_metrics_json(losses)

            self.optimiser.zero_grad(set_to_none=True)

            for _ in range(self.gradient_accum):
                x, y = self._get_batch("train")
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                    loss = loss / self.gradient_accum
                tokens += x.numel()
                self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.iter_num += 1

            if self.iter_num % 50 == 0:
                elapsed = time.time() - start_time
                print(f"iter {self.iter_num:5d} | loss {loss.item() * self.gradient_accum:.4f} | {tokens / max(elapsed, 1e-8):,.0f} tok/s")
                tokens = 0
                start_time = time.time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate Enhinged.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt_path", default=None, help="Resume from this checkpoint (e.g. a previous Kaggle session's latest.pt)")
    parser.add_argument(
        "--init_from",
        choices=["scratch", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2",
        help="Default changed from 'scratch' to 'gpt2': fine-tune real pretrained GPT-2 "
        "instead of training bilingual competence from random init. See module docstring.",
    )
    # These four are IGNORED when --init_from is not "scratch" (from_pretrained sets the
    # architecture for you), kept only so `--init_from scratch` experiments still work.
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max_iters", type=int, default=16000,
        help="Total iterations to run. Resume from best.pt at iter ~8000 -> set this to 16000 for 8000 more steps.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5,
        help="Fine-tuning default. Use ~6e-4 only with --init_from scratch.")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=0,
        help="LR warmup steps. Default 0 because when resuming from a checkpoint the model is already warm.")
    parser.add_argument("--lr_decay_iters", type=int, default=16000,
        help="Cosine decay runs from warmup end to this iteration. Match max_iters.")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument(
        "--english_ratio",
        type=float,
        default=0.5,
        help="Fraction of every training batch drawn from the English pool (0.0-1.0). "
        "Ignored automatically if you only prepared a Hinglish-only (legacy) data dir.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        # ------------------------------------------------------------------
        # Build the model.
        #
        # CRITICAL ORDER OF OPERATIONS:
        #   If --ckpt_path is given, peek at its model_config FIRST and
        #   build the model from that config -- NOT from --init_from /
        #   --n_layer / --n_embd etc. The checkpoint already stores its
        #   architecture in the "model_config" key (written by
        #   save_checkpoint). Trying to load a 124M checkpoint into a 30M
        #   model built from scratch args causes the shape mismatch you saw:
        #       transformer.wte.weight [50257,768] vs [50257,384]
        #   We fix this by reading the checkpoint config before torch
        #   allocates any tensors for the wrong shape.
        # ------------------------------------------------------------------
        if args.ckpt_path:
            print(f"Peeking at checkpoint: {args.ckpt_path}")
            try:
                _peek = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                _peek = torch.load(args.ckpt_path, map_location="cpu")

            if "model_config" not in _peek:
                raise KeyError(
                    f"Checkpoint at '{args.ckpt_path}' has no 'model_config' key. "
                    "This checkpoint was saved by a different codebase. "
                    "Inspect it with torch.load() and check its keys."
                )
            _cfg = GPTConfig(**_peek["model_config"])
            print(
                f"  Checkpoint model_config: n_layer={_cfg.n_layer}, "
                f"n_head={_cfg.n_head}, n_embd={_cfg.n_embd}, "
                f"block_size={_cfg.block_size}, vocab_size={_cfg.vocab_size}"
            )
            # Free the peeked tensors -- we'll load them properly later via
            # load_checkpoint() which maps to the correct device.
            del _peek
            model = HinglishGPT(_cfg)

        elif args.init_from == "scratch":
            model = HinglishGPT(
                GPTConfig(
                    block_size=args.block_size,
                    vocab_size=50257,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    dropout=args.dropout,
                )
            )
        else:
            print(f"Loading pretrained {args.init_from} weights (architecture args like --n_layer are ignored)...")
            model = HinglishGPT.from_pretrained(args.init_from)

        model.to(device)
        print(f"Model parameters: {model.count_params() / 1e6:.2f}M")

        dataset = _resolve_dataset(
            data_dir=args.data_dir,
            block_size=model.config.block_size,
            batch_size=args.batch_size,
            device=device,
        )
        trainer = Trainer(
            model=model,
            dataset=dataset,
            out_dir=args.out_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            gradient_accum_steps=args.grad_accum_steps,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            english_ratio=args.english_ratio,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
        )

        if args.ckpt_path:
            print(f"Loading checkpoint weights + optimiser + iter_num from: {args.ckpt_path}")
            trainer.load_checkpoint(args.ckpt_path)
            print(f"Resumed at iter {trainer.iter_num}, best_val={trainer.best_val:.4f}")
        trainer.train()
        return


    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "best.pt")
    model, _, device = load_model_from_checkpoint(ckpt_path, device)
    dataset = _resolve_dataset(
        data_dir=args.data_dir,
        block_size=model.config.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    evaluate_model(model, dataset, device)


if __name__ == "__main__":
    main()
