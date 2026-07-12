"""benchmark_latency.py — Measure generation speed before/after KV-caching.

Run this after Phase 1 to get the actual speedup number to report on
the Metrics page.

Usage:
    python benchmark_latency.py --ckpt_path checkpoints/best.pt

Outputs:
  - Wall-clock time for 100-token generation (cached and uncached)
  - Tokens/second for both
  - Actual speedup ratio

These numbers should be recorded and used in the frontend Metrics section,
NOT estimated or made up.
"""

from __future__ import annotations

import argparse
import time

import torch

from model import HinglishGPT, generate, load_model_from_checkpoint


@torch.no_grad()
def _generate_uncached(
    model: HinglishGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
) -> tuple[torch.Tensor, float]:
    """Reference uncached generation. Returns (output, elapsed_seconds)."""
    model.eval()
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        context = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _, _ = model(context)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    elapsed = time.perf_counter() - start
    return idx, elapsed


@torch.no_grad()
def _generate_cached(
    model: HinglishGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
) -> tuple[torch.Tensor, float]:
    """KV-cached generation. Returns (output, elapsed_seconds)."""
    model.eval()
    start = time.perf_counter()
    out = generate(model, idx, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start
    return out, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KV-cached vs uncached generation latency.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=5, help="Average over N runs.")
    parser.add_argument("--prompt", default="User: Kya hal hai yaar?\nAssistant:")
    args = parser.parse_args()

    model, enc, device = load_model_from_checkpoint(args.ckpt_path)
    print(f"\nBenchmarking on {device} | max_new_tokens={args.max_new_tokens} | {args.n_runs} runs")
    print(f"Prompt: {args.prompt!r}\n")

    prompt_ids = enc.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Warm-up.
    _generate_cached(model, idx.clone(), max_new_tokens=10)

    uncached_times = []
    cached_times = []

    for run in range(args.n_runs):
        torch.manual_seed(42)
        _, t_uncached = _generate_uncached(model, idx.clone(), max_new_tokens=args.max_new_tokens)
        uncached_times.append(t_uncached)

        torch.manual_seed(42)
        _, t_cached = _generate_cached(model, idx.clone(), max_new_tokens=args.max_new_tokens)
        cached_times.append(t_cached)

        print(f"  Run {run + 1}: uncached={t_uncached:.3f}s  cached={t_cached:.3f}s  speedup={t_uncached/t_cached:.2f}x")

    avg_uncached = sum(uncached_times) / len(uncached_times)
    avg_cached = sum(cached_times) / len(cached_times)
    speedup = avg_uncached / avg_cached
    tok_per_sec_cached = args.max_new_tokens / avg_cached
    tok_per_sec_uncached = args.max_new_tokens / avg_uncached

    print(f"\n{'='*60}")
    print(f"RESULTS ({args.max_new_tokens} tokens, avg over {args.n_runs} runs)")
    print(f"  Uncached: {avg_uncached:.3f}s  ({tok_per_sec_uncached:.1f} tok/s)")
    print(f"  Cached:   {avg_cached:.3f}s  ({tok_per_sec_cached:.1f} tok/s)")
    print(f"  Speedup:  {speedup:.2f}x")
    print(f"{'='*60}")
    print(f"\n→ Use these numbers on the Metrics page (NOT estimates):")
    print(f"    V1 generation speed: {tok_per_sec_uncached:.0f} tok/s")
    print(f"    V2 generation speed: {tok_per_sec_cached:.0f} tok/s")
    print(f"    Speedup: {speedup:.1f}x faster than V1")


if __name__ == "__main__":
    main()
