"""verify_kvcache.py — Verify that KV-cached generate() produces identical
outputs to a reference uncached implementation.

Run this BEFORE trusting the cached generation path:
    python verify_kvcache.py --ckpt_path checkpoints/best.pt

Expected output: PASS on every test prompt.

The reference implementation re-runs a full forward pass over the ENTIRE
growing sequence at every single new token step — this is the original V1
behavior. The new cached implementation feeds only the newest single token
at each step. Both must produce IDENTICAL token sequences for the same
seed and greedy decoding (do_sample=False).
"""

from __future__ import annotations

import argparse

import torch

from model import HinglishGPT, load_model_from_checkpoint, generate


@torch.no_grad()
def _generate_uncached(
    model: HinglishGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 50,
) -> torch.Tensor:
    """Reference uncached generation — re-runs full forward pass every step."""
    model.eval()
    for _ in range(max_new_tokens):
        context = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _, _ = model(context)          # full sequence every time
        logits = logits[:, -1, :]              # take last-token logits
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


def _test(
    model: HinglishGPT,
    enc,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int = 50,
) -> bool:
    all_passed = True
    for prompt in prompts:
        token_ids = enc.encode(f"User: {prompt}\nAssistant:")
        idx_ref = torch.tensor([token_ids], dtype=torch.long, device=device)
        idx_cached = idx_ref.clone()

        # Greedy (do_sample=False) with fixed seed for determinism.
        torch.manual_seed(42)
        ref_out = _generate_uncached(model, idx_ref, max_new_tokens=max_new_tokens)

        torch.manual_seed(42)
        cached_out = generate(model, idx_cached, max_new_tokens=max_new_tokens, do_sample=False)

        # Compare only the newly generated tokens (not the shared prompt).
        prompt_len = len(token_ids)
        ref_new = ref_out[0][prompt_len:].tolist()
        cached_new = cached_out[0][prompt_len:].tolist()

        if ref_new == cached_new:
            print(f"  PASS | prompt: {prompt!r}")
        else:
            print(f"  FAIL | prompt: {prompt!r}")
            print(f"         ref:    {ref_new}")
            print(f"         cached: {cached_new}")
            all_passed = False

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify KV-cached generation produces identical outputs.")
    parser.add_argument("--ckpt_path", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    model, enc, device = load_model_from_checkpoint(args.ckpt_path)

    prompts = [
        "Hello, how are you?",
        "Kya hal hai bhai",
        "Mujhe ek joke sunao",
        "What is machine learning?",
        "Aaj ka weather kaisa hai",
        "Tell me something funny in Hinglish",
    ]

    print(f"\nVerifying KV-cached generation against uncached reference\n{'='*60}")
    passed = _test(model, enc, device, prompts, max_new_tokens=args.max_new_tokens)

    print("\n" + ("=" * 60))
    if passed:
        print("ALL TESTS PASSED — cached and uncached generation are identical.")
    else:
        print("SOME TESTS FAILED — review the diffs above before proceeding.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
