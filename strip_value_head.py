"""strip_value_head.py — Phase 4: strip the value head from a PPO checkpoint.

The PPO-trained checkpoint has value_head.* keys in its state_dict that
the plain HinglishGPT inference path doesn't expect. This script creates
a clean, deployment-ready checkpoint by:

1. Loading the PPO checkpoint (which contains value_head.* keys).
2. Removing all value_head.* keys from the state_dict.
3. Saving the cleaned state dict in the same format as a normal training
   checkpoint (model_state + model_config keys), so load_model_from_checkpoint
   in model.py can load it without any code changes.

The original PPO checkpoint is preserved — this creates a NEW file.

Usage:
    python strip_value_head.py \
        --ppo_ckpt checkpoints/rlhf/ppo_best.pt \
        --out_path checkpoints/rlhf_best.pt

Verification:
    The script loads the stripped checkpoint through load_model_from_checkpoint
    to confirm it loads cleanly before declaring success.
"""

from __future__ import annotations

import argparse
import os

import torch

from model import load_model_from_checkpoint


def strip_value_head(ppo_ckpt_path: str, out_path: str) -> None:
    print(f"Loading PPO checkpoint from: {ppo_ckpt_path}")
    try:
        ckpt = torch.load(ppo_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ppo_ckpt_path, map_location="cpu")

    state = ckpt["model_state"]
    config = ckpt["model_config"]

    # Count value-head keys.
    value_head_keys = [k for k in state if k.startswith("value_head.")]
    print(f"Found {len(value_head_keys)} value head keys to strip: {value_head_keys}")

    # Also strip the 'base.' prefix that PolicyWithValueHead adds.
    # PolicyWithValueHead stores base model parameters as 'base.transformer.*'
    # and 'base.lm_head.*'. We need to strip 'base.' prefix to match
    # HinglishGPT's expected key names.
    cleaned: dict = {}
    for k, v in state.items():
        if k.startswith("value_head."):
            continue  # drop value head
        if k.startswith("base."):
            cleaned[k[5:]] = v  # strip 'base.' prefix
        else:
            cleaned[k] = v

    print(f"Original keys: {len(state)}")
    print(f"Cleaned keys:  {len(cleaned)}")

    # Build the output checkpoint in the same format as train.py's save_checkpoint.
    out_ckpt = {
        "model_state": cleaned,
        "model_config": config,
        # Preserve iteration metadata if present.
        "iter_num": ckpt.get("iteration", ckpt.get("iter_num", 0)),
        "best_val": ckpt.get("best_val", float("inf")),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(out_ckpt, out_path)
    print(f"Stripped checkpoint saved to: {out_path}")

    # Verify: load through the standard inference path.
    print("\nVerifying stripped checkpoint loads cleanly through load_model_from_checkpoint...")
    try:
        model, enc, device = load_model_from_checkpoint(out_path)
        n_params = model.count_params()
        print(f"  ✓ Loaded successfully. Parameters: {n_params / 1e6:.2f}M")
        print(f"  ✓ Config: n_layer={model.config.n_layer}, n_embd={model.config.n_embd}")
        print(f"\nDeployment checkpoint ready: {out_path}")
        print("This checkpoint can be uploaded to inpersonin/HinGPTv2 and used directly by inference.py.")
    except Exception as exc:
        print(f"  ✗ VERIFICATION FAILED: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip value head from a PPO checkpoint for deployment.")
    parser.add_argument("--ppo_ckpt", required=True, help="Path to PPO checkpoint with value head.")
    parser.add_argument("--out_path", required=True, help="Output path for stripped checkpoint.")
    args = parser.parse_args()

    if not os.path.exists(args.ppo_ckpt):
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_ckpt}")
    if os.path.abspath(args.ppo_ckpt) == os.path.abspath(args.out_path):
        raise ValueError("--ppo_ckpt and --out_path must be different files (never overwrite the PPO checkpoint).")

    strip_value_head(args.ppo_ckpt, args.out_path)


if __name__ == "__main__":
    main()
