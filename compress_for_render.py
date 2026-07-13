"""
compress_for_render.py

Run this in Kaggle after PPO training.
Converts rlhf_best.pt weights from float32 → float16,
cutting the file from ~548 MB to ~274 MB so it fits inside
Render's 512 MB free-tier RAM.

Usage:
    !python compress_for_render.py \
        --in_path  checkpoints/rlhf/ppo_best.pt \
        --out_path checkpoints/rlhf_best_fp16.pt

Then upload:
    !python upload_model.py \
        --ckpt checkpoints/rlhf_best_fp16.pt \
        --repo  inpersonin/HinGPTv2 \
        --filename rlhf_best.pt        # overwrites the old one
"""

import argparse
import torch
import os

def compress(in_path: str, out_path: str) -> None:
    print(f"Loading  {in_path}  …")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)

    model_state = ckpt.get("model", ckpt)  # handle both wrapped and bare state_dicts

    print("Converting weights to float16 …")
    fp16_state = {}
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            fp16_state[k] = v.to(torch.float16)
        else:
            fp16_state[k] = v

    # Preserve the full checkpoint wrapper if present
    if "model" in ckpt:
        out_ckpt = {**ckpt, "model": fp16_state}
    else:
        out_ckpt = fp16_state

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(out_ckpt, out_path)

    in_mb  = os.path.getsize(in_path)  / 1024**2
    out_mb = os.path.getsize(out_path) / 1024**2
    print(f"Saved  {out_path}")
    print(f"  {in_mb:.1f} MB  →  {out_mb:.1f} MB  ({100*(1-out_mb/in_mb):.0f}% smaller)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",  required=True,
                        help="Input checkpoint (float32), e.g. checkpoints/rlhf/ppo_best.pt")
    parser.add_argument("--out_path", required=True,
                        help="Output checkpoint (float16), e.g. checkpoints/rlhf_best_fp16.pt")
    args = parser.parse_args()
    compress(args.in_path, args.out_path)
