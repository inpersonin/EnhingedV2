"""
export_onnx.py

Exports the HinglishGPT model to ONNX format for lightweight CPU inference.
onnxruntime uses ~50MB RAM vs PyTorch's ~200MB, making the full backend fit
comfortably inside Render's 512MB free-tier container.

The exported model takes (input_ids,) and returns (next_token_logits,) — i.e.,
only the logits for the LAST position. The autoregressive loop runs in Python.

Usage:
    python export_onnx.py \
        --ckpt checkpoints/rlhf_best_fp16.pt \
        --out  model.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# ── minimal model wrapper for ONNX export ──────────────────────────────────────

class _ONNXWrapper(nn.Module):
    """Thin wrapper that strips use_cache/capture_attn and returns
    only the last-position logits (vocab_size,) for sampling."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (1, seq_len)  — batch size fixed to 1 for generation
        # forward returns (logits, loss, kv_cache) — we only want logits
        logits, _loss, _kv = self.gpt(input_ids, targets=None, use_cache=False)
        return logits[:, -1, :]   # (1, vocab_size)


def export(ckpt_path: str, out_path: str) -> None:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    from config import GPTConfig
    from model import HinglishGPT

    config = GPTConfig(**ckpt["model_config"])
    state  = {k: v for k, v in ckpt["model_state"].items()
               if not k.startswith("value_head.")}

    # Detect fp16 and pre-allocate in the right dtype to keep peak RAM low
    sample = next(iter(state.values()))
    use_fp16 = isinstance(sample, torch.Tensor) and sample.dtype == torch.float16

    model = HinglishGPT(config)
    if use_fp16:
        model = model.half()
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:
        model.load_state_dict(state)

    model.eval()
    del ckpt, state

    wrapper = _ONNXWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, 8, dtype=torch.long)

    print("Exporting to ONNX ...")
    # Use dynamo=False to avoid onnxscript requirement
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            out_path,
            input_names=["input_ids"],
            output_names=["next_token_logits"],
            dynamic_axes={"input_ids": {1: "seq_len"}},
            opset_version=13,
            do_constant_folding=True,
            dynamo=False,
        )

    in_mb  = os.path.getsize(ckpt_path) / 1024**2
    out_mb = os.path.getsize(out_path)  / 1024**2
    print(f"Done! {in_mb:.0f} MB checkpoint -> {out_mb:.0f} MB ONNX model")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/rlhf_best_fp16.pt")
    parser.add_argument("--out",  default="model.onnx")
    args = parser.parse_args()
    export(args.ckpt, args.out)
