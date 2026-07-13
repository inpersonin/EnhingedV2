FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# ── Bake the fp16 model into the image at BUILD time ──────────────────────────
# The Docker builder has no RAM limit, so we can freely load the fp32 checkpoint
# (522 MB), convert it to fp16 (335 MB), and save the result.
# At container start the model is already on disk — zero download, zero OOM.
RUN python - <<'EOF'
import os, sys, torch
from huggingface_hub import hf_hub_download

os.makedirs("checkpoints", exist_ok=True)
out = "checkpoints/rlhf_best_fp16.pt"

print("==> Downloading fp32 checkpoint from HuggingFace …", flush=True)
fp32_path = hf_hub_download(
    repo_id="inpersonin/HinGPTv2",
    filename="rlhf_best.pt",
    local_dir="/tmp/hf_dl",
    local_dir_use_symlinks=False,
)

print("==> Converting to fp16 …", flush=True)
ckpt = torch.load(fp32_path, map_location="cpu", weights_only=False)
state = {
    k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
    for k, v in ckpt["model_state"].items()
}
ckpt["model_state"] = state
torch.save(ckpt, out)

size_mb = os.path.getsize(out) / 1024 / 1024
print(f"==> Done — {out} ({size_mb:.0f} MB fp16)", flush=True)
EOF

EXPOSE 8000

# sh -c is required so Railway's $PORT env var is expanded at runtime.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
