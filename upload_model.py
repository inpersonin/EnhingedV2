"""upload_model.py — Upload the RLHF-trained checkpoint to inpersonin/HinGPTv2.

This script uploads to the V2 model repository ONLY — never to the V1
repo (inpersonin/HinGPT) or the V1 Space (inpersonin/Enhinged).

Usage:
    python upload_model.py --ckpt_path checkpoints/rlhf_best.pt
"""

import argparse
import os
from huggingface_hub import HfApi


def upload(ckpt_path: str, filename: str) -> None:
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print("--- Enhinged V2 Model Uploader ---")
    token = input("Please enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        return

    api = HfApi(token=token)

    # Upload to V2 model repo ONLY.
    repo_id = "inpersonin/HinGPTv2"
    print(f"\nUploading {ckpt_path} to model repo: {repo_id} ...")
    try:
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"\nSuccess! {ckpt_path} uploaded as {filename} to {repo_id}")
        print("The Enhinged V2 backend will pick this up automatically.")
    except Exception as e:
        print(f"\nUpload failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload V2 checkpoint to HuggingFace.")
    parser.add_argument("--ckpt_path", default="checkpoints/rlhf_best.pt")
    parser.add_argument("--filename", default="rlhf_best.pt")
    args = parser.parse_args()
    upload(args.ckpt_path, args.filename)


if __name__ == "__main__":
    main()
