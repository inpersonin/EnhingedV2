"""
===================================================================
Enhinged — Kaggle Training Notebook
===================================================================
Run this cell-by-cell in a Kaggle GPU session.

REQUIRED INPUTS (add via "Add Input" in the Kaggle notebook UI):
  1. Your model files (model.py, train.py, utils.py, prepare_data.py,
     config.py, inference.py) — upload as a Dataset or copy inline.
  2. Your Hinglish corpus.txt — already in your dataset.
  3. best.pt checkpoint — your current best model weights.
  4. DailyDialog dataset  — search "dailydialog" on Kaggle Datasets.
  5. Cornell Movie-Dialogs — search "cornell movie dialog" on Kaggle.

WHAT THIS NOTEBOOK DOES:
  1. Installs dependencies.
  2. Runs prepare_data.py to rebuild the .bin files WITH per-pair EOS
     tokens and boundary arrays (the root-cause fix).
  3. Runs train.py resuming from best.pt for 8000 more steps.
  4. Saves best.pt as the notebook output.

===================================================================
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q", "tiktoken", "transformers"], check=True)
print("Dependencies installed.")


# ============================================================
# CELL 2 — Paths (edit these to match your Kaggle input paths)
# ============================================================
import os

# ---- Edit these paths to match your Kaggle inputs ----
CORPUS_TXT      = "/kaggle/input/enhinged/corpus.txt"
BEST_PT         = "/kaggle/input/enhinged/best.pt"   # your uploaded best.pt
DAILYDIALOG_DIR = "/kaggle/input/daily-dialog"        # Kaggle DailyDialog dataset
CORNELL_LINES   = "/kaggle/input/cornell-movie-dialog/movie_lines.txt"
CORNELL_CONVOS  = "/kaggle/input/cornell-movie-dialog/movie_conversations.txt"

DATA_OUT        = "/kaggle/working/data"
CKPT_OUT        = "/kaggle/working/checkpoints"
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(CKPT_OUT, exist_ok=True)

# ---- Verify paths exist ----
for label, path in [
    ("corpus.txt", CORPUS_TXT),
    ("best.pt", BEST_PT),
]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1e6
        print(f"  ✓ {label}: {path} ({size:.1f} MB)")
    else:
        print(f"  ✗ MISSING: {label}: {path}  <-- fix this path!")

for label, path in [
    ("DailyDialog", DAILYDIALOG_DIR),
    ("Cornell lines", CORNELL_LINES),
    ("Cornell convos", CORNELL_CONVOS),
]:
    if os.path.exists(path):
        print(f"  ✓ {label}: {path}")
    else:
        print(f"  ~ {label}: {path} (optional, skipped if missing)")


# ============================================================
# CELL 3 — Rebuild data with per-pair EOS + boundary arrays
# ============================================================
# This is the key fix. The old .bin files had ONE EOT token for the
# whole pool. The new ones have EOT after every (User, Assistant) pair
# and boundary .npy arrays so the sampler always starts at a pair.

cornell_args = []
if os.path.exists(CORNELL_LINES) and os.path.exists(CORNELL_CONVOS):
    cornell_args = [
        "--cornell_lines_file", CORNELL_LINES,
        "--cornell_conversations_file", CORNELL_CONVOS,
    ]

dailydialog_args = []
if os.path.exists(DAILYDIALOG_DIR):
    dailydialog_args = ["--dailydialog_file", DAILYDIALOG_DIR]

cmd = [
    "python", "prepare_data.py",
    "--mode", "bilingual",
    "--hinglish_file", CORPUS_TXT,
    *dailydialog_args,
    *cornell_args,
    "--output_dir", DATA_OUT,
    "--val_ratio", "0.1",
    "--encoding", "gpt2",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("\nData preparation complete!")


# ============================================================
# CELL 4 — Verify the fix: count EOT tokens in new bin files
# ============================================================
import numpy as np

print("EOT token counts in new .bin files (should be >> 1 now):")
for fname in sorted(os.listdir(DATA_OUT)):
    if fname.endswith(".bin"):
        path = os.path.join(DATA_OUT, fname)
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        n_eot = int(np.sum(tokens == 50256))
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {fname}: {len(tokens):,} tokens, {n_eot:,} EOT tokens ({size_mb:.1f} MB)")

print("\nBoundary files:")
for fname in sorted(os.listdir(DATA_OUT)):
    if fname.endswith("_boundaries.npy"):
        path = os.path.join(DATA_OUT, fname)
        bounds = np.load(path)
        print(f"  {fname}: {len(bounds):,} pairs")


# ============================================================
# CELL 5 — Train: resume from best(1).pt, run 8000 more steps
# ============================================================
# The fixed train.py reads model_config FROM the checkpoint before
# building the model, so the 124M architecture is restored correctly.
# DO NOT pass --init_from scratch -- it is now ignored when
# --ckpt_path is present. The architecture comes from the checkpoint.
#
# iter_num is restored from the checkpoint (e.g. 8000).
# max_iters=16000 means the loop runs from 8000 to 16000.
# warmup_iters=0: model is already warm, no ramp-up needed.
# lr_decay_iters=16000: cosine decay reaches min_lr at iter 16000.

CKPT_PATH = "/kaggle/input/models/uehewbrv/hingpt3/transformers/default/1/best(1).pt"

train_cmd = [
    "python", "train.py",
    "--mode", "train",
    "--data_dir", DATA_OUT,
    "--ckpt_path", CKPT_PATH,
    "--out_dir", CKPT_OUT,
    # NOTE: --init_from is intentionally OMITTED.
    # When --ckpt_path is set, train.py reads model_config from the
    # checkpoint and builds the 124M GPT-2 model automatically.
    "--max_iters", "16000",
    "--lr_decay_iters", "16000",
    "--warmup_iters", "0",
    "--learning_rate", "2e-5",
    "--english_ratio", "0.4",
    "--batch_size", "8",        # 124M model needs smaller batch on T4
    "--grad_accum_steps", "4",  # effective batch = 32 via gradient accumulation
    "--eval_interval", "200",
    "--eval_iters", "50",
    "--dtype", "bfloat16",
    "--seed", "42",
]
print("Running:", " ".join(train_cmd))
subprocess.run(train_cmd, check=True)



# ============================================================
# CELL 6 — Copy best.pt to output (so Kaggle saves it)
# ============================================================
import shutil

best_src = os.path.join(CKPT_OUT, "best.pt")
latest_src = os.path.join(CKPT_OUT, "latest.pt")

if os.path.exists(best_src):
    shutil.copy(best_src, "/kaggle/working/best.pt")
    size = os.path.getsize("/kaggle/working/best.pt") / 1e6
    print(f"Saved best.pt to /kaggle/working/best.pt ({size:.1f} MB)")
else:
    print("WARNING: best.pt not found in checkpoint dir!")

if os.path.exists(latest_src):
    shutil.copy(latest_src, "/kaggle/working/latest.pt")
    print(f"Saved latest.pt to /kaggle/working/latest.pt")


# ============================================================
# CELL 7 — Quick inference test
# ============================================================
import sys
sys.path.insert(0, ".")

from inference import load_model, generate_response

load_model("/kaggle/working/best.pt")

test_prompts = [
    "kaisa hai bhai",
    "kya kar rahe ho aaj",
    "Hey, how's it going",
    "what are you up to",
    "monsoon mein kaafi baarish ho rahi hai",
]

print("\n=== Inference Test ===")
for prompt in test_prompts:
    response = generate_response(prompt, max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.95)
    print(f"\nUser: {prompt}")
    print(f"Assistant: {response}")
