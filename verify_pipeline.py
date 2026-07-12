"""Optional: a two-minute sanity check to run BEFORE you commit Kaggle GPU
hours to a full training run.

It builds a tiny bilingual pool from whatever real source files you point
it at, then prints back a few decoded (input, output) pairs from each
language so you can eyeball that:
  - the Hinglish pairs still look like your original corpus
  - the DailyDialog/Cornell pairs are genuinely single-turn (one input,
    one output), not chained multi-turn dialogue
  - nothing is mojibake / mis-encoded

Usage (on Kaggle, after attaching your datasets):

    python verify_pipeline.py \\
        --hinglish_file /kaggle/input/your-hinglish-corpus/corpus.txt \\
        --dailydialog_file /kaggle/input/dailydialog/dialogues_text.txt \\
        --cornell_lines_file /kaggle/input/cornell-movie-dialogs-corpusdialog-datasets/movie_lines.txt \\
        --cornell_conversations_file /kaggle/input/cornell-movie-dialogs-corpusdialog-datasets/movie_conversations.txt

This uses the REAL tiktoken GPT-2 tokenizer (unlike the automated tests,
this script needs internet access the first time it runs, same as your
actual training run will).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np
import tiktoken

import prepare_data as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hinglish_file", default=None)
    parser.add_argument("--dailydialog_file", default=None)
    parser.add_argument("--cornell_lines_file", default=None)
    parser.add_argument("--cornell_conversations_file", default=None)
    parser.add_argument("--n_samples", type=int, default=5, help="How many example pairs to print per language")
    args = parser.parse_args()

    if not any([args.hinglish_file, args.dailydialog_file, args.cornell_lines_file]):
        raise SystemExit("Provide at least one of --hinglish_file / --dailydialog_file / --cornell_lines_file")

    tmp_dir = tempfile.mkdtemp(prefix="enhinged_verify_")
    print(f"(writing a small test output to {tmp_dir} -- safe to ignore/delete afterwards)\n")

    try:
        pd.prepare_bilingual_corpus(
            output_dir=tmp_dir,
            val_ratio=0.1,
            encoding_name="gpt2",
            hinglish_file=args.hinglish_file,
            dailydialog_file=args.dailydialog_file,
            cornell_lines_file=args.cornell_lines_file,
            cornell_conversations_file=args.cornell_conversations_file,
        )

        encoding = tiktoken.get_encoding("gpt2")

        for pool_name in ("hinglish", "english"):
            path = os.path.join(tmp_dir, f"{pool_name}_train.bin")
            if not os.path.exists(path):
                continue
            tokens = np.memmap(path, dtype=np.uint16, mode="r")
            text = encoding.decode(tokens[: min(len(tokens), 4000)].tolist())
            blocks = [b for b in text.split("\n\n") if b.strip()][: args.n_samples]

            print(f"\n{'=' * 70}\n{pool_name.upper()} POOL -- {args.n_samples} example pairs\n{'=' * 70}")
            for block in blocks:
                print(block)
                print("-" * 40)

        print(
            "\nLook closely at the blocks above:\n"
            "  - Each one should be exactly 'User: ...' followed by 'Assistant: ...'\n"
            "    (one input, one output -- not a long chain of multiple turns)\n"
            "  - The text should read cleanly, no garbled characters\n"
            "  - The English examples should look like plausible casual chat,\n"
            "    even though they were extracted from dialogue/movie sources\n"
            "\nIf all of that looks right, you're good to launch the real Kaggle run."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
