"""Prepare Enhinged training data.

=====================================================================
WHAT CHANGED AND WHY (read this before the code)
=====================================================================
The original version of this file only knew how to do one thing:
take a single Hinglish text file (or a folder of them), tokenise it,
and split it into train.bin / val.bin. That is still fully supported
below (see `prepare_single_file` / `prepare_directory_corpus` and the
old `--input_file` / `--input_dir` CLI flags) -- nothing about that
path has been removed.

What's new is a *bilingual* mode (`--mode bilingual`) that builds TWO
separate token pools instead of one:

    - a "hinglish" pool, from your existing corpus.txt
    - an "english" pool, built by merging DailyDialog + the Cornell
      Movie-Dialogs Corpus (both are well-known, citable, casual/
      conversational English datasets -- not obscure scrapes)

Why two pools instead of just concatenating everything into one file?
Your Hinglish corpus is roughly 1M exchanges. DailyDialog is ~13k
dialogues and Cornell is ~220k exchanges -- so English is a real
minority by raw volume (order ~300-400k exchanges vs ~1M). If you
just concatenated the files and shuffled, the model would see
Hinglish 3-4x more often than English, and "full bilingual fluency"
would quietly become "mostly Hinglish, a little English on the side".

Keeping the pools separate lets `utils.py`'s BilingualDataset draw a
*controlled* fraction of every training batch from each pool (see
`english_ratio` in train.py), independent of how much raw text exists
in each language. That's the actual fix for the volume imbalance --
not something this file does alone, but this file is what makes it
possible, by keeping "how much of each language exists" and "how much
of each language the model sees per step" as two separate knobs.

=====================================================================
KEY FIX: PER-PAIR EOS TOKENS + BOUNDARY ARRAYS
=====================================================================
The original version concatenated every (User, Assistant) pair into
one giant string with "\\n\\n" separators, then appended ONE eot_token
at the very end of the whole pool. The sampler in utils.py drew
random windows from anywhere in that stream, so training examples
routinely spanned the end of one conversation and the start of a
completely unrelated one with no boundary marker between them.

The fix: tokenize every pair INDIVIDUALLY and append eot_token after
each one. The binary layout is identical (still one big .bin file,
still np.memmap-compatible), but now there is an EOS after every pair
instead of just one at the end.

In parallel, we save a *boundaries array* (`{pool}_{split}_boundaries.npy`)
-- a uint32 array of token start positions for every pair. utils.py's
sampler loads this array and always starts a training window at a
pair boundary (i.e. right after an EOS, at the start of "User: ..."),
so training examples are always exactly one conversation unit, never
a random mid-stream fragment.

=====================================================================
WHERE YOUR ENGLISH DATA COMES FROM (Kaggle-native, on purpose)
=====================================================================
DailyDialog and Cornell Movie-Dialogs are both available as ready-made
Kaggle Datasets. On Kaggle, click "Add Input" in your notebook and
search for them -- they'll be mounted read-only under
`/kaggle/input/<dataset-slug>/...` with zero download code needed.
This deliberately avoids going through Hugging Face's `datasets`
library for these two: at the time of writing, the classic
`daily_dialog` HF dataset raises
`RuntimeError: Dataset scripts are no longer supported` on current
`datasets` versions (HF deprecated script-based dataset loading), so
depending on it live is fragile. Reading a plain file Kaggle already
mounted for you is far more robust.

The parsers below (`parse_dailydialog_raw`, `parse_cornell_raw`) each
auto-detect which of two formats they've been given, so either the
original release or the modern Kaggle mirror works with no CLI or
downstream changes:

  - DailyDialog:
      * Original release: a file (commonly `dialogues_text.txt`) where
        each line is one whole dialogue, with individual utterances
        joined by the literal token `__eou__` (end-of-utterance).
      * Kaggle mirror (thedevastator/dailydialog-unlock-the-conversation...):
        train.csv/validation.csv/test.csv, each with a `dialog` column
        holding one conversation per row. Point `--dailydialog_file` at
        any one of these CSVs, or at the directory containing them --
        all splits found are merged automatically.

  - Cornell Movie-Dialogs:
      * Original release: `movie_lines.txt` (each line: lineID
        +++$+++ characterID +++$+++ movieID +++$+++ characterName
        +++$+++ text) and `movie_conversations.txt` (each line ends
        with a Python-list-looking string of lineIDs, e.g.
        "['L194', 'L195', 'L196', 'L197']"), joined by ' +++$+++ '.
      * Kaggle mirror (Cornell-University/movie-dialog-corpus):
        movie_lines.tsv / movie_conversations.tsv, same field layout
        but tab-separated. Either .txt/.tsv filename works; the
        separator is sniffed from the file's first line rather than
        assumed.

If a parser ever prints a suspiciously low dialogue count on a dataset
that isn't one of the two mirrors above, open the file with `head` and
compare it against the formats described here -- you may be looking at
a third, differently-repackaged mirror that needs its own branch added
to the relevant `parse_*` function.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# =====================================================================
# Shared low-level helpers
# =====================================================================

def clean_text(filepath: str) -> str:
    """Read and normalise a text corpus."""

    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def encode_text(text: str, encoding_name: str) -> tuple[np.ndarray, str]:
    """Tokenise text with tiktoken and return the chosen dtype.

    NOTE: This is kept for the legacy single-file path only.
    For bilingual mode, use encode_pairs() which tokenises each pair
    individually and returns a boundary array.
    """

    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    start_time = time.time()
    token_ids = encoding.encode_ordinary(text)
    token_ids.append(encoding.eot_token)
    dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32
    array = np.array(token_ids, dtype=dtype)
    elapsed = time.time() - start_time
    print(f"Tokenised {len(array):,} tokens in {elapsed:.2f}s")
    return array, str(array.dtype)


def encode_pairs(
    pairs: list[str],
    encoding_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Tokenise a list of text pairs, appending eot_token after EACH one.

    Returns:
        token_array   -- flat uint16/uint32 array of all tokens concatenated
        boundaries    -- uint32 array of start positions (in tokens) for each
                         pair; len(boundaries) == len(pairs). The sampler in
                         utils.py loads this to ensure training windows always
                         start at a pair boundary, never mid-conversation.
        dtype_string  -- "uint16" or "uint32", for metadata.

    Why tokenise per-pair rather than the whole string at once?
    Because the only way to insert eot_token after EVERY pair (not just
    at the very end of the pool) is to tokenise each pair separately and
    append eot_token individually. The concatenated output is byte-for-byte
    identical to what you'd get by tokenising the full string EXCEPT that
    now every pair ends with eot_token instead of bare "\\n\\n".
    """

    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32
    eot = encoding.eot_token

    start_time = time.time()
    chunks: list[np.ndarray] = []
    boundary_starts: list[int] = []
    cursor = 0

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        boundary_starts.append(cursor)
        ids = encoding.encode_ordinary(pair)
        ids.append(eot)
        arr = np.array(ids, dtype=dtype)
        chunks.append(arr)
        cursor += len(arr)

    if not chunks:
        token_array = np.array([], dtype=dtype)
        boundaries = np.array([], dtype=np.uint32)
        return token_array, boundaries, str(dtype)

    token_array = np.concatenate(chunks)
    boundaries = np.array(boundary_starts, dtype=np.uint32)
    elapsed = time.time() - start_time
    print(
        f"Tokenised {len(pairs):,} pairs → {len(token_array):,} tokens "
        f"({len(boundaries):,} boundaries) in {elapsed:.2f}s"
    )
    return token_array, boundaries, str(dtype)


def split_tokens(ids: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split token IDs into train and validation segments."""

    val_size = int(len(ids) * val_ratio)
    train_size = len(ids) - val_size
    return ids[:train_size], ids[train_size:]


def split_pairs(
    pairs: list[str],
    val_ratio: float,
) -> tuple[list[str], list[str]]:
    """Split pairs list into train and validation subsets.

    Splitting at the pair level (not the token level) ensures that the
    train and val boundary arrays are consistent with their respective
    token arrays -- no pair is ever split across the boundary.
    """

    n_val = max(1, int(len(pairs) * val_ratio))
    n_train = len(pairs) - n_val
    return pairs[:n_train], pairs[n_train:]


def save_bin(ids: np.ndarray, filepath: str) -> None:
    """Write token IDs to a binary file."""

    ids.tofile(filepath)
    print(f"Saved {len(ids):,} tokens to {filepath}")


def save_boundaries(boundaries: np.ndarray, filepath: str) -> None:
    """Write pair-start boundary positions alongside a .bin file."""

    np.save(filepath, boundaries)
    print(f"Saved {len(boundaries):,} pair boundaries to {filepath}")


# =====================================================================
# Original single-source modes (backward compatible, unchanged behaviour)
# =====================================================================

def save_metadata_legacy(output_dir: str, encoding_name: str, token_dtype: str, val_ratio: float, total_tokens: int) -> None:
    """Persist preprocessing metadata next to the binary files (single-pool mode)."""

    metadata = {
        "encoding": encoding_name,
        "token_dtype": token_dtype,
        "val_ratio": val_ratio,
        "total_tokens": total_tokens,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def prepare_single_file(input_file: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    """Prepare a single text corpus file (original behaviour, unchanged)."""

    os.makedirs(output_dir, exist_ok=True)
    text = clean_text(input_file)
    ids, token_dtype = encode_text(text, encoding_name)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, "train.bin"))
    save_bin(val_ids, os.path.join(output_dir, "val.bin"))
    save_metadata_legacy(output_dir, encoding_name, token_dtype, val_ratio, len(ids))
    print("Preprocessing complete.")


def prepare_directory_corpus(input_dir: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    """Prepare a directory of text files as one combined corpus (original behaviour, unchanged)."""

    os.makedirs(output_dir, exist_ok=True)
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    txt_files = sorted(Path(input_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    token_chunks: list[np.ndarray] = []
    token_dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32

    for path in txt_files:
        text = clean_text(str(path))
        token_ids = encoding.encode_ordinary(text)
        token_ids.append(encoding.eot_token)
        token_chunks.append(np.array(token_ids, dtype=token_dtype))
        print(f"{path.name}: {len(token_ids):,} tokens")

    ids = np.concatenate(token_chunks)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, "train.bin"))
    save_bin(val_ids, os.path.join(output_dir, "val.bin"))
    save_metadata_legacy(output_dir, encoding_name, str(ids.dtype), val_ratio, len(ids))
    print("Preprocessing complete.")


# =====================================================================
# NEW: English-source parsers
# =====================================================================
# Both parsers return the *same* shape: a list of dialogues, where each
# dialogue is a list of utterance strings in chronological order, e.g.
#   [["Hey, how's it going?", "Not bad, you?", "Same old, same old."], ...]
# That common shape is what lets a single downstream function
# (`utterances_to_pairs`) turn either source into the same
# "User: ...\nAssistant: ..." schema your Hinglish data already uses.

def _parse_dailydialog_original(path: str) -> list[list[str]]:
    """Parse the original DailyDialog release: dialogues_text.txt, one
    dialogue per line, utterances joined by the literal `__eou__` marker."""

    dialogues: list[list[str]] = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
            if len(utterances) >= 2:
                dialogues.append(utterances)
            else:
                skipped += 1

    if skipped:
        print(f"  (DailyDialog: skipped {skipped:,} lines that didn't look like dialogues)")
    return dialogues


def _split_dailydialog_dialog_field(raw_dialog: str) -> list[str]:
    """Split one CSV `dialog` cell into individual utterances.

    The Kaggle mirror's `dialog` column has been seen in the wild in two
    shapes: a Python-list-looking string (e.g. "['Hi.', 'Hello!']"), or a
    plain `__eou__`-joined string like the original release. Handle both.
    """

    raw_dialog = raw_dialog.strip()
    if not raw_dialog:
        return []

    if raw_dialog.startswith("[") and raw_dialog.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_dialog)
            if isinstance(parsed, (list, tuple)):
                return [str(u).strip() for u in parsed if str(u).strip()]
        except (ValueError, SyntaxError):
            pass  # fall through to __eou__ / plain-string handling below

    if "__eou__" in raw_dialog:
        return [u.strip() for u in raw_dialog.split("__eou__") if u.strip()]

    return [raw_dialog] if raw_dialog else []


def _parse_dailydialog_kaggle_csv(csv_paths: list[str]) -> list[list[str]]:
    """Parse the Kaggle CSV mirror (train.csv/validation.csv/test.csv),
    each with a `dialog` column holding one full conversation per row.

    Returns the SAME shape as the original parser: list[list[str]].
    """

    import csv

    dialogues: list[list[str]] = []
    skipped = 0

    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "dialog" not in reader.fieldnames:
                print(f"  WARNING: {csv_path} has no 'dialog' column (found {reader.fieldnames}); skipping this file.")
                continue
            for row in reader:
                utterances = _split_dailydialog_dialog_field(row.get("dialog", ""))
                if len(utterances) >= 2:
                    dialogues.append(utterances)
                else:
                    skipped += 1

    if skipped:
        print(f"  (DailyDialog CSV: skipped {skipped:,} rows that didn't look like dialogues)")
    return dialogues


def _resolve_dailydialog_kaggle_csv_paths(path: str) -> list[str]:
    """Given whatever path the user pointed at, find train/validation/test CSVs.

    Accepts a directory containing them, or any one of the three CSV
    files directly (siblings are picked up automatically).
    """

    candidate_dir = path if os.path.isdir(path) else os.path.dirname(path) or "."
    names = ["train.csv", "validation.csv", "test.csv"]
    found = [os.path.join(candidate_dir, name) for name in names if os.path.exists(os.path.join(candidate_dir, name))]
    return found


def parse_dailydialog_raw(path: str) -> list[list[str]]:
    """Parse DailyDialog from EITHER the original release or the Kaggle
    CSV mirror, auto-detecting which one `path` points at.

    - Original release: `path` is (or points at) `dialogues_text.txt`,
      one dialogue per line, utterances joined by `__eou__`.
    - Kaggle mirror (thedevastator/dailydialog-...): `path` is a
      directory containing train.csv/validation.csv/test.csv (or one of
      those files directly), each with a `dialog` column. All three
      splits found are merged.

    Either way, returns the same shape as before: list[list[str]], so
    nothing downstream (utterances_to_pairs, ...) needs to change.
    """

    if path.lower().endswith(".csv"):
        csv_paths = _resolve_dailydialog_kaggle_csv_paths(path)
        if not csv_paths:
            csv_paths = [path]  # just the one CSV given, no siblings found
        print(f"  Detected DailyDialog Kaggle CSV format ({len(csv_paths)} file(s): {[os.path.basename(p) for p in csv_paths]})")
        return _parse_dailydialog_kaggle_csv(csv_paths)

    if os.path.isdir(path):
        csv_paths = _resolve_dailydialog_kaggle_csv_paths(path)
        if csv_paths:
            print(f"  Detected DailyDialog Kaggle CSV format ({len(csv_paths)} file(s): {[os.path.basename(p) for p in csv_paths]})")
            return _parse_dailydialog_kaggle_csv(csv_paths)
        raise FileNotFoundError(
            f"'{path}' is a directory but contains none of train.csv/validation.csv/test.csv. "
            "Point --dailydialog_file at dialogues_text.txt, at a CSV file, or at a directory "
            "containing the Kaggle CSVs."
        )

    print("  Detected DailyDialog original format (dialogues_text.txt, __eou__-separated)")
    return _parse_dailydialog_original(path)


def _detect_cornell_separator(sample_path: str) -> str:
    """Sniff whether a Cornell file uses the original ' +++$+++ ' separator
    or the Kaggle TSV mirror's tab character.

    Reads just the first non-empty line -- cheap, and the separator is
    consistent throughout a given file.
    """

    ORIGINAL_SEP = " +++$+++ "
    with open(sample_path, "r", encoding="iso-8859-1", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.strip():
                if ORIGINAL_SEP in raw_line:
                    return ORIGINAL_SEP
                if "\t" in raw_line:
                    return "\t"
                # Neither marker found on the first non-empty line -- default
                # to the original separator and let downstream malformed-line
                # counting surface the problem if this guess is wrong.
                return ORIGINAL_SEP
    return ORIGINAL_SEP


def parse_cornell_raw(lines_path: str, conversations_path: str) -> list[list[str]]:
    """Parse the Cornell Movie-Dialogs Corpus from EITHER the original
    release (movie_lines.txt / movie_conversations.txt, ' +++$+++ '
    separated) or the Kaggle TSV mirror (movie_lines.tsv /
    movie_conversations.tsv, tab-separated). The separator is
    auto-detected per file rather than assumed, and either .txt or .tsv
    filenames work with no other code changes needed.

    `movie_lines`: each row maps a lineID to its spoken text, as fields
    [lineID, characterID, movieID, characterName, text].
    `movie_conversations`: each row's last field lists the ordered
    lineIDs making up one conversation, as a Python-list-looking string,
    e.g. "['L194', 'L195', 'L196', 'L197']".

    Returns the same shape as before: list[list[str]].
    """

    lines_sep = _detect_cornell_separator(lines_path)
    convos_sep = _detect_cornell_separator(conversations_path)
    print(f"  Detected Cornell separator: lines={lines_sep!r}, conversations={convos_sep!r}")

    line_text: dict[str, str] = {}
    malformed_lines = 0

    with open(lines_path, "r", encoding="iso-8859-1", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split(lines_sep)
            if len(parts) < 5:
                malformed_lines += 1
                continue
            line_id, text = parts[0], parts[4]
            line_text[line_id.strip()] = text.strip()

    dialogues: list[list[str]] = []
    malformed_convos = 0

    with open(conversations_path, "r", encoding="iso-8859-1", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split(convos_sep)
            if len(parts) < 4:
                malformed_convos += 1
                continue
            try:
                # The last field looks like "['L194', 'L195', ...]" --
                # ast.literal_eval safely parses that Python-list syntax
                # without executing arbitrary code (unlike eval()).
                line_ids = ast.literal_eval(parts[3].strip())
            except (ValueError, SyntaxError):
                malformed_convos += 1
                continue

            utterances = [line_text[line_id] for line_id in line_ids if line_id in line_text]
            utterances = [u for u in utterances if u]
            if len(utterances) >= 2:
                dialogues.append(utterances)

    if malformed_lines or malformed_convos:
        print(
            f"  (Cornell: skipped {malformed_lines:,} malformed lines and "
            f"{malformed_convos:,} malformed conversations)"
        )
    return dialogues


def utterances_to_pairs(utterances: list[str]) -> list[str]:
    """Decompose ONE multi-turn dialogue into independent (input, output) blocks.

    ---- WHY THIS FUNCTION EXISTS ----
    The first version of this file chained an entire dialogue into one
    long block, alternating User/Assistant across every line. That's a
    different *shape* of training example than your Hinglish corpus,
    which is pure single-turn (input, output) pairs -- one prompt, one
    reply, nothing chained. Mixing those two shapes into one training
    run would teach the model two different behaviours depending on
    which pool an example came from, and for Cornell specifically,
    consecutive movie lines are often not real question-and-answer
    exchanges at all, so chaining them risked teaching the "answer the
    question" behaviour actively wrong lessons.

    The fix: a sliding window. For a dialogue [u0, u1, u2, u3], this
    yields THREE independent, self-contained pairs:
        (input=u0, output=u1)
        (input=u1, output=u2)
        (input=u2, output=u3)
    instead of one 4-line chained block. Every example is now exactly
    the same "one input, one output" shape as your Hinglish data.

    Returns a list of formatted pair strings:
        ["User: u0\\nAssistant: u1", "User: u1\\nAssistant: u2", ...]
    """

    pairs: list[str] = []
    for index in range(len(utterances) - 1):
        input_utterance = utterances[index].strip()
        output_utterance = utterances[index + 1].strip()
        if input_utterance and output_utterance:
            pairs.append(f"User: {input_utterance}\nAssistant: {output_utterance}")
    return pairs


def corpus_txt_to_pairs(text: str) -> list[str]:
    """Split a corpus.txt-style Hinglish file into individual pair strings.

    corpus.txt uses blank lines to separate pairs. Each block is expected
    to look like:
        User: <prompt>
        Assistant: <reply>

    This function splits on blank lines and returns each non-empty block
    as one element. Blocks that don't start with "User:" are kept anyway
    (they'll tokenise fine and include the EOS marker) but a warning is
    printed so you can spot malformed lines.

    This replaces the old `dialogues_to_text()` approach of joining
    everything into one giant string before tokenising -- we need the
    pair list so we can tokenise per-pair in `encode_pairs()`.
    """

    raw_blocks = re.split(r"\n\s*\n", text)
    pairs: list[str] = []
    n_malformed = 0
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if not block.startswith("User:"):
            n_malformed += 1
        pairs.append(block)
    if n_malformed:
        print(f"  WARNING: {n_malformed:,} Hinglish blocks don't start with 'User:' -- check corpus.txt formatting.")
    return pairs


# =====================================================================
# NEW: bilingual preparation pipeline
# =====================================================================

def save_metadata_bilingual(
    output_dir: str,
    encoding_name: str,
    token_dtype: str,
    val_ratio: float,
    pools: dict[str, dict],
) -> None:
    """Persist metadata describing BOTH token pools.

    `pools` looks like:
        {
            "hinglish": {"total_tokens": N, "n_pairs": P, "sources": [...]},
            "english":  {"total_tokens": M, "n_pairs": Q, "sources": [...]},
        }
    utils.py's BilingualDataset reads this file to know the token
    dtype/encoding, and train.py prints it at startup so you can see
    the actual language balance you built, not just assume it.
    """

    metadata = {
        "encoding": encoding_name,
        "token_dtype": token_dtype,
        "val_ratio": val_ratio,
        "bilingual": True,
        "boundary_aligned": True,   # flag so utils.py knows to look for .npy files
        "pools": pools,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _tokenise_and_write_pool(
    pool_name: str,
    pairs: list[str],
    output_dir: str,
    val_ratio: float,
    encoding_name: str,
) -> tuple[int, int, str]:
    """Tokenise `pairs` and write {pool_name}_{train,val}.bin + boundary .npy files.

    Each pair is tokenised individually with its own EOS appended.
    The boundary array records the token start position of each pair,
    split consistently with the token array (train pairs → train boundaries,
    val pairs → val boundaries).

    Returns (total_token_count, total_pair_count, token_dtype_string).
    """

    if not pairs:
        print(f"WARNING: pool '{pool_name}' is empty -- writing empty bin files.")
        empty_tokens = np.array([], dtype=np.uint16)
        empty_bounds = np.array([], dtype=np.uint32)
        save_bin(empty_tokens, os.path.join(output_dir, f"{pool_name}_train.bin"))
        save_bin(empty_tokens, os.path.join(output_dir, f"{pool_name}_val.bin"))
        save_boundaries(empty_bounds, os.path.join(output_dir, f"{pool_name}_train_boundaries.npy"))
        save_boundaries(empty_bounds, os.path.join(output_dir, f"{pool_name}_val_boundaries.npy"))
        return 0, 0, "uint16"

    # Split pairs BEFORE tokenising so boundary arrays are consistent
    # with their respective token arrays.
    train_pairs, val_pairs = split_pairs(pairs, val_ratio)

    train_tokens, train_bounds, token_dtype = encode_pairs(train_pairs, encoding_name)
    val_tokens, val_bounds, _ = encode_pairs(val_pairs, encoding_name)

    # Validation boundary positions are relative to the start of the val
    # token array (which starts at 0), so they're already correct.

    save_bin(train_tokens, os.path.join(output_dir, f"{pool_name}_train.bin"))
    save_bin(val_tokens, os.path.join(output_dir, f"{pool_name}_val.bin"))
    save_boundaries(train_bounds, os.path.join(output_dir, f"{pool_name}_train_boundaries.npy"))
    save_boundaries(val_bounds, os.path.join(output_dir, f"{pool_name}_val_boundaries.npy"))

    total_tokens = len(train_tokens) + len(val_tokens)
    total_pairs = len(train_pairs) + len(val_pairs)
    return total_tokens, total_pairs, token_dtype


def prepare_bilingual_corpus(
    output_dir: str,
    val_ratio: float,
    encoding_name: str,
    hinglish_file: Optional[str] = None,
    dailydialog_file: Optional[str] = None,
    cornell_lines_file: Optional[str] = None,
    cornell_conversations_file: Optional[str] = None,
) -> None:
    """Build the hinglish/english token pools used by BilingualDataset.

    Any of the sources can be omitted -- e.g. if you only have
    DailyDialog attached in a given Kaggle session and want to add
    Cornell later, just leave `cornell_*` unset and re-run once you
    have it. The hinglish pool is untouched either way.

    KEY CHANGE vs. the old version:
    - Pairs are tokenised INDIVIDUALLY (one EOS per pair) instead of as
      one giant concatenated string (one EOS at the end).
    - Boundary arrays are saved alongside every .bin file.
    This fixes the boundary-spanning training bug described in the module
    docstring.
    """

    os.makedirs(output_dir, exist_ok=True)
    pools: dict[str, dict] = {}
    final_token_dtype = "uint16"

    # ---- Hinglish pool (your existing corpus, unchanged content) ----
    if hinglish_file:
        print(f"\n=== Hinglish pool: {hinglish_file} ===")
        hinglish_text = clean_text(hinglish_file)
        hinglish_pairs = corpus_txt_to_pairs(hinglish_text)
        print(f"  Split into {len(hinglish_pairs):,} individual pairs.")
        total_tokens, total_pairs, token_dtype = _tokenise_and_write_pool(
            "hinglish", hinglish_pairs, output_dir, val_ratio, encoding_name
        )
        final_token_dtype = token_dtype
        pools["hinglish"] = {
            "total_tokens": total_tokens,
            "n_pairs": total_pairs,
            "sources": [hinglish_file],
        }
    else:
        print("No --hinglish_file given -- skipping the Hinglish pool.")

    # ---- English pool (DailyDialog + Cornell, merged) ----
    english_pairs: list[str] = []
    english_sources: list[str] = []

    if dailydialog_file:
        print(f"\n=== Parsing DailyDialog: {dailydialog_file} ===")
        dd_dialogues = parse_dailydialog_raw(dailydialog_file)
        print(f"  Parsed {len(dd_dialogues):,} dialogues from DailyDialog.")
        dd_pairs: list[str] = []
        for utterances in dd_dialogues:
            dd_pairs.extend(utterances_to_pairs(utterances))
        print(f"  Decomposed into {len(dd_pairs):,} (User, Assistant) pairs.")
        english_pairs.extend(dd_pairs)
        english_sources.append(f"dailydialog ({len(dd_dialogues):,} dialogues, {len(dd_pairs):,} pairs): {dailydialog_file}")

    if cornell_lines_file and cornell_conversations_file:
        print(f"\n=== Parsing Cornell Movie-Dialogs: {cornell_lines_file} + {cornell_conversations_file} ===")
        cornell_dialogues = parse_cornell_raw(cornell_lines_file, cornell_conversations_file)
        print(f"  Parsed {len(cornell_dialogues):,} conversations from Cornell.")
        cornell_pairs: list[str] = []
        for utterances in cornell_dialogues:
            cornell_pairs.extend(utterances_to_pairs(utterances))
        print(f"  Decomposed into {len(cornell_pairs):,} (User, Assistant) pairs.")
        english_pairs.extend(cornell_pairs)
        english_sources.append(f"cornell ({len(cornell_dialogues):,} conversations, {len(cornell_pairs):,} pairs)")
    elif cornell_lines_file or cornell_conversations_file:
        print(
            "WARNING: Cornell needs BOTH --cornell_lines_file and "
            "--cornell_conversations_file -- only one was given, skipping Cornell."
        )

    if english_pairs:
        print(f"\n=== English pool: {len(english_pairs):,} pairs total ===")
        total_tokens, total_pairs, token_dtype = _tokenise_and_write_pool(
            "english", english_pairs, output_dir, val_ratio, encoding_name
        )
        final_token_dtype = token_dtype
        pools["english"] = {
            "total_tokens": total_tokens,
            "n_pairs": total_pairs,
            "sources": english_sources,
        }
    else:
        print("\nNo English sources given -- skipping the English pool.")

    if not pools:
        raise ValueError("No sources provided at all -- nothing to prepare.")

    save_metadata_bilingual(output_dir, encoding_name, final_token_dtype, val_ratio, pools)

    # Summary
    print("\n=== Summary ===")
    for pool_name, info in pools.items():
        print(f"  {pool_name}: {info['total_tokens']:,} tokens, {info['n_pairs']:,} pairs")
    if "hinglish" in pools and "english" in pools:
        total = pools["hinglish"]["total_tokens"] + pools["english"]["total_tokens"]
        if total > 0:
            en_share = pools["english"]["total_tokens"] / total * 100
            print(f"  Raw English share of combined tokens: {en_share:.1f}%")
            print(
                "  (This will very likely be well under 50%. That's expected and "
                "fine -- train.py's english_ratio batch mixing, not this raw "
                "share, is what controls actual training exposure.)"
            )
    print("\nBilingual preprocessing complete.")
    print("\nEOS token verification:")
    _verify_eos_counts(output_dir, pools, encoding_name)


def _verify_eos_counts(output_dir: str, pools: dict, encoding_name: str) -> None:
    """Quick sanity check: count EOT tokens in each bin and compare to pair count."""

    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    eot = encoding.eot_token
    dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32

    for pool_name, info in pools.items():
        for split in ("train", "val"):
            bin_path = os.path.join(output_dir, f"{pool_name}_{split}.bin")
            npy_path = os.path.join(output_dir, f"{pool_name}_{split}_boundaries.npy")
            if not os.path.exists(bin_path):
                continue
            tokens = np.memmap(bin_path, dtype=dtype, mode="r")
            n_eot = int(np.sum(tokens == eot))
            n_bounds = len(np.load(npy_path)) if os.path.exists(npy_path) else "n/a"
            status = "✓" if isinstance(n_bounds, str) or n_eot == n_bounds else "MISMATCH"
            print(f"  {pool_name}_{split}: {n_eot:,} EOT tokens, {n_bounds} boundaries  [{status}]")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Prepare Enhinged conversational data for GPT training.")
    parser.add_argument(
        "--mode",
        choices=["legacy", "bilingual"],
        default="legacy",
        help="'legacy' = original single-pool behaviour (--input_file/--input_dir). "
        "'bilingual' = build separate hinglish/english pools for BilingualDataset.",
    )
    # Legacy single-pool args (unchanged)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    # Bilingual args
    parser.add_argument("--hinglish_file", type=str, default=None, help="Path to your Hinglish corpus.txt")
    parser.add_argument("--dailydialog_file", type=str, default=None, help="Path to DailyDialog: either dialogues_text.txt (original), or a Kaggle CSV / directory of train.csv+validation.csv+test.csv (auto-detected)")
    parser.add_argument("--cornell_lines_file", type=str, default=None, help="Path to Cornell movie_lines.txt (original) or movie_lines.tsv (Kaggle, tab-separated; auto-detected)")
    parser.add_argument("--cornell_conversations_file", type=str, default=None, help="Path to Cornell movie_conversations.txt (original) or movie_conversations.tsv (Kaggle, tab-separated; auto-detected)")
    # Shared args
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--encoding", type=str, default="gpt2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "bilingual":
        if not any([args.hinglish_file, args.dailydialog_file, args.cornell_lines_file]):
            print("Bilingual mode needs at least one of: --hinglish_file, --dailydialog_file, --cornell_lines_file")
            sys.exit(1)
        prepare_bilingual_corpus(
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            encoding_name=args.encoding,
            hinglish_file=args.hinglish_file,
            dailydialog_file=args.dailydialog_file,
            cornell_lines_file=args.cornell_lines_file,
            cornell_conversations_file=args.cornell_conversations_file,
        )
        sys.exit(0)

    # --- legacy mode, exactly as before ---
    if args.input_file and args.input_dir:
        print("Specify either --input_file or --input_dir, not both.")
        sys.exit(1)

    if args.input_file:
        prepare_single_file(args.input_file, args.output_dir, args.val_ratio, args.encoding)
    elif args.input_dir:
        prepare_directory_corpus(args.input_dir, args.output_dir, args.val_ratio, args.encoding)
    else:
        print("Provide --input_file or --input_dir (or use --mode bilingual).")
        sys.exit(1)
