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

def clean_text(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

def encode_text(text: str, encoding_name: str) -> tuple[np.ndarray, str]:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    start_time = time.time()
    token_ids = encoding.encode_ordinary(text)
    token_ids.append(encoding.eot_token)
    dtype = np.uint16 if encoding.n_vocab <= 65535 else np.uint32
    array = np.array(token_ids, dtype=dtype)
    elapsed = time.time() - start_time
    print(f'Tokenised {len(array):,} tokens in {elapsed:.2f}s')
    return (array, str(array.dtype))

def encode_pairs(pairs: list[str], encoding_name: str) -> tuple[np.ndarray, np.ndarray, str]:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    dtype = np.uint16 if encoding.n_vocab <= 65535 else np.uint32
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
        return (token_array, boundaries, str(dtype))
    token_array = np.concatenate(chunks)
    boundaries = np.array(boundary_starts, dtype=np.uint32)
    elapsed = time.time() - start_time
    print(f'Tokenised {len(pairs):,} pairs → {len(token_array):,} tokens ({len(boundaries):,} boundaries) in {elapsed:.2f}s')
    return (token_array, boundaries, str(dtype))

def split_tokens(ids: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    val_size = int(len(ids) * val_ratio)
    train_size = len(ids) - val_size
    return (ids[:train_size], ids[train_size:])

def split_pairs(pairs: list[str], val_ratio: float) -> tuple[list[str], list[str]]:
    n_val = max(1, int(len(pairs) * val_ratio))
    n_train = len(pairs) - n_val
    return (pairs[:n_train], pairs[n_train:])

def save_bin(ids: np.ndarray, filepath: str) -> None:
    ids.tofile(filepath)
    print(f'Saved {len(ids):,} tokens to {filepath}')

def save_boundaries(boundaries: np.ndarray, filepath: str) -> None:
    np.save(filepath, boundaries)
    print(f'Saved {len(boundaries):,} pair boundaries to {filepath}')

def save_metadata_legacy(output_dir: str, encoding_name: str, token_dtype: str, val_ratio: float, total_tokens: int) -> None:
    metadata = {'encoding': encoding_name, 'token_dtype': token_dtype, 'val_ratio': val_ratio, 'total_tokens': total_tokens}
    with open(os.path.join(output_dir, 'meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

def prepare_single_file(input_file: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    text = clean_text(input_file)
    ids, token_dtype = encode_text(text, encoding_name)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, 'train.bin'))
    save_bin(val_ids, os.path.join(output_dir, 'val.bin'))
    save_metadata_legacy(output_dir, encoding_name, token_dtype, val_ratio, len(ids))
    print('Preprocessing complete.')

def prepare_directory_corpus(input_dir: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    txt_files = sorted(Path(input_dir).glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f'No .txt files found in {input_dir}')
    token_chunks: list[np.ndarray] = []
    token_dtype = np.uint16 if encoding.n_vocab <= 65535 else np.uint32
    for path in txt_files:
        text = clean_text(str(path))
        token_ids = encoding.encode_ordinary(text)
        token_ids.append(encoding.eot_token)
        token_chunks.append(np.array(token_ids, dtype=token_dtype))
        print(f'{path.name}: {len(token_ids):,} tokens')
    ids = np.concatenate(token_chunks)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, 'train.bin'))
    save_bin(val_ids, os.path.join(output_dir, 'val.bin'))
    save_metadata_legacy(output_dir, encoding_name, str(ids.dtype), val_ratio, len(ids))
    print('Preprocessing complete.')

def _parse_dailydialog_original(path: str) -> list[list[str]]:
    dialogues: list[list[str]] = []
    skipped = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            utterances = [u.strip() for u in line.split('__eou__') if u.strip()]
            if len(utterances) >= 2:
                dialogues.append(utterances)
            else:
                skipped += 1
    if skipped:
        print(f"  (DailyDialog: skipped {skipped:,} lines that didn't look like dialogues)")
    return dialogues

def _split_dailydialog_dialog_field(raw_dialog: str) -> list[str]:
    raw_dialog = raw_dialog.strip()
    if not raw_dialog:
        return []
    if raw_dialog.startswith('[') and raw_dialog.endswith(']'):
        try:
            parsed = ast.literal_eval(raw_dialog)
            if isinstance(parsed, (list, tuple)):
                return [str(u).strip() for u in parsed if str(u).strip()]
        except (ValueError, SyntaxError):
            pass
    if '__eou__' in raw_dialog:
        return [u.strip() for u in raw_dialog.split('__eou__') if u.strip()]
    return [raw_dialog] if raw_dialog else []

def _parse_dailydialog_kaggle_csv(csv_paths: list[str]) -> list[list[str]]:
    import csv
    dialogues: list[list[str]] = []
    skipped = 0
    for csv_path in csv_paths:
        with open(csv_path, 'r', encoding='utf-8', errors='replace', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or 'dialog' not in reader.fieldnames:
                print(f"  WARNING: {csv_path} has no 'dialog' column (found {reader.fieldnames}); skipping this file.")
                continue
            for row in reader:
                utterances = _split_dailydialog_dialog_field(row.get('dialog', ''))
                if len(utterances) >= 2:
                    dialogues.append(utterances)
                else:
                    skipped += 1
    if skipped:
        print(f"  (DailyDialog CSV: skipped {skipped:,} rows that didn't look like dialogues)")
    return dialogues

def _resolve_dailydialog_kaggle_csv_paths(path: str) -> list[str]:
    candidate_dir = path if os.path.isdir(path) else os.path.dirname(path) or '.'
    names = ['train.csv', 'validation.csv', 'test.csv']
    found = [os.path.join(candidate_dir, name) for name in names if os.path.exists(os.path.join(candidate_dir, name))]
    return found

def parse_dailydialog_raw(path: str) -> list[list[str]]:
    if path.lower().endswith('.csv'):
        csv_paths = _resolve_dailydialog_kaggle_csv_paths(path)
        if not csv_paths:
            csv_paths = [path]
        print(f'  Detected DailyDialog Kaggle CSV format ({len(csv_paths)} file(s): {[os.path.basename(p) for p in csv_paths]})')
        return _parse_dailydialog_kaggle_csv(csv_paths)
    if os.path.isdir(path):
        csv_paths = _resolve_dailydialog_kaggle_csv_paths(path)
        if csv_paths:
            print(f'  Detected DailyDialog Kaggle CSV format ({len(csv_paths)} file(s): {[os.path.basename(p) for p in csv_paths]})')
            return _parse_dailydialog_kaggle_csv(csv_paths)
        raise FileNotFoundError(f"'{path}' is a directory but contains none of train.csv/validation.csv/test.csv. Point --dailydialog_file at dialogues_text.txt, at a CSV file, or at a directory containing the Kaggle CSVs.")
    print('  Detected DailyDialog original format (dialogues_text.txt, __eou__-separated)')
    return _parse_dailydialog_original(path)

def _detect_cornell_separator(sample_path: str) -> str:
    ORIGINAL_SEP = ' +++$+++ '
    with open(sample_path, 'r', encoding='iso-8859-1', errors='replace') as handle:
        for raw_line in handle:
            if raw_line.strip():
                if ORIGINAL_SEP in raw_line:
                    return ORIGINAL_SEP
                if '\t' in raw_line:
                    return '\t'
                return ORIGINAL_SEP
    return ORIGINAL_SEP

def parse_cornell_raw(lines_path: str, conversations_path: str) -> list[list[str]]:
    lines_sep = _detect_cornell_separator(lines_path)
    convos_sep = _detect_cornell_separator(conversations_path)
    print(f'  Detected Cornell separator: lines={lines_sep!r}, conversations={convos_sep!r}')
    line_text: dict[str, str] = {}
    malformed_lines = 0
    with open(lines_path, 'r', encoding='iso-8859-1', errors='replace') as handle:
        for raw_line in handle:
            parts = raw_line.rstrip('\n').split(lines_sep)
            if len(parts) < 5:
                malformed_lines += 1
                continue
            line_id, text = (parts[0], parts[4])
            line_text[line_id.strip()] = text.strip()
    dialogues: list[list[str]] = []
    malformed_convos = 0
    with open(conversations_path, 'r', encoding='iso-8859-1', errors='replace') as handle:
        for raw_line in handle:
            parts = raw_line.rstrip('\n').split(convos_sep)
            if len(parts) < 4:
                malformed_convos += 1
                continue
            try:
                line_ids = ast.literal_eval(parts[3].strip())
            except (ValueError, SyntaxError):
                malformed_convos += 1
                continue
            utterances = [line_text[line_id] for line_id in line_ids if line_id in line_text]
            utterances = [u for u in utterances if u]
            if len(utterances) >= 2:
                dialogues.append(utterances)
    if malformed_lines or malformed_convos:
        print(f'  (Cornell: skipped {malformed_lines:,} malformed lines and {malformed_convos:,} malformed conversations)')
    return dialogues

def utterances_to_pairs(utterances: list[str]) -> list[str]:
    pairs: list[str] = []
    for index in range(len(utterances) - 1):
        input_utterance = utterances[index].strip()
        output_utterance = utterances[index + 1].strip()
        if input_utterance and output_utterance:
            pairs.append(f'User: {input_utterance}\nAssistant: {output_utterance}')
    return pairs

def corpus_txt_to_pairs(text: str) -> list[str]:
    raw_blocks = re.split('\\n\\s*\\n', text)
    pairs: list[str] = []
    n_malformed = 0
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if not block.startswith('User:'):
            n_malformed += 1
        pairs.append(block)
    if n_malformed:
        print(f"  WARNING: {n_malformed:,} Hinglish blocks don't start with 'User:' -- check corpus.txt formatting.")
    return pairs

def save_metadata_bilingual(output_dir: str, encoding_name: str, token_dtype: str, val_ratio: float, pools: dict[str, dict]) -> None:
    metadata = {'encoding': encoding_name, 'token_dtype': token_dtype, 'val_ratio': val_ratio, 'bilingual': True, 'boundary_aligned': True, 'pools': pools}
    with open(os.path.join(output_dir, 'meta.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

def _tokenise_and_write_pool(pool_name: str, pairs: list[str], output_dir: str, val_ratio: float, encoding_name: str) -> tuple[int, int, str]:
    if not pairs:
        print(f"WARNING: pool '{pool_name}' is empty -- writing empty bin files.")
        empty_tokens = np.array([], dtype=np.uint16)
        empty_bounds = np.array([], dtype=np.uint32)
        save_bin(empty_tokens, os.path.join(output_dir, f'{pool_name}_train.bin'))
        save_bin(empty_tokens, os.path.join(output_dir, f'{pool_name}_val.bin'))
        save_boundaries(empty_bounds, os.path.join(output_dir, f'{pool_name}_train_boundaries.npy'))
        save_boundaries(empty_bounds, os.path.join(output_dir, f'{pool_name}_val_boundaries.npy'))
        return (0, 0, 'uint16')
    train_pairs, val_pairs = split_pairs(pairs, val_ratio)
    train_tokens, train_bounds, token_dtype = encode_pairs(train_pairs, encoding_name)
    val_tokens, val_bounds, _ = encode_pairs(val_pairs, encoding_name)
    save_bin(train_tokens, os.path.join(output_dir, f'{pool_name}_train.bin'))
    save_bin(val_tokens, os.path.join(output_dir, f'{pool_name}_val.bin'))
    save_boundaries(train_bounds, os.path.join(output_dir, f'{pool_name}_train_boundaries.npy'))
    save_boundaries(val_bounds, os.path.join(output_dir, f'{pool_name}_val_boundaries.npy'))
    total_tokens = len(train_tokens) + len(val_tokens)
    total_pairs = len(train_pairs) + len(val_pairs)
    return (total_tokens, total_pairs, token_dtype)

def prepare_bilingual_corpus(output_dir: str, val_ratio: float, encoding_name: str, hinglish_file: Optional[str]=None, dailydialog_file: Optional[str]=None, cornell_lines_file: Optional[str]=None, cornell_conversations_file: Optional[str]=None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pools: dict[str, dict] = {}
    final_token_dtype = 'uint16'
    if hinglish_file:
        print(f'\n=== Hinglish pool: {hinglish_file} ===')
        hinglish_text = clean_text(hinglish_file)
        hinglish_pairs = corpus_txt_to_pairs(hinglish_text)
        print(f'  Split into {len(hinglish_pairs):,} individual pairs.')
        total_tokens, total_pairs, token_dtype = _tokenise_and_write_pool('hinglish', hinglish_pairs, output_dir, val_ratio, encoding_name)
        final_token_dtype = token_dtype
        pools['hinglish'] = {'total_tokens': total_tokens, 'n_pairs': total_pairs, 'sources': [hinglish_file]}
    else:
        print('No --hinglish_file given -- skipping the Hinglish pool.')
    english_pairs: list[str] = []
    english_sources: list[str] = []
    if dailydialog_file:
        print(f'\n=== Parsing DailyDialog: {dailydialog_file} ===')
        dd_dialogues = parse_dailydialog_raw(dailydialog_file)
        print(f'  Parsed {len(dd_dialogues):,} dialogues from DailyDialog.')
        dd_pairs: list[str] = []
        for utterances in dd_dialogues:
            dd_pairs.extend(utterances_to_pairs(utterances))
        print(f'  Decomposed into {len(dd_pairs):,} (User, Assistant) pairs.')
        english_pairs.extend(dd_pairs)
        english_sources.append(f'dailydialog ({len(dd_dialogues):,} dialogues, {len(dd_pairs):,} pairs): {dailydialog_file}')
    if cornell_lines_file and cornell_conversations_file:
        print(f'\n=== Parsing Cornell Movie-Dialogs: {cornell_lines_file} + {cornell_conversations_file} ===')
        cornell_dialogues = parse_cornell_raw(cornell_lines_file, cornell_conversations_file)
        print(f'  Parsed {len(cornell_dialogues):,} conversations from Cornell.')
        cornell_pairs: list[str] = []
        for utterances in cornell_dialogues:
            cornell_pairs.extend(utterances_to_pairs(utterances))
        print(f'  Decomposed into {len(cornell_pairs):,} (User, Assistant) pairs.')
        english_pairs.extend(cornell_pairs)
        english_sources.append(f'cornell ({len(cornell_dialogues):,} conversations, {len(cornell_pairs):,} pairs)')
    elif cornell_lines_file or cornell_conversations_file:
        print('WARNING: Cornell needs BOTH --cornell_lines_file and --cornell_conversations_file -- only one was given, skipping Cornell.')
    if english_pairs:
        print(f'\n=== English pool: {len(english_pairs):,} pairs total ===')
        total_tokens, total_pairs, token_dtype = _tokenise_and_write_pool('english', english_pairs, output_dir, val_ratio, encoding_name)
        final_token_dtype = token_dtype
        pools['english'] = {'total_tokens': total_tokens, 'n_pairs': total_pairs, 'sources': english_sources}
    else:
        print('\nNo English sources given -- skipping the English pool.')
    if not pools:
        raise ValueError('No sources provided at all -- nothing to prepare.')
    save_metadata_bilingual(output_dir, encoding_name, final_token_dtype, val_ratio, pools)
    print('\n=== Summary ===')
    for pool_name, info in pools.items():
        print(f"  {pool_name}: {info['total_tokens']:,} tokens, {info['n_pairs']:,} pairs")
    if 'hinglish' in pools and 'english' in pools:
        total = pools['hinglish']['total_tokens'] + pools['english']['total_tokens']
        if total > 0:
            en_share = pools['english']['total_tokens'] / total * 100
            print(f'  Raw English share of combined tokens: {en_share:.1f}%')
            print("  (This will very likely be well under 50%. That's expected and fine -- train.py's english_ratio batch mixing, not this raw share, is what controls actual training exposure.)")
    print('\nBilingual preprocessing complete.')
    print('\nEOS token verification:')
    _verify_eos_counts(output_dir, pools, encoding_name)

def _verify_eos_counts(output_dir: str, pools: dict, encoding_name: str) -> None:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    eot = encoding.eot_token
    dtype = np.uint16 if encoding.n_vocab <= 65535 else np.uint32
    for pool_name, info in pools.items():
        for split in ('train', 'val'):
            bin_path = os.path.join(output_dir, f'{pool_name}_{split}.bin')
            npy_path = os.path.join(output_dir, f'{pool_name}_{split}_boundaries.npy')
            if not os.path.exists(bin_path):
                continue
            tokens = np.memmap(bin_path, dtype=dtype, mode='r')
            n_eot = int(np.sum(tokens == eot))
            n_bounds = len(np.load(npy_path)) if os.path.exists(npy_path) else 'n/a'
            status = '✓' if isinstance(n_bounds, str) or n_eot == n_bounds else 'MISMATCH'
            print(f'  {pool_name}_{split}: {n_eot:,} EOT tokens, {n_bounds} boundaries  [{status}]')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare Enhinged conversational data for GPT training.')
    parser.add_argument('--mode', choices=['legacy', 'bilingual'], default='legacy', help="'legacy' = original single-pool behaviour (--input_file/--input_dir). 'bilingual' = build separate hinglish/english pools for BilingualDataset.")
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--hinglish_file', type=str, default=None, help='Path to your Hinglish corpus.txt')
    parser.add_argument('--dailydialog_file', type=str, default=None, help='Path to DailyDialog: either dialogues_text.txt (original), or a Kaggle CSV / directory of train.csv+validation.csv+test.csv (auto-detected)')
    parser.add_argument('--cornell_lines_file', type=str, default=None, help='Path to Cornell movie_lines.txt (original) or movie_lines.tsv (Kaggle, tab-separated; auto-detected)')
    parser.add_argument('--cornell_conversations_file', type=str, default=None, help='Path to Cornell movie_conversations.txt (original) or movie_conversations.tsv (Kaggle, tab-separated; auto-detected)')
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--encoding', type=str, default='gpt2')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'bilingual':
        if not any([args.hinglish_file, args.dailydialog_file, args.cornell_lines_file]):
            print('Bilingual mode needs at least one of: --hinglish_file, --dailydialog_file, --cornell_lines_file')
            sys.exit(1)
        prepare_bilingual_corpus(output_dir=args.output_dir, val_ratio=args.val_ratio, encoding_name=args.encoding, hinglish_file=args.hinglish_file, dailydialog_file=args.dailydialog_file, cornell_lines_file=args.cornell_lines_file, cornell_conversations_file=args.cornell_conversations_file)
        sys.exit(0)
    if args.input_file and args.input_dir:
        print('Specify either --input_file or --input_dir, not both.')
        sys.exit(1)
    if args.input_file:
        prepare_single_file(args.input_file, args.output_dir, args.val_ratio, args.encoding)
    elif args.input_dir:
        prepare_directory_corpus(args.input_dir, args.output_dir, args.val_ratio, args.encoding)
    else:
        print('Provide --input_file or --input_dir (or use --mode bilingual).')
        sys.exit(1)