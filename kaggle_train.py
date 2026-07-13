import subprocess
subprocess.run(['pip', 'install', '-q', 'tiktoken', 'transformers'], check=True)
print('Dependencies installed.')
import os
CORPUS_TXT = '/kaggle/input/datasets/uehewbrv/corpus/corpus.txt'
BEST_PT = '/kaggle/input/models/uehewbrv/enhinged/transformers/default/1/best.pt'
DAILYDIALOG_DIR = '/kaggle/input/daily-dialog'
CORNELL_LINES = '/kaggle/input/cornell-movie-dialog/movie_lines.txt'
CORNELL_CONVOS = '/kaggle/input/cornell-movie-dialog/movie_conversations.txt'
DATA_OUT = '/kaggle/working/data'
CKPT_OUT = '/kaggle/working/checkpoints'
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(CKPT_OUT, exist_ok=True)
for label, path in [('corpus.txt', CORPUS_TXT), ('best.pt', BEST_PT)]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1000000.0
        print(f'  ✓ {label}: {path} ({size:.1f} MB)')
    else:
        print(f'  ✗ MISSING: {label}: {path}  <-- fix this path!')
for label, path in [('DailyDialog', DAILYDIALOG_DIR), ('Cornell lines', CORNELL_LINES), ('Cornell convos', CORNELL_CONVOS)]:
    if os.path.exists(path):
        print(f'  ✓ {label}: {path}')
    else:
        print(f'  ~ {label}: {path} (optional, skipped if missing)')
cornell_args = []
if os.path.exists(CORNELL_LINES) and os.path.exists(CORNELL_CONVOS):
    cornell_args = ['--cornell_lines_file', CORNELL_LINES, '--cornell_conversations_file', CORNELL_CONVOS]
dailydialog_args = []
if os.path.exists(DAILYDIALOG_DIR):
    dailydialog_args = ['--dailydialog_file', DAILYDIALOG_DIR]
cmd = ['python', 'prepare_data.py', '--mode', 'bilingual', '--hinglish_file', CORPUS_TXT, *dailydialog_args, *cornell_args, '--output_dir', DATA_OUT, '--val_ratio', '0.1', '--encoding', 'gpt2']
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
print('\nData preparation complete!')
import numpy as np
print('EOT token counts in new .bin files (should be >> 1 now):')
for fname in sorted(os.listdir(DATA_OUT)):
    if fname.endswith('.bin'):
        path = os.path.join(DATA_OUT, fname)
        tokens = np.memmap(path, dtype=np.uint16, mode='r')
        n_eot = int(np.sum(tokens == 50256))
        size_mb = os.path.getsize(path) / 1000000.0
        print(f'  {fname}: {len(tokens):,} tokens, {n_eot:,} EOT tokens ({size_mb:.1f} MB)')
print('\nBoundary files:')
for fname in sorted(os.listdir(DATA_OUT)):
    if fname.endswith('_boundaries.npy'):
        path = os.path.join(DATA_OUT, fname)
        bounds = np.load(path)
        print(f'  {fname}: {len(bounds):,} pairs')
CKPT_PATH = '/kaggle/input/models/uehewbrv/enhinged/transformers/default/1/best.pt'
train_cmd = ['python', 'train.py', '--mode', 'train', '--data_dir', DATA_OUT, '--ckpt_path', CKPT_PATH, '--out_dir', CKPT_OUT, '--max_iters', '16000', '--lr_decay_iters', '16000', '--warmup_iters', '0', '--learning_rate', '2e-5', '--english_ratio', '0.4', '--batch_size', '8', '--grad_accum_steps', '4', '--eval_interval', '200', '--eval_iters', '50', '--dtype', 'bfloat16', '--seed', '42']
print('Running:', ' '.join(train_cmd))
subprocess.run(train_cmd, check=True)
import shutil
best_src = os.path.join(CKPT_OUT, 'best.pt')
latest_src = os.path.join(CKPT_OUT, 'latest.pt')
if os.path.exists(best_src):
    shutil.copy(best_src, '/kaggle/working/best.pt')
    size = os.path.getsize('/kaggle/working/best.pt') / 1000000.0
    print(f'Saved best.pt to /kaggle/working/best.pt ({size:.1f} MB)')
else:
    print('WARNING: best.pt not found in checkpoint dir!')
if os.path.exists(latest_src):
    shutil.copy(latest_src, '/kaggle/working/latest.pt')
    print(f'Saved latest.pt to /kaggle/working/latest.pt')
import sys
sys.path.insert(0, '.')
from inference import load_model, generate_response
load_model('/kaggle/working/best.pt')
test_prompts = ['kaisa hai bhai', 'kya kar rahe ho aaj', "Hey, how's it going", 'what are you up to', 'monsoon mein kaafi baarish ho rahi hai']
print('\n=== Inference Test ===')
for prompt in test_prompts:
    response = generate_response(prompt, max_new_tokens=80, temperature=0.8, top_k=50, top_p=0.95)
    print(f'\nUser: {prompt}')
    print(f'Assistant: {response}')