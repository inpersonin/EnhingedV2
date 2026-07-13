---
title: Enhinged V2
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_port: 7860
pinned: false
---

# Enhinged V2

Enhinged V2 is a GPT-2-small (124M parameter) conversational language model fine-tuned bilingually on Hinglish and English conversational data, with a reinforcement learning from AI feedback (RLHF) pass to further improve response quality. The repository includes the FastAPI backend, all training and RLHF scripts, and the Next.js frontend scaffold.

## What's New in V2

- **Fixed data pipeline bug** that caused disjointed, unrelated responses by ensuring training windows always start at conversation pair boundaries (no more cross-pair gradient leakage).
- **Added English fluency** alongside Hinglish — the bilingual training pipeline now mixes both language pools with a configurable ratio.
- **RLHF response-quality tuning** — a reinforcement learning pass using AI-judged preference data further improves conversational naturalness, relevance, and absence of repetition.
- **Faster generation** — KV-caching in the attention layer reduces generation from quadratic to linear cost per step, delivering a significant speedup on CPU hosting.

## Model Architecture

V2 is based on GPT-2-small fine-tuned on bilingual Hinglish+English conversational data:

- `n_layer = 12`, `n_head = 12`, `n_embd = 768`
- `block_size = 1024`, `vocab_size = 50257`
- Total parameters: ~124M
- Tokenizer: GPT-2 BPE via tiktoken

## RLHF Pipeline

The RLHF pipeline follows established techniques (RLAIF — reinforcement learning from AI feedback):

1. **Phase 1**: Fine-tune GPT-2-small on bilingual data with corrected pipeline.
2. **Phase 2**: Generate completions offline → AI judge labels pairwise preferences → train small local reward model on Bradley-Terry loss.
3. **Phase 3**: PPO fine-tuning with the local reward model scoring rollouts + KL penalty against the Phase-1 reference model.
4. **Phase 4**: Strip the value head from the PPO checkpoint before deployment.

## Project Structure

```text
.
├── api.py                  # FastAPI inference service
├── config.py               # Shared configuration
├── inference.py            # Runtime model loading and generation
├── model.py                # HinglishGPT architecture + KV-cached generate()
├── train.py                # Supervised fine-tuning loop
├── utils.py                # Bilingual dataset + boundary-aligned sampling
├── prepare_data.py         # Data preprocessing
├── reward_model.py         # RLHF reward model (Phase 2)
├── label_preference_data.py # AI-judge labeling (Phase 2)
├── ppo_train.py            # PPO training loop (Phase 3)
├── strip_value_head.py     # Value head stripping for deployment (Phase 4)
├── benchmark_latency.py    # KV-cache speedup benchmarking (Phase 1)
├── verify_kvcache.py       # Verify cached/uncached output identity (Phase 1)
├── upload_model.py         # Upload checkpoint to inpersonin/HinGPTv2
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Inference

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Deployment

- **Backend**: Hugging Face Space `inpersonin/EnhingedV2` (Docker SDK, port 7860)
- **Frontend**: GitHub Pages from `inpersonin/EnhingedV2` repo
- **Model weights**: Stored in `inpersonin/HinGPTv2` HF model repo (separate from the Space to avoid the 1 GB storage limit)

Environment variables:
- `ENHINGED_CKPT_PATH`: path to the checkpoint (defaults to `checkpoints/best.pt`, auto-downloads from HF Hub if not found)
- `ENHINGED_CORS_ORIGINS`: comma-separated allowed origins for CORS (defaults to `*`)
- `ENHINGED_QUANTIZE`: set to `1` to enable int8 dynamic quantization at inference time (toggleable, default off)

## Notes

- Do NOT push to or deploy over `inpersonin/Enhinged` (GitHub or HF Space) — those are the V1 resources. All V2 work goes to `inpersonin/EnhingedV2` and `inpersonin/HinGPTv2`.
- The AI judge labeling (`label_preference_data.py`) is an offline one-time pass. Never call the judge inside the PPO loop.
- Verify KV-cache correctness with `verify_kvcache.py` before trusting generation output.
- Benchmark actual latency with `benchmark_latency.py` and use those real numbers on the Metrics page.
