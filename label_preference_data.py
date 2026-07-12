"""label_preference_data.py — Phase 2: offline AI-judge labeling for RLHF.

This script:
1. Loads prompts from the validation set (both languages) plus optional
   hand-written openers.
2. Generates 3–4 completions per prompt from the Phase-1 policy checkpoint
   using sampling at varied temperatures.
3. Calls the judge model (via OpenAI API or any compatible endpoint) with
   a pairwise ranking rubric.
4. Persists raw judge outputs + parsed rankings to pref_data/judge_pairs.jsonl.

The judge is called OFFLINE in a one-time labeling pass only — never inside
the PPO training loop.

Usage:
    # Set GEMINI_API_KEY in environment
    python label_preference_data.py \
        --ckpt_path checkpoints/best.pt \
        --hinglish_val hinglish_val.bin \
        --english_val english_val.bin \
        --out_dir pref_data/ \
        --n_prompts 300 \
        --n_completions 4 \
        --judge_model gemini-2.5-flash

    # Resume from an existing partial run:
    python label_preference_data.py ... --resume

The judge prompt is designed specifically for Enhinged:
  - Pairwise comparison (not raw 1-10 scoring) for reliability.
  - Rubric: relevance, language-appropriateness (natural Hinglish
    code-switching), conversational naturalness, absence of repetition/
    gibberish, and reasonable length (not truncated, not rambling).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from config import DEFAULT_TOKENIZER_NAME, GPTConfig
from model import HinglishGPT, generate, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Rubric prompt for the judge.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a bilingual (Hinglish + English) conversational AI called Enhinged.
Your job is to compare two AI-generated responses to a user prompt and determine which one is BETTER.

When comparing, consider these criteria (in rough order of importance):
1. RELEVANCE & COHERENCE: Does the response actually address the user's message in a sensible, on-topic way?
2. LANGUAGE APPROPRIATENESS: If the prompt is in Hinglish, the response should feel natural in that register. Natural code-switching (mixing Hindi/English words) is fine and expected. Jarring, random language switching or inappropriate formality is NOT.
3. CONVERSATIONAL NATURALNESS: Does it sound like something a real person would say in a casual chat? Avoid robotic or template-like replies.
4. ABSENCE OF REPETITION/ARTIFACTS: No repeated phrases, no gibberish, no sentence fragments mid-thought.
5. LENGTH: Not truncated mid-sentence. Not rambling unnecessarily. Concise is usually better.

Your output must be ONLY a JSON object with these fields:
{
  "winner": "A" or "B",
  "reasoning": "One or two sentences explaining why the winner is better."
}
Do not output anything else outside the JSON."""


def _judge_prompt(user_prompt: str, completion_a: str, completion_b: str) -> str:
    return f"""User prompt: {user_prompt!r}

Response A:
{completion_a}

Response B:
{completion_b}

Which response is better? Reply with only the JSON as instructed."""


# ---------------------------------------------------------------------------
# Prompt extraction from validation binary files.
# ---------------------------------------------------------------------------

def _extract_prompts_from_bin(
    bin_path: str,
    n: int,
    seed: int = 42,
) -> list[str]:
    """Extract up to `n` unique user prompt strings from a .bin file.

    Decodes the memory-mapped token pool and looks for 'User: ... \\nAssistant:'
    patterns, extracting the user-side only.
    """
    import tiktoken
    enc = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)

    try:
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    except Exception as exc:
        print(f"WARNING: could not open {bin_path}: {exc}")
        return []

    # Decode a sample of the pool to find User: ... patterns.
    rng = random.Random(seed)
    prompts: list[str] = []
    seen: set[str] = set()
    n_attempts = 0

    chunk_size = 2048
    total_tokens = len(data)

    for _ in range(min(n * 10, 5000)):
        if len(prompts) >= n:
            break
        start = rng.randint(0, max(0, total_tokens - chunk_size))
        chunk = data[start: start + chunk_size]
        try:
            text = enc.decode(chunk.astype(np.int64).tolist())
        except Exception:
            continue

        # Find "User: ... \nAssistant:" pattern.
        import re
        matches = re.findall(r"User:\s*(.+?)\n\s*Assistant:", text, re.DOTALL)
        for m in matches:
            p = m.strip()
            if p and len(p) > 5 and len(p) < 200 and p not in seen:
                seen.add(p)
                prompts.append(p)
                if len(prompts) >= n:
                    break

    print(f"  Extracted {len(prompts)} prompts from {bin_path}")
    return prompts


# ---------------------------------------------------------------------------
# Completion generation.
# ---------------------------------------------------------------------------

def _generate_completions(
    model: HinglishGPT,
    enc,
    device: torch.device,
    prompt: str,
    n_completions: int = 4,
    max_new_tokens: int = 80,
) -> list[str]:
    """Generate n_completions diverse completions from the policy checkpoint."""
    temperatures = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2]
    completions: list[str] = []
    prompt_text = f"User: {prompt}\nAssistant:"
    prompt_ids = enc.encode(prompt_text)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for i in range(n_completions):
        temp = temperatures[i % len(temperatures)]
        torch.manual_seed(i * 1337 + 42)
        with torch.no_grad():
            out = generate(
                model, idx.clone(),
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=50256,
            )
        new_ids = out[0][len(prompt_ids):].tolist()
        text = enc.decode(new_ids)
        for stop in ("User:", "\nUser:", "Assistant:", "\nAssistant:", "<|endoftext|>"):
            if stop in text:
                text = text[:text.index(stop)]
        text = text.strip()
        if text:
            completions.append(text)

    # Deduplicate.
    seen: set[str] = set()
    unique: list[str] = []
    for c in completions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Judge API call.
# ---------------------------------------------------------------------------

def _call_judge(
    client,
    judge_model: str,
    user_prompt: str,
    completion_a: str,
    completion_b: str,
    max_retries: int = 10,
) -> Optional[dict]:
    """Call the judge model and parse its ranking. Returns None on failure."""
    from google.genai import types
    import time
    
    prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{_judge_prompt(user_prompt, completion_a, completion_b)}"
    
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=judge_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                    response_mime_type="application/json",
                )
            )
            raw = resp.text.strip()
            # Parse JSON from the response.
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if "winner" in parsed and parsed["winner"] in ("A", "B"):
                    return {"winner": parsed["winner"], "reasoning": parsed.get("reasoning", "")}
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
                import re
                match = re.search(r"Retry in (\d+) seconds", err_msg)
                sleep_time = int(match.group(1)) + 2 if match else 65
                print(f"    Rate limit hit (Error: {err_msg.strip()}).\n    Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"    Judge API error (attempt {attempt + 1}): {exc}")
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Main labeling loop.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI judge preference labels for Enhinged V2 RLHF.")
    parser.add_argument("--ckpt_path", required=True, help="Phase-1 policy checkpoint.")
    parser.add_argument("--hinglish_val", default="hinglish_val.bin")
    parser.add_argument("--english_val", default="english_val.bin")
    parser.add_argument("--out_dir", default="pref_data")
    parser.add_argument("--n_prompts", type=int, default=300,
                        help="Total prompts to label (split ~50/50 between languages).")
    parser.add_argument("--n_completions", type=int, default=4,
                        help="Completions per prompt (3-4 recommended).")
    parser.add_argument("--judge_model", default="gemini-2.0-flash",
                        help="Judge model name. Must be a valid Gemini model.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_pairs_per_prompt", type=int, default=1,
                        help="Maximum pairwise comparisons per prompt to save API quota.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip prompts already in the output file.")
    parser.add_argument("--extra_prompts_file", default=None,
                        help="Optional .txt file with one hand-written prompt per line.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "judge_pairs.jsonl")
    raw_path = os.path.join(args.out_dir, "judge_raw.jsonl")

    # Load already-done prompts if resuming.
    done_prompts: set[str] = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_prompts.add(row.get("prompt", ""))
                except Exception:
                    pass
        print(f"Resuming: {len(done_prompts)} prompts already labeled.")

    # Load model.
    print("Loading policy checkpoint...")
    model, enc, device = load_model_from_checkpoint(args.ckpt_path)
    model.eval()

    # Collect prompts.
    half = args.n_prompts // 2
    hinglish_prompts = _extract_prompts_from_bin(args.hinglish_val, half)
    english_prompts = _extract_prompts_from_bin(args.english_val, args.n_prompts - half)

    # Hand-written casual openers (both languages).
    handwritten = [
        "Kya chal raha hai yaar",
        "Hey, what's up?",
        "Bhai ek joke batao",
        "Tell me something interesting",
        "Kuch funny bolo",
        "What do you think about AI?",
        "Aaj ka din kaisa tha",
        "How's the weather today?",
        "Yaar thak gaya hoon aaj",
        "Can you explain machine learning simply?",
        "Bhai kya kar raha hai tu",
        "Give me a quick fun fact",
        "Koi acchi si baat batao",
        "What's your favorite thing to talk about?",
        "Arey yaar kuch bolo na",
        "I'm bored, entertain me",
        "Explain blockchain in 2 sentences",
        "Kya lagta hai future mein AI kaisa hoga",
        "Tell me a short story",
        "Bhai mujhe motivate karo",
    ]
    if args.extra_prompts_file and os.path.exists(args.extra_prompts_file):
        with open(args.extra_prompts_file, "r", encoding="utf-8") as f:
            handwritten += [line.strip() for line in f if line.strip()]

    all_prompts = list(set(hinglish_prompts + english_prompts + handwritten))
    all_prompts = [p for p in all_prompts if p not in done_prompts]
    random.shuffle(all_prompts)
    all_prompts = all_prompts[:args.n_prompts]
    print(f"Total prompts to process: {len(all_prompts)}")

    # Set up judge client and model.
    judge_model = args.judge_model
    try:
        from google import genai
        client = genai.Client()
        
        # Check if the requested model is valid
        client.models.get(model=judge_model)
    except ImportError:
        raise ImportError("google-genai package required: pip install google-genai")
    except Exception:
        print(f"Model {judge_model} not found or unavailable. Querying available models...")
        try:
            available_models = [m.name for m in client.models.list() if "flash" in m.name.lower() and "gemini" in m.name.lower()]
            if available_models:
                preferred = [m for m in available_models if "2.0" in m] or [m for m in available_models if "1.5" in m] or available_models
                judge_model = preferred[0]
                print(f"Auto-selected supported model: {judge_model}")
            else:
                print("Warning: Could not automatically find a Flash model. Using gemini-1.5-flash as fallback.")
                judge_model = "gemini-1.5-flash"
        except Exception as e:
            print(f"Warning: Model auto-discovery failed ({e}). Using gemini-1.5-flash.")
            judge_model = "gemini-1.5-flash"

    # Main labeling loop.
    n_labeled = 0
    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(raw_path, "a", encoding="utf-8") as raw_f:

        for prompt_idx, prompt in enumerate(all_prompts):
            print(f"\n[{prompt_idx + 1}/{len(all_prompts)}] Prompt: {prompt!r}")

            completions = _generate_completions(
                model, enc, device, prompt,
                n_completions=args.n_completions,
                max_new_tokens=args.max_new_tokens,
            )
            if len(completions) < 2:
                print(f"  Skipping: only {len(completions)} unique completions generated.")
                continue

            # Do all pairwise comparisons (up to C(n,2) pairs).
            # For n=4 completions this gives 6 pairs; for n=3 this gives 3.
            from itertools import combinations
            pairs = list(combinations(range(len(completions)), 2))
            random.shuffle(pairs)
            
            if args.max_pairs_per_prompt > 0:
                pairs = pairs[:args.max_pairs_per_prompt]

            for i, j in pairs:
                ca, cb = completions[i], completions[j]
                print(f"  Judging pair ({i}, {j})...")
                result = _call_judge(client, judge_model, prompt, ca, cb)
                if result is None:
                    print("  Judge call failed, skipping pair.")
                    continue

                winner_idx = i if result["winner"] == "A" else j
                loser_idx = j if result["winner"] == "A" else i

                pair_record = {
                    "prompt": prompt,
                    "winner": completions[winner_idx],
                    "loser": completions[loser_idx],
                    "judge_model": judge_model,
                    "reasoning": result["reasoning"],
                }
                raw_record = {
                    "prompt": prompt,
                    "completion_a": ca,
                    "completion_b": cb,
                    "winner": result["winner"],
                    "reasoning": result["reasoning"],
                }

                out_f.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
                out_f.flush()
                raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                raw_f.flush()

                n_labeled += 1
                print(f"  Winner: completion {winner_idx} | {result['reasoning'][:80]}")

            # Small delay to respect API rate limits.
            time.sleep(0.5)

    print(f"\nLabeling complete. {n_labeled} preference pairs written to {out_path}")
    print(f"Raw judge outputs written to {raw_path}")
    print(f"\nNext step: python reward_model.py --base_ckpt {args.ckpt_path} --pref_data {out_path}")


if __name__ == "__main__":
    main()
