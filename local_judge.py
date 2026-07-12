"""
local_judge.py

Drop-in replacement for the OpenRouter/Gemini API judge in your DPO
preference-labeling pipeline. Runs Qwen2.5-7B-Instruct locally in 4-bit
on a single Kaggle T4 (~5GB VRAM), so it comfortably fits alongside your
124M-param policy model on the same GPU, or you can put the policy model
on cuda:1 and the judge on cuda:0 to run both in parallel.

No API keys, no rate limits, no quotas.

Usage in label_preference_data.py:

    from local_judge import judge_pair

    # replace your old call_judge_api(prompt, resp_a, resp_b) with:
    verdict = judge_pair(prompt, resp_a, resp_b)
    if verdict is None:
        continue  # judge disagreed with itself on the swap check, skip this pair
    winner = resp_a if verdict == 'A' else resp_b
"""

import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---- config ------------------------------------------------------------

# 3B fits comfortably alongside the 124M policy model on a single T4 (~2.5GB in 4-bit).
# If you have a GPU with >12GB free VRAM you can bump this back to Qwen2.5-7B-Instruct.
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def load_judge():
    """Lazy-loads the judge model once. Call this explicitly at the start
    of your script if you want to fail fast on OOM rather than on the
    first judged pair."""
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading local judge: {JUDGE_MODEL} on {JUDGE_DEVICE} ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 is Turing -> use fp16, not bf16
    )
    _tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        quantization_config=quant_config,
        device_map={"": JUDGE_DEVICE},
    )
    _model.eval()
    print("Judge model loaded.")


_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating two chatbot responses to the same "
    "conversation prompt. Judge only on relevance, coherence, and helpfulness. "
    "Reply with exactly one letter: A or B. Do not explain your reasoning."
)


def _build_messages(prompt, resp_a, resp_b):
    user_msg = (
        f"Conversation prompt:\n{prompt}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        "Which response is better? Reply with exactly one letter: A or B."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _parse_verdict(text):
    text = text.strip().upper()
    match = re.search(r"\b(A|B)\b", text)
    return match.group(1) if match else None


@torch.no_grad()
def _ask(prompt, a, b):
    messages = _build_messages(prompt, a, b)
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # force a dict of {input_ids, attention_mask}, not a bare tensor
    ).to(JUDGE_DEVICE)

    output = _model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=_tokenizer.eos_token_id,
    )
    gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = _tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return _parse_verdict(text)


def judge_pair(prompt, resp_a, resp_b, swap_check=True):
    """
    Returns 'A', 'B', or None (only when swap_check=True and the judge
    contradicts itself, meaning the pair is too ambiguous to trust).

    swap_check=True asks the judge twice, in both orderings, to cancel out
    position bias (a known LLM-judge artifact where it slightly favors
    whichever response it saw first). This roughly doubles judge calls
    but meaningfully improves preference-label quality -- worth it since
    you only have ~300 prompts. Set swap_check=False if you need it faster.
    """
    load_judge()

    verdict_1 = _ask(prompt, resp_a, resp_b)  # A=resp_a, B=resp_b

    if not swap_check:
        if verdict_1 in ("A", "B"):
            return verdict_1
        return random.choice(["A", "B"])  # judge gave garbage, coin-flip

    verdict_2 = _ask(prompt, resp_b, resp_a)  # A=resp_b, B=resp_a (swapped)
    verdict_2_mapped = {"A": "B", "B": "A"}.get(verdict_2)

    if verdict_1 is not None and verdict_1 == verdict_2_mapped:
        return verdict_1  # consistent across both orderings -> trust it

    return None  # disagreement or parse failure -> skip this pair


if __name__ == "__main__":
    # quick smoke test
    load_judge()
    result = judge_pair(
        prompt="Kaise ho?",
        resp_a="Main theek hoon, tum batao kaisa chal raha hai sab?",
        resp_b="Purple elephant airplane 42 the.",
    )
    print("Verdict:", result)
