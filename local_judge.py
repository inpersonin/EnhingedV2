import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
JUDGE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'
JUDGE_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
_tokenizer = None
_model = None

def load_judge():
    global _tokenizer, _model
    if _model is not None:
        return
    print(f'Loading local judge: {JUDGE_MODEL} on {JUDGE_DEVICE} ...')
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
    _tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, quantization_config=quant_config, device_map={'': JUDGE_DEVICE})
    _model.eval()
    print('Judge model loaded.')
_SYSTEM_PROMPT = 'You are an impartial judge evaluating two chatbot responses to the same conversation prompt. Judge only on relevance, coherence, and helpfulness. Reply with exactly one letter: A or B. Do not explain your reasoning.'

def _build_messages(prompt, resp_a, resp_b):
    user_msg = f'Conversation prompt:\n{prompt}\n\nResponse A:\n{resp_a}\n\nResponse B:\n{resp_b}\n\nWhich response is better? Reply with exactly one letter: A or B.'
    return [{'role': 'system', 'content': _SYSTEM_PROMPT}, {'role': 'user', 'content': user_msg}]

def _parse_verdict(text):
    text = text.strip().upper()
    match = re.search('\\b(A|B)\\b', text)
    return match.group(1) if match else None

@torch.no_grad()
def _ask(prompt, a, b):
    messages = _build_messages(prompt, a, b)
    inputs = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(JUDGE_DEVICE)
    output = _model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=_tokenizer.eos_token_id)
    gen_tokens = output[0][inputs['input_ids'].shape[-1]:]
    text = _tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return _parse_verdict(text)

def judge_pair(prompt, resp_a, resp_b, swap_check=True):
    load_judge()
    verdict_1 = _ask(prompt, resp_a, resp_b)
    if not swap_check:
        if verdict_1 in ('A', 'B'):
            return verdict_1
        return random.choice(['A', 'B'])
    verdict_2 = _ask(prompt, resp_b, resp_a)
    verdict_2_mapped = {'A': 'B', 'B': 'A'}.get(verdict_2)
    if verdict_1 is not None and verdict_1 == verdict_2_mapped:
        return verdict_1
    return None
if __name__ == '__main__':
    load_judge()
    result = judge_pair(prompt='Kaise ho?', resp_a='Main theek hoon, tum batao kaisa chal raha hai sab?', resp_b='Purple elephant airplane 42 the.')
    print('Verdict:', result)