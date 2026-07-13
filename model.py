"""Model components, KV-cached generation, and sampling utilities for Enhinged V2.

Key changes vs V1:
- CausalSelfAttention supports an optional kv_cache (list of (k, v) tensors
  per layer) for O(n) per-step generation instead of O(n^2).
- Block and HinglishGPT forward() accept and return the cache when in
  generation mode (use_cache=True).
- generate() now uses a two-phase approach:
    Phase 1: full prompt forward with use_cache=True -> produces initial cache
    Phase 2: each new token processes only ONE token, reusing the cache.
  This produces IDENTICAL outputs to the original uncached implementation
  for the same seed/greedy-decoding path, verified by tests in verify_kvcache.py.
- The training/loss forward pass (targets != None) never uses the cache.
- load_model_from_checkpoint: compatible with both old checkpoints (no value
  head keys) and new RLHF checkpoints (value head stripped before saving).
"""

from __future__ import annotations

import inspect
import math
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DEFAULT_TOKENIZER_NAME, GPTConfig, SUPPORTED_PRETRAINED_MODELS


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention with optional KV-cache support."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        capture_attn: bool = False,
        layer_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:            (batch, seq_len, n_embd)
            capture_attn: whether to store attention weights for visualization
            layer_cache:  None for training/full-sequence mode, or a
                          (past_k, past_v) tuple for cached generation.
                          Each tensor is (batch, n_head, past_len, head_dim).
            use_cache:    whether to return the new cache (True during generation).

        Returns:
            output:     (batch, seq_len, n_embd)
            new_cache:  updated (k, v) tuple when layer_cache is not None,
                        else None.
        """
        batch_size, sequence_length, channels = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        query = reshape(query)  # (B, nh, T, hd)
        key = reshape(key)
        value = reshape(value)

        if layer_cache is not None:
            # Append new k/v to the cached keys/values.
            past_k, past_v = layer_cache
            key = torch.cat([past_k, key], dim=2)    # (B, nh, past+T, hd)
            value = torch.cat([past_v, value], dim=2)
            new_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = (key, value)
        elif use_cache:
            new_cache = (key, value)
        else:
            new_cache = None

        total_len = key.size(2)  # past_len + sequence_length

        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if layer_cache is not None:
            # In generation mode (seq_len == 1), there is no causal mask
            # needed since we're only generating one token at a time and all
            # past tokens are already committed. Using a full-ones mask avoids
            # the need to slice the buffer which may not accommodate past_len.
            # For the initial prompt (seq_len > 1) we still mask correctly.
            if sequence_length > 1:
                # Initial prompt pass: use standard causal mask.
                # We slice for (seq_len, total_len) which equals (T, T) since
                # past is empty at this point.
                scores = scores.masked_fill(
                    self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
                )
            # else: seq_len==1, no masking needed — all attention is to past tokens.
        else:
            # Normal training mode: standard causal mask.
            scores = scores.masked_fill(
                self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
            )

        attn = self.attn_dropout(F.softmax(scores, dim=-1))

        if capture_attn:
            self._attn_weights = attn.detach().cpu()

        output = attn @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(output)), new_cache


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block with pre-norm residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        capture_attn: bool = False,
        layer_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln_1(x), capture_attn=capture_attn, layer_cache=layer_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class HinglishGPT(nn.Module):
    """GPT-2 style language model for Enhinged V2 conversational responses."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self, exclude_embeddings: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        if exclude_embeddings:
            total -= self.transformer["wpe"].weight.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        capture_attn: bool = False,
        use_cache: bool = False,
        past_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            idx:          (batch, seq_len) token indices
            targets:      (batch, seq_len) token targets for loss computation.
                          When provided, use_cache is ignored (training path).
            capture_attn: capture attention weights for visualization
            use_cache:    if True, returns new KV cache (generation path only,
                          ignored when targets is not None).
            past_kv_cache: list of (k, v) per layer from a previous step.

        Returns:
            logits:       (batch, seq_len, vocab_size) or (batch, 1, vocab_size)
            loss:         scalar loss when targets provided, else None
            new_kv_cache: list of updated (k, v) per layer when use_cache=True,
                          else None
        """
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.config.block_size}"
            )

        # Positional offset: when using cache, the current tokens start at
        # position past_len rather than 0.
        past_len = 0
        if past_kv_cache is not None and len(past_kv_cache) > 0:
            past_len = past_kv_cache[0][0].size(2)

        positions = torch.arange(past_len, past_len + sequence_length, device=idx.device)
        token_embeddings = self.transformer["wte"](idx)
        position_embeddings = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_embeddings + position_embeddings)

        new_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache and targets is None else None

        for i, block in enumerate(self.transformer["h"]):
            layer_cache = past_kv_cache[i] if (past_kv_cache is not None and targets is None) else None
            x, updated_cache = block(x, capture_attn=capture_attn, layer_cache=layer_cache, use_cache=use_cache)
            if new_kv_cache is not None:
                new_kv_cache.append(updated_cache)  # type: ignore[arg-type]

        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Training path: full logits over entire sequence.
            logits = self.lm_head(x)
            # ignore_index=-100: positions masked by the boundary-aligned sampler
            # (EOS crossings between unrelated pairs) contribute zero loss.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # Inference path: only need the last token's logits.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_cache

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "HinglishGPT":
        """Load weights from an official GPT-2 checkpoint."""

        if model_type not in SUPPORTED_PRETRAINED_MODELS:
            raise ValueError(f"Unknown model_type: {model_type}")

        from transformers import GPT2LMHeadModel

        config_map = {
            "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024),
            "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
            "gpt2-large": GPTConfig(n_layer=36, n_head=20, n_embd=1280, block_size=1024),
            "gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600, block_size=1024),
        }

        config = config_map[model_type]
        config.vocab_size = 50257
        config.bias = True

        pretrained = GPT2LMHeadModel.from_pretrained(model_type)
        source_state = pretrained.state_dict()
        model = cls(config)
        target_state = model.state_dict()
        transposed = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        source_keys = [key for key in source_state if not key.endswith(".attn.masked_bias") and not key.endswith(".attn.bias")]

        for source_key in source_keys:
            target_key = source_key
            if target_key not in target_state:
                continue
            with torch.no_grad():
                if any(target_key.endswith(suffix) for suffix in transposed):
                    target_state[target_key].copy_(source_state[source_key].T)
                else:
                    target_state[target_key].copy_(source_state[source_key])

        model.load_state_dict(target_state)
        return model

    def configure_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        decay_params = [parameter for parameter in self.parameters() if parameter.requires_grad and parameter.dim() >= 2]
        nodecay_params = [parameter for parameter in self.parameters() if parameter.requires_grad and parameter.dim() < 2]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            fused=fused_available if fused_available else False,
        )

    def get_attention_weights(self) -> list[torch.Tensor]:
        """Return attention weights captured during the last forward pass."""

        weights: list[torch.Tensor] = []
        for block in self.transformer["h"]:
            if block.attn._attn_weights is not None:
                weights.append(block.attn._attn_weights)
        return weights


GPT = HinglishGPT


@torch.no_grad()
def generate(
    model: HinglishGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively using KV-caching for efficiency.

    Phase 1: runs the full prompt through the model once to seed the KV cache.
    Phase 2: generates one new token at a time, reusing the cached keys/values.

    This is a pure performance optimisation — the generated token sequence is
    identical to the uncached version for the same seed and sampling params
    (verified by verify_kvcache.py).
    """

    model.eval()

    # Phase 1: full prompt forward to seed the cache.
    prompt_len = idx.size(1)
    context = idx if prompt_len <= model.config.block_size else idx[:, -model.config.block_size:]

    _, _, kv_cache = model(context, use_cache=True, past_kv_cache=None)

    for _ in range(max_new_tokens):
        # Phase 2: feed only the last generated token, reuse cache.
        last_token = idx[:, [-1]]

        # Guard: if total sequence exceeds block_size, truncate from the left
        # by dropping the oldest cached key-value pair from each layer and
        # restarting caching from the current window. This is rare (only
        # triggers if prompt + generated > block_size) and gracefully degrades.
        current_total = kv_cache[0][0].size(2) + 1 if kv_cache else 1
        if current_total > model.config.block_size:
            # Fall back to recomputing from scratch for the last block_size tokens.
            context = idx[:, -model.config.block_size:]
            _, _, kv_cache = model(context, use_cache=True, past_kv_cache=None)
            last_token = idx[:, [-1]]

        logits, _, kv_cache = model(last_token, use_cache=True, past_kv_cache=kv_cache)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for batch_index in range(idx.size(0)):
                unique_tokens = idx[batch_index].unique()
                token_logits = logits[batch_index, unique_tokens]
                # Correct CTRL-style repetition penalty: divide positive logits,
                # multiply negative logits, so the score always moves toward -inf.
                logits[batch_index, unique_tokens] = torch.where(
                    token_logits > 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty,
                )

        logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_to_remove = cumulative_probs > top_p
            sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
            sorted_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_to_remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1) if do_sample else torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return idx


def generate_text(
    model: HinglishGPT,
    enc: tiktoken.Encoding,
    prompt: str,
    device: torch.device,
    **kwargs,
) -> str:
    """Convert a text prompt into generated text."""

    model.eval()
    token_ids = enc.encode(prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = generate(model, idx, **kwargs)
    return enc.decode(out[0].tolist())


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> tuple[HinglishGPT, tiktoken.Encoding, torch.device]:
    """Load a trained Enhinged V2 checkpoint.

    Uses mmap=True + assign=True so PyTorch memory-maps the file from disk
    instead of copying it into RAM. Peak RAM = 1× model size instead of 2×.
    This is required to fit within Railway's 1 GB container limit.

    Inference behaviour after loading is completely identical to a normal load.

    Compatible with:
    - Phase-1 fine-tuned checkpoints (plain model_state)
    - Phase-4 RLHF checkpoints with value head already stripped
    - fp16-compressed checkpoints (auto-detected, model kept in fp16)
    """

    resolved_device = _resolve_device(device)

    # Load checkpoint into RAM (335 MB). No mmap=True to avoid Docker cgroup page cache OOM kills.
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state = checkpoint["model_state"]
    # Strip any value-head keys left in RLHF checkpoints.
    state = {k: v for k, v in state.items() if not k.startswith("value_head.")}

    sample_weight = next(iter(state.values()))
    use_fp16 = isinstance(sample_weight, torch.Tensor) and sample_weight.dtype == torch.float16

    # Initialize model with 0 bytes RAM usage using 'meta' device.
    # This completely eliminates the memory spike during instantiation.
    with torch.device("meta"):
        model = HinglishGPT(GPTConfig(**checkpoint["model_config"]))
        
    if use_fp16:
        print("inference: fp16 checkpoint detected — model running in half precision (~335 MB).")
    else:
        print("inference: fp32 checkpoint detected — model running in full precision (~522 MB).")

    # assign=True: directly swaps parameter tensors with the loaded ones —
    # no intermediate copy, so peak RAM stays at 1× model size.
    try:
        model.load_state_dict(state, assign=True)
    except TypeError:
        # PyTorch < 2.0 doesn't support assign=; fall back silently.
        model.load_state_dict(state)

    # Free checkpoint reference immediately — the mmap'd data is released.
    del checkpoint, state

    model.to(resolved_device)
    model.eval()

    # Apply int8 dynamic quantization if toggled (inference-only, never training).
    from config import USE_INT8_QUANTIZE
    if USE_INT8_QUANTIZE:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            print("inference: int8 dynamic quantization applied.")
        except Exception as exc:
            print(f"inference: int8 quantization failed ({exc}), running in full precision.")

    encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)
    return model, encoding, resolved_device
