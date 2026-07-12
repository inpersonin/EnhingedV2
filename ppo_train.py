"""ppo_train.py — Phase 3: PPO fine-tuning for Enhinged V2.

Architecture:
  - Policy: HinglishGPT wrapped with a thin value head (PolicyWithValueHead).
    The base model's forward/loss path is NOT modified.
  - Reference model: a frozen copy of the Phase-1 checkpoint for KL penalty.
  - Reward model: the small local RewardModel from Phase 2, frozen.

PPO loop per iteration:
  1. Sample a batch of prompts.
  2. Generate completions with the policy (sampling, for exploration).
  3. Score completions with the frozen reward model.
  4. Compute per-token KL penalty (policy vs reference logits).
  5. Compute advantages via Generalized Advantage Estimation (GAE).
  6. PPO clipped-surrogate update on policy + value head.

Monitoring (logged every eval_interval steps):
  - Mean reward (should rise over training)
  - Mean KL divergence from reference (watch for blow-up)
  - Sample generations for manual spot-checking (reward hacking detection)

Usage:
    python ppo_train.py \
        --policy_ckpt checkpoints/best.pt \
        --reward_ckpt checkpoints/reward_model/best_reward_model.pt \
        --prompt_file pref_data/judge_pairs.jsonl \
        --out_dir checkpoints/rlhf/

    # After PPO, strip the value head:
    python strip_value_head.py --ppo_ckpt checkpoints/rlhf/best_ppo.pt \
                               --out_path checkpoints/rlhf_best.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DEFAULT_TOKENIZER_NAME, GPTConfig
from model import HinglishGPT, generate, load_model_from_checkpoint
from reward_model import RewardModel


# ---------------------------------------------------------------------------
# Policy with value head.
# ---------------------------------------------------------------------------

class PolicyWithValueHead(nn.Module):
    """Thin wrapper: HinglishGPT policy + a scalar value head.

    The base model's forward() and loss path are completely untouched.
    The value head is added as a separate linear layer that reads from the
    final hidden state of the last token.

    Saving/loading: value head parameters are stored under 'value_head.*'
    keys in the state_dict, which are explicitly stripped in strip_value_head.py
    before deployment.
    """

    def __init__(self, base_model: HinglishGPT) -> None:
        super().__init__()
        self.base = base_model
        self.config = base_model.config
        n_embd = base_model.config.n_embd
        self.value_head = nn.Linear(n_embd, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            logits:  (batch, seq, vocab) — from base model
            loss:    scalar when targets provided, else None
            values:  (batch,) scalar value estimates, always returned
        """
        logits, loss, _ = self.base(idx, targets=targets)

        # Value estimate: hidden state at the last token position.
        # We need the full hidden state, not just the logit-projected one.
        # Run a lightweight pass to get it.
        with torch.no_grad() if targets is not None else torch.enable_grad():
            hidden = self._get_last_hidden(idx)
        values = self.value_head(hidden).squeeze(-1)  # (batch,)

        return logits, loss, values

    def _get_last_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """Get the hidden state at the last token position."""
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            seq_len = self.config.block_size

        positions = torch.arange(seq_len, device=idx.device)
        x = self.base.transformer["drop"](
            self.base.transformer["wte"](idx) + self.base.transformer["wpe"](positions)
        )
        for block in self.base.transformer["h"]:
            x, _ = block(x)
        x = self.base.transformer["ln_f"](x)
        return x[:, -1, :]  # (batch, n_embd)

    def get_logprobs(self, idx: torch.Tensor) -> torch.Tensor:
        """Return per-token log-probabilities for the given sequence.

        Returns: (batch, seq_len, vocab_size) log-probs.
        """
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.base.transformer["drop"](
            self.base.transformer["wte"](idx) + self.base.transformer["wpe"](positions)
        )
        for block in self.base.transformer["h"]:
            x, _ = block(x)
        x = self.base.transformer["ln_f"](x)
        logits = self.base.lm_head(x)
        return F.log_softmax(logits, dim=-1)

    def parameters_for_opt(self):
        """Yield parameters that should be optimized (policy + value head)."""
        return list(self.base.parameters()) + list(self.value_head.parameters())


# ---------------------------------------------------------------------------
# Reference model helpers.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_ref_logprobs(
    ref_model: HinglishGPT,
    idx: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities from the frozen reference model."""
    _, seq_len = idx.shape
    if seq_len > ref_model.config.block_size:
        idx = idx[:, -ref_model.config.block_size:]
    positions = torch.arange(idx.size(1), device=idx.device)
    x = ref_model.transformer["drop"](
        ref_model.transformer["wte"](idx) + ref_model.transformer["wpe"](positions)
    )
    for block in ref_model.transformer["h"]:
        x, _ = block(x)
    x = ref_model.transformer["ln_f"](x)
    logits = ref_model.lm_head(x)
    return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# GAE advantage computation.
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    rewards: (T,)
    values:  (T + 1,)  — last entry is bootstrap value
    dones:   (T,)      — 1.0 where episode ended
    Returns advantages (T,) and returns (T,).
    """
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        next_val = values[t + 1] * (1.0 - dones[t])
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update step.
# ---------------------------------------------------------------------------

def ppo_step(
    policy: PolicyWithValueHead,
    optimiser: torch.optim.Optimizer,
    old_logprobs: torch.Tensor,   # (batch, gen_len, vocab)
    gen_tokens: torch.Tensor,     # (batch, gen_len)
    advantages: torch.Tensor,     # (batch,)
    returns: torch.Tensor,        # (batch,)
    values: torch.Tensor,         # (batch,)
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """One PPO clipped-surrogate update."""

    # New log-probs for generated tokens under current policy.
    new_logprobs_full = policy.get_logprobs(gen_tokens)  # (batch, seq, vocab)

    # Gather the log-probs of the actual generated tokens.
    # old_logprobs: (batch, seq, vocab), gen_tokens: (batch, seq)
    gen_tokens_idx = gen_tokens.unsqueeze(-1)  # (batch, seq, 1)
    new_lp = new_logprobs_full.gather(-1, gen_tokens_idx).squeeze(-1)  # (batch, seq)
    old_lp = old_logprobs.gather(-1, gen_tokens_idx).squeeze(-1)       # (batch, seq)

    # Sequence-level: sum log-probs over generated tokens.
    new_lp_sum = new_lp.sum(dim=-1)  # (batch,)
    old_lp_sum = old_lp.sum(dim=-1)  # (batch,)

    # Ratio (importance weight).
    ratio = torch.exp(new_lp_sum - old_lp_sum)

    # Normalize advantages.
    adv = advantages.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # PPO clipped objective.
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss.
    _, _, new_values = policy(gen_tokens)
    value_loss = F.mse_loss(new_values, returns.detach())

    # Entropy bonus (encourages exploration).
    probs = new_logprobs_full.exp()
    entropy = -(probs * new_logprobs_full).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    total_loss = policy_loss + value_coef * value_loss + entropy_loss

    optimiser.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters_for_opt(), max_grad_norm)
    optimiser.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": total_loss.item(),
        "mean_ratio": ratio.mean().item(),
    }


# ---------------------------------------------------------------------------
# KL divergence between policy and reference.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mean_kl(
    policy: PolicyWithValueHead,
    ref_model: HinglishGPT,
    gen_tokens: torch.Tensor,
) -> float:
    """Mean per-token KL divergence: KL(policy || reference)."""
    new_lp = policy.get_logprobs(gen_tokens)           # (batch, seq, vocab)
    ref_lp = _get_ref_logprobs(ref_model, gen_tokens)  # (batch, seq, vocab)
    kl = (new_lp.exp() * (new_lp - ref_lp)).sum(dim=-1).mean().item()
    return kl


# ---------------------------------------------------------------------------
# Prompt loading.
# ---------------------------------------------------------------------------

def _load_prompts(prompt_file: str, extra_prompts: Optional[list[str]] = None) -> list[str]:
    """Load prompts from a JSONL preference file and optional extras."""
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "prompt" in row:
                    prompts.append(row["prompt"])
            except Exception:
                pass
    prompts = list(set(prompts))
    if extra_prompts:
        prompts += extra_prompts
    random.shuffle(prompts)
    print(f"Loaded {len(prompts)} unique prompts for PPO rollouts.")
    return prompts


# ---------------------------------------------------------------------------
# Checkpoint save/load.
# ---------------------------------------------------------------------------

def _save_ppo_checkpoint(
    policy: PolicyWithValueHead,
    optimiser: torch.optim.Optimizer,
    iteration: int,
    best_reward: float,
    out_dir: str,
    tag: str = "latest",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ppo_{tag}.pt")
    torch.save({
        "model_state": policy.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "model_config": dataclasses.asdict(policy.config),
        "iteration": iteration,
        "best_reward": best_reward,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Main PPO training loop.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO fine-tuning for Enhinged V2.")
    parser.add_argument("--policy_ckpt", required=True)
    parser.add_argument("--reward_ckpt", required=True)
    parser.add_argument("--prompt_file", required=True, help="JSONL with 'prompt' keys.")
    parser.add_argument("--out_dir", default="checkpoints/rlhf")
    parser.add_argument("--max_iters", type=int, default=500,
                        help="Total PPO iterations. Start conservative (200-500).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Prompts per PPO iteration.")
    parser.add_argument("--max_new_tokens", type=int, default=60,
                        help="Generated tokens per rollout (shorter = faster).")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Conservative default. Increase only if reward doesn't rise.")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL penalty coefficient. Increase if KL blows up.")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Log metrics every N iterations.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPO training on {device}")

    # Load policy backbone.
    print(f"Loading policy from {args.policy_ckpt}")
    base_model, enc, _ = load_model_from_checkpoint(args.policy_ckpt, device=device)
    policy = PolicyWithValueHead(base_model)
    policy.to(device)
    policy.base.train()

    # Load frozen reference model.
    print(f"Loading reference model (frozen) from {args.policy_ckpt}")
    ref_model, _, _ = load_model_from_checkpoint(args.policy_ckpt, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Load frozen reward model.
    print(f"Loading reward model (frozen) from {args.reward_ckpt}")
    reward_model = RewardModel.load(args.reward_ckpt, device=device)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # Optimiser (policy params + value head params).
    optimiser = torch.optim.Adam(policy.parameters_for_opt(), lr=args.lr)

    # Prompts.
    prompts = _load_prompts(args.prompt_file)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "ppo_log.jsonl")
    samples_path = os.path.join(args.out_dir, "ppo_samples.txt")

    best_reward = float("-inf")
    reward_history = []
    kl_history = []

    print(f"\nStarting PPO training for {args.max_iters} iterations")
    print(f"  lr={args.lr}, kl_coef={args.kl_coef}, batch_size={args.batch_size}")
    print(f"  clip_eps={args.clip_eps}, max_new_tokens={args.max_new_tokens}")
    print(f"{'=' * 60}\n")

    for iteration in range(args.max_iters):
        # ---------------------------------------------------------------
        # Rollout: generate completions for a batch of prompts.
        # ---------------------------------------------------------------
        batch_prompts = random.choices(prompts, k=args.batch_size)
        gen_tokens_list = []
        rewards_list = []
        old_logprobs_list = []
        values_list = []

        policy.eval()
        with torch.no_grad():
            for prompt in batch_prompts:
                prompt_text = f"User: {prompt}\nAssistant:"
                prompt_ids = enc.encode(prompt_text)
                max_prompt = max(1, base_model.config.block_size - args.max_new_tokens - 5)
                prompt_ids = prompt_ids[-max_prompt:]

                idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                torch.manual_seed(iteration * 1337 + hash(prompt) % 10000)
                out = generate(
                    base_model, idx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=50256,
                )

                # Generated tokens only (not prompt).
                gen_part = out  # keep full sequence for logprob computation

                # Get old log-probs.
                old_lp = policy.get_logprobs(gen_part)  # (1, seq, vocab)

                # Value estimate.
                _, _, val = policy(gen_part)  # (1,)

                # Reward from frozen reward model.
                reward = reward_model(gen_part).item()

                gen_tokens_list.append(gen_part)
                old_logprobs_list.append(old_lp)
                rewards_list.append(reward)
                values_list.append(val.item())

        # ---------------------------------------------------------------
        # Compute KL penalties and shaped rewards.
        # ---------------------------------------------------------------
        policy.train()
        kl_list = []
        shaped_rewards = []
        for i, gen_part in enumerate(gen_tokens_list):
            kl = compute_mean_kl(policy, ref_model, gen_part)
            kl_list.append(kl)
            shaped_rewards.append(rewards_list[i] - args.kl_coef * kl)

        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_kl = sum(kl_list) / len(kl_list)
        mean_shaped = sum(shaped_rewards) / len(shaped_rewards)
        reward_history.append(mean_reward)
        kl_history.append(mean_kl)

        # ---------------------------------------------------------------
        # PPO update (each item in batch independently — simplest approach).
        # ---------------------------------------------------------------
        total_stats: dict[str, float] = {}
        for i in range(len(gen_tokens_list)):
            gen = gen_tokens_list[i]
            old_lp = old_logprobs_list[i]
            r = torch.tensor([shaped_rewards[i]], device=device)
            v = torch.tensor([values_list[i]], device=device)
            # Bootstrap value = 0 (episode end).
            bootstrap = torch.tensor([0.0], device=device)
            advantages, returns = compute_gae(r, torch.cat([v, bootstrap]), torch.ones(1, device=device))

            stats = ppo_step(
                policy, optimiser,
                old_logprobs=old_lp,
                gen_tokens=gen,
                advantages=advantages,
                returns=returns,
                values=v,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )
            for k, vv in stats.items():
                total_stats[k] = total_stats.get(k, 0.0) + vv / len(gen_tokens_list)

        # ---------------------------------------------------------------
        # Logging.
        # ---------------------------------------------------------------
        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            log_entry = {
                "iter": iteration + 1,
                "mean_reward": mean_reward,
                "mean_kl": mean_kl,
                "mean_shaped_reward": mean_shaped,
                **total_stats,
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry) + "\n")

            print(
                f"iter {iteration + 1:4d} | "
                f"reward {mean_reward:.4f} | "
                f"kl {mean_kl:.4f} | "
                f"shaped {mean_shaped:.4f} | "
                f"policy_loss {total_stats.get('policy_loss', 0):.4f}"
            )

            # KL blow-up warning.
            if mean_kl > 2.0:
                print(f"  ⚠️  KL divergence {mean_kl:.3f} is HIGH — consider increasing --kl_coef or reducing --lr")

            # Save sample generations for manual quality check.
            policy.eval()
            with torch.no_grad():
                with open(samples_path, "a", encoding="utf-8") as sf:
                    sf.write(f"\n{'='*60}\nIteration {iteration + 1} | reward={mean_reward:.4f}\n")
                    for check_prompt in batch_prompts[:3]:
                        prompt_text = f"User: {check_prompt}\nAssistant:"
                        ids = enc.encode(prompt_text)[-200:]
                        idx = torch.tensor([ids], dtype=torch.long, device=device)
                        out = generate(base_model, idx, max_new_tokens=50, do_sample=False)
                        new_ids = out[0][len(ids):].tolist()
                        reply = enc.decode(new_ids).strip()
                        sf.write(f"  Prompt: {check_prompt!r}\n  Reply:  {reply!r}\n\n")
            policy.train()

            # Save checkpoint.
            _save_ppo_checkpoint(policy, optimiser, iteration, mean_reward, args.out_dir, "latest")
            if mean_reward > best_reward:
                best_reward = mean_reward
                _save_ppo_checkpoint(policy, optimiser, iteration, mean_reward, args.out_dir, "best")
                print(f"  ★  New best reward: {best_reward:.4f} — saved ppo_best.pt")

    print(f"\nPPO training complete.")
    print(f"  Best reward achieved: {best_reward:.4f}")
    print(f"  Final checkpoint: {os.path.join(args.out_dir, 'ppo_latest.pt')}")
    print(f"  Best checkpoint:  {os.path.join(args.out_dir, 'ppo_best.pt')}")
    print(f"\nNext step: strip the value head before deployment:")
    print(f"  python strip_value_head.py --ppo_ckpt {os.path.join(args.out_dir, 'ppo_best.pt')} --out_path checkpoints/rlhf_best.pt")


if __name__ == "__main__":
    main()
