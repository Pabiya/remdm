# -*- coding: utf-8 -*-
"""
refine_ent3_llada.py

Refine-Ent-3 sampler (in-step 2-forward refine with entropy overwrite)
adapted for diffusion-style language models (e.g., LLaDA-8B).

This file is **model-agnostic** via a pluggable `step_fn`:
    step_fn(x_tokens: LongTensor[B, L], t_step: int) -> Namespace(
        p_x0: FloatTensor[B, L, V],   # token probs for x0 at current step (after top-p if you want)
        alpha_t: float,               # alpha(t)
        alpha_s: float,               # alpha(s=t-dt)
        prev_mask: BoolTensor[B, L],  # mask flags BEFORE applying the step (for conf update)
    )

You must implement a factory that returns such a step_fn for your LLaDA HF model.

CLI:
    python refine_ent3_llada.py \
        --model_id GSAI-ML/LLaDA-8B-Instruct \
        --mode refine_ent3 \
        --steps 1024 \
        --nucleus_p 0.9 \
        --refine_every 1 \
        --prompts_file prompts.txt \
        --out completions.jsonl

"""
from __future__ import annotations

import math
import json
import argparse
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None


# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_p_probs(probs: torch.Tensor, p: float, eps: float = 1e-12) -> torch.Tensor:
    """Apply nucleus (top-p) in the last dimension and renormalize. probs shape: (..., V)"""
    if p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumsum <= p)
    # always keep first token
    keep[..., 0] = True
    kept = sorted_probs * keep
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, kept)
    return out


def sample_categorical(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample indices from categorical distribution along last dim."""
    B, L, V = probs.shape
    probs = probs.view(-1, V)
    idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).view(B, L)
    return idx


def entropy_from_probs(probs: torch.Tensor, mask_index: int, remove_mask_prob: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """Compute token-level Shannon entropy H over vocab (optionally excluding [MASK])."""
    P = probs.clamp_min(eps)
    if remove_mask_prob:
        Pw = P.clone()
        Pw[..., mask_index] = 0.0
        Z = Pw.sum(dim=-1, keepdim=True).clamp_min(eps)
        Q = Pw / Z
    else:
        Z = P.sum(dim=-1, keepdim=True).clamp_min(eps)
        Q = P / Z
    H = -(Q * (Q + eps).log()).sum(dim=-1)  # (...,)
    return H


def build_q_xs2_from_p_x0(p_x0: torch.Tensor, alpha_t: float, alpha_s: float, mask_index: int, eps: float = 1e-12) -> torch.Tensor:
    """
    MDLM posterior for [MASK]->token transition used in our step-1 default unmasking.
    q_xs2(token)   = p_x0(token) * ((alpha_s - alpha_t)/(1 - alpha_t))
    q_xs2([MASK])  = (1 - alpha_s)/(1 - alpha_t)
    """
    denom = max(eps, (1.0 - alpha_t))
    q = p_x0 * ((alpha_s - alpha_t) / denom)
    q[..., mask_index] = (1.0 - alpha_s) / denom
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q


def left_pad_and_mask(prompts_ids: List[List[int]], max_len: int, mask_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: LongTensor[B, L] filled with mask_id except the prompt span
        prompt_mask: BoolTensor[B, L] True where prompt tokens are placed (frozen)
    """
    B = len(prompts_ids)
    L = max_len
    x = torch.full((B, L), fill_value=mask_id, dtype=torch.long, device=device)
    prompt_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i, ids in enumerate(prompts_ids):
        ids = ids[:L]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        prompt_mask[i, :len(ids)] = True
    return x, prompt_mask


# ------------------------------
# Refine-Ent-3 Sampler
# ------------------------------

class RefineEnt3Sampler:
    """
    In-step 2-forward refine with entropy overwrite, periodic execution.

    Args:
        step_fn:  callable (x_tokens, t_idx) -> SimpleNamespace(p_x0, alpha_t, alpha_s, prev_mask)
        tokenizer: AutoTokenizer
        nucleus_p: float for nucleus on p_x0 and p_x0_2
        refine_every: int, k=1 means refine every step
        remove_mask_prob_for_entropy: bool
        mask_token_id: int (if None, will use tokenizer.mask_token_id or last vocab id)
    """
    def __init__(
        self,
        step_fn: Callable[[torch.Tensor, int], SimpleNamespace],
        tokenizer,
        nucleus_p: float = 0.9,
        refine_every: int = 1,
        remove_mask_prob_for_entropy: bool = True,
        mask_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.step_fn = step_fn
        self.tok = tokenizer
        self.nucleus_p = float(nucleus_p)
        self.refine_every = int(refine_every)
        self.remove_mask_prob = bool(remove_mask_prob_for_entropy)
        self.device = device
        # mask id
        if mask_token_id is not None:
            self.mask_id = int(mask_token_id)
        else:
            mid = getattr(self.tok, "mask_token_id", None)
            if mid is None:
                # fallback: reserve last id as [MASK]-like
                mid = self.tok.vocab_size - 1
            self.mask_id = int(mid)

        self.conf = None  # (B, L) entropy cache at unmask time
        self.step_idx = 0

    @torch.no_grad()
    def sample(
        self,
        prompts: List[str],
        max_len: int = 1024,
        steps: int = 1024,
        seed: int = 1,
    ) -> List[str]:
        set_seed(seed)
        device = self.device or (self.tok._device if hasattr(self.tok, "_device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # tokenize
        prompts_ids = self.tok(prompts, add_special_tokens=False)["input_ids"]
        x, prompt_mask = left_pad_and_mask(prompts_ids, max_len=max_len, mask_id=self.mask_id, device=device)
        B, L = x.shape
        self.conf = torch.zeros((B, L), device=device)
        self.step_idx = 0

        # main diffusion loop
        for t in range(steps):
            # 1) first forward
            out = self.step_fn(x, t)
            # nucleus on p_x0
            p_x0 = top_p_probs(out.p_x0, self.nucleus_p)
            alpha_t = float(out.alpha_t)
            alpha_s = float(out.alpha_s)
            prev_mask = out.prev_mask  # Bool[B, L]

            # 1-a) default unmask for current [MASK] positions via q_xs2
            masked_flag = (x == self.mask_id)
            if masked_flag.any():
                q_xs2 = build_q_xs2_from_p_x0(p_x0, alpha_t, alpha_s, self.mask_id)
                new_tok = sample_categorical(q_xs2)
                # respect prompt prefix: do not overwrite prompt tokens
                write_mask = masked_flag & (~prompt_mask)
                x[write_mask] = new_tok[write_mask]

            # 2) refine (periodic)
            if (self.step_idx % max(1, self.refine_every)) == 0:
                # selection like remdm-ent-2: eta=softmax(conf), masked excluded
                eta = torch.softmax(self.conf, dim=-1)
                eta = eta.masked_fill(masked_flag, 0.0)
                sigma_max = min(1.0, (1.0 - alpha_s) / max(1e-12, alpha_t))
                sigma = (eta * sigma_max).clamp_(0.0, 1.0)
                unmasked_flag = ~masked_flag
                R = (torch.rand_like(sigma) < sigma) & unmasked_flag & (~prompt_mask)
                if R.any():
                    # 2nd forward on x_tmp
                    x_tmp = x.clone()
                    x_tmp[R] = self.mask_id
                    out2 = self.step_fn(x_tmp, t)
                    p_x0_2 = top_p_probs(out2.p_x0, self.nucleus_p)
                    # refill only R
                    refill_tok = sample_categorical(p_x0_2)
                    x[R] = refill_tok[R]
                    # overwrite conf at R with entropy from 2nd forward
                    H2 = entropy_from_probs(p_x0_2, mask_index=self.mask_id, remove_mask_prob=self.remove_mask_prob)
                    self.conf[R] = H2[R]

            # 3) conf update for newly unmasked ([MASK]->token) positions from first forward
            H1 = entropy_from_probs(p_x0, mask_index=self.mask_id, remove_mask_prob=self.remove_mask_prob)
            became_unmasked = (prev_mask & (x != self.mask_id))
            self.conf[became_unmasked] = H1[became_unmasked]

            self.step_idx += 1

        # decode
        return self._decode_sequences(x, prompt_mask)

    def _decode_sequences(self, x: torch.Tensor, prompt_mask: torch.Tensor) -> List[str]:
        # simple decoding: take the entire sequence (or drop prompt tokens?)
        # Here we drop the prompt prefix and decode the rest (first contiguous non-prompt span).
        outs = []
        for i in range(x.size(0)):
            # find first non-prompt position
            pm = prompt_mask[i].tolist()
            try:
                start = pm.index(False)  # first non-prompt
            except ValueError:
                start = len(pm)
            seq = x[i, start:].tolist()
            outs.append(self.tok.decode(seq, skip_special_tokens=True))
        return outs


# ------------------------------
# LLaDA adapter (skeleton)
# ------------------------------

def llada_step_fn_factory(
    model,
    tokenizer,
    mask_id: Optional[int] = None,
    *,
    steps_total: int = 1024,   # 샘플러의 steps 와 반드시 일치
    schedule: str = "linear",  # 'linear' or 'cosine'
):
    """
    LLaDA 한 스텝용 step_fn을 생성.
    입력:  x_tokens(LongTensor[B,L]), t_step(int)
    출력:  SimpleNamespace(p_x0, alpha_t, alpha_s, prev_mask)

    시간 스케줄:
      t \in (0,1]  (0은 너무 약해서 1 스텝 밀어 올림)
      alpha_t = 1 - t, alpha_s = 1 - s  (s = t - dt, dt=1/steps_total)
    """
    # [MASK] 토큰 id
    if mask_id is None:
        mid = getattr(tokenizer, "mask_token_id", None)
        if mid is None:
            mid = tokenizer.vocab_size - 1  # 폴백: 마지막 vocab id를 [MASK]로 사용
        mask_id = int(mid)

    # t \in (0,1] 로 매핑
    def _to01_linear(t_idx: int) -> float:
        # 0..steps_total-1  -> (t_idx+1)/steps_total
        return min(1.0, max(0.0, (t_idx + 1) / float(steps_total)))

    def _to01_cosine(t_idx: int) -> float:
        # 느린 시작, 후반 가속 (cosine)
        x = (t_idx + 1) / float(steps_total)
        return 0.5 * (1.0 - math.cos(math.pi * x))

    def _sched01(t_idx: int) -> float:
        return _to01_linear(t_idx) if schedule == "linear" else _to01_cosine(t_idx)

    @torch.no_grad()
    def step_fn(x_tokens: torch.Tensor, t_step: int) -> SimpleNamespace:
        """
        한 reverse step:
          1) prev_mask = (x == [MASK])
          2) 모델 forward -> logits -> p_x0
          3) alpha_t, alpha_s 계산 후 반환
        """
        # x_tokens 는 샘플러에서 이미 적절한 device 로 생성됨
        prev_mask = (x_tokens == mask_id)

        # HF remote code: 일반적으로 forward(input_ids=...) -> logits(B,L,V)
        # 필요 시 attention_mask를 ones로 줄 수도 있음.
        outputs = model(input_ids=x_tokens)
        logits = outputs.logits  # (B, L, V)
        p_x0 = F.softmax(logits, dim=-1)

        # masking ratio 스케줄
        t = _sched01(t_step)                # in (0,1]
        dt = 1.0 / float(steps_total)
        s = max(0.0, t - dt)                # 이전 스텝의 비율
        alpha_t = 1.0 - t
        alpha_s = 1.0 - s

        return SimpleNamespace(
            p_x0=p_x0,
            alpha_t=alpha_t,
            alpha_s=alpha_s,
            prev_mask=prev_mask,
        )

    return step_fn



# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--mode", type=str, choices=["refine_ent3", "vanilla_generate"], default="refine_ent3")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--nucleus_p", type=float, default=0.9)
    parser.add_argument("--refine_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompts_file", type=str, required=True, help="one prompt per line")
    parser.add_argument("--out", type=str, required=True, help="output JSONL (fields: prompt, completion)")
    args = parser.parse_args()

    # load tokenizer/model (HF)
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("Install transformers to use this script.")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]

    if args.mode == "vanilla_generate":
        # simple baseline using HF generate (no refine). Useful sanity check.
        completions = []
        for p in prompts:
            inputs = tok(p, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=args.max_len, do_sample=False)
            text = tok.decode(out[0], skip_special_tokens=True)
            completions.append({"prompt": p, "completion": text})
        with open(args.out, "w", encoding="utf-8") as f:
            for item in completions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    # refine-ent-3 path
    step_fn = llada_step_fn_factory(
        model, tok,
        steps_total=args.steps,   # 샘플러 steps와 일치
        schedule="linear",        # 필요시 "cosine"
    )
    sampler = RefineEnt3Sampler(
        step_fn=step_fn,
        tokenizer=tok,
        nucleus_p=args.nucleus_p,
        refine_every=args.refine_every,
        remove_mask_prob_for_entropy=True,
        mask_token_id=getattr(tok, "mask_token_id", None),
    )

    # NOTE: sampler.sample will raise NotImplementedError until step_fn is implemented
    completions_text = sampler.sample(prompts=prompts, max_len=args.max_len, steps=args.steps, seed=args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for p, c in zip(prompts, completions_text):
            f.write(json.dumps({"prompt": p, "completion": c}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
