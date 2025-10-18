#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_code.py

Generate code on HumanEval/MBPP with LLaDA-8B (HF) in two modes:
  - baseline: standard HF generate
  - refine  : Refine-Ent-3 (in-step 2-forward refine with entropy overwrite)

Outputs JSONL:
  {"task_id": str, "prompt": str, "completion": str, "mode": "baseline|refine", "model": "...", "seed": int}

Example:
  python gen_code.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --dataset humaneval \
    --mode refine \
    --steps 256 --refine_every 1 --nucleus_p 0.9 \
    --out /root/results/llada8b_refine_he10.jsonl \
    --limit 10 --add_mask_token
"""

from __future__ import annotations
import argparse, json, os, math
from pathlib import Path
from typing import List, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_jsonl(path, rows):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ensure_mask_token(tokenizer, model, add_mask_token: bool = True) -> int:
    """
    Ensure tokenizer has a [MASK]-like token and the model embeddings match.
    Returns mask_token_id.
    """
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is not None:
        return int(mask_id)
    if not add_mask_token:
        return tokenizer.vocab_size - 1  # fallback: 마지막 vocab id를 [MASK]처럼 사용
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    model.resize_token_embeddings(len(tokenizer))
    return int(tokenizer.convert_tokens_to_ids("<mask>"))

def top_p_probs(probs: torch.Tensor, p: float, eps: float = 1e-12) -> torch.Tensor:
    if p >= 1.0:
        return probs
    P = probs.float()
    sorted_probs, sorted_idx = torch.sort(P, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumsum <= p)
    keep[..., 0] = True
    kept = sorted_probs * keep
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.zeros_like(P)
    out.scatter_(-1, sorted_idx, kept)
    return out.to(probs.dtype)

def sample_categorical(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    B, L, V = probs.shape
    P = probs.reshape(-1, V)
    idx = torch.multinomial(P, num_samples=1, replacement=True, generator=generator).view(B, L)
    return idx

def entropy_from_probs(probs: torch.Tensor, mask_index: int, remove_mask_prob: bool = True, eps: float = 1e-12) -> torch.Tensor:
    P = probs.float().clamp_min(eps)
    if remove_mask_prob:
        Pw = P.clone(); Pw[..., mask_index] = 0.0
        Z = Pw.sum(dim=-1, keepdim=True).clamp_min(eps)
        Q = Pw / Z
    else:
        Z = P.sum(dim=-1, keepdim=True).clamp_min(eps)
        Q = P / Z
    H = -(Q * (Q + eps).log()).sum(dim=-1)
    return H  # fp32

def build_q_xs2_from_p_x0(p_x0: torch.Tensor, alpha_t: float, alpha_s: float, mask_index: int, eps: float = 1e-12) -> torch.Tensor:
    denom = max(eps, (1.0 - alpha_t))
    q = p_x0 * ((alpha_s - alpha_t) / denom)
    q[..., mask_index] = (1.0 - alpha_s) / denom
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q

def left_pad_and_mask(prompts_ids: List[List[int]], max_len: int, mask_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L = len(prompts_ids), max_len
    x = torch.full((B, L), fill_value=mask_id, dtype=torch.long, device=device)
    prompt_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i, ids in enumerate(prompts_ids):
        ids = ids[:L]
        if len(ids) == 0: continue
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        prompt_mask[i, :len(ids)] = True
    return x, prompt_mask

# ------------------------------
# Dataset iterators
# ------------------------------

def iter_humaneval(limit: int = 0):
    ds = load_dataset("openai_humaneval", split="test")
    it = ds if (not limit or limit <= 0) else ds.select(range(min(limit, len(ds))))
    for ex in it:
        yield {"task_id": ex["task_id"], "prompt": ex["prompt"]}

def iter_mbpp(limit: int = 0, sanitized=True):
    name = "nlile/mbpp" if sanitized else "Muennighoff/mbpp"
    ds = load_dataset(name, split="test")
    it = ds if (not limit or limit <= 0) else ds.select(range(min(limit, len(ds))))
    for i, ex in enumerate(it):
        task_id = str(ex.get("task_id", f"mbpp-{i}"))
        text = ex.get("text") or ex.get("prompt") or ex.get("description")
        yield {"task_id": task_id, "prompt": text}

# ------------------------------
# LLaDA step_fn factory (skeleton)
# ------------------------------

def llada_step_fn_factory(model, tokenizer, mask_id: int, *, steps_total: int = 256, schedule: str = "linear"):
    def _to01_linear(t_idx: int) -> float:
        return min(1.0, max(0.0, (t_idx + 1) / float(steps_total)))
    def _to01_cosine(t_idx: int) -> float:
        x = (t_idx + 1) / float(steps_total)
        return 0.5 * (1.0 - math.cos(math.pi * x))
    def _sched01(t_idx: int) -> float:
        return _to01_linear(t_idx) if schedule == "linear" else _to01_cosine(t_idx)

    @torch.no_grad()
    def step_fn(x_tokens: torch.Tensor, t_step: int) -> SimpleNamespace:
        prev_mask = (x_tokens == mask_id)
        attn = torch.ones_like(x_tokens, dtype=torch.long, device=x_tokens.device)
        outputs = model(input_ids=x_tokens, attention_mask=attn)
        logits = outputs.logits.float()
        p_x0 = F.softmax(logits, dim=-1).to(logits.dtype)
        t = _sched01(t_step); dt = 1.0 / float(steps_total); s = max(0.0, t - dt)
        alpha_t = 1.0 - t; alpha_s = 1.0 - s
        return SimpleNamespace(p_x0=p_x0, alpha_t=alpha_t, alpha_s=alpha_s, prev_mask=prev_mask)

    return step_fn

# ------------------------------
# Refine-Ent-3 sampler
# ------------------------------

class RefineEnt3Sampler:
    def __init__(self, step_fn, tokenizer, nucleus_p=0.9, refine_every=1, remove_mask_prob_for_entropy=True,
                 mask_token_id=None, device=None):
        self.step_fn = step_fn
        self.tok = tokenizer
        self.nucleus_p = float(nucleus_p)
        self.refine_every = int(refine_every)
        self.remove_mask_prob = bool(remove_mask_prob_for_entropy)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_id = int(mask_token_id)
        self.conf = None
        self.step_idx = 0

    @torch.no_grad()
    def sample_one(self, prompt: str, max_len: int, steps: int, seed: int) -> str:
        set_seed(seed)
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        gen.manual_seed(seed)

        toks = self.tok([prompt], add_special_tokens=False)
        x, prompt_mask = left_pad_and_mask(toks["input_ids"], max_len=max_len, mask_id=self.mask_id, device=self.device)
        B, L = x.shape
        self.conf = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        self.step_idx = 0

        for t in range(steps):
            # first forward
            out = self.step_fn(x, t)
            p_x0 = top_p_probs(out.p_x0, self.nucleus_p)
            alpha_t = float(out.alpha_t); alpha_s = float(out.alpha_s)
            prev_mask = out.prev_mask

            # default unmask via q_xs2
            masked_flag = (x == self.mask_id)
            if masked_flag.any():
                q_xs2 = build_q_xs2_from_p_x0(p_x0, alpha_t, alpha_s, self.mask_id)
                new_tok = sample_categorical(q_xs2, generator=gen)
                write_mask = masked_flag & (~prompt_mask)
                x[write_mask] = new_tok[write_mask]

            # refine periodically
            if (self.step_idx % max(1, self.refine_every)) == 0:
                eta = torch.softmax(self.conf, dim=-1)
                eta = eta.masked_fill(masked_flag, 0.0)
                sigma_max = min(1.0, (1.0 - alpha_s) / max(1e-12, alpha_t))
                sigma = (eta * sigma_max).clamp_(0.0, 1.0)
                unmasked_flag = ~masked_flag
                R = (torch.rand_like(sigma) < sigma) & unmasked_flag & (~prompt_mask)
                if R.any():
                    x_tmp = x.clone(); x_tmp[R] = self.mask_id
                    out2 = self.step_fn(x_tmp, t)
                    p_x0_2 = top_p_probs(out2.p_x0, self.nucleus_p)
                    refill_tok = sample_categorical(p_x0_2, generator=gen)
                    x[R] = refill_tok[R]
                    H2 = entropy_from_probs(p_x0_2, mask_index=self.mask_id, remove_mask_prob=self.remove_mask_prob)
                    self.conf[R] = H2[R].to(self.conf.dtype)

            # conf update for newly unmasked (from first forward)
            H1 = entropy_from_probs(p_x0, mask_index=self.mask_id, remove_mask_prob=self.remove_mask_prob)
            became_unmasked = (prev_mask & (x != self.mask_id))
            self.conf[became_unmasked] = H1[became_unmasked].to(self.conf.dtype)

            self.step_idx += 1

        # decode completion (drop prompt prefix)
        pm = prompt_mask[0].tolist()
        try:
            start = pm.index(False)
        except ValueError:
            start = len(pm)
        seq = x[0, start:].tolist()
        return self.tok.decode(seq, skip_special_tokens=True)

# ------------------------------
# Baseline generation
# ------------------------------

@torch.no_grad()
def generate_baseline(tok, mdl, prompt, max_new_tokens=256, temperature=0.2, top_p=0.9) -> str:
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids(tok.eos_token or "</s>")
    pad_id = eos_id
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):]

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--mode", default="baseline", choices=["baseline", "refine"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only first N samples")
    ap.add_argument("--seed", type=int, default=1)

    # baseline knobs
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    # refine knobs
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--refine_every", type=int, default=1)
    ap.add_argument("--nucleus_p", type=float, default=0.9)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--add_mask_token", action="store_true", help="add <mask> token if tokenizer lacks one")

    args = ap.parse_args()

    set_seed(args.seed)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch_dtype, device_map="auto")
    mdl.eval()

    # dataset iterator
    if args.dataset == "humaneval":
        data_iter = iter_humaneval(limit=args.limit)
        total = min(args.limit, 164) if args.limit else 164
    else:
        data_iter = iter_mbpp(limit=args.limit, sanitized=True)
        total = args.limit if args.limit else None

    rows = []
    if args.mode == "baseline":
        pbar = tqdm(data_iter, total=total, desc=f"gen-{args.dataset}-baseline", dynamic_ncols=True)
        for ex in pbar:
            comp = generate_baseline(tok, mdl, ex["prompt"],
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature,
                                     top_p=args.top_p)
            rows.append({
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "completion": comp,
                "mode": "baseline",
                "model": args.model,
                "seed": args.seed,
            })
            pbar.set_postfix_str(f"task={ex['task_id']}")
        pbar.close()
        save_jsonl(args.out, rows)
        print(f"[gen/baseline] wrote {len(rows)} -> {args.out}")
        return

    # refine mode
    mask_id = ensure_mask_token(tok, mdl, add_mask_token=args.add_mask_token)
    step_fn = llada_step_fn_factory(mdl, tok, mask_id=mask_id, steps_total=args.steps, schedule="linear")
    sampler = RefineEnt3Sampler(step_fn, tok, nucleus_p=args.nucleus_p, refine_every=args.refine_every,
                                remove_mask_prob_for_entropy=True, mask_token_id=mask_id)

    pbar = tqdm(data_iter, total=total, desc=f"gen-{args.dataset}-refine", dynamic_ncols=True)
    for ex in pbar:
        comp = sampler.sample_one(prompt=ex["prompt"], max_len=args.max_len, steps=args.steps, seed=args.seed)
        rows.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "completion": comp,
            "mode": "refine",
            "model": args.model,
            "seed": args.seed,
        })
        pbar.set_postfix_str(f"task={ex['task_id']}")
    pbar.close()

    save_jsonl(args.out, rows)
    print(f"[gen/refine] wrote {len(rows)} -> {args.out}")

if __name__ == "__main__":
    main()
