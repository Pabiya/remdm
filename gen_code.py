#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate code with LLaDA-8B on HumanEval / MBPP.
Modes:
  - baseline: use model.generate (HF, trust_remote_code=True)
  - refine:   apply in-step 2-forward refine (refine-ent-3) via a per-step hook

Outputs JSONL per dataset:
  {"task_id": str, "prompt": str, "completion": str, "mode": "baseline|refine", "model": "...", "seed": int}
"""

import argparse, json, os, random
from pathlib import Path
from itertools import islice

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ------------------------------
# Utils
# ------------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_jsonl(path, rows):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_llada(model_name: str, dtype: str = "bfloat16", device: str = "cuda"):
    torch_dtype = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
    mdl.eval()
    return tok, mdl


# ------------------------------
# Datasets
# ------------------------------
def iter_humaneval(split="test"):
    ds = load_dataset("openai_humaneval", split=split)
    for ex in ds:
        yield {"task_id": ex["task_id"], "prompt": ex["prompt"], "tests": ex["test"]}

def iter_mbpp(split="test", sanitized=True):
    name = "nlile/mbpp" if sanitized else "Muennighoff/mbpp"
    ds = load_dataset(name, split=split)
    for i, ex in enumerate(ds):
        task_id = str(ex.get("task_id", f"mbpp-{i}"))
        text = ex.get("text") or ex.get("prompt") or ex.get("description")
        test_list = ex.get("test_list")
        test = ex.get("test")
        yield {"task_id": task_id, "prompt": text, "tests": test or test_list}


# ------------------------------
# Baseline generation
# ------------------------------
@torch.no_grad()
def generate_baseline(tok, mdl, prompt, max_new_tokens=256, temperature=0.2, top_p=0.9):
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):]


# ------------------------------
# Refine (refine-ent-3): in-step 2-forward, overwrite conf with entropy of 2nd forward
# ------------------------------
@torch.no_grad()
def generate_refine_ent3(tok, mdl, prompt, steps=256, top_p=0.9, refine_every=1, seed=1):
    """
    Assumes the model's custom generate supports a per-step hook (trust_remote_code=True).
    The hook:
      - eta = softmax(conf); mask positions excluded
      - sigma_max = min(1, (1 - alpha_s) / alpha_t)
      - sigma = eta * sigma_max; R ~ Bernoulli(sigma) on unmasked
      - 2nd forward on x_tmp (R -> [MASK]); resample R from p_x0'
      - conf[R] = entropy(p_x0')  # overwrite; no EMA / cooldown
    """
    def refine_hook(state):
        # quick guard: periodic refine
        if (state["step_idx"] % max(1, refine_every)) != 0:
            return None

        x = state["x_t"]; xs = state["xs"]
        conf = state["conf"]; masked_flag = state["masked_flag"]
        alpha_t = state["alpha_t"]; alpha_s = state["alpha_s"]

        sigma_max = min(1.0, (1.0 - alpha_s) / max(1e-12, alpha_t))
        eta = torch.softmax(conf, dim=-1)
        eta = eta.masked_fill(masked_flag, 0.0)
        sigma = (eta * sigma_max).clamp_(0.0, 1.0)

        unmasked_flag = ~masked_flag
        R = (torch.rand_like(sigma) < sigma) & unmasked_flag
        if not R.any():
            return None

        # second forward with R masked
        x_tmp = xs.clone()
        x_tmp[R] = state["mask_index"]
        log_p_x0_2 = state["forward_fn"](x_tmp)           # same t
        p_x0_2 = state["sample_with_top_p"](log_p_x0_2)   # apply top-p & renorm

        # resample only R
        sampled = state["sample_categorical"](p_x0_2)
        xs[R] = sampled[R]

        # overwrite conf on R with entropy(p_x0_2)
        P2 = p_x0_2.clamp_min(1e-12)
        if state["entropy_remove_mask_prob"]:
            Pw = P2.clone(); Pw[..., state["mask_index"]] = 0.0
            Z = Pw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            Q = Pw / Z
        else:
            Z = P2.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            Q = P2 / Z
        H2 = -(Q * (Q + 1e-12).log()).sum(dim=-1)
        conf[R] = H2[R]
        return {"xs_override": xs, "conf_override": conf}

    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        do_sample=True,
        top_p=top_p,
        diffusion_steps=steps,             # custom kwarg provided by the model (adapt if needed)
        refine_hook=refine_hook,           # our hook
        return_dict_in_generate=True,
        output_scores=False,
    )
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    return text[len(prompt):]


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="baseline", choices=["baseline", "refine"])
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=256, help="diffusion steps for refine")
    ap.add_argument("--refine_every", type=int, default=1, help="k=1 => refine every step")
    ap.add_argument("--limit", type=int, default=0, help="if >0, only first N samples; show tqdm")
    args = ap.parse_args()

    set_seed(args.seed)
    tok, mdl = load_llada(args.model, dtype=args.dtype, device=args.device)

    it = iter_humaneval(args.split) if args.dataset == "humaneval" else iter_mbpp(args.split, sanitized=True)
    iterable = islice(it, args.limit) if args.limit and args.limit > 0 else it
    rows = []

    pbar = tqdm(iterable, total=args.limit if args.limit else None,
                desc=f"gen-{args.dataset}-{args.mode}", dynamic_ncols=True)
    for ex in pbar:
        prompt = ex["prompt"]
        if args.mode == "baseline":
            comp = generate_baseline(tok, mdl, prompt,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature,
                                     top_p=args.top_p)
        else:
            comp = generate_refine_ent3(tok, mdl, prompt,
                                        steps=args.steps, top_p=args.top_p,
                                        refine_every=args.refine_every, seed=args.seed)
        rows.append({
            "task_id": ex["task_id"],
            "prompt": prompt,
            "completion": comp,
            "mode": args.mode,
            "model": args.model,
            "seed": args.seed,
        })
        pbar.set_postfix_str(f"task={ex['task_id']}")
    pbar.close()

    save_jsonl(args.out, rows)
    print(f"[gen] wrote {len(rows)} samples -> {args.out}")

if __name__ == "__main__":
    main()
