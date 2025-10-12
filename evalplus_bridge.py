# -*- coding: utf-8 -*-
"""
evalplus_bridge.py

Bridge script to evaluate Refine-Ent-3 (refine_ent3_llada.RefineEnt3Sampler)
on HumanEval+ / MBPP+ using EvalPlus **CLI**.

Usage:
    python evalplus_bridge.py \
      --model_id GSAI-ML/LLaDA-8B-Instruct \
      --dataset humanevalplus \
      --steps 1024 --nucleus_p 0.9 --refine_every 1 \
      --out_dir results_llada_refine3

It will:
  1) Load tasks/prompts from 🤗 datasets
     - HumanEval:   dataset "openai_humaneval" (task_id keys)
     - MBPP (san.): dataset "google-research-datasets/mbpp", config "sanitized"
  2) Generate ONE completion per task using RefineEnt3Sampler
  3) Save to JSONL (task_id, completion) in out_dir
  4) Call: evalplus.evaluate --dataset {humaneval|mbpp} --samples <jsonl>
"""
from __future__ import annotations

import os
import json
import argparse
import subprocess
from typing import List, Tuple

import torch
import datasets
from transformers import AutoTokenizer, AutoModel

# Import our sampler
from refine_ent3_llada import RefineEnt3Sampler, llada_step_fn_factory


def load_humaneval_prompts() -> Tuple[List[str], List[str]]:
    """Return (prompts, task_ids) for HumanEval."""
    ds = datasets.load_dataset("openai_humaneval")["test"]
    keys = [ex["task_id"] for ex in ds]
    prompts = [ex["prompt"] for ex in ds]
    return prompts, keys


def load_mbpp_prompts() -> Tuple[List[str], List[int]]:
    """Return (prompts, task_ids) for MBPP (sanitized split)."""
    ds = datasets.load_dataset("google-research-datasets/mbpp", "sanitized")["test"]
    prompts, ids = [], []
    for ex in ds:
        p = ex["text"]
        if ex.get("code", ""):
            p = p + "\n" + ex["code"]
        prompts.append(p)
        ids.append(ex["task_id"])
    return prompts, ids


@torch.no_grad()
def generate_with_refine3(model_id: str, steps: int, nucleus_p: float, refine_every: int,
                          dataset: str, out_dir: str, max_len: int = 1024, seed: int = 1):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # IMPORTANT: steps_total must match sampler steps for sigma_max consistency
    step_fn = llada_step_fn_factory(model, tok, steps_total=steps, schedule="linear")

    sampler = RefineEnt3Sampler(
        step_fn=step_fn,
        tokenizer=tok,
        nucleus_p=nucleus_p,
        refine_every=refine_every,
        remove_mask_prob_for_entropy=True,
        mask_token_id=getattr(tok, "mask_token_id", None),
    )

    os.makedirs(out_dir, exist_ok=True)

    if dataset.lower() in ["humaneval", "humanevalplus", "he", "he+"]:
        prompts, keys = load_humaneval_prompts()
        completions = sampler.sample(prompts=prompts, max_len=max_len, steps=steps, seed=seed)
        out_path = os.path.join(out_dir, "samples_humanevalplus.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for key, code in zip(keys, completions):
                rec = {"task_id": key, "completion": code}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # EvalPlus CLI (will run HumanEval+ by default when dataset=humaneval)
        cmd = ["evalplus.evaluate", "--dataset", "humaneval", "--samples", out_path]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return out_path

    elif dataset.lower() in ["mbpp", "mbppplus", "mbpp+", "mbpp-plus"]:
        prompts, ids = load_mbpp_prompts()
        completions = sampler.sample(prompts=prompts, max_len=max_len, steps=steps, seed=seed)
        out_path = os.path.join(out_dir, "samples_mbppplus.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for key, code in zip(ids, completions):
                rec = {"task_id": key, "completion": code}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # EvalPlus CLI (will run MBPP+ by default when dataset=mbpp)
        cmd = ["evalplus.evaluate", "--dataset", "mbpp", "--samples", out_path]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return out_path

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--nucleus_p", type=float, default=0.9)
    ap.add_argument("--refine_every", type=int, default=1)
    ap.add_argument("--dataset", type=str, choices=["humanevalplus", "mbppplus"], required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    path = generate_with_refine3(
        model_id=args.model_id,
        steps=args.steps,
        nucleus_p=args.nucleus_p,
        refine_every=args.refine_every,
        dataset=args.dataset,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    print("Saved samples to:", path)


if __name__ == "__main__":
    main()
