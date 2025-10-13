# -*- coding: utf-8 -*-
"""
evalplus_bridge.py (batched, robust MBPP loader)

- Uses EvalPlus CLI for scoring (no Python API import).
- Loads prompts via 🤗 datasets.
- **Batched generation** to prevent GPU OOM.
- Robust MBPP field handling (prompt/text/description, starter_code/code).

Usage (examples):
  python evalplus_bridge.py \
    --model_id GSAI-ML/LLaDA-8B-Instruct \
    --dataset humanevalplus \
    --steps 1024 --nucleus_p 0.9 --refine_every 4 \
    --batch_size 1 --max_len 512 \
    --out_dir results_llada_refine3

  python evalplus_bridge.py \
    --model_id GSAI-ML/LLaDA-8B-Instruct \
    --dataset mbppplus \
    --steps 1024 --nucleus_p 0.9 --refine_every 4 \
    --batch_size 1 --max_len 512 \
    --out_dir results_llada_refine3
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

from refine_ent3_llada import RefineEnt3Sampler, llada_step_fn_factory


def load_humaneval_prompts() -> Tuple[List[str], List[str]]:
    ds = datasets.load_dataset("openai_humaneval")["test"]
    keys = [ex["task_id"] for ex in ds]
    prompts = [ex["prompt"] for ex in ds]
    return prompts, keys


def _first_present(ex: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in ex and ex[k]:
            return ex[k]
    return default


def load_mbpp_prompts() -> Tuple[List[str], List[int]]:
    # Prefer sanitized split; fallbacks supported
    ds_dict = datasets.load_dataset("google-research-datasets/mbpp", "sanitized")
    split = "test" if "test" in ds_dict else ("validation" if "validation" in ds_dict else "train")
    ds = ds_dict[split]

    prompts, ids = [], []
    for ex in ds:
        prompt_txt = _first_present(ex, ["prompt", "text", "description", "task_description", "problem"], "")
        starter = _first_present(ex, ["code", "starter_code", "starter"], "")
        p = prompt_txt + ("\n" + starter if starter else "")
        prompts.append(p)
        ids.append(ex.get("task_id", ex.get("id", len(ids))))
    return prompts, ids


@torch.no_grad()
def generate_with_refine3(
    model_id: str,
    steps: int,
    nucleus_p: float,
    refine_every: int,
    dataset: str,
    out_dir: str,
    max_len: int = 512,
    seed: int = 1,
    batch_size: int = 1,
):
    # Robust device_map handling (accelerate optional)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=dtype, device_map="auto"
        )
    except ValueError as e:
        if "requires `accelerate`" in str(e):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True,
                torch_dtype=dtype, device_map=None
            ).to(device)
        else:
            raise
    model.eval()

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
        out_path = os.path.join(out_dir, "samples_humanevalplus.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            # batched generation
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_comps = sampler.sample(prompts=batch_prompts, max_len=max_len, steps=steps, seed=seed+i)
                for key, code in zip(keys[i:i+batch_size], batch_comps):
                    f.write(json.dumps({"task_id": key, "completion": code}, ensure_ascii=False) + "\n")
        cmd = ["evalplus.evaluate", "--dataset", "humaneval", "--samples", out_path]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return out_path

    elif dataset.lower() in ["mbpp", "mbppplus", "mbpp+", "mbpp-plus"]:
        prompts, ids = load_mbpp_prompts()
        out_path = os.path.join(out_dir, "samples_mbppplus.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_comps = sampler.sample(prompts=batch_prompts, max_len=max_len, steps=steps, seed=seed+i)
                for key, code in zip(ids[i:i+batch_size], batch_comps):
                    f.write(json.dumps({"task_id": key, "completion": code}, ensure_ascii=False) + "\n")
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
    ap.add_argument("--refine_every", type=int, default=4)  # default less frequent to save mem
    ap.add_argument("--dataset", type=str, choices=["humanevalplus", "mbppplus"], required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    path = generate_with_refine3(
        model_id=args.model_id,
        steps=args.steps,
        nucleus_p=args.nucleus_p,
        refine_every=args.refine_every,
        dataset=args.dataset,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print("Saved samples to:", path)


if __name__ == "__main__":
    main()
