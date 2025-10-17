# -*- coding: utf-8 -*-
"""
gen_samples.py
- 모델 로드 전에 HF dataset을 먼저 로드(빠른 실패)
- 배치 단위로 생성 -> JSONL에 즉시 append + flush
- 기존 파일이 있으면 task_id 기준으로 이어하기(resume)
- tqdm로 진행 상황 표시
"""

from __future__ import annotations
import os, json, argparse
from typing import List, Tuple, Set

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# 당신의 샘플러/스텝 팩토리 (기존 파일 그대로 활용)
from refine_ent3_llada import RefineEnt3Sampler, llada_step_fn_factory


def _first_present(ex: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in ex and ex[k]:
            return ex[k]
    return default


def _norm_mbpp_id(x) -> str:
    if isinstance(x, int):
        return f"MBPP/{x}"
    s = str(x)
    return s if s.startswith("MBPP/") else f"MBPP/{int(s)}"


def _norm_he_id(x) -> str:
    s = str(x)
    return s if s.startswith("HumanEval/") else s  # HF는 원래 HumanEval/<n> 형식


def load_humaneval_prompts() -> Tuple[List[str], List[str]]:
    ds = load_dataset("openai_humaneval")["test"]
    keys = [ex["task_id"] for ex in ds]                # 이미 HumanEval/<n>
    prompts = [ex["prompt"] for ex in ds]
    return prompts, keys


def load_mbppplus_prompts() -> Tuple[List[str], List[str]]:
    # EvalPlus 릴리스와 맞는 HF 세트
    ds = load_dataset("evalplus/mbppplus")["test"]
    prompts, ids = [], []
    for ex in ds:
        prompt_txt = _first_present(
            ex, ["prompt", "text", "description", "task_description", "problem"], ""
        )
        starter = _first_present(ex, ["starter_code", "code", "starter"], "")
        p = prompt_txt + (("\n" + starter) if starter else "")
        prompts.append(p)
        raw_id = ex.get("task_id", ex.get("id"))
        ids.append(_norm_mbpp_id(raw_id))
    return prompts, ids


def read_existing_ids(jsonl_path: str) -> Set[str]:
    ids = set()
    if not os.path.exists(jsonl_path):
        return ids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = str(obj.get("task_id"))
                ids.add(tid)
            except Exception:
                pass
    return ids


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, type=str)
    ap.add_argument("--dataset", required=True, choices=["humanevalplus", "mbppplus"])
    ap.add_argument("--out_jsonl", required=True, type=str)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--nucleus_p", type=float, default=0.9)
    ap.add_argument("--refine_every", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    # 1) Dataset 먼저 로드 (빠른 실패 지점)
    if args.dataset == "humanevalplus":
        prompts, keys = load_humaneval_prompts()
    else:
        prompts, keys = load_mbppplus_prompts()

    # 2) Resume: 기존에 쓴 task_id는 스킵
    existing = read_existing_ids(args.out_jsonl)
    todo_indices = [i for i, k in enumerate(keys) if k not in existing]
    print(f"[Info] total={len(keys)} / existing={len(existing)} / to_generate={len(todo_indices)}")

    if not todo_indices:
        print("[Info] Nothing to generate. All done.")
        return

    # 3) 모델 로드 (dataset 로드 성공 후)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            args.model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
        )
    except ValueError as e:
        if "requires `accelerate`" in str(e):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModel.from_pretrained(
                args.model_id, trust_remote_code=True, torch_dtype=dtype, device_map=None
            ).to(device)
        else:
            raise
    model.eval()

    step_fn = llada_step_fn_factory(model, tok, steps_total=args.steps, schedule="linear")
    sampler = RefineEnt3Sampler(
        step_fn=step_fn,
        tokenizer=tok,
        nucleus_p=args.nucleus_p,
        refine_every=args.refine_every,
        remove_mask_prob_for_entropy=True,
        mask_token_id=getattr(tok, "mask_token_id", None),
    )

    # 4) 생성 → 즉시 append 저장 + flush (에러나도 저장본은 남는다)
    with open(args.out_jsonl, "a", encoding="utf-8") as fout:
        for s in tqdm(range(0, len(todo_indices), args.batch_size), desc="Generating", unit="batch"):
            idxs = todo_indices[s:s + args.batch_size]
            batch_prompts = [prompts[i] for i in idxs]
            batch_keys = [keys[i] for i in idxs]

            comps = sampler.sample(
                prompts=batch_prompts, max_len=args.max_len, steps=args.steps, seed=args.seed + s
            )

            for k, code in zip(batch_keys, comps):
                obj = {"task_id": k, "completion": code}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()  # 중요! 배치마다 디스크 기록 고정

    print(f"[Done] Saved to: {args.out_jsonl}")


if __name__ == "__main__":
    main()
