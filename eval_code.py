#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved generations on HumanEval / MBPP without re-generating.
- Input: JSONL from gen_llada.py
- Output: pass@1 (HumanEval harness / MBPP unit-test runner)
"""

import argparse, json, sys, subprocess, tempfile, textwrap
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ------------------------------
# HumanEval evaluation (OpenAI harness)
# ------------------------------
def eval_humaneval(jsonl_path, limit=0):
    rows = load_jsonl(jsonl_path)
    if limit and limit > 0:
        rows = rows[:limit]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    for r in tqdm(rows, desc="prepare-humaneval", total=len(rows), dynamic_ncols=True):
        obj = {"task_id": r["task_id"], "completion": r["completion"]}
        tmp.write((json.dumps(obj) + "\n").encode("utf-8"))
    tmp.close()

    cmd = [
        sys.executable, "-m", "human_eval.evaluation",
        "--samples", tmp.name,
        "--problem_file", "human_eval/data/HumanEval.jsonl",
        "--n_workers", "8"
    ]
    print("[eval] Running HumanEval harness:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
    return out.returncode == 0


# ------------------------------
# MBPP evaluation (unit-test runner)
# ------------------------------
def eval_mbpp(jsonl_path, sanitized=True, limit=0):
    rows = load_jsonl(jsonl_path); rows_by_id = {r["task_id"]: r for r in rows}
    name = "nlile/mbpp" if sanitized else "Muennighoff/mbpp"
    ds = load_dataset(name, split="test")
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    passed = 0; total = 0; failed_cases = []
    for ex in tqdm(ds, desc="eval-mbpp", total=len(ds), dynamic_ncols=True):
        task_id = str(ex.get("task_id"))
        if task_id not in rows_by_id:
            continue
        code = rows_by_id[task_id]["completion"]
        tests = ex.get("test") or ex.get("test_list")
        tests_src = "\n".join(tests) if isinstance(tests, list) else tests

        prog = textwrap.dedent(f"""
        import sys
        {code}

        if __name__ == "__main__":
            {tests_src}
            print("PASSED")
        """)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tf:
            tf.write(prog); tmp_py = tf.name

        try:
            proc = subprocess.run([sys.executable, tmp_py], timeout=10, capture_output=True, text=True)
            ok = (proc.returncode == 0) and ("PASSED" in proc.stdout)
        except subprocess.TimeoutExpired:
            ok = False

        total += 1
        if ok: passed += 1
        else: failed_cases.append(task_id)

    print(f"[MBPP] pass@1: {passed}/{total} = {passed/total if total else 0.0:.3f}")
    if failed_cases:
        print("[MBPP] failed task_ids (first 20):", failed_cases[:20])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--pred", required=True, help="path to JSONL from gen_llada.py")
    ap.add_argument("--limit", type=int, default=0, help="if >0, evaluate only first N")
    args = ap.parse_args()

    if args.dataset == "humaneval":
        ok = eval_humaneval(args.pred, limit=args.limit)
    else:
        ok = eval_mbpp(args.pred, sanitized=True, limit=args.limit)

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
