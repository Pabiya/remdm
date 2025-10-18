#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved generations on HumanEval / MBPP without re-generating.
- Input: JSONL from gen_code.py (or compatible)
- Output:
    * HumanEval: pass@k (stdout), optional JSON summary (--report_out)
    * MBPP: pass@1 (stdout), optional JSON summary (--report_out) and per-task detail (--detail_out)
"""

import argparse, json, sys, subprocess, tempfile, textwrap, os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ------------------------------
# HumanEval evaluation (OpenAI harness)
# ------------------------------
def eval_humaneval(jsonl_path, limit=0, workers=8, report_out=None):
    rows = load_jsonl(jsonl_path)
    if limit and limit > 0:
        rows = rows[:limit]

    # prepare samples file for harness
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    for r in tqdm(rows, desc="prepare-humaneval", total=len(rows), dynamic_ncols=True):
        obj = {"task_id": r["task_id"], "completion": r["completion"]}
        tmp.write((json.dumps(obj) + "\n").encode("utf-8"))
    tmp.close()

    cmd = [
        sys.executable, "-m", "human_eval.evaluation",
        "--samples", tmp.name,
        "--problem_file", "human_eval/data/HumanEval.jsonl",
        "--n_workers", str(workers),
    ]
    print("[eval] Running HumanEval harness:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)

    # HumanEval harness prints a json line summary, e.g. {"pass@1": 0.35, "pass@10": ..., ...}
    summary = None
    stdout = out.stdout.strip()
    if stdout:
        print(stdout)  # keep raw output for logs
        # try parse last non-empty line as JSON
        last = stdout.splitlines()[-1]
        try:
            summary = json.loads(last)
        except Exception:
            try:
                summary = json.loads(stdout)
            except Exception:
                summary = None

    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)

    if summary:
        pa1 = summary.get("pass@1")
        pa10 = summary.get("pass@10")
        pa100 = summary.get("pass@100")
        print(f"[HumanEval] pass@1={pa1}, pass@10={pa10}, pass@100={pa100}")
        if report_out:
            Path(os.path.dirname(report_out) or ".").mkdir(parents=True, exist_ok=True)
            with open(report_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[HumanEval] report saved -> {report_out}")
    else:
        print("[HumanEval] Could not parse summary JSON from harness stdout.")

    return out.returncode == 0


# ------------------------------
# MBPP evaluation (unit-test runner)
# ------------------------------
def eval_mbpp(jsonl_path, sanitized=True, limit=0, timeout=10, report_out=None, detail_out=None):
    rows = load_jsonl(jsonl_path)
    rows_by_id = {r["task_id"]: r for r in rows}

    name = "nlile/mbpp" if sanitized else "Muennighoff/mbpp"
    ds = load_dataset(name, split="test")
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    passed = 0
    total = 0
    failed_cases = []
    details = []  # for JSONL detail output

    for ex in tqdm(ds, desc="eval-mbpp", total=len(ds), dynamic_ncols=True):
        task_id = str(ex.get("task_id"))
        # skip if no prediction for this task
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
            tf.write(prog)
            tmp_py = tf.name

        ok = False
        err = ""
        try:
            proc = subprocess.run([sys.executable, tmp_py], timeout=timeout, capture_output=True, text=True)
            ok = (proc.returncode == 0) and ("PASSED" in proc.stdout)
            if not ok:
                err = (proc.stderr or "") + "\n" + (proc.stdout or "")
        except subprocess.TimeoutExpired as e:
            ok = False
            err = f"Timeout after {timeout}s"

        total += 1
        if ok:
            passed += 1
        else:
            failed_cases.append(task_id)

        details.append({
            "task_id": task_id,
            "passed": ok,
            "error": err[:2000] if err else "",  # truncate long output
        })

    # summary
    ratio = (passed / total) if total else 0.0
    print(f"[MBPP] pass@1: {passed}/{total} = {ratio:.3f}")
    if failed_cases:
        print("[MBPP] failed task_ids (first 20):", failed_cases[:20])

    # save summaries
    if report_out:
        Path(os.path.dirname(report_out) or ".").mkdir(parents=True, exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump({"pass@1": ratio, "passed": passed, "total": total}, f, ensure_ascii=False, indent=2)
        print(f"[MBPP] report saved -> {report_out}")

    if detail_out:
        Path(os.path.dirname(detail_out) or ".").mkdir(parents=True, exist_ok=True)
        with open(detail_out, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[MBPP] details saved -> {detail_out}")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--pred", required=True, help="path to JSONL from gen_code.py")
    ap.add_argument("--limit", type=int, default=0, help="if >0, evaluate only first N")
    ap.add_argument("--report_out", type=str, default="", help="save summary JSON to this path")
    # HumanEval-only
    ap.add_argument("--workers", type=int, default=8, help="HumanEval harness n_workers")
    # MBPP-only
    ap.add_argument("--timeout", type=int, default=10, help="MBPP per-task timeout (seconds)")
    ap.add_argument("--detail_out", type=str, default="", help="save per-task MBPP results (JSONL)")
    args = ap.parse_args()

    if args.dataset == "humaneval":
        ok = eval_humaneval(
            args.pred,
            limit=args.limit,
            workers=args.workers,
            report_out=(args.report_out or None),
        )
    else:
        ok = eval_mbpp(
            args.pred,
            sanitized=True,
            limit=args.limit,
            timeout=args.timeout,
            report_out=(args.report_out or None),
            detail_out=(args.detail_out or None),
        )

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
