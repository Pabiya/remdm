#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved generations on HumanEval / MBPP without re-generating.
- Input JSONL format (per line):
    {"task_id": "...", "completion": "..."}     # prompt field is ignored for eval
- Outputs:
    * HumanEval: pass@k to stdout, optional JSON summary (--report_out)
    * MBPP: pass@1 to stdout, optional JSON summary (--report_out) and per-task detail (--detail_out)
"""

from __future__ import annotations
import argparse, json, sys, subprocess, tempfile, textwrap, os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ------------------------------
# Small utils (optional post-process)
# ------------------------------
def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


# ------------------------------
# HumanEval helpers
# ------------------------------
def _materialize_humaneval_problem_file_subset(task_ids: set[str]) -> str:
    """
    HumanEval 문제를 human_eval.data.read_problems()에서 읽어
    '샘플에 존재하는 task_id'만 포함한 임시 JSONL을 만들어 경로를 반환.
    """
    try:
        from human_eval.data import read_problems
        problems = read_problems()
    except Exception as e:
        # 패키지가 없다면 HF datasets에서 가져와서 필터
        ds = load_dataset("openai_humaneval", split="test")
        problems = {
            ex["task_id"]: {
                "prompt": ex["prompt"],
                "canonical_solution": ex.get("canonical_solution", ""),
                "test": ex["test"],
                "entry_point": ex.get("entry_point", ""),
            }
            for ex in ds
        }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    with open(tmp.name, "w", encoding="utf-8") as f:
        for tid in sorted(task_ids):
            ex = problems.get(tid)
            if not ex:  # 샘플에만 있고 문제사전에 없으면 건너뜀
                continue
            obj = {
                "task_id": tid,
                "prompt": ex.get("prompt", ""),
                "canonical_solution": ex.get("canonical_solution", ""),
                "test": ex.get("test", ""),
                "entry_point": ex.get("entry_point", ""),
            }
            f.write(json.dumps(obj) + "\n")
    return tmp.name

def _print_and_maybe_save_he_summary(summary: dict, report_out: str | None):
    pa1 = summary.get("pass@1"); pa10 = summary.get("pass@10"); pa100 = summary.get("pass@100")
    print(f"[HumanEval] pass@1={pa1}, pass@10={pa10}, pass@100={pa100}")
    if report_out:
        Path(os.path.dirname(report_out) or ".").mkdir(parents=True, exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[HumanEval] report saved -> {report_out}")

def _eval_humaneval_via_subprocess(samples_path: str, problem_file: str, workers: int, report_out: str | None):
    cmd = [
        sys.executable, "-m", "human_eval.evaluation",
        "--samples", samples_path,
        "--problem_file", problem_file,
        "--n_workers", str(workers),
    ]
    print("[eval] Running HumanEval harness (subprocess):", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)

    stdout = (out.stdout or "").strip()
    if stdout:
        print(stdout)  # keep raw output

    # find the largest JSON object within stdout
    summary = None
    try:
        s = stdout
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            summary = json.loads(s[l:r+1])
    except Exception:
        summary = None

    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)

    if not summary:
        print("[HumanEval] Could not parse summary JSON from harness stdout.")
        return False

    _print_and_maybe_save_he_summary(summary, report_out)
    return True


# ------------------------------
# HumanEval evaluation
# ------------------------------
def eval_humaneval(jsonl_path: str, limit=0, workers=8, report_out: str | None = None,
                   apply_filter_code=False, apply_fix_indents=False):
    """
    HumanEval은 샘플에 포함된 task_id만 담은 문제파일 서브셋을 만들어 평가한다.
    Python API 우선, 실패 시 subprocess 백업.
    """
    rows = load_jsonl(jsonl_path)
    if limit and limit > 0:
        rows = rows[:limit]

    # optional completion post-process
    if apply_filter_code or apply_fix_indents:
        for r in rows:
            c = r.get("completion", "")
            if apply_fix_indents:
                c = fix_indents(c)
            if apply_filter_code:
                c = filter_code(c)
            r["completion"] = c

    # 샘플 JSONL (task_id, completion)
    tmp_samples = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    for r in tqdm(rows, desc="prepare-humaneval", total=len(rows), dynamic_ncols=True):
        obj = {"task_id": r["task_id"], "completion": r["completion"]}
        tmp_samples.write((json.dumps(obj) + "\n").encode("utf-8"))
    tmp_samples.close()

    # 문제파일: 샘플에 포함된 task_id 서브셋으로 한정
    task_ids = {r["task_id"] for r in rows}
    problem_file = _materialize_humaneval_problem_file_subset(task_ids)

    # Python API 먼저
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        print(f"[eval] Running HumanEval (python API): samples={tmp_samples.name}, problem_file={problem_file}, n_workers={workers}")
        summary = evaluate_functional_correctness(
            sample_file=tmp_samples.name,
            problem_file=problem_file,
            k=[1, 10, 100],
            n_workers=workers,
        )
        _print_and_maybe_save_he_summary(summary, report_out)
        return True
    except Exception as e:
        print("[eval] human_eval API failed, falling back to subprocess:", e)

    # 백업: subprocess
    return _eval_humaneval_via_subprocess(tmp_samples.name, problem_file, workers, report_out)


# ------------------------------
# MBPP evaluation (unit-test runner)
# ------------------------------
def eval_mbpp(jsonl_path: str, sanitized=True, limit=0, timeout=10,
              report_out: str | None = None, detail_out: str | None = None,
              apply_filter_code=False, apply_fix_indents=False):
    rows = load_jsonl(jsonl_path)
    if apply_filter_code or apply_fix_indents:
        for r in rows:
            c = r.get("completion", "")
            if apply_fix_indents:
                c = fix_indents(c)
            if apply_filter_code:
                c = filter_code(c)
            r["completion"] = c

    rows_by_id = {r["task_id"]: r for r in rows}

    name = "nlile/mbpp" if sanitized else "Muennighoff/mbpp"
    ds = load_dataset(name, split="test")
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    passed = 0
    total = 0
    failed_cases = []
    details = []

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
            tf.write(prog)
            tmp_py = tf.name

        ok = False
        err = ""
        try:
            proc = subprocess.run([sys.executable, tmp_py], timeout=timeout, capture_output=True, text=True)
            ok = (proc.returncode == 0) and ("PASSED" in proc.stdout)
            if not ok:
                err = (proc.stderr or "") + "\n" + (proc.stdout or "")
        except subprocess.TimeoutExpired:
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
            "error": err[:2000] if err else "",
        })

    ratio = (passed / total) if total else 0.0
    print(f"[MBPP] pass@1: {passed}/{total} = {ratio:.3f}")
    if failed_cases:
        print("[MBPP] failed task_ids (first 20):", failed_cases[:20])

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


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--pred", required=True, help="path to JSONL predictions")
    ap.add_argument("--limit", type=int, default=0, help="if >0, evaluate only first N")
    ap.add_argument("--report_out", type=str, default="", help="save summary JSON to this path")

    # HumanEval-only
    ap.add_argument("--workers", type=int, default=8, help="HumanEval harness n_workers")

    # MBPP-only
    ap.add_argument("--timeout", type=int, default=10, help="MBPP per-task timeout (seconds)")
    ap.add_argument("--detail_out", type=str, default="", help="save per-task MBPP results (JSONL)")

    # Optional completion post-process
    ap.add_argument("--apply_filter_code", action="store_true", help="strip to the first function block before eval")
    ap.add_argument("--fix_indents", action="store_true", help="replace tabs with 4 spaces before eval")

    args = ap.parse_args()

    if args.dataset == "humaneval":
        ok = eval_humaneval(
            args.pred,
            limit=args.limit,
            workers=args.workers,
            report_out=(args.report_out or None),
            apply_filter_code=args.apply_filter_code,
            apply_fix_indents=args.fix_indents,
        )
    else:
        ok = eval_mbpp(
            args.pred,
            sanitized=True,
            limit=args.limit,
            timeout=args.timeout,
            report_out=(args.report_out or None),
            detail_out=(args.detail_out or None),
            apply_filter_code=args.apply_filter_code,
            apply_fix_indents=args.fix_indents,
        )

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
