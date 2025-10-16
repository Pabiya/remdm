import os, json, argparse, subprocess
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datasets import load_dataset

from refine_ent3_llada import RefineEnt3Sampler, llada_step_fn_factory


def load_humaneval_prompts() -> Tuple[List[str], List[str]]:
    ds = load_dataset("evalplus/humanevalplus")["test"]  # 164개
    prompts = [ex["prompt"] for ex in ds]
    ids = [str(ex["task_id"]) for ex in ds]              # "HumanEval/0" 형식
    return prompts, ids


def _first_present(ex: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in ex and ex[k]:
            return ex[k]
    return default


def load_mbpp_prompts() -> Tuple[List[str], List[str]]:
    ds = load_dataset("evalplus/mbppplus")["test"]       # MBPP+ 정합 데이터
    prompts, ids = [], []
    for ex in ds:
        # prompt + (starter_code 있으면 붙이기)
        prompt_txt = _first_present(ex, ["prompt", "text", "description", "task_description", "problem"], "")
        starter = _first_present(ex, ["starter_code", "code", "starter"], "")
        p = prompt_txt + (("\n" + starter) if starter else "")
        prompts.append(p)
        # 이미 "MBPP/xxx" 문자열일 가능성이 큼
        ids.append(str(ex.get("task_id", ex.get("id"))))
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
    os.makedirs(out_dir, exist_ok=True)

    # 1) 데이터셋 먼저 로드 (fail-fast)
    dataset_key = dataset.lower()
    if dataset_key in ["humaneval", "humanevalplus", "he", "he+"]:
        prompts, keys = load_humaneval_prompts()
        out_path = os.path.join(out_dir, "samples_humanevalplus.jsonl")
        eval_cmd = ["evalplus.evaluate", "--dataset", "humaneval", "--samples", out_path]
    elif dataset_key in ["mbpp", "mbppplus", "mbpp+", "mbpp-plus"]:
        prompts, keys = load_mbpp_prompts()
        out_path = os.path.join(out_dir, "samples_mbppplus.jsonl")
        eval_cmd = ["evalplus.evaluate", "--dataset", "mbpp", "--samples", out_path]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"[Info] Loaded dataset: {dataset} (#problems={len(prompts)})", flush=True)

    # 2) 모델은 그 다음에 로드
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
    except ValueError as e:
        if "requires `accelerate`" in str(e):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype, device_map=None).to(device)
        else:
            raise
    model.eval()

    step_fn = llada_step_fn_factory(model, tok, steps_total=steps, schedule="linear")
    sampler = RefineEnt3Sampler(
        step_fn=step_fn, tokenizer=tok, nucleus_p=nucleus_p, refine_every=refine_every,
        remove_mask_prob_for_entropy=True, mask_token_id=getattr(tok, "mask_token_id", None),
    )

    # 3) tqdm으로 배치 진행 표시
    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
            batch_prompts = prompts[i:i+batch_size]
            batch_comps = sampler.sample(prompts=batch_prompts, max_len=max_len, steps=steps, seed=seed+i)
            for key, code in zip(keys[i:i+batch_size], batch_comps):
                f.write(json.dumps({"task_id": str(key), "completion": code}, ensure_ascii=False) + "\n")

    print("Running:", " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True)
    return out_path


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

    out_path = generate_with_refine3(
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
    print("Saved samples to:", out_path)


if __name__ == "__main__":
    main()
