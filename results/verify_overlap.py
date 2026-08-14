#!/usr/bin/env python3
"""
Минимальный скрипт проверки overlap_area — независимый от src/eval/.

Считает overlap_area at position 0 (first 1 token) для baseline drafter'а
на всех flan validation кластерах. Результат — JSON + print.

Usage:
    python results/verify_overlap.py
    python results/verify_overlap.py --draft tiny-mixtral --dtype float32
    python results/verify_overlap.py --cluster aeslc_10templates --max-samples 20
    python results/verify_overlap.py --all-clusters --max-samples 50
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent


def load_models(target_dir, draft_dir, dtype_str, device):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    print(f"Loading tokenizer from {target_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(target_dir), trust_remote_code=True
    )

    print(f"Loading target model: {target_dir} ({dtype_str}) ...")
    target_model = AutoModelForCausalLM.from_pretrained(
        str(target_dir), trust_remote_code=True, dtype=torch_dtype
    ).to(device).eval()

    print(f"Loading draft model: {draft_dir} ({dtype_str}) ...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(draft_dir), dtype=torch_dtype
    ).to(device).eval()

    draft_vocab = draft_model.config.vocab_size
    print(f"Draft vocab size: {draft_vocab}")
    print(f"Target vocab size: {target_model.config.vocab_size}")

    return tokenizer, target_model, draft_model, draft_vocab


def eval_cluster_position0(
    tokenizer, target_model, draft_model, draft_vocab,
    flan_file, max_samples, topk_K, device,
):
    """Compute per-sample overlap_area at position 0 for one cluster."""
    records = []
    with open(flan_file) as f:
        for line in f:
            records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break

    overlaps = []
    skipped = 0

    for rec in records:
        prompt_ids = tokenizer.encode(
            rec["inputs"], add_special_tokens=True,
            truncation=True, max_length=2047,
        )
        trunk_ids = tokenizer.encode(rec["targets"], add_special_tokens=False)
        if not trunk_ids:
            skipped += 1
            continue

        full_ids = prompt_ids + trunk_ids[:1]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)
        gen_start = len(prompt_ids)

        with torch.no_grad():
            # Target
            t_logits = target_model(
                input_ids=input_ids, attention_mask=attn_mask
            ).logits
            t_probs = torch.softmax(
                t_logits[0, gen_start - 1].float(), dim=-1
            )
            t_topk = torch.topk(t_probs, k=topk_K)
            t_ids = t_topk.indices.cpu().numpy().astype(np.int64)
            t_probs_np = t_topk.values.cpu().numpy().astype(np.float32)

            if int(t_ids[0]) >= draft_vocab:
                skipped += 1
                continue

            t_sum = t_probs_np.sum()
            if t_sum > 0:
                t_probs_np = t_probs_np / t_sum

            # Draft
            input_ids_safe = input_ids.clone()
            input_ids_safe[input_ids_safe >= draft_vocab] = 0

            d_logits = draft_model(
                input_ids=input_ids_safe, attention_mask=attn_mask
            ).logits
            d_probs = torch.softmax(
                d_logits[0, gen_start - 1].float(), dim=-1
            )

            safe_ids = np.where(
                (t_ids >= 0) & (t_ids < draft_vocab), t_ids, 0
            )
            d_at_target = d_probs[
                torch.tensor(safe_ids, device=d_probs.device)
            ].cpu().numpy().astype(np.float32)

            oob = (t_ids < 0) | (t_ids >= draft_vocab)
            d_at_target[oob] = 0.0

            d_sum = d_at_target.sum()
            if d_sum > 0:
                d_aligned = d_at_target / d_sum
            else:
                d_aligned = np.full(topk_K, 1.0 / topk_K, dtype=np.float32)

            oa = float(np.minimum(d_aligned, t_probs_np).sum())
            overlaps.append(oa)

    mean_oa = float(np.mean(overlaps)) if overlaps else 0.0
    std_oa = float(np.std(overlaps)) if overlaps else 0.0
    return {
        "mean": round(mean_oa, 4),
        "std": round(std_oa, 4),
        "n_samples": len(overlaps),
        "n_skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify overlap_area at position 0 for baseline drafter"
    )
    parser.add_argument(
        "--target", default="TurboSparse-Mistral-Instruct",
        help="Target model directory"
    )
    parser.add_argument(
        "--draft", default="Lite-Mistral-150M-v2-Instruct",
        help="Draft model directory"
    )
    parser.add_argument(
        "--cluster", default=None,
        help="Single cluster name (e.g. aeslc_10templates)"
    )
    parser.add_argument(
        "--all-clusters", action="store_true",
        help="Run on all flan validation clusters"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per cluster (None = all)"
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--topk-K", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file (default: print only)"
    )
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / args.target
    draft_dir = PROJECT_ROOT / args.draft

    tokenizer, target_model, draft_model, draft_vocab = load_models(
        str(target_dir), str(draft_dir), args.dtype, args.device
    )

    flan_dir = PROJECT_ROOT / "flan" / "validation"
    if args.all_clusters:
        files = sorted(flan_dir.glob("*_validation.jsonl"))
    elif args.cluster:
        f = flan_dir / f"{args.cluster}_validation.jsonl"
        if not f.exists():
            sys.exit(f"Not found: {f}")
        files = [f]
    else:
        # Default: all clusters
        files = sorted(flan_dir.glob("*_validation.jsonl"))

    print(f"\nEvaluating {len(files)} cluster(s), topk_K={args.topk_K}, "
          f"dtype={args.dtype}, draft={args.draft}")
    print("=" * 60)

    results = {}
    all_means = []
    t_start = time.time()

    for flan_file in files:
        cluster = flan_file.stem.replace("_validation", "").replace(
            "_10templates", ""
        )
        r = eval_cluster_position0(
            tokenizer, target_model, draft_model, draft_vocab,
            str(flan_file), args.max_samples, args.topk_K, args.device,
        )
        results[cluster] = r
        all_means.append(r["mean"])
        print(f"  {cluster:40s}  mean={r['mean']:.4f}  std={r['std']:.4f}  "
              f"n={r['n_samples']}  skip={r['n_skipped']}")

    elapsed = time.time() - t_start
    global_mean = float(np.mean(all_means)) if all_means else 0.0
    global_std = float(np.std(all_means)) if all_means else 0.0

    print("=" * 60)
    print(f"Overall: mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"Reference (first 1 token): baseline mean=0.373, std=0.181")
    print(f"Elapsed: {elapsed:.1f}s")

    output = {
        "config": {
            "target": args.target,
            "draft": args.draft,
            "dtype": args.dtype,
            "topk_K": args.topk_K,
            "max_samples": args.max_samples,
            "device": args.device,
        },
        "summary": {
            "mean_of_cluster_means": round(global_mean, 4),
            "std_of_cluster_means": round(global_std, 4),
            "n_clusters": len(results),
        },
        "reference": {
            "first_1_token": {"baseline_mean": 0.373, "baseline_std": 0.181},
            "first_100_tokens": {"baseline_mean": 0.480, "baseline_std": 0.159},
        },
        "per_cluster": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}")
    else:
        print(f"\nUse --output <file.json> to save results")


if __name__ == "__main__":
    main()
