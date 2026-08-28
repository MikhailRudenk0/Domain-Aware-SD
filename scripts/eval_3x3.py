#!/usr/bin/env python
"""
Evaluate 3 draft models × 3 domain splits on validation data.

Models:
  1. drafter_understanding   — trained on Understanding (21 clusters)
  2. drafter_text_reformulation — trained on Text Reformulation (11 clusters)
  3. drafter_mixed_ut        — trained on Understanding + Text Reformulation (32 clusters)

Domains (validation):
  1. Understanding  — 21 cluster val files
  2. Text Reformulation — 11 cluster val files
  3. Mixed (U+T)    — 32 cluster val files

Output: eval_results/ar_3x3/ with JSON per (model, domain) and a summary CSV.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import SpecDecDataset
from src.eval.draft_runner import DraftRunner
from src.eval.evaluator import evaluate_dataset
from src.eval.target_provider import DatasetTargetProvider
from src.eval.output_writer import mode_label

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "understanding": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_understanding/checkpoint-52803",
    "text_reformulation": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_text_reformulation/final",
    "mixed_ut": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_mixed_ut/final",
}

CLUSTER_CONFIGS = {
    "understanding": PROJECT_ROOT / "configs/clusters/understanding.json",
    "text_reformulation": PROJECT_ROOT / "configs/clusters/text_reformulation.json",
    "mixed_ut": PROJECT_ROOT / "configs/clusters/mixed_ut.json",
}

VAL_DIR = PROJECT_ROOT / "data/synthetic/validation/v3"
TARGET_DIR = PROJECT_ROOT / "TurboSparse-Mistral-Instruct"

N_POSITIONS = 100
BATCH_SIZE = 8
DEVICE = "cuda"
DTYPE = "bfloat16"

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_clusters(path: Path) -> List[str]:
    with open(path) as f:
        return json.load(f)


def get_val_files(clusters: List[str], val_dir: Path) -> List[Path]:
    files = []
    for c in clusters:
        p = val_dir / f"{c}.jsonl"
        if p.exists():
            files.append(p)
        else:
            print(f"  [warn] missing validation file: {p}")
    return files


def mean_overlap(results: dict, n_positions: int) -> float:
    """Extract mean overlap_area across positions."""
    for key, values in results.items():
        if "overlap_area" in key:
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else 0.0
    return 0.0


def mean_top1_match(results: dict) -> float:
    for key, values in results.items():
        if "top1_match" in key:
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else 0.0
    return 0.0


def mean_kl(results: dict) -> float:
    for key, values in results.items():
        if "__kl__" in key:
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else 0.0
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    print("Loading target tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TARGET_DIR), trust_remote_code=True
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Target provider: dataset mode (use top-10 from JSONL)
    target_provider = DatasetTargetProvider()

    output_dir = PROJECT_ROOT / "eval_results" / "ar_3x3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_rows: List[dict] = []

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Loading draft model: {model_name} ({model_path})")
        print(f"{'='*70}")

        runner = DraftRunner(
            model_dir=Path(model_path),
            device=DEVICE,
            dtype=DTYPE,
            topk_K=10,
            trust_remote_code=False,
        )
        draft_vocab_size = runner.vocab_size

        for domain_name, cluster_config in CLUSTER_CONFIGS.items():
            clusters = load_clusters(cluster_config)
            val_files = get_val_files(clusters, VAL_DIR)

            if not val_files:
                print(f"  [skip] no val files for domain {domain_name}")
                continue

            print(f"\n  ── {model_name} × {domain_name} ({len(val_files)} files) ──")

            # Load all val files for this domain as one dataset
            dataset = SpecDecDataset(
                val_files,
                tokenizer=tokenizer,
                mode="distillation",
                max_length=2048,
                max_gen_length=N_POSITIONS,
            )

            mode_str = mode_label("dataset", 10)

            result = evaluate_dataset(
                dataset=dataset,
                draft_runners=[runner],
                target_provider=target_provider,
                n_positions=N_POSITIONS,
                batch_size=BATCH_SIZE,
                pad_token_id=pad_id,
                metrics=["overlap_area", "top1_match", "kl"],
                aggregations=[],
                mode_label_str=mode_str,
                draft_vocab_size=draft_vocab_size,
                max_samples=None,
            )

            ar = mean_overlap(result.results, N_POSITIONS)
            t1 = mean_top1_match(result.results)
            kl = mean_kl(result.results)

            print(f"    samples={result.n_samples_total}, "
                  f"AR(overlap_area)={ar:.4f}, top1_match={t1:.4f}, KL={kl:.4f}")

            summary_rows.append({
                "drafter": model_name,
                "eval_domain": domain_name,
                "n_samples": result.n_samples_total,
                "overlap_area_mean": round(ar, 4),
                "top1_match_mean": round(t1, 4),
                "kl_mean": round(kl, 4),
            })

            # Save per-pair JSON
            pair_out = output_dir / f"{model_name}_on_{domain_name}.json"
            with open(pair_out, "w") as f:
                json.dump({
                    "drafter": model_name,
                    "drafter_path": model_path,
                    "eval_domain": domain_name,
                    "n_clusters": len(clusters),
                    "n_samples": result.n_samples_total,
                    "n_positions": N_POSITIONS,
                    "overlap_area_mean": round(ar, 4),
                    "top1_match_mean": round(t1, 4),
                    "kl_mean": round(kl, 4),
                    "results": {k: v for k, v in result.results.items()},
                }, f, indent=2, default=str)
            print(f"    wrote {pair_out}")

        # Unload model
        del runner
        torch.cuda.empty_cache()

    # Write summary CSV
    csv_path = output_dir / "summary_3x3.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "drafter", "eval_domain", "n_samples",
            "overlap_area_mean", "top1_match_mean", "kl_mean"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n{'='*70}")
    print(f"Summary written to {csv_path}")
    print(f"{'='*70}")

    # Print table
    print("\n┌─────────────────────────┬──────────────────────────┬──────────┬──────────────┬────────────┬────────┐")
    print("│ Drafter                 │ Eval Domain              │ Samples  │ Overlap Area │ Top1 Match │ KL     │")
    print("├─────────────────────────┼──────────────────────────┼──────────┼──────────────┼────────────┼────────┤")
    for r in summary_rows:
        print(f"│ {r['drafter']:<23} │ {r['eval_domain']:<24} │ {r['n_samples']:<8} │ {r['overlap_area_mean']:<12.4f} │ {r['top1_match_mean']:<10.4f} │ {r['kl_mean']:<6.4f} │")
    print("└─────────────────────────┴──────────────────────────┴──────────┴──────────────┴────────────┴────────┘")


if __name__ == "__main__":
    main()
