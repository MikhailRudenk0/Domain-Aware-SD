#!/usr/bin/env python
"""
Evaluate 7 draft models × 5 WMT16 translation clusters (model mode).

Models:
  1–5. Per-cluster WMT16 drafters (ruen, csen, fien, tren, deen)
  6.   text_reformulation (baseline from previous training round)
  7.   base (untrained Lite-Mistral-150M-v2-Instruct)

Uses ModelTargetProvider (full target forward pass) — the canonical eval mode.
Dataset mode (DatasetTargetProvider) is DEPRECATED since 2026-09-23; see JOURNAL.md.

Clusters (validation, 200 samples each):
  wmt16_translate_{ruen,csen,fien,tren,deen}_10templates

Metrics: overlap_area, top1_match, topk_overlap, kl

Output: eval_results/wmt16_per_cluster/ with JSON per pair and a summary CSV.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import SpecDecDataset
from src.eval.draft_runner import DraftRunner
from src.eval.evaluator import evaluate_dataset
from src.eval.target_provider import ModelTargetProvider
from src.eval.output_writer import mode_label

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "wmt16_ruen": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_ruen/final",
    "wmt16_csen": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_csen/final",
    "wmt16_fien": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_fien/final",
    "wmt16_tren": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_tren/final",
    "wmt16_deen": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_deen/final",
    "text_reformulation": "/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_text_reformulation/final",
    "base_untrained": str(PROJECT_ROOT / "Lite-Mistral-150M-v2-Instruct"),
}

CLUSTERS = [
    "wmt16_translate_ruen_10templates",
    "wmt16_translate_csen_10templates",
    "wmt16_translate_fien_10templates",
    "wmt16_translate_tren_10templates",
    "wmt16_translate_deen_10templates",
]

VAL_DIR = PROJECT_ROOT / "data/synthetic/validation/v3"
TARGET_DIR = PROJECT_ROOT / "TurboSparse-Mistral-Instruct"

N_POSITIONS = 100
BATCH_SIZE = 8
DEVICE = "cuda"
DTYPE = "bfloat16"
METRICS = ["overlap_area", "top1_match", "topk_overlap", "kl"]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_mean(results: dict, metric_substr: str) -> float:
    for key, values in results.items():
        if metric_substr in key:
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

    # Model mode: full target forward pass (canonical eval mode).
    print("Loading target model for model-mode evaluation...")
    target_provider = ModelTargetProvider(
        model_dir=TARGET_DIR,
        device=DEVICE,
        dtype=DTYPE,
        trust_remote_code=True,
        topk_K=10,
    )

    output_dir = PROJECT_ROOT / "eval_results" / "wmt16_per_cluster"
    output_dir.mkdir(parents=True, exist_ok=True)

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

        for cluster in CLUSTERS:
            val_file = VAL_DIR / f"{cluster}.jsonl"
            if not val_file.exists():
                print(f"  [skip] missing: {val_file}")
                continue

            # Short name for cluster (e.g., "ruen")
            cluster_short = cluster.replace("wmt16_translate_", "").replace("_10templates", "")

            print(f"\n  ── {model_name} × {cluster_short} ──")

            dataset = SpecDecDataset(
                val_file,
                tokenizer=tokenizer,
                mode="distillation",
                max_length=2048,
                max_gen_length=N_POSITIONS,
            )

            mode_str = mode_label("model", None)

            result = evaluate_dataset(
                dataset=dataset,
                draft_runners=[runner],
                target_provider=target_provider,
                n_positions=N_POSITIONS,
                batch_size=BATCH_SIZE,
                pad_token_id=pad_id,
                metrics=METRICS,
                aggregations=[],
                mode_label_str=mode_str,
                draft_vocab_size=draft_vocab_size,
                max_samples=None,
            )

            oa = _extract_mean(result.results, "overlap_area")
            t1 = _extract_mean(result.results, "top1_match")
            tk = _extract_mean(result.results, "topk_overlap")
            kl = _extract_mean(result.results, "__kl__")

            print(f"    samples={result.n_samples_total}, "
                  f"overlap_area={oa:.4f}, top1_match={t1:.4f}, "
                  f"topk_overlap={tk:.4f}, kl={kl:.4f}")

            row = {
                "drafter": model_name,
                "eval_cluster": cluster_short,
                "n_samples": result.n_samples_total,
                "overlap_area": round(oa, 4),
                "top1_match": round(t1, 4),
                "topk_overlap": round(tk, 4),
                "kl": round(kl, 4),
            }
            summary_rows.append(row)

            # Save per-pair JSON
            pair_out = output_dir / f"{model_name}_on_{cluster_short}.json"
            with open(pair_out, "w") as f:
                json.dump({
                    **row,
                    "drafter_path": model_path,
                    "cluster_full": cluster,
                    "n_positions": N_POSITIONS,
                    "results": {k: v for k, v in result.results.items()},
                }, f, indent=2, default=str)
            print(f"    wrote {pair_out}")

        del runner
        torch.cuda.empty_cache()

    # Free target model GPU memory
    target_provider.free()
    torch.cuda.empty_cache()

    # Write summary CSV
    csv_path = output_dir / "summary_wmt16.csv"
    fieldnames = ["drafter", "eval_cluster", "n_samples",
                  "overlap_area", "top1_match", "topk_overlap", "kl"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*70}")
    print(f"Summary written to {csv_path}")
    print(f"{'='*70}")

    # Print table
    print(f"\n{'='*90}")
    hdr = f"{'Drafter':<22} {'Cluster':<8} {'N':>5} {'OA':>8} {'Top1':>8} {'TopK':>8} {'KL':>8}"
    print(hdr)
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['drafter']:<22} {r['eval_cluster']:<8} {r['n_samples']:>5} "
              f"{r['overlap_area']:>8.4f} {r['top1_match']:>8.4f} "
              f"{r['topk_overlap']:>8.4f} {r['kl']:>8.4f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
