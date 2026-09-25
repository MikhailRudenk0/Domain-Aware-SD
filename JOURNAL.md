# Experiment Journal

> **Format**: each entry is max 5 lines — date, what was done, key result, and link to report/data. No long descriptions here; details go in per-experiment reports.

---

### 2026-09-23 — Metric fix: union-support overlap_area + deprecate dataset eval mode

Fixed `overlap_area` computation: now uses union of draft top-K and target top-K (each side's own probabilities, zeros elsewhere). Old code renormalized draft onto target's support, inflating overlap_area by ~3x for untrained drafters. Dataset eval mode (`target_source: dataset`) deprecated — always use `target_source: model` (full target forward pass). Correlation between modes improved from r=0.34 to r=0.68 (overlap_area).
Files changed: `src/eval/evaluator.py`, `src/eval/metrics.py`, `src/eval/target_provider.py`, `src/eval/draft_runner.py`.

### 2026-09-23 — Re-eval base drafter on 66 Flan clusters (model mode vs dataset mode)

Ran base (untrained) Lite-Mistral-150M-v2-Instruct on all 66 Flan validation clusters in both modes with fixed metrics. Model mode: overlap_area mean=0.191, dataset mode: mean=0.131. Pearson r: overlap_area=0.68, topk_overlap=0.87, KL=0.92.
Script: `scripts/compare_model_vs_dataset.py`. Results: `eval_results/model_mode_v2/`, `eval_results/*.json`.

### 2026-09-23 — Re-eval WMT16 per-cluster drafters (model mode, fixed metrics)

Re-ran 7 drafters (5 per-cluster + text_reformulation + base) x 5 WMT16 clusters with `ModelTargetProvider` and fixed union-support metrics. Previous run (2026-09-11) used dataset mode.
Report: `results/REPORT_wmt16_per_cluster_drafters.pdf`. Data: `eval_results/wmt16_per_cluster/`.

### 2026-09-11 — WMT16 per-cluster eval (dataset mode, OLD metrics)

Evaluated 6 drafters x 5 WMT16 clusters using `DatasetTargetProvider`. Results are superseded by 2026-09-23 re-eval (model mode + fixed metrics). Kept for reference only.
Report: `results/REPORT_wmt16_per_cluster_drafters.md`. Data: `eval_results/wmt16_per_cluster/` (overwritten by 2026-09-23 run).

### 2026-09-10..11 — Train 5 WMT16 per-cluster drafters

Trained 5 domain-specific drafters on WMT16 translation clusters (ruen, csen, fien, tren, deen), 28.5K samples each, 10 epochs, 1x RTX 3090. Best: tren (54.91% top-1 acc, eval_loss 1.998). All plateau by epoch 5-6.
Report: `results/REPORT_wmt16_per_cluster_drafters.md`. Checkpoints: `/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_wmt16_translate_*/final`.

### 2026-08-29 — 3x3 AR matrix (3 drafters x 3 clusters)

Cross-evaluated text_reformulation, mixed_ut, and understanding drafters on 3 Flan meta-clusters. Text Reformulation drafter generalizes best across domains.
Data: `eval_results/ar_3x3/`.

### 2026-08 — Train Text Reformulation drafter (11 clusters, 327K samples)

Fine-tuned Lite-Mistral-150M-v2-Instruct on 11 Text Reformulation clusters. Final: eval_loss 2.151, top-1 acc 54.34%.
Checkpoint: `/media/public/rudenko/projects/Domain-Aware-SD/outputs/drafter_text_reformulation/final`.

### 2026-07-03 — First eval of tiny-mixtral on 66 clusters (dataset mode, OLD metrics)

Initial baseline eval of untrained tiny-mixtral on all 66 Flan clusters. Used dataset mode with old (broken) metric computation. Results deleted 2026-09-23 (were in `eval_results/true/`).
