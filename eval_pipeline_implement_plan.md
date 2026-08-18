# Eval Pipeline — Implementation Plan

## Goal

Measure distributional similarity between draft and target models per trunk
position, as a proxy for Speculative Decoding acceptance rate. Configurable
via Hydra, logged to MLflow.

Two main modes:

1. **Target source = `dataset`** — read target's top-K from synthetic data files
   (cheap, fast, K is whatever was captured during generation, usually 10).
2. **Target source = `model`** — load target locally and compute its full
   distribution at each position on the fly (slow but exact).

## Methodology

* **Teacher forcing.** For each sample, feed `prompt + trunk[:N]` to the draft.
  Take the logits at positions `prompt_len-1 .. prompt_len+N-2` — these
  predict `trunk[0..N-1]`. Same recipe for target in mode `model`.
* **First N positions per sample**, default `N=100`. If `len(trunk) < N`, mask
  the missing positions out of the per-position average.
* **Per-position dataset average**: result file holds `array[N]` per
  (aggregation × metric × mode) cell.

## Config (`configs/eval.yaml`)

```yaml
draft_models:
  - tiny-mixtral           # tried locally first, then s3://<bucket>/models/<name>

target:
  path: TurboSparse-Mistral-Instruct
  trust_remote_code: true
  dtype: bfloat16

s3:
  bucket: domain-aware-sd
  models_prefix: models

datasets:
  - data/synthetic/v1/aeslc_10templates.jsonl

target_source: dataset        # "dataset" | "model"
n_positions: 100              # N
batch_size: 8
max_samples_per_file: null    # null = all

# Inference backend: HuggingFace transformers only. vLLM is intentionally not
# used in this pipeline (it stays in the data-generation path). On MPS the
# bfloat16 dtype is silently downgraded to float16 (MPS does not support bf16).
device: cuda                  # "cuda" | "cpu" | "mps"
dtype: bfloat16               # bfloat16 | float16 | float32

metrics:                      # multiple allowed; overlap_area is the canonical one
  - overlap_area
  - top1_match
  - topk_overlap
  - kl

# Only applied when len(draft_models) > 1. Each strategy is computed
# independently and written as a separate result row.
aggregations:
  - best
  - arithmetic_average
  - geometric_average
  - softmax_sum
  - softmax_hadamard
  - most_confident

# K used by topk_overlap when running in full-target mode (in dataset mode
# K is whatever the dataset provides per position).
topk_K: 10

output:
  dir: eval_results

mlflow:
  tracking_uri: ./mlruns
  experiment_name: eval_drafters
  run_name: null              # auto from models + dataset
```

## Output JSON

**Filename**: `eval__<draft_model_tag>__<dataset_basename>.json`
* `draft_model_tag` = single model name, or `multi-<hash6>` when >1 model
  (full list is in metadata).
* `dataset_basename` = filename stem (e.g. `aeslc_10templates`).

**Schema**:

```json
{
  "metadata": {
    "draft_models": ["tiny-mixtral"],
    "target_model": "TurboSparse-Mistral-Instruct",
    "target_source": "dataset",
    "dataset": "data/synthetic/v1/aeslc_10templates.jsonl",
    "n_positions": 100,
    "n_samples_total": 1234,
    "n_samples_per_position": [1234, 1234, 1233, "..."],
    "n_skipped_per_position": [0, 0, 1, "..."],
    "n_special_target_per_position": [0, "..."],
    "top_k_from_dataset": 10,
    "target_renormalized": true,
    "metrics": ["overlap_area", "top1_match", "topk_overlap", "kl"],
    "aggregations": ["single"],
    "config_snapshot": { "...": "full Hydra config..." },
    "timestamp": "2026-06-30T12:34:56Z"
  },
  "results": {
    "single__overlap_area__top10_from_dataset": [0.87, 0.85, "..."],
    "single__top1_match__top10_from_dataset":   [0.91, 0.88, "..."],
    "single__topk_overlap__top10_from_dataset": [0.74, 0.70, "..."],
    "single__kl__top10_from_dataset":           [0.31, 0.34, "..."]
  }
}
```

**Field naming**: `<aggregation>__<metric>__<mode>` where:
* `aggregation`: `single` (when 1 draft), or one of the configured aggregations
  (`best`, `arithmetic_average`, etc.).
* `metric`: `overlap_area`, `top1_match`, `topk_overlap`, `kl`.
* `mode`: `top{K}_from_dataset` (dataset target) or `full_target` (model target).

## Module layout

```
src/eval/
├── __init__.py
├── main.py                # Hydra entry point
├── model_loader.py        # local→S3 fallback for drafters
├── metrics.py             # Metric registry — see contract below
├── aggregations.py        # Aggregation registry — see contract below
├── target_provider.py     # DatasetTargetProvider | ModelTargetProvider
├── draft_runner.py        # batched teacher-forcing draft inference
├── output_writer.py       # JSON output with the naming convention above
└── mlflow_logger.py       # params, timeseries metrics, artifacts

configs/
└── eval.yaml

tests/eval/
├── __init__.py
├── test_metrics.py
├── test_aggregations.py
├── test_model_loader.py
├── test_output_writer.py
└── test_runner_smoke.py   # end-to-end on tiny fixture
```

## Algorithm (per dataset file)

```
1. Build SpecDecDataset(file, tokenizer, mode="distillation",
                        max_gen_length=N).
2. Resolve all draft model paths (local → S3 download).
3. If target_source=="model": load target.
4. For each sample (batched):
   a. Run each draft via teacher forcing → draft_logp[m]: [B, N, V_d]
   b. Per position, build a "draft restricted distribution":
        - dataset mode: restrict to target's top-K indices, renormalize
        - model mode:   keep full V_d
   c. Get target distribution:
        - dataset mode: top10_ids/probs (renormalized within K)
        - model mode:   target_logp at the same positions, full V_t
   d. For each aggregation × metric:
        - aggregation merges draft distributions across M models → 1 dist
          (the "best" aggregation is special: compute metric M times,
           take max across models per position per sample)
        - compute metric(draft_agg, target) → scalar per sample-position
   e. Accumulate per-position sums + sample counts (mask invalid positions)
5. Per-position averages → results[key] arrays of length N.
6. Write JSON + log MLflow.
```

## Metrics

All metrics receive `target` as either a full distribution or a `TopK`
struct (`ids: [K]`, `probs: [K]`), and `draft` as either a full distribution
or values aligned to the same `ids`.

| Metric | Full mode | Dataset (top-K) mode |
|---|---|---|
| `overlap_area`  | `Σ_v min(p_d, p_t)` | Both renormalized within target's top-K ids; `Σ_k min(p_d_k, p_t_k)` |
| `top1_match`    | `argmax(p_d) == argmax(p_t)` | argmax(target) = top-K id at index 0; argmax(draft) over full vocab |
| `topk_overlap`  | `|topK(p_d) ∩ topK(p_t)| / K`, K from config | K from dataset; draft topK from full vocab |
| `kl`            | `Σ_v p_t (log p_t − log p_d)` | Sum over target's top-K ids only (target zero elsewhere); add tiny eps to draft for log |

Registry pattern: `METRICS = {name: callable(draft, target, mode_info)}`.

## Aggregations

Each aggregation takes `K` distributions of identical shape `[V or K]` and
returns one distribution of the same shape. They are applied **after**
restricting to target's top-K in dataset mode.

| Aggregation | Definition |
|---|---|
| `best` | Special — see algorithm step 4d (per-position max metric across models) |
| `arithmetic_average` | `mean(p_k)` |
| `geometric_average`  | `exp(mean(log p_k))` then renormalize |
| `softmax_sum`        | `softmax(Σ p_k)` — literal sum of probabilities, then softmax |
| `softmax_hadamard`   | `softmax(Π p_k)` — literal product, then softmax |
| `most_confident`     | Return the distribution whose `max(p)` is largest |

Notes:
* `softmax_sum` / `softmax_hadamard` are **literal** per the spec (softmax is
  applied to probabilities, not logits). They behave like sharpening / mixing
  transforms. If the user wants logit-space sum, that is `geometric_average`.

Registry pattern: `AGGREGATIONS = {name: callable(list_of_dists) → dist}`.

## Model loading (`model_loader.py`)

```
def resolve_model_dir(name: str, s3_prefix: str = "models") -> Path:
    1. local = PROJECT_ROOT / name
    2. if local.exists(): return local
    3. download from s3://<bucket>/<s3_prefix>/<name>/ into local
       (reusing helpers from src/download_from_s3.py)
    4. raise if S3 also has nothing
```

Target model is assumed to be present locally — no S3 fallback for it,
just a clear error if missing.

## Inference (`draft_runner.py`, `target_provider.py`)

HuggingFace `transformers` only. Single batched forward pass per model; we
slice logits at positions `prompt_len-1 .. prompt_len+N-2` and softmax.

vLLM is intentionally not used in this pipeline:
* The eval (val) split is small enough that HF is fast enough.
* vLLM has known compatibility friction on the current server stack — we
  prefer to keep that battle on the generation path only.

In dataset mode we extract draft probabilities at the target's top-K ids
directly from the full softmax; in model mode we also extract target's
top-K from its full softmax (the full distribution is materialized but not
stored — we only keep top-K).

For MPS (Mac) we silently downgrade `bfloat16` → `float16`. Use this path
for very small local sanity tests (e.g. tiny-mixtral on a handful of
samples). Real eval runs go on the GPU server.

## Edge cases

* **Trunk shorter than N**: mask, decrement `n_samples_per_position[i]`.
* **Dataset skipped position** (target top10 == `[]`): mask, increment
  `n_skipped_per_position[i]`.
* **Vocab mismatch** (draft 32000, target 32064): if target's argmax falls in
  ids ≥ 32000 (chat tokens), draft cannot predict it. Mask the position and
  increment `n_special_target_per_position[i]`.
* **Token id outside draft vocab in top-K**: clamp prob to 0, log a counter.
* **One draft model**: aggregations are skipped entirely; key prefix is
  `single` (not the aggregation name).

## MLflow

* **Params**: all flattened Hydra config keys.
* **Timeseries metrics** (`step` = position 0..N-1): one per
  `<agg>__<metric>__<mode>` key.
* **Scalars**: `<key>__mean_all`, `<key>__mean_first10`, `<key>__mean_last10`.
* **Artifacts**: the JSON output and the resolved config snapshot.

## Tests

* `test_metrics.py` — toy distributions; check exact values for
  `overlap_area`, `top1_match`, `topk_overlap`, `kl` in both modes.
* `test_aggregations.py` — toy lists of distributions; verify shapes,
  normalization, and known-output cases (e.g. arithmetic mean of identical
  dists = the dist).
* `test_model_loader.py` — mock S3 client; verify local-hit short-circuit
  and S3 download path.
* `test_output_writer.py` — round-trip key naming; metadata fields present.
* `test_runner_smoke.py` — tiny dummy dataset (5 records, N=10) + a stub
  draft model that returns uniform distributions; verifies pipeline runs
  end-to-end and produces a well-formed JSON.

## Implementation order

1. Plan file (this).
2. `configs/eval.yaml`.
3. `src/eval/__init__.py` (empty).
4. `src/eval/metrics.py` + `tests/eval/test_metrics.py`.
5. `src/eval/aggregations.py` + `tests/eval/test_aggregations.py`.
6. `src/eval/model_loader.py` + `tests/eval/test_model_loader.py`.
7. `src/eval/output_writer.py` + `tests/eval/test_output_writer.py`.
8. `src/eval/target_provider.py`.
9. `src/eval/draft_runner.py`.
10. `src/eval/mlflow_logger.py`.
11. `src/eval/main.py`.
12. `tests/eval/test_runner_smoke.py`.
13. `pytest` — must be green; report what was skipped (CUDA-dependent).

## Out of scope

* Caching target distributions back into the dataset (user said no).
* Real SD acceptance rate via rejection sampling (use `overlap_area` as
  the proxy; mathematically `1 − TVD = Σ min(p, q)` is exactly the
  acceptance probability of perfect-distribution speculative decoding).
* HuggingFace Hub downloads (always either local or S3).
