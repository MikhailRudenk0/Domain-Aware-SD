# Synthetic Dataset Format

Reference for the on-disk and in-memory schemas produced by `src/generate_synthetic_data.py` and consumed by `src/data/SpecDecDataset`.

There are **two on-disk storage formats** (JSONL and NPZ) that hold the **same logical record**. The reader (`SpecDecDataset`) loads both transparently and yields identical Python/Tensor objects.

---

## 1. Logical record

One record per generated sample. Field-by-field:

| Field | Type | Required | Description |
|------|------|---|---|
| `cluster` | string | yes | Cluster name (e.g. `aeslc_10templates`). Identifies the domain partition. |
| `prompt` | string | yes | The exact Flan `inputs` field that was fed to the target model. |
| `reference` | string | (often present) | The original Flan `targets` field. Not used in training; kept for diagnostics. |
| `trunk` | `list[int]` | yes | Token IDs sampled from the target — the "primary" sequence (top-p sampling, length ≤ `generation.max_new_tokens`). |
| `top10_ids` | `list[list[int]]` | yes | One row per trunk position. Each row is up to K=10 token IDs sorted by probability descending. May be `[]` (see *Skipped positions*). |
| `top10_probs` | `list[list[float]]` | yes | Parallel to `top10_ids`. Probabilities in `[0, 1]`, rounded to 3 decimal places. Same row length and same skip semantics as `top10_ids`. |

**Alignment invariant**: `len(top10_ids) == len(top10_probs) == len(trunk)`. Row `i` describes the target's distribution at the step that emitted `trunk[i]`. Probs do not necessarily sum to 1.0 — they are the top-K marginals of a 32k-token softmax, plus 3-d.p. rounding.

### Skipped positions

A row in `top10_ids` / `top10_probs` may be empty (`[]`). This signals that **top-K was not captured at that step**, while `trunk[i]` is still the actual sampled token. Two causes:

1. **Normal mode** — `generation.logprobs=0`. No top-K is captured anywhere; every row is empty (or both fields are absent / empty at the record level).
2. **Skip threshold** — `generation.skip_top10_above_prob=p`. At positions where the target's top-1 prob exceeded `p`, the distribution is not stored. The trunk token is still emitted and kept.

At training/inference time, skipped rows materialize as all-zero `[10]` slices in the tensor. Any KL-style distillation loss should mask these (a row of zeros has zero prob mass, which corrupts a naive KL).

### Backward compatibility

Legacy records with the previous schema:
```jsonc
{
  "trunk": [...],
  "top10": [
    [{"token_id": 1234, "prob": 0.45, "token": "The"}, ...],
    ...
  ]
}
```
are auto-normalized to the new schema at load time:
- `top10` → `top10_ids` + `top10_probs` parallel arrays
- The `"token"` decoded string is dropped (unused downstream)

The legacy schema is read-only — the writer never produces it.

---

## 2. JSONL format (`.jsonl`)

Recommended for: validation / test splits, ad-hoc inspection, debugging.

One JSON record per line, UTF-8, no trailing whitespace. Example (formatted for readability, in the actual file it's one line):

```json
{
  "cluster":     "aeslc_10templates",
  "prompt":      "Write a subject for the following email: ...",
  "reference":   "Re: meeting on Thursday",
  "trunk":       [1234, 567, 89, 200, 2],
  "top10_ids":   [
                   [1234, 99, 12, 4, 8, 15, 16, 23, 42, 77],
                   [],
                   [89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
                   [200, 201, 5, 6, 7, 8, 9, 10, 11, 12],
                   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                 ],
  "top10_probs": [
                   [0.45, 0.20, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01],
                   [],
                   [0.32, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02],
                   [0.50, 0.25, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01],
                   [0.90, 0.04, 0.02, 0.01, 0.005, 0.005, 0.003, 0.002, 0.001, 0.001]
                 ]
}
```

Empty row at index 1 above means the top-1 prob at that step exceeded `skip_top10_above_prob` (or `logprobs=0` for the whole run).

---

## 3. NPZ format (`.npz`)

Recommended for: train split. ~10× smaller than JSONL on real generations (uint16 ids and uint16 quantized probs beat the text representation by a large margin; gzip compression on top).

One `.npz` file per cluster (`<cluster_name>.npz`). It is a `np.savez_compressed` archive with these arrays:

| Array | dtype | Shape | Meaning |
|------|-------|-------|---------|
| `cluster` | `str_` | scalar | Cluster name (same for all records in the file). |
| `prompts` | `object` | `[N]` | UTF-8 strings, one per sample. |
| `references` | `object` | `[N]` | UTF-8 strings, one per sample. |
| `trunk_lens` | `int32` | `[N]` | Length of each sample's trunk. Lets you slice the concatenated arrays. |
| `trunk_ids` | `uint16` | `[T]` where `T = sum(trunk_lens)` | All trunks concatenated. |
| `top10_ids` | `uint16` | `[T, K]` where K=10 (or 0 in normal mode) | Top-K token IDs per position. Rows where `top10_mask=False` are all zeros. |
| `top10_probs_q` | `uint16` | `[T, K]` | Top-K probs **quantized to integers in `[0, 1000]`** (= `round(prob * 1000)`). Divide by 1000.0 to recover the rounded float. |
| `top10_mask` | `bool` | `[T]` | `True` at positions where top-K was captured; `False` for skipped/normal-mode positions. The reader uses this to produce `[]` rows in the logical record. |
| `top_k` | `uint8` | scalar | K (= 10 in practice, 0 in normal mode). |

### Reconstructing a sample

```python
import numpy as np
ar = np.load("aeslc_10templates.npz", allow_pickle=True)
trunk_lens = ar["trunk_lens"]
offsets = np.concatenate([[0], np.cumsum(trunk_lens)])
i = 0   # sample index
start, end = offsets[i], offsets[i + 1]

prompt    = str(ar["prompts"][i])
reference = str(ar["references"][i])
trunk     = ar["trunk_ids"][start:end].astype(int).tolist()
mask      = ar["top10_mask"][start:end]
ids       = ar["top10_ids"][start:end]                   # [trunk_len, K], uint16
probs     = ar["top10_probs_q"][start:end].astype(np.float32) / 1000.0
```

Rows where `mask[j] == False` should be treated as "no top-K data here" — both `ids[j]` and `probs[j]` are all-zero.

### Why uint16

- Target vocab is 32064 (≤ 65535), so token IDs fit in uint16 cleanly.
- Probs are already rounded to 3 decimal places at generation time, so the quantized range `[0, 1000]` is *exact*: no information is lost by going through `uint16` vs `float32`.
- Halves the size compared to `int32` / `float32` and is the same speed in practice.

---

## 4. Reader API (`SpecDecDataset`)

```python
from src.data import SpecDecDataset, DistillationCollator

# Single file, list of files, or recursive directory scan (mixes .jsonl + .npz)
ds = SpecDecDataset("data/synthetic/v1/aeslc.jsonl", tokenizer)
ds = SpecDecDataset.from_dir("data/synthetic/v1", tokenizer, mode="distillation")
```

`__getitem__(i)` returns a dict:

| Key | Shape / type | Description |
|-----|------|-------------|
| `input_ids` | `LongTensor [L]` | `prompt_ids + trunk_ids`, truncated to `max_length`. |
| `attention_mask` | `LongTensor [L]` | all ones. |
| `labels` | `LongTensor [L]` | `-100` on prompt positions, trunk token IDs on generated positions. |
| `gen_start` | `int` | Index where generation begins inside `input_ids`. |
| `cluster` | `str` | Cluster name. |
| `top10_ids` | `LongTensor [gen_len, 10]` | Distillation mode only, present iff at least one row carries data. Skipped rows are all zeros. |
| `top10_probs` | `FloatTensor [gen_len, 10]` | Same. |

`DistillationCollator` pads to `[B, max_gen_len, 10]` with zeros. Samples in the batch that lack `top10_ids` (normal-mode rows) are zero-padded for the whole batch slot — i.e. they contribute zero KL loss naturally.

---

## 5. Picking format at generation time

In `configs/generation.yaml`:

```yaml
output:
  format: jsonl     # or "npz"

generation:
  logprobs: 10                  # 0 → normal mode (no top-K)
  skip_top10_above_prob: 1.0    # < 1.0 → drop top-K at confident positions
```

Or on the CLI (Hydra):

```bash
python src/generate_synthetic_data.py output.format=npz
python src/generate_synthetic_data.py generation.logprobs=0
python src/generate_synthetic_data.py generation.skip_top10_above_prob=0.95
```

Both formats can coexist in the same `data/synthetic/<version>/` directory and `SpecDecDataset.from_dir` will pick up all of them.
