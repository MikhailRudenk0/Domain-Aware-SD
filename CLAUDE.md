# Domain-Aware Speculative Decoding — Project Guide

## Research Goal

Test whether **domain-specific draft models** achieve higher **acceptance rates** in speculative decoding than a single draft model trained on the full dataset.

**Hypothesis**: A draft model fine-tuned on cluster-specific data (e.g., summarization tasks) will better approximate the target model's distribution on that domain, leading to more draft tokens being accepted → faster inference.

---

## Terminology

| Term | Meaning in this project |
|------|------------------------|
| **SD** | Speculative Decoding — a lossless inference acceleration technique where a small *drafter* proposes tokens that a large *target* verifies in parallel. |
| **AR** (Acceptance Rate) | Fraction of draft tokens accepted by the target per decoding step. Higher AR → more speedup. AR = 1 means every draft token was accepted; AR = 0 means none were. |
| **Target** | **TurboSparse-Mistral-Instruct** — a 7B Mistral-based model with Bamboo architecture (sparse activations). This is the large, accurate model we accelerate. Vocab size: 32064. Located at `TurboSparse-Mistral-Instruct/`. |
| **Drafter** | Small, fast model that proposes tokens. Must share the target's tokenizer. Current candidate: **tiny-mixtral** (2-layer MixtralForCausalLM, vocab 32000 ⊂ target vocab). Located at `tiny-mixtral/`. |
| **Domain partition** | One of 66 task clusters from the Flan dataset (e.g., `aeslc_10templates`, `hellaswag`). Each cluster is a distinct NLP task with its own linguistic distribution. |
| **Trunk** | The primary generated sequence (top-1 sample at each step). |
| **Top-10** | Top-10 token probabilities captured at each generation step from the target model. Forms the training signal for draft models (knowledge distillation). |

---

## Models

### Target: TurboSparse-Mistral-Instruct
- **Path**: `TurboSparse-Mistral-Instruct/`
- **Architecture**: `BambooForCausalLM` (custom — requires `trust_remote_code=True`)
- **Size**: 7B parameters
- **Vocab**: 32064 tokens (Mistral base 32000 + 64 special tokens incl. `<|im_start|>`, `<|im_end|>`)
- **Tokenizer**: `LlamaTokenizer` (SentencePiece-based)
- **S3**: `s3://domain-aware-sd/models/TurboSparse-Mistral-Instruct/`

### Draft candidate: tiny-mixtral ✅ COMPATIBLE
- **Path**: `tiny-mixtral/`
- **Architecture**: `MixtralForCausalLM` (standard — no custom code needed)
- **Size**: ~2-layer MoE (tiny)
- **Vocab**: 32000 tokens — subset of target vocab; token IDs 0–31999 map to the same strings in both tokenizers
- **Tokenizer**: `LlamaTokenizer` (same SentencePiece model as target base)
- **S3**: `s3://domain-aware-sd/models/tiny-mixtral/`

### Rejected candidates
| Model | Reason |
|-------|--------|
| `flant5-tuned-30` | T5Tokenizer — completely different tokenization; incompatible with SD |
| `t5-small-finetuned` | T5Tokenizer — same issue; also encoder-decoder architecture |

---

## Dataset: Flan (66 clusters)

- **Location**: `flan/` with `train/`, `test/`, `validation/` splits
- **Format**: JSONL with fields `{"inputs": "...", "targets": "...", "task": "..."}`
- **Clusters**: 66 files per split (e.g., `aeslc_10templates_train.jsonl`)
- **Total train samples**: ~1.4M across all clusters
- **Cluster sizes**: 500 – 30,000 samples each

These 66 clusters define the domain partition. Each cluster corresponds to a distinct NLP task, which serves as a proxy for a "domain."

---

## Tech Stack

| Tool | Role |
|------|------|
| **vLLM** | Fast inference for target model data generation (requires Linux + CUDA) |
| **HuggingFace Transformers** | Model loading, fine-tuning, fallback generation |
| **DeepSpeed** | Distributed training of draft models (requires Linux + CUDA) |
| **Hydra** | Config management for all scripts |
| **MLflow** | Experiment tracking (AR, loss, generation metadata) |
| **S3 (Timeweb)** | Storage for models, generated data, checkpoints |
| **boto3** | S3 client; credentials from `.env` |

---

## Path conventions

**Always use paths relative to the project root.** Never put absolute paths in config files, `.env.example`, or any committed file — the repo is used on multiple machines and servers.

- `configs/*.yaml` — all `path:` / `dir:` / `flan_dir:` values are relative to the project root
- `.env.example` — `TARGET_MODEL_PATH`, `FLAN_DIR` are relative to the project root
- Python scripts resolve relative paths via `PROJECT_ROOT = Path(__file__).parent.parent`

Always run scripts from the project root so relative paths resolve correctly:
```bash
cd /path/to/Domain-Aware-SD
python src/generate_synthetic_data.py   # correct
```

---

## Project Structure

```
Domain-Aware-SD/
├── TurboSparse-Mistral-Instruct/   # target model (7B)
├── tiny-mixtral/                   # draft model — compatible tokenizer
├── flant5-tuned-30/               # rejected — T5 tokenizer
├── t5-small-finetuned/            # rejected — T5 tokenizer
├── flan/                          # dataset
│   ├── train/                     # 66 *.jsonl files
│   ├── test/
│   └── validation/
├── src/
│   ├── check_tokenizers.py        # Task 2: tokenizer check + S3 upload
│   ├── generate_synthetic_data.py # Task 3: vLLM top-10 data generation
│   └── data/                      # Dataset module (see §Data Pipeline below)
│       ├── __init__.py
│       ├── dataset.py             # SpecDecDataset
│       ├── collator.py            # DistillationCollator
│       └── utils.py               # walk_data_files, detect_format, build_index
├── configs/
│   └── generation.yaml            # Hydra config for data generation
├── data/
│   └── synthetic/
│       └── v1/                    # generated JSONL output (66 files)
├── environment.yml                # conda env: domain_sd
├── DATASET_FORMAT.md              # synthetic dataset schema (JSONL + NPZ)
├── S3_INFO.md                     # S3 setup & usage docs
├── .env.example                   # credential template
├── .env                           # actual credentials (gitignored)
└── CLAUDE.md                      # this file
```

---

## Data Pipeline (`src/data/`)

### Record formats (auto-detected per record)

| Format | Key fields | Source |
|--------|-----------|--------|
| `synthetic` | `cluster`, `prompt`, `reference`, `trunk`, `top10_ids`, `top10_probs` | `generate_synthetic_data.py` output |
| `flan` | `inputs`, `targets` | Raw Flan JSONL |
| `plain` | `text` | Generic text |

Synthetic records are stored as either `.jsonl` (human-readable, recommended for validation/test) or `.npz` (compact binary, ~10× smaller, recommended for train). Both formats hold the same logical fields and are loaded transparently by `SpecDecDataset.from_dir`.

Full schema, dtype layout, and skip-position semantics: see **[DATASET_FORMAT.md](DATASET_FORMAT.md)**.

### `SpecDecDataset`

```python
from src.data import SpecDecDataset, DistillationCollator

# Single file, list of files, or recursive directory scan
ds = SpecDecDataset("data/synthetic/v1/aeslc.jsonl", tokenizer)
ds = SpecDecDataset(["f1.jsonl", "f2.jsonl"], tokenizer, mode="standard")
ds = SpecDecDataset.from_dir("data/synthetic/v1", tokenizer, mode="distillation")

# Cluster API
ds.cluster_names()               # sorted list of cluster names
ds.get_cluster_subset("aeslc")   # torch.utils.data.Subset

# Stratified train/val/test split (proportional per cluster)
train, val, test = ds.split((0.8, 0.1, 0.1), seed=42)

# Summary statistics
ds.stats()   # total_samples, cluster_counts, gen_len, mean_top1_prob
```

Constructor kwargs: `mode` (`"distillation"` | `"standard"`), `max_length` (default 512), `max_gen_length` (default 256), `clusters_filter` (list of names), `min_top1_prob` (quality filter — drops samples where mean top-1 prob < threshold).

### `__getitem__` output

Always present:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `[L]` | `prompt_ids + trunk_ids`, truncated to `max_length` |
| `attention_mask` | `[L]` | all ones |
| `labels` | `[L]` | `-100` for prompt positions, trunk token IDs for generated positions |
| `gen_start` | `int` | index where generation begins in `input_ids` |
| `cluster` | `str` | cluster name |

Distillation mode only (when `mode="distillation"` and record is `synthetic`):

| Key | Shape | Description |
|-----|-------|-------------|
| `top10_ids` | `[gen_len, 10]` | token IDs of top-10 candidates per position |
| `top10_probs` | `[gen_len, 10]` | probabilities (float32, rounded to 3 d.p.) |

Labels are designed for HuggingFace causal LM models that internally shift logits/labels (logits[i] predicts labels[i+1]). With `-100` masking on prompt positions, the loss covers only the generated span.

### `DistillationCollator`

```python
from torch.utils.data import DataLoader
collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
loader = DataLoader(ds, batch_size=16, collate_fn=collator)
```

Pads `input_ids` / `attention_mask` / `labels` to batch max length. Pads `top10_ids` / `top10_probs` to `[B, max_gen_len, 10]` with zeros.

---

## Setup

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate domain_sd

# On GPU server (Linux + CUDA), also install:
pip install vllm>=0.4.0
pip install deepspeed>=0.14.0
```

### 2. Credentials

```bash
cp .env.example .env
# Edit .env with actual S3 credentials
```

### 3. Create S3 bucket (first time only)

```python
# Run once
import boto3, os
from dotenv import load_dotenv
load_dotenv(".env")
s3 = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT"),
                  aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                  aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
                  region_name=os.getenv("S3_REGION"))
s3.create_bucket(Bucket="domain-aware-sd")
```

---

## Pipeline

### Step 1 — Tokenizer check + upload (Task 2)

```bash
conda activate domain_sd
cd /Users/mike/projects/Domain-Aware-SD

# Dry run (no upload)
python src/check_tokenizers.py

# Check + upload compatible drafts
python src/check_tokenizers.py --upload

# Also upload target model
python src/check_tokenizers.py --upload --also-upload-target
```

**Result**: `tiny-mixtral` is compatible (32000-token subset of target's 32064-token vocab).

### Step 2 — Synthetic data generation (Task 3)

```bash
# Must run on a GPU server with vLLM installed
# Falls back to transformers (slow) if vLLM is unavailable

# Required env vars on the current GPU server (cbg-ai-blue-l-1):
#   VLLM_USE_FLASHINFER_SAMPLER=0  — flashinfer's sampler tries to JIT-compile
#       a CUDA kernel with nvcc; the server has no CUDA toolkit, so we use
#       the default torch sampler instead. vLLM otherwise loads Bamboo fine.
#   MLFLOW_ALLOW_FILE_STORE=true   — modern MLflow refuses the ./mlruns
#       filesystem backend without explicit opt-in.
VLLM_USE_FLASHINFER_SAMPLER=0 MLFLOW_ALLOW_FILE_STORE=true \
  python src/generate_synthetic_data.py

# Override config on the command line (Hydra syntax):
python src/generate_synthetic_data.py generation.max_samples_per_cluster=100
python src/generate_synthetic_data.py model.tensor_parallel_size=4
python src/generate_synthetic_data.py s3.upload_after_generation=true

# Memory-saving knobs:
python src/generate_synthetic_data.py generation.logprobs=0                  # normal mode — no top-K at all
python src/generate_synthetic_data.py generation.skip_top10_above_prob=0.95  # skip top-K at confident positions
python src/generate_synthetic_data.py output.format=npz                      # compact binary archive (recommended for train split)
```

**Output**: `data/synthetic/v1/<cluster_name>.jsonl` (or `.npz` if `output.format=npz`) — one file per cluster.

**Output schema** (JSONL per line; npz holds the same fields as numpy arrays):
```json
{
  "cluster":     "aeslc_10templates",
  "prompt":      "<original flan input>",
  "reference":   "<original flan target>",
  "trunk":       [1234, 567, ...],
  "top10_ids":   [[1234, 99, 12, ...], [], [567, 5, 8, ...], ...],
  "top10_probs": [[0.45, 0.20, 0.05, ...], [], [0.32, 0.15, 0.10, ...], ...]
}
```
Empty rows `[]` indicate skipped positions (normal mode or above the `skip_top10_above_prob` threshold).

### Step 3 — Fine-tune domain-specific draft models (TODO)

For each of the 66 clusters, fine-tune `tiny-mixtral` on that cluster's synthetic data using knowledge distillation against the top-10 distributions.

```bash
# Planned — DeepSpeed + transformers Trainer
python src/train_drafter.py cluster=aeslc_10templates
```

### Step 4 — Evaluate AR (TODO)

Run SD evaluation: for each cluster, use the domain-specific draft model and measure AR vs. the baseline (draft trained on all data).

---

## Known Issues / Notes

### vLLM + TurboSparse custom architecture
TurboSparse uses `BambooForCausalLM` (defined in `modeling_bamboo.py`). vLLM has its own model registry and may not support this architecture by default. If `llm = LLM(model=..., trust_remote_code=True)` fails, the script automatically falls back to HuggingFace `transformers.generate()` which respects `trust_remote_code`.

To check if vLLM supports it:
```python
from vllm import LLM
llm = LLM("TurboSparse-Mistral-Instruct", trust_remote_code=True)
```
If it errors with "unsupported model type", the fallback path in `generate_synthetic_data.py` handles it.

### GPU server env vars (cbg-ai-blue-l-1)
The synthetic data generation requires two env vars on the current GPU server:

| Variable | Value | Why |
|----------|-------|-----|
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | vLLM's default flashinfer top-k/top-p sampler JIT-compiles a CUDA kernel via `nvcc`. The server has no CUDA toolkit installed (`/usr/local/cuda` missing), so the JIT step crashes during `profile_run`. Setting this to `0` falls back to the torch-native sampler, which works without `nvcc`. vLLM loads Bamboo and runs inference normally — only the sampler kernel was the problem. |
| `MLFLOW_ALLOW_FILE_STORE` | `true` | MLflow ≥ 3.x refuses the `./mlruns` filesystem backend without explicit opt-in (it tells you to migrate to SQLite). We keep the file backend for simplicity, so this opt-out must be set. |

Working invocation:
```bash
VLLM_USE_FLASHINFER_SAMPLER=0 MLFLOW_ALLOW_FILE_STORE=true \
  python src/generate_synthetic_data.py
```

Alternatives if you'd rather fix the underlying issue:
- Install nvcc (`conda install -c nvidia cuda-nvcc cuda-cccl`) — lets flashinfer JIT-compile and you can drop `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Switch MLflow to SQLite: `python src/generate_synthetic_data.py mlflow.tracking_uri=sqlite:///mlflow.db`.

### macOS / Apple Silicon
vLLM and DeepSpeed require Linux + CUDA. On macOS:
- The conda env installs without those two packages
- `generate_synthetic_data.py` falls back to transformers with MPS backend
- Use a remote GPU server for production generation

### Token ID mapping
Target vocab has 32064 tokens; tiny-mixtral has 32000. Token IDs 0–31999 are identical in both (same Mistral SentencePiece model). The 64 extra tokens in the target (32000–32063) are chat special tokens (`<|im_end|>`, `<|im_start|>`, etc.) that the draft doesn't need to predict.

---

## S3 Reference

See `S3_INFO.md` for full S3 setup, boto3 usage examples, and bucket structure.

---

## MLflow

Experiments are tracked locally in `./mlruns`. To view:
```bash
conda activate domain_sd
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

To use a remote MLflow server, set `MLFLOW_TRACKING_URI` in `.env`.
