#!/usr/bin/env python3
"""
Generate synthetic data from TurboSparse-Mistral-Instruct (target model).

For each of 66 flan clusters, generates text completions and captures
top-10 token probabilities at each generation step. This data trains
domain-specific draft models for speculative decoding.

Output schema (JSONL — one record per line):
{
  "cluster":     "aeslc_10templates",
  "prompt":      "...",
  "reference":   "...",
  "trunk":       [tok_id_0, tok_id_1, ...],        # sampled sequence
  "top10_ids":   [[id_0_0, ..., id_0_9], ...],     # one row per generated position
  "top10_probs": [[p_0_0,  ..., p_0_9 ], ...]      # rounded to 3 d.p.
}

Rows in top10_ids/top10_probs are aligned with trunk: position i corresponds to
trunk[i]. A skipped row (empty list []) means top-K was not captured at that
position — either because logprobs=0 (normal mode) or because the top-1 prob
exceeded skip_top10_above_prob.

When output.format=npz, the same fields are written as a compact binary numpy
archive (uint16 ids, uint16 quantized probs ×1000) — readable transparently
by SpecDecDataset.

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py generation.max_samples_per_cluster=100
    python src/generate_synthetic_data.py model.tensor_parallel_size=2
"""

import os
import sys
import json
import math
import random
import glob
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

load_dotenv_path = Path(__file__).parent.parent / ".env"
if load_dotenv_path.exists():
    from dotenv import load_dotenv
    load_dotenv(load_dotenv_path)

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cluster(jsonl_path: Path, max_samples: Optional[int], shuffle: bool, seed: int) -> list[dict]:
    """Load samples from one cluster JSONL file."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def cluster_name(jsonl_path: Path) -> str:
    """Extract cluster name from filename: 'aeslc_10templates_train.jsonl' → 'aeslc_10templates'."""
    name = jsonl_path.stem  # removes .jsonl
    # Remove trailing _train / _test / _validation
    for suffix in ("_train", "_test", "_validation"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def logprobs_to_topk(
    logprobs_at_position: dict,
    k: int,
    skip_above_prob: float = 1.0,
) -> tuple[list[int], list[float]]:
    """
    Convert vLLM logprobs at one generated position to parallel (ids, probs) arrays.

    vLLM logprobs format: {token_id: Logprob(logprob=float, rank=int, decoded_token=str)}.
    Returned arrays are sorted by probability descending and truncated to k.
    If the top-1 prob > skip_above_prob, returns ([], []) — caller should record
    this as a skipped position so trunk and top10 stay aligned.
    """
    entries: list[tuple[int, float]] = []
    for token_id, lp_obj in logprobs_at_position.items():
        logprob = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
        prob = math.exp(logprob)
        entries.append((int(token_id), round(prob, 3)))
    entries.sort(key=lambda x: x[1], reverse=True)
    entries = entries[:k]
    if entries and entries[0][1] > skip_above_prob:
        return [], []
    ids = [e[0] for e in entries]
    probs = [e[1] for e in entries]
    return ids, probs


def batch_prompts(samples: list[dict], batch_size: int):
    """Yield batches of (index, prompt) pairs."""
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        yield i, batch


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_npz(records: list[dict], path: Path) -> None:
    """
    Pack a cluster's records into a compact .npz archive.

    Layout (all arrays cluster-wide):
      prompts        object [N]          UTF-8 strings
      references     object [N]
      cluster        scalar str
      trunk_lens     int32  [N]          length of each sample's trunk
      trunk_ids      uint16 [sum lens]   concatenated trunk token IDs
      top10_ids      uint16 [sum lens, K] zeros at skipped/normal positions
      top10_probs_q  uint16 [sum lens, K] prob × 1000; zeros at skipped positions
      top10_mask     bool   [sum lens]   True where top-K was captured
      top_k          scalar uint8        K (10 by default; 0 in normal mode)

    Empty cluster (no records) → still writes the file but with N=0.
    """
    import numpy as np

    if not records:
        np.savez_compressed(path, prompts=np.array([], dtype=object))
        return

    cluster = records[0]["cluster"]
    prompts = np.array([r["prompt"] for r in records], dtype=object)
    references = np.array([r.get("reference", "") for r in records], dtype=object)

    trunk_lens = np.array([len(r["trunk"]) for r in records], dtype=np.int32)
    trunk_ids = np.fromiter(
        (tid for r in records for tid in r["trunk"]),
        dtype=np.uint16,
        count=int(trunk_lens.sum()),
    )

    # Infer K from the first non-empty top10 row across all records.
    k = 0
    for r in records:
        for row in r.get("top10_ids", []):
            if row:
                k = len(row)
                break
        if k:
            break

    total = int(trunk_lens.sum())
    if k > 0 and total > 0:
        top10_ids = np.zeros((total, k), dtype=np.uint16)
        top10_probs_q = np.zeros((total, k), dtype=np.uint16)
        top10_mask = np.zeros(total, dtype=bool)
        offset = 0
        for r in records:
            ids_rows = r.get("top10_ids", [])
            probs_rows = r.get("top10_probs", [])
            for i in range(len(r["trunk"])):
                pos_ids = ids_rows[i] if i < len(ids_rows) else []
                pos_probs = probs_rows[i] if i < len(probs_rows) else []
                if pos_ids:
                    n = min(len(pos_ids), k)
                    top10_ids[offset + i, :n] = pos_ids[:n]
                    # quantize probs to uint16 in [0, 1000]
                    top10_probs_q[offset + i, :n] = [
                        max(0, min(1000, int(round(p * 1000)))) for p in pos_probs[:n]
                    ]
                    top10_mask[offset + i] = True
            offset += len(r["trunk"])
    else:
        # normal mode (k == 0) — still write empty arrays so the reader has a
        # consistent schema.
        top10_ids = np.zeros((total, 0), dtype=np.uint16)
        top10_probs_q = np.zeros((total, 0), dtype=np.uint16)
        top10_mask = np.zeros(total, dtype=bool)

    np.savez_compressed(
        path,
        cluster=np.array(cluster),
        prompts=prompts,
        references=references,
        trunk_lens=trunk_lens,
        trunk_ids=trunk_ids,
        top10_ids=top10_ids,
        top10_probs_q=top10_probs_q,
        top10_mask=top10_mask,
        top_k=np.uint8(k),
    )


def write_cluster(records: list[dict], out_dir: Path, cname: str, fmt: str) -> Path:
    """Dispatch to the configured writer; return the path written."""
    if fmt == "jsonl":
        path = out_dir / f"{cname}.jsonl"
        write_jsonl(records, path)
    elif fmt == "npz":
        path = out_dir / f"{cname}.npz"
        write_npz(records, path)
    else:
        raise ValueError(f"Unknown output.format: {fmt!r} (expected 'jsonl' or 'npz')")
    return path


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def build_vllm_engine(cfg: DictConfig):
    """Initialize vLLM LLM engine from config."""
    import os
    from vllm import LLM

    # flashinfer's sampling kernel requires nvcc for JIT compilation.
    # If nvcc is missing (common when CUDA is installed via conda or a
    # non-standard path), disable the flashinfer sampler so vLLM falls back
    # to its built-in PyTorch sampler instead of crashing.
    import shutil
    if not shutil.which("nvcc"):
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    return LLM(
        model=cfg.model.path,
        trust_remote_code=cfg.model.trust_remote_code,
        dtype=cfg.model.dtype,
        tensor_parallel_size=cfg.model.tensor_parallel_size,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
    )


def build_sampling_params(cfg: DictConfig):
    """Build vLLM SamplingParams from config."""
    from vllm import SamplingParams

    # logprobs=0 (or null) → normal mode: don't request top-K from vLLM at all.
    k = cfg.generation.logprobs
    return SamplingParams(
        max_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logprobs=k if k else None,
    )


def _vllm_max_model_len(llm, default: int = 32768) -> int:
    """Best-effort lookup of the engine's max_model_len across vLLM versions."""
    for attr_path in (
        ("llm_engine", "model_config", "max_model_len"),
        ("llm_engine", "vllm_config", "model_config", "max_model_len"),
    ):
        obj = llm
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            return int(obj)
        except AttributeError:
            continue
    return default


def generate_cluster_vllm(
    llm,
    sampling_params,
    samples: list[dict],
    cluster: str,
    cfg: DictConfig,
) -> list[dict]:
    """Generate synthetic data for one cluster using vLLM."""
    results = []
    batch_size = cfg.generation.batch_size

    # Drop prompts that won't fit in the model's context. vLLM raises
    # VLLMValidationError mid-batch otherwise, killing the whole run.
    max_model_len = _vllm_max_model_len(llm)
    max_prompt_tokens = max_model_len - cfg.generation.max_new_tokens - 16
    tokenizer = llm.get_tokenizer()
    kept, skipped = [], 0
    for s in samples:
        if len(tokenizer.encode(s["inputs"])) <= max_prompt_tokens:
            kept.append(s)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped}/{len(samples)} samples over {max_prompt_tokens} prompt tokens")
    samples = kept

    k = cfg.generation.logprobs or 0
    skip_above = float(cfg.generation.skip_top10_above_prob)

    for batch_start, batch in batch_prompts(samples, batch_size):
        prompts = [s["inputs"] for s in batch]
        print(f"    Batch {batch_start // batch_size + 1}: {len(prompts)} prompts")

        outputs = llm.generate(prompts, sampling_params)

        for sample, output in zip(batch, outputs):
            if not output.outputs:
                continue
            out = output.outputs[0]

            top10_ids: list[list[int]] = []
            top10_probs: list[list[float]] = []
            if k and out.logprobs:
                for pos_logprobs in out.logprobs:
                    ids, probs = logprobs_to_topk(pos_logprobs, k, skip_above)
                    top10_ids.append(ids)
                    top10_probs.append(probs)

            results.append({
                "cluster": cluster,
                "prompt": sample["inputs"],
                "reference": sample.get("targets", ""),
                "trunk": list(out.token_ids),
                "top10_ids": top10_ids,
                "top10_probs": top10_probs,
            })

    return results


# ---------------------------------------------------------------------------
# Transformers fallback (CPU / MPS — no vLLM required)
# ---------------------------------------------------------------------------

def generate_cluster_transformers(
    model,
    tokenizer,
    samples: list[dict],
    cluster: str,
    cfg: DictConfig,
    device: str,
) -> list[dict]:
    """
    Fallback generator using HuggingFace transformers.
    Slower than vLLM but works with custom architectures and without CUDA.
    """
    import torch

    results = []
    batch_size = min(cfg.generation.batch_size, 4)  # conservative for CPU/MPS
    k = cfg.generation.logprobs or 0
    skip_above = float(cfg.generation.skip_top10_above_prob)

    for batch_start, batch in batch_prompts(samples, batch_size):
        prompts = [s["inputs"] for s in batch]
        print(f"    Batch {batch_start // batch_size + 1}: {len(prompts)} prompts")

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=True,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                return_dict_in_generate=True,
                output_scores=bool(k),  # only request logits when top-K is wanted
            )

        sequences = gen_out.sequences                       # (batch, prompt_len + gen_len)
        scores = gen_out.scores if k else None              # tuple of (batch, vocab)
        prompt_len = inputs["input_ids"].shape[1]

        for b_idx, sample in enumerate(batch):
            generated_ids = sequences[b_idx, prompt_len:].tolist()
            top10_ids: list[list[int]] = []
            top10_probs: list[list[float]] = []

            if k and scores is not None:
                for step_scores in scores:
                    probs = torch.softmax(step_scores[b_idx], dim=-1)
                    top_probs, top_ids = torch.topk(probs, k=k)
                    top1 = round(float(top_probs[0]), 3)
                    if top1 > skip_above:
                        top10_ids.append([])
                        top10_probs.append([])
                    else:
                        top10_ids.append([int(t) for t in top_ids.tolist()])
                        top10_probs.append([round(float(p), 3) for p in top_probs.tolist()])

            results.append({
                "cluster": cluster,
                "prompt": sample["inputs"],
                "reference": sample.get("targets", ""),
                "trunk": generated_ids,
                "top10_ids": top10_ids,
                "top10_probs": top10_probs,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="generation", version_base=None)
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # MLflow tracking
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "generation_config.json")

        # Locate cluster files
        flan_dir = Path(cfg.data.flan_dir) / cfg.data.split
        cluster_files = sorted(flan_dir.glob("*.jsonl"))
        print(f"\nFound {len(cluster_files)} cluster files in {flan_dir}")

        # Prepare output dir
        out_dir = Path(cfg.output.dir) / cfg.output.version
        out_dir.mkdir(parents=True, exist_ok=True)

        # Try vLLM first, fall back to transformers
        use_vllm = True
        llm = None
        sampling_params = None
        hf_model = None
        hf_tokenizer = None
        device = "cpu"

        try:
            print("\nLoading model with vLLM...")
            llm = build_vllm_engine(cfg)
            sampling_params = build_sampling_params(cfg)
            print("vLLM engine ready.")
        except Exception as e:
            print(f"vLLM unavailable ({e}). Falling back to HuggingFace transformers.")
            use_vllm = False

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"Using device: {device}")

            hf_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.path, trust_remote_code=cfg.model.trust_remote_code
            )
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token

            torch_dtype = (
                torch.bfloat16
                if cfg.model.dtype == "bfloat16" and device != "cpu"
                else torch.float32
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.path,
                trust_remote_code=cfg.model.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            hf_model.eval()
            print("HuggingFace model loaded.")

        # Process each cluster
        out_fmt = cfg.output.format
        total_samples = 0
        for cluster_file in cluster_files:
            cname = cluster_name(cluster_file)
            # Skip if either format is already present — same cluster, same version.
            if (out_dir / f"{cname}.jsonl").exists() or (out_dir / f"{cname}.npz").exists():
                print(f"\n[SKIP] {cname} — output already exists")
                continue

            print(f"\n[{cluster_files.index(cluster_file)+1}/{len(cluster_files)}] Cluster: {cname}")
            samples = load_cluster(
                cluster_file,
                max_samples=cfg.data.max_samples_per_cluster,
                shuffle=cfg.data.shuffle,
                seed=cfg.data.shuffle_seed,
            )
            print(f"  Loaded {len(samples)} samples")

            if use_vllm:
                results = generate_cluster_vllm(llm, sampling_params, samples, cname, cfg)
            else:
                results = generate_cluster_transformers(
                    hf_model, hf_tokenizer, samples, cname, cfg, device
                )

            out_file = write_cluster(results, out_dir, cname, out_fmt)
            total_samples += len(results)
            print(f"  Saved {len(results)} samples → {out_file}")
            mlflow.log_metric("clusters_done", cluster_files.index(cluster_file) + 1)

        mlflow.log_metric("total_samples_generated", total_samples)
        print(f"\nDone. Total samples generated: {total_samples}")
        print(f"Output: {out_dir}")

        # Optionally upload to S3
        if cfg.s3.upload_after_generation:
            print("\nUploading to S3...")
            _upload_to_s3(out_dir, cfg)


def _upload_to_s3(out_dir: Path, cfg: DictConfig):
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT", "https://s3.twcstorage.ru"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name=os.getenv("S3_REGION", "ru-1-hot"),
    )
    bucket = cfg.s3.bucket
    prefix = f"{cfg.s3.prefix}/{cfg.output.version}"

    files = sorted(list(out_dir.glob("*.jsonl")) + list(out_dir.glob("*.npz")))
    for f in files:
        key = f"{prefix}/{f.name}"
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name} ({size_mb:.1f} MB) → s3://{bucket}/{key}")
        s3.upload_file(str(f), bucket, key)


if __name__ == "__main__":
    main()
