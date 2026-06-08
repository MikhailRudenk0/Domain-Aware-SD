#!/usr/bin/env python3
"""
Generate synthetic data from TurboSparse-Mistral-Instruct (target model).

For each of 66 flan clusters, generates text completions and captures
top-10 token probabilities at each generation step. This data trains
domain-specific draft models for speculative decoding.

Output format per sample (JSONL):
{
  "cluster":  "aeslc_10templates",
  "prompt":   "...",
  "trunk":    [tok_id_0, tok_id_1, ...],      # sampled sequence (top-1 at each step)
  "top10":    [                                # one entry per generated position
    [{"token_id": X, "prob": Y, "token": "..."},  ...],   # 10 entries sorted by prob desc
    ...
  ]
}

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


def logprobs_to_top10(logprobs_at_position: dict) -> list[dict]:
    """
    Convert vLLM logprobs dict at one position to sorted top-10 list.

    vLLM logprobs format: {token_id: Logprob(logprob=float, rank=int, decoded_token=str)}
    """
    import math
    entries = []
    for token_id, lp_obj in logprobs_at_position.items():
        logprob = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
        prob = math.exp(logprob)
        token_str = lp_obj.decoded_token if hasattr(lp_obj, "decoded_token") else ""
        entries.append({"token_id": int(token_id), "prob": round(prob, 6), "token": token_str})
    entries.sort(key=lambda x: x["prob"], reverse=True)
    return entries[:10]


def batch_prompts(samples: list[dict], batch_size: int):
    """Yield batches of (index, prompt) pairs."""
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        yield i, batch


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def build_vllm_engine(cfg: DictConfig):
    """Initialize vLLM LLM engine from config."""
    from vllm import LLM

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

    return SamplingParams(
        max_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        logprobs=cfg.generation.logprobs,  # capture top-k logprobs per step
    )


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

    for batch_start, batch in batch_prompts(samples, batch_size):
        prompts = [s["inputs"] for s in batch]
        print(f"    Batch {batch_start // batch_size + 1}: {len(prompts)} prompts")

        outputs = llm.generate(prompts, sampling_params)

        for sample, output in zip(batch, outputs):
            if not output.outputs:
                continue
            out = output.outputs[0]

            top10_per_pos = []
            if out.logprobs:
                for pos_logprobs in out.logprobs:
                    top10_per_pos.append(logprobs_to_top10(pos_logprobs))

            results.append({
                "cluster": cluster,
                "prompt": sample["inputs"],
                "reference": sample.get("targets", ""),
                "trunk": list(out.token_ids),
                "top10": top10_per_pos,
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
                output_scores=True,  # logits at each step
            )

        # gen_out.scores: tuple of (batch, vocab) tensors, one per generated step
        sequences = gen_out.sequences  # (batch, prompt_len + gen_len)
        scores = gen_out.scores        # tuple of (batch, vocab)

        prompt_len = inputs["input_ids"].shape[1]

        for b_idx, sample in enumerate(batch):
            generated_ids = sequences[b_idx, prompt_len:].tolist()
            top10_per_pos = []

            for step, step_scores in enumerate(scores):
                logits = step_scores[b_idx]
                # Convert logits → probabilities
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_ids = torch.topk(probs, k=10)
                top10_at_step = [
                    {
                        "token_id": int(top_ids[k]),
                        "prob": round(float(top_probs[k]), 6),
                        "token": tokenizer.decode([top_ids[k]]),
                    }
                    for k in range(10)
                ]
                top10_per_pos.append(top10_at_step)

            results.append({
                "cluster": cluster,
                "prompt": sample["inputs"],
                "reference": sample.get("targets", ""),
                "trunk": generated_ids,
                "top10": top10_per_pos,
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
        total_samples = 0
        for cluster_file in cluster_files:
            cname = cluster_name(cluster_file)
            out_file = out_dir / f"{cname}.jsonl"

            if out_file.exists():
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

            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

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

    for f in sorted(out_dir.glob("*.jsonl")):
        key = f"{prefix}/{f.name}"
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name} ({size_mb:.1f} MB) → s3://{bucket}/{key}")
        s3.upload_file(str(f), bucket, key)


if __name__ == "__main__":
    main()
