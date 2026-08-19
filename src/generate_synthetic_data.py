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
    """Dispatch to the configured writer; return the path written.

    Writes to a temporary file first and renames atomically, so a killed run
    never leaves a truncated output that a restart would mistake for a
    finished cluster.
    """
    if fmt == "jsonl":
        path = out_dir / f"{cname}.jsonl"
        writer = write_jsonl
    elif fmt == "npz":
        path = out_dir / f"{cname}.npz"
        writer = write_npz
    else:
        raise ValueError(f"Unknown output.format: {fmt!r} (expected 'jsonl' or 'npz')")
    tmp = path.with_name(path.name + ".tmp")
    if fmt == "npz":
        # pass an open file handle so numpy does not append ".npz" to the name
        with open(tmp, "wb") as fh:
            writer(records, fh)
    else:
        writer(records, tmp)
    os.replace(tmp, path)
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
    HuggingFace generator with a manual batched sampling loop.

    We do NOT use ``model.generate()``: modeling_bamboo.py predates the
    transformers-5 generation/cache API and its ``prepare_inputs_for_generation``
    breaks there. A manual loop with an explicit ``DynamicCache``, left padding
    and explicit ``position_ids`` works correctly with the patched model code.

    Batching: prompts are sorted by token length and grouped under a
    ``token_budget`` of (prompt + max_new_tokens) tokens per batch, so short
    prompts run at a large batch size while rare long prompts get small batches
    instead of blowing up the KV cache.

    Top-K capture uses the raw (temperature=1) softmax distribution, matching
    what vLLM's ``logprobs`` returned in the original pipeline.
    """
    import time

    import torch
    from transformers.cache_utils import DynamicCache

    k = cfg.generation.logprobs or 0
    skip_above = float(cfg.generation.skip_top10_above_prob)
    temperature = float(cfg.generation.temperature)
    top_p = float(cfg.generation.top_p)
    max_new = int(cfg.generation.max_new_tokens)
    token_budget = int(cfg.generation.get("token_budget", 16000))
    max_prompt = int(cfg.generation.get("max_prompt_tokens", 4096))
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Encode all prompts (tokenizer adds BOS); skip the rare over-long ones.
    encoded, skipped = [], 0
    for s in samples:
        ids = tokenizer.encode(s["inputs"])
        if len(ids) > max_prompt:
            skipped += 1
            continue
        encoded.append((s, ids))
    if skipped:
        print(f"  Skipped {skipped}/{len(samples)} prompts over {max_prompt} tokens")

    encoded.sort(key=lambda x: len(x[1]))
    batches, cur, cur_tokens = [], [], 0
    for item in encoded:
        cost = len(item[1]) + max_new
        if cur and cur_tokens + cost > token_budget:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(item)
        cur_tokens += cost
    if cur:
        batches.append(cur)

    # Prefill through the base model and apply lm_head to the LAST position
    # only. BambooForCausalLM.forward materializes float32 logits for every
    # prompt position (B x L x 32064) — several GB per batch and the main
    # cause of prefill OOM on a 24 GB GPU.
    base_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)

    def forward_last_logits(step_ids, step_attn, step_pos, cache):
        with torch.no_grad():
            if base_model is not None and lm_head is not None:
                out = base_model(input_ids=step_ids, attention_mask=step_attn,
                                 position_ids=step_pos, past_key_values=cache,
                                 use_cache=True)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                return lm_head(hidden[:, -1, :])
            out = model(input_ids=step_ids, attention_mask=step_attn,
                        position_ids=step_pos, past_key_values=cache, use_cache=True)
            return out.logits[:, -1, :]

    def run_batch(batch):
        """Generate one batch; on CUDA OOM, split it in half and retry."""
        try:
            return _run_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                print("    OOM on a single sample — skipping it", flush=True)
                return []
            mid = len(batch) // 2
            print(f"    OOM at batch size {len(batch)} — splitting in half", flush=True)
            return run_batch(batch[:mid]) + run_batch(batch[mid:])

    def _run_batch(batch):
        B = len(batch)
        maxlen = max(len(ids) for _, ids in batch)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for i, (_, ids) in enumerate(batch):  # left padding
            input_ids[i, maxlen - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[i, maxlen - len(ids):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)

        cache = DynamicCache()
        logits = forward_last_logits(input_ids, attn, pos, cache)
        cur_pos = pos[:, -1]

        trunks = [[] for _ in range(B)]
        tops_i: list[list[list[int]]] = [[] for _ in range(B)]
        tops_p: list[list[list[float]]] = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _step in range(max_new):
            probs = torch.softmax(logits.float(), dim=-1)
            if k:
                tp, ti = torch.topk(probs, k=k, dim=-1)
                tp_cpu, ti_cpu = tp.tolist(), ti.tolist()
            # sampling distribution: temperature -> top_p -> renormalize
            sp = torch.softmax(logits.float() / max(temperature, 1e-4), dim=-1)
            if top_p < 1.0:
                srt, idx = torch.sort(sp, descending=True, dim=-1)
                drop = srt.cumsum(-1) - srt > top_p
                srt = srt.masked_fill(drop, 0.0)
                nxt = idx.gather(-1, torch.multinomial(srt, 1))
            else:
                nxt = torch.multinomial(sp, 1)
            nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, pad_id), nxt)
            nxt_cpu = nxt.squeeze(1).tolist()
            for i in range(B):
                if done[i]:
                    continue
                trunks[i].append(nxt_cpu[i])
                if k:
                    if round(tp_cpu[i][0], 3) > skip_above:
                        tops_i[i].append([])
                        tops_p[i].append([])
                    else:
                        tops_i[i].append([int(t) for t in ti_cpu[i]])
                        tops_p[i].append([round(float(p), 3) for p in tp_cpu[i]])
            done = done | (nxt.squeeze(1) == eos_id)
            if bool(done.all()):
                break
            cur_pos = cur_pos + 1
            # done rows keep receiving pad tokens; their outputs are ignored.
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=device)], dim=1)
            logits = forward_last_logits(nxt, attn, cur_pos.unsqueeze(1), cache)

        return [
            {
                "cluster": cluster,
                "prompt": sample["inputs"],
                "reference": sample.get("targets", ""),
                "trunk": trunks[i],
                "top10_ids": tops_i[i],
                "top10_probs": tops_p[i],
            }
            for i, (sample, _ids) in enumerate(batch)
        ]

    results = []
    torch.manual_seed(int(cfg.data.shuffle_seed))
    t_cluster = time.time()
    for bi, batch in enumerate(batches):
        results.extend(run_batch(batch))
        rate = len(results) / max(time.time() - t_cluster, 1e-6)
        print(f"    Batch {bi + 1}/{len(batches)}: {len(batch)} prompts, "
              f"{len(results)}/{len(encoded)} samples, {rate:.1f} samples/s", flush=True)

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

        # Backend selection.
        #
        # Default is "hf": vLLM 0.22 does load BambooForCausalLM, but it does
        # NOT run the repaired remote code — its output is near-uniform word
        # salad (the broken-RoPE signature; see results/SYNTHETIC_DATA_VALIDATION.md).
        # Set model.backend=vllm only after verifying its output by eye.
        backend = str(cfg.model.get("backend", "hf")).lower()
        use_vllm = backend in ("vllm", "auto")
        llm = None
        sampling_params = None
        hf_model = None
        hf_tokenizer = None
        device = "cpu"

        if use_vllm:
            try:
                print("\nLoading model with vLLM...")
                llm = build_vllm_engine(cfg)
                sampling_params = build_sampling_params(cfg)
                print("vLLM engine ready.")
            except Exception as e:
                if backend == "vllm":
                    raise
                print(f"vLLM unavailable ({e}). Falling back to HuggingFace transformers.")
                use_vllm = False

        if not use_vllm:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            sys.path.insert(0, str(PROJECT_ROOT))
            from src.repro.bamboo_fix import fix_rotary

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
            hf_tokenizer.padding_side = "left"

            torch_dtype = (
                torch.bfloat16
                if cfg.model.dtype == "bfloat16" and device != "cpu"
                else torch.float32
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.path,
                trust_remote_code=cfg.model.trust_remote_code,
                dtype=torch_dtype,
            )
            # Repair the RoPE buffers that transformers >= 5 leaves
            # uninitialized (belt and braces: the patched modeling_bamboo.py
            # also lazily rebuilds them on first forward).
            n_fixed = fix_rotary(hf_model)
            print(f"fix_rotary: repaired {n_fixed} rotary modules")
            hf_model = hf_model.to(device)
            hf_model.eval()
            print("HuggingFace model loaded.")

        # Optional sharding: run N independent processes (e.g. one per GPU),
        # each taking clusters where index % num_shards == shard_id.
        num_shards = int(cfg.data.get("num_shards", 1))
        shard_id = int(cfg.data.get("shard_id", 0))

        # Process each cluster
        out_fmt = cfg.output.format
        total_samples = 0
        for cluster_idx, cluster_file in enumerate(cluster_files):
            cname = cluster_name(cluster_file)
            if cluster_idx % num_shards != shard_id:
                continue
            # Skip if either format is already present — same cluster, same version.
            if (out_dir / f"{cname}.jsonl").exists() or (out_dir / f"{cname}.npz").exists():
                print(f"\n[SKIP] {cname} — output already exists")
                continue

            print(f"\n[{cluster_idx+1}/{len(cluster_files)}] Cluster: {cname}", flush=True)
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
            print(f"  Saved {len(results)} samples → {out_file}", flush=True)
            mlflow.log_metric("clusters_done", cluster_idx + 1)

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
