#!/usr/bin/env python3
"""
Step 1 of the first-token AR reproduction: cache target/draft logits.

For every cluster of ``flan/validation`` we take the first N records, build the
prompt according to ``--variant``, append the first ``--offsets`` reference
answer tokens and run both models once. The logits that predict answer tokens
0..max_offset are written to ``data/repro_cache/<variant>/<cluster>.npz``.

Logits do not depend on temperature / top_p / top_k, so the whole sampling
sweep (step 2, ``sweep_first_token.py``) runs off this cache without a GPU
forward pass.

Usage:
    python src/repro/cache_first_token.py --variant chatml --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.repro.bamboo_fix import fix_rotary  # noqa: E402

TARGET_DIR = PROJECT_ROOT / "TurboSparse-Mistral-Instruct"
DRAFT_DIR = PROJECT_ROOT / "Lite-Mistral-150M-v2-Instruct"
FLAN_VAL = PROJECT_ROOT / "flan" / "validation"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ── prompt variants ───────────────────────────────────────────────────────────


def build_prompt_ids(tok, inputs: str, variant: str, max_len: int) -> list[int]:
    """Return prompt token ids for one of the four prompt variants."""
    if variant == "plain":
        ids = tok.encode(inputs, add_special_tokens=True, truncation=True, max_length=max_len)
    elif variant == "plain_nl":
        ids = tok.encode(inputs + "\n", add_special_tokens=True, truncation=True, max_length=max_len)
    elif variant == "draftchat":
        # Lite-Mistral's own chat template: <s>role\n{content}</s>\n ... <s>assistant\n
        # (bos/eos are real special tokens, the rest is plain text)
        text = f"user\n{inputs}"
        body = tok.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)
        # NB: draft's eos is </s> (id 2); the target tokenizer maps eos to
        # <|im_end|> (32000), so use the SentencePiece ids directly.
        ids = ([1] + body + [2]
               + tok.encode("\n", add_special_tokens=False)
               + [1] + tok.encode("assistant\n", add_special_tokens=False))
    elif variant in ("chatml", "chatml_bos"):
        text = f"<|im_start|>user\n{inputs}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > max_len:
            # truncate the user text from the left, keep the ChatML tail intact
            tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
            head = tok.encode("<|im_start|>user\n", add_special_tokens=False)
            body = ids[len(head): len(ids) - len(tail)]
            body = body[: max_len - len(head) - len(tail)]
            ids = head + body + tail
        if variant == "chatml_bos":
            ids = [tok.bos_token_id] + ids
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return ids


# ── batching ──────────────────────────────────────────────────────────────────


def make_batches(items: list[dict], max_tokens: int, max_bs: int) -> list[list[dict]]:
    """Length-sorted batches with a token budget (keeps padding cheap)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    batches, cur = [], []
    for i in order:
        cand = cur + [items[i]]
        L = max(len(s["input_ids"]) for s in cand)
        if cur and (L * len(cand) > max_tokens or len(cand) > max_bs):
            batches.append(cur)
            cur = [items[i]]
        else:
            cur = cand
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def run_model(model, batches, n_samples: int, n_off: int, pad_id: int,
              device: str, vocab: int) -> np.ndarray:
    """Return [n_samples, n_off, vocab] float16 logits, indexed by sample idx."""
    out = None
    for batch in batches:
        L = max(len(s["input_ids"]) for s in batch)
        B = len(batch)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(batch):
            n = len(s["input_ids"])
            ids[i, :n] = torch.tensor(s["input_ids"], dtype=torch.long)
            att[i, :n] = 1
        # Draft vocab may be smaller than some target-only ids (ChatML specials).
        ids_safe = ids.clone()
        ids_safe[ids_safe >= vocab] = 0
        logits = model(input_ids=ids_safe.to(device), attention_mask=att.to(device)).logits
        if out is None:
            out = np.zeros((n_samples, n_off, logits.shape[-1]), dtype=np.float16)
        for i, s in enumerate(batch):
            # logits[j] predicts token j+1 → answer token k is predicted at
            # index prompt_len - 1 + k
            base = s["prompt_len"] - 1
            for k in range(n_off):
                idx = base + k
                if idx < logits.shape[1] and k < s["n_answer_avail"]:
                    out[s["idx"], k] = logits[i, idx].float().cpu().numpy().astype(np.float16)
                else:
                    out[s["idx"], k] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="chatml",
                    choices=["plain", "plain_nl", "chatml", "chatml_bos", "draftchat"])
    ap.add_argument("--n-per-cluster", type=int, default=50)
    ap.add_argument("--offsets", type=int, default=3, help="cache logits for answer tokens 0..offsets-1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=8192, help="token budget per batch")
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--clusters", default=None, help="comma-separated subset (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "data" / "repro_cache" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(TARGET_DIR), trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    torch_dtype = DTYPES[args.dtype]
    print(f"loading target ({args.dtype}) …", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(TARGET_DIR), trust_remote_code=True, dtype=torch_dtype, attn_implementation="eager"
    )
    # transformers>=5 leaves the vendored rotary buffers uninitialized -> garbage
    # logits. See src/repro/bamboo_fix.py.
    print(f"repaired rotary modules: {fix_rotary(target)}", flush=True)
    target = target.to(args.device).eval()
    t_vocab = target.config.vocab_size
    print(f"loading draft ({args.dtype}) …", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(str(DRAFT_DIR), dtype=torch_dtype).to(args.device).eval()
    d_vocab = draft.config.vocab_size
    print(f"target vocab={t_vocab}  draft vocab={d_vocab}", flush=True)

    files = sorted(FLAN_VAL.glob("*.jsonl"))
    if args.clusters:
        keep = set(args.clusters.split(","))
        files = [f for f in files if f.name.replace("_validation.jsonl", "") in keep]

    t_start = time.time()
    for fi, f in enumerate(files):
        cluster = f.name.replace("_validation.jsonl", "").replace("_10templates", "")
        dst = out_dir / f"{cluster}.npz"
        if dst.exists():
            print(f"[{fi+1}/{len(files)}] {cluster}: cached, skip", flush=True)
            continue

        records = []
        with open(f) as fh:
            for line in fh:
                records.append(json.loads(line))
                if len(records) >= args.n_per_cluster:
                    break

        samples = []
        for i, rec in enumerate(records):
            prompt_ids = build_prompt_ids(tok, rec["inputs"], args.variant, args.max_prompt_len)
            answer_ids = tok.encode(rec["targets"], add_special_tokens=False)
            if not answer_ids:
                continue
            ctx = answer_ids[: args.offsets - 1]
            samples.append({
                "idx": len(samples),
                "input_ids": prompt_ids + ctx,
                "prompt_len": len(prompt_ids),
                "n_answer_avail": min(len(answer_ids), args.offsets),
                "gold": (answer_ids + [-1] * args.offsets)[: args.offsets],
            })
        if not samples:
            print(f"[{fi+1}/{len(files)}] {cluster}: no usable records, skip", flush=True)
            continue

        batches = make_batches(samples, args.max_tokens, args.max_bs)
        t0 = time.time()
        n = len(samples)
        t_logits = run_model(target, batches, n, args.offsets, pad_id, args.device, t_vocab)
        d_logits = run_model(draft, batches, n, args.offsets, pad_id, args.device, d_vocab)
        np.savez(
            dst,
            target_logits=t_logits,
            draft_logits=d_logits,
            gold_ids=np.array([s["gold"] for s in samples], dtype=np.int64),
            prompt_lens=np.array([s["prompt_len"] for s in samples], dtype=np.int64),
            variant=args.variant,
            dtype=args.dtype,
        )
        el = time.time() - t0
        print(f"[{fi+1}/{len(files)}] {cluster}: {len(samples)} samples, "
              f"{el:.1f}s ({el/len(samples):.2f}s/sample), total {(time.time()-t_start)/60:.1f}m",
              flush=True)

    print("done →", out_dir)


if __name__ == "__main__":
    main()
