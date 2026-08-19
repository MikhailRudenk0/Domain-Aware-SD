#!/usr/bin/env python3
"""
Sanity-check the generated synthetic datasets: is the trunk text reasonable?

For every file of a split, decode a few trunks with the target tokenizer and
compute quality heuristics:

* frac_ascii        — fraction of ASCII characters in the decoded trunk (garbage
                      generations are full of CJK/Cyrillic pieces on English tasks)
* frac_repeat       — fraction of positions where a 4-gram of token ids repeats
                      an earlier 4-gram (degenerate loops)
* trunk_len         — mean generated length
* top1_prob         — mean captured top-1 probability (confident target ≈ 0.6+;
                      a broken target is near-uniform ≈ 0.001)
* oov_rate          — fraction of trunk ids outside the target vocab (must be 0)

Works for both storage formats: JSONL (validation) and NPZ (train/test).

Usage:
    python src/repro/validate_synthetic.py --split data/synthetic/v1/validation --n-files 66
    python src/repro/validate_synthetic.py --split data/synthetic/train/v2 --sample-text 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def iter_records_jsonl(path: Path, limit: int):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            r = json.loads(line)
            probs = [row[0] if row else None for row in r.get("top10_probs", [])]
            yield r["prompt"], r["reference"], list(r["trunk"]), probs


def iter_records_npz(path: Path, limit: int):
    z = np.load(path, allow_pickle=True)
    lens = z["trunk_lens"]
    ids = z["trunk_ids"]
    probs_q = z["top10_probs_q"]
    mask = z["top10_mask"]
    # uint16 quantized probs: value / 65535
    off = 0
    for i in range(min(limit, len(lens))):
        L = int(lens[i])
        seg = ids[off:off + L].astype(np.int64).tolist()
        p1 = [(float(probs_q[off + j, 0]) / 65535.0) if mask[off + j] else None for j in range(L)]
        yield str(z["prompts"][i]), str(z["references"][i]), seg, p1
        off += L


def ngram_repeat_frac(ids: list[int], n: int = 4) -> float:
    if len(ids) < 2 * n:
        return 0.0
    seen = set()
    rep = 0
    total = 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        total += 1
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-files", type=int, default=66)
    ap.add_argument("--n-per-file", type=int, default=25)
    ap.add_argument("--sample-text", type=int, default=0,
                    help="print N decoded trunks per file (first files only)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "TurboSparse-Mistral-Instruct"),
                                        trust_remote_code=True)
    vocab = 32064

    split = Path(args.split)
    files = sorted(list(split.glob("*.jsonl")) + list(split.glob("*.npz")))[: args.n_files]
    print(f"split={split}  files={len(files)}")
    print(f"{'cluster':38s} {'n':>4} {'len':>6} {'ascii':>6} {'rep4':>6} {'top1':>6} {'oov':>5}")

    agg = []
    for fi, f in enumerate(files):
        it = iter_records_jsonl(f, args.n_per_file) if f.suffix == ".jsonl" \
            else iter_records_npz(f, args.n_per_file)
        lens, asciis, reps, top1s, oovs, texts = [], [], [], [], [], []
        for prompt, refr, trunk, p1 in it:
            if not trunk:
                continue
            text = tok.decode(trunk)
            lens.append(len(trunk))
            asciis.append(sum(c.isascii() for c in text) / max(len(text), 1))
            reps.append(ngram_repeat_frac(trunk))
            valid_p = [p for p in p1 if p is not None]
            if valid_p:
                top1s.append(float(np.mean(valid_p)))
            oovs.append(sum(t >= vocab for t in trunk) / len(trunk))
            texts.append((prompt, refr, text))
        name = f.stem.replace("_10templates", "")
        row = dict(cluster=name, n=len(lens), length=float(np.mean(lens)) if lens else 0,
                   ascii=float(np.mean(asciis)) if asciis else 0,
                   rep4=float(np.mean(reps)) if reps else 0,
                   top1=float(np.mean(top1s)) if top1s else float("nan"),
                   oov=float(np.mean(oovs)) if oovs else 0)
        agg.append(row)
        print(f"{name:38.38s} {row['n']:>4} {row['length']:>6.1f} {row['ascii']:>6.3f} "
              f"{row['rep4']:>6.3f} {row['top1']:>6.3f} {row['oov']:>5.3f}")
        if args.sample_text and fi < 3:
            for prompt, refr, text in texts[: args.sample_text]:
                print(f"    PROMPT: {prompt[:110]!r}")
                print(f"    REF   : {refr[:110]!r}")
                print(f"    TRUNK : {text[:220]!r}")

    a = {k: float(np.nanmean([r[k] for r in agg])) for k in ("length", "ascii", "rep4", "top1", "oov")}
    print(f"\nOVERALL: len={a['length']:.1f} ascii={a['ascii']:.3f} rep4={a['rep4']:.3f} "
          f"top1={a['top1']:.3f} oov={a['oov']:.4f}")
    bad = [r["cluster"] for r in agg if r["ascii"] < 0.85 or r["rep4"] > 0.5 or (r["top1"] == r["top1"] and r["top1"] < 0.2)]
    print("suspicious clusters:", bad if bad else "none")


if __name__ == "__main__":
    main()
