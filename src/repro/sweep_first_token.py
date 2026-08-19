#!/usr/bin/env python3
"""
Step 2 of the first-token AR reproduction: sweep sampling parameters on the
cached logits and score every combination against the reference chart values.

AR = sum_v min(p_draft(v), p_target(v)) over the shared vocabulary (ids 0..31999),
with both distributions transformed by the sampling parameters first
(temperature -> top_k -> top_p -> renormalize), and NOT renormalized onto the
target's top-K.

Usage:
    python src/repro/sweep_first_token.py --cache data/repro_cache/chatml \
        --offsets 0,1 --t-target 0.7,1.0,1.3 --t-draft 0.7,1.0,1.3
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "results" / "reference_ar_first1_token.json"
SHARED_VOCAB = 32000
# the three rightmost bars of the reference chart are occluded by the legend box
LOW_CONFIDENCE = {"e2e_nlg", "anli_r1", "wic"}


def transform(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """logits [N, V] -> sampling distribution [N, V] (float32, rows sum to 1)."""
    if temperature <= 1e-3:          # greedy: distribution collapses to argmax
        p = torch.zeros_like(logits, dtype=torch.float32)
        p.scatter_(-1, logits.float().argmax(-1, keepdim=True), 1.0)
        return p
    x = logits.float() / temperature
    if top_k and top_k < x.shape[-1]:
        kth = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    p = torch.softmax(x, dim=-1)
    if top_p < 1.0:
        srt, idx = torch.sort(p, dim=-1, descending=True)
        cum = srt.cumsum(-1)
        # keep tokens up to and including the one that crosses top_p
        drop = cum - srt > top_p
        srt = srt.masked_fill(drop, 0.0)
        p = torch.zeros_like(p).scatter_(-1, idx, srt)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return p


def load_cache(cache_dir: Path, offset: int, device: str):
    """Return {cluster: (target_logits, draft_logits)} sliced to one offset."""
    out = {}
    for f in sorted(cache_dir.glob("*.npz")):
        z = np.load(f)
        t = z["target_logits"][:, offset, :]
        d = z["draft_logits"][:, offset, :]
        ok = ~(np.isnan(t).any(1) | np.isnan(d).any(1))
        if not ok.any():
            continue
        out[f.stem] = (
            torch.from_numpy(t[ok]).to(device),
            torch.from_numpy(d[ok]).to(device),
        )
    return out


def score(values: dict[str, float], ref: dict[str, float]) -> dict:
    keys = [k for k in values if k in ref and k not in LOW_CONFIDENCE]
    v = np.array([values[k] for k in keys])
    r = np.array([ref[k] for k in keys])
    mae = float(np.abs(v - r).mean())
    rmse = float(np.sqrt(((v - r) ** 2).mean()))
    pear = float(np.corrcoef(v, r)[0, 1]) if v.std() > 1e-9 else 0.0
    rv = np.argsort(np.argsort(v))
    rr = np.argsort(np.argsort(r))
    spear = float(np.corrcoef(rv, rr)[0, 1])
    allv = np.array([values[k] for k in values])
    return {"mae": mae, "rmse": rmse, "pearson": pear, "spearman": spear,
            "mean": float(allv.mean()), "std": float(allv.std()), "n": len(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache dir, e.g. data/repro_cache/chatml")
    ap.add_argument("--offsets", default="0", help="comma-separated answer-token offsets")
    ap.add_argument("--t-target", default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.5")
    ap.add_argument("--t-draft", default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.5")
    ap.add_argument("--top-p", default="1.0")
    ap.add_argument("--top-k", default="0")
    ap.add_argument("--vocab", default="full", choices=["full", "shared"],
                    help="full = softmax over each model's own vocab, then min over shared ids; "
                         "shared = slice logits to 32000 before softmax")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-n", type=int, default=15, help="how many best combos to print")
    ap.add_argument("--out", default=None, help="write full sweep table as JSON")
    args = ap.parse_args()

    ref = json.load(open(REFERENCE))
    cache_dir = Path(args.cache)
    offsets = [int(x) for x in args.offsets.split(",")]
    tts = [float(x) for x in args.t_target.split(",")]
    tds = [float(x) for x in args.t_draft.split(",")]
    tps = [float(x) for x in args.top_p.split(",")]
    tks = [int(x) for x in args.top_k.split(",")]

    rows = []
    for off in offsets:
        data = load_cache(cache_dir, off, args.device)
        if not data:
            print(f"offset {off}: no usable cache")
            continue
        print(f"offset {off}: {len(data)} clusters, "
              f"{sum(v[0].shape[0] for v in data.values())} samples", flush=True)
        for tt, td, tp, tk in itertools.product(tts, tds, tps, tks):
            per_cluster = {}
            for cl, (tl, dl) in data.items():
                if args.vocab == "shared":
                    pt = transform(tl[:, :SHARED_VOCAB], tt, tk, tp)
                    pd = transform(dl[:, :SHARED_VOCAB], td, tk, tp)
                else:
                    pt = transform(tl, tt, tk, tp)[:, :SHARED_VOCAB]
                    pd = transform(dl, td, tk, tp)[:, :SHARED_VOCAB]
                per_cluster[cl] = float(torch.minimum(pt, pd).sum(-1).mean())
            s = score(per_cluster, ref)
            s.update({"offset": off, "t_target": tt, "t_draft": td, "top_p": tp, "top_k": tk,
                      "values": per_cluster})
            rows.append(s)

    rows.sort(key=lambda r: r["mae"])
    print(f"\n{'off':>3} {'T_tgt':>6} {'T_drf':>6} {'top_p':>6} {'top_k':>6} "
          f"{'MAE':>6} {'RMSE':>6} {'pear':>6} {'spear':>6} {'mean':>6} {'std':>6}")
    print("-" * 78)
    for r in rows[:args.top_n]:
        print(f"{r['offset']:>3} {r['t_target']:>6.2f} {r['t_draft']:>6.2f} {r['top_p']:>6.2f} "
              f"{r['top_k']:>6d} {r['mae']:>6.3f} {r['rmse']:>6.3f} {r['pearson']:>6.3f} "
              f"{r['spearman']:>6.3f} {r['mean']:>6.3f} {r['std']:>6.3f}")
    best_corr = max(rows, key=lambda r: r["pearson"])
    print(f"\nbest pearson: off={best_corr['offset']} T_tgt={best_corr['t_target']} "
          f"T_drf={best_corr['t_draft']} top_p={best_corr['top_p']} top_k={best_corr['top_k']} "
          f"r={best_corr['pearson']:.3f} MAE={best_corr['mae']:.3f} "
          f"mean={best_corr['mean']:.3f} std={best_corr['std']:.3f}")
    print("reference: mean=0.373 std=0.181")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
