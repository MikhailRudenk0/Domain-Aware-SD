#!/usr/bin/env python3
"""
Step 3 of the first-token AR reproduction: exhaustive search over the sampling
parameters, the answer-token offset and the metric definition, scored against
the reference chart values.

Search space
------------
* answer-token ``offset`` (0..N-1 as cached) and, optionally, the running mean
  over offsets 0..k (``--windows``)
* ``T_target`` x ``T_draft`` grid (independent)
* ``top_p``, ``top_k`` (applied to both models)
* metric: ``overlap_full`` (sum of min over the shared 32000-token vocabulary),
  ``overlap_top10`` (both distributions renormalized on the target's top-10 —
  what src/eval currently does), ``overlap_top10_draftraw``

Usage:
    python src/repro/optimize_first_token.py --cache data/repro_cache/plain_full \
        --out results/repro_best_first1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "results" / "reference_ar_first1_token.json"
SHARED_VOCAB = 32000
LOW_CONFIDENCE = {"e2e_nlg", "anli_r1", "wic"}   # legend-occluded bars


def transform(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 1e-3:
        p = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
        p.scatter_(-1, logits.float().argmax(-1, keepdim=True), 1.0)
        return p
    x = logits.float() / temperature
    if top_k and top_k < x.shape[-1]:
        kth = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    p = torch.softmax(x, dim=-1)
    if top_p < 1.0:
        srt, idx = torch.sort(p, dim=-1, descending=True)
        drop = srt.cumsum(-1) - srt > top_p
        srt = srt.masked_fill(drop, 0.0)
        p = torch.zeros_like(p).scatter_(-1, idx, srt)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return p


def ar(pt: torch.Tensor, pd: torch.Tensor, metric: str) -> torch.Tensor:
    pt, pd = pt[:, :SHARED_VOCAB], pd[:, :SHARED_VOCAB]
    if metric == "overlap_full":
        return torch.minimum(pt, pd).sum(-1)
    tk = torch.topk(pt, 10, dim=-1)
    t = tk.values / tk.values.sum(-1, keepdim=True).clamp_min(1e-12)
    d = torch.gather(pd, 1, tk.indices)
    if metric == "overlap_top10_draftraw":
        return torch.minimum(d, t).sum(-1)
    d = d / d.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.minimum(d, t).sum(-1)


def score(values: dict[str, float], ref: dict[str, float]) -> dict:
    keys = [k for k in values if k in ref and k not in LOW_CONFIDENCE]
    v = np.array([values[k] for k in keys])
    r = np.array([ref[k] for k in keys])
    allv = np.array(list(values.values()))
    return {
        "mae": float(np.abs(v - r).mean()),
        "rmse": float(np.sqrt(((v - r) ** 2).mean())),
        "pearson": float(np.corrcoef(v, r)[0, 1]) if v.std() > 1e-9 else 0.0,
        "spearman": float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(r)))[0, 1]),
        "mean": float(allv.mean()), "std": float(allv.std()), "n_scored": len(keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--offsets", default="0,1,2,3", help="single offsets to try")
    ap.add_argument("--windows", default="", help="also try running means 0..k, e.g. '1,2,3'")
    ap.add_argument("--temps", default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.7,2.0")
    ap.add_argument("--top-p", default="1.0,0.95")
    ap.add_argument("--top-k", default="0")
    ap.add_argument("--metrics", default="overlap_full,overlap_top10,overlap_top10_draftraw")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = json.load(open(REFERENCE))
    temps = [float(x) for x in args.temps.split(",")]
    tps = [float(x) for x in args.top_p.split(",")]
    tks = [int(x) for x in args.top_k.split(",")]
    mets = args.metrics.split(",")
    offs = [int(x) for x in args.offsets.split(",") if x != ""]
    wins = [int(x) for x in args.windows.split(",") if x != ""]

    # cache[offset][cluster] = (target_logits, draft_logits) on device
    need = sorted(set(offs) | set(range(max(wins) + 1)) if wins else set(offs))
    cache: dict[int, dict[str, tuple]] = {o: {} for o in need}
    for f in sorted(Path(args.cache).glob("*.npz")):
        z = np.load(f)
        for o in need:
            t, d = z["target_logits"][:, o, :], z["draft_logits"][:, o, :]
            ok = ~(np.isnan(t).any(1) | np.isnan(d).any(1))
            if ok.any():
                cache[o][f.stem] = (torch.from_numpy(t[ok]).to(args.device),
                                    torch.from_numpy(d[ok]).to(args.device))
    print(f"loaded offsets {need}: " +
          ", ".join(f"off{o}={sum(v[0].shape[0] for v in cache[o].values())} samples" for o in need),
          flush=True)

    rows = []
    for tp in tps:
        for tk in tks:
            for met in mets:
                for tt in temps:
                    pts = {o: {cl: transform(v[0], tt, tk, tp) for cl, v in cache[o].items()} for o in need}
                    for td in temps:
                        per_off = {}
                        for o in need:
                            per_off[o] = {}
                            for cl, v in cache[o].items():
                                pd = transform(v[1], td, tk, tp)
                                per_off[o][cl] = float(ar(pts[o][cl], pd, met).mean())
                        for o in offs:
                            rows.append(dict(score(per_off[o], ref), pos=f"off{o}", t_target=tt,
                                             t_draft=td, top_p=tp, top_k=tk, metric=met,
                                             values=per_off[o]))
                        for k in wins:
                            cum = {}
                            for cl in per_off[0]:
                                vals = [per_off[o][cl] for o in range(k + 1) if cl in per_off[o]]
                                cum[cl] = float(np.mean(vals))
                            rows.append(dict(score(cum, ref), pos=f"mean0..{k}", t_target=tt,
                                             t_draft=td, top_p=tp, top_k=tk, metric=met, values=cum))
                    del pts
                    torch.cuda.empty_cache()
                print(f"  done top_p={tp} top_k={tk} metric={met} T_t={tt}", flush=True)

    rows.sort(key=lambda r: r["mae"])
    hdr = (f"{'pos':>10} {'metric':>22} {'T_tgt':>6} {'T_drf':>6} {'top_p':>6} {'top_k':>6} "
           f"{'MAE':>6} {'RMSE':>6} {'pear':>6} {'spear':>6} {'mean':>6} {'std':>6}")
    print("\n=== best by MAE ===\n" + hdr)
    for r in rows[:args.top_n]:
        print(f"{r['pos']:>10} {r['metric']:>22} {r['t_target']:>6.2f} {r['t_draft']:>6.2f} "
              f"{r['top_p']:>6.2f} {r['top_k']:>6d} {r['mae']:>6.3f} {r['rmse']:>6.3f} "
              f"{r['pearson']:>6.3f} {r['spearman']:>6.3f} {r['mean']:>6.3f} {r['std']:>6.3f}")
    by_r = sorted(rows, key=lambda r: -r["pearson"])
    print("\n=== best by Pearson ===\n" + hdr)
    for r in by_r[:args.top_n]:
        print(f"{r['pos']:>10} {r['metric']:>22} {r['t_target']:>6.2f} {r['t_draft']:>6.2f} "
              f"{r['top_p']:>6.2f} {r['top_k']:>6d} {r['mae']:>6.3f} {r['rmse']:>6.3f} "
              f"{r['pearson']:>6.3f} {r['spearman']:>6.3f} {r['mean']:>6.3f} {r['std']:>6.3f}")
    print("\nreference: mean=0.373 std=0.181")

    if args.out:
        keep = rows[:50] + by_r[:50]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(keep, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
