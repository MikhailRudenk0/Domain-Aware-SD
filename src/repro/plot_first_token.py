#!/usr/bin/env python3
"""
Step 4 of the first-token AR reproduction: plot reproduced vs reference values.

Input is a JSON produced by ``sweep_first_token.py --out`` (a list of sweep
rows); the row to plot is picked by ``--pick`` (``mae`` = lowest MAE, or an
explicit index), or a plain ``{cluster: value}`` mapping.

Usage:
    python src/repro/plot_first_token.py --sweep results/sweep_chatml.json \
        --out results/ar_per_cluster_first1_token_REPRO.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "results" / "reference_ar_first1_token.json"
LOW_CONFIDENCE = {"e2e_nlg", "anli_r1", "wic"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--pick", default="mae", help="'mae', 'pearson' or a row index")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "ar_per_cluster_first1_token_REPRO.png"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    ref = json.load(open(REFERENCE))
    blob = json.load(open(args.sweep))
    if isinstance(blob, list):
        if args.pick == "mae":
            row = min(blob, key=lambda r: r["mae"])
        elif args.pick == "pearson":
            row = max(blob, key=lambda r: r["pearson"])
        else:
            row = blob[int(args.pick)]
        vals, meta = row["values"], row
    else:
        vals, meta = blob, {}

    names = [k for k in ref if k in vals]          # reference order (ascending by SFT)
    rep = np.array([vals[k] for k in names])
    rf = np.array([ref[k] for k in names])
    good = np.array([k not in LOW_CONFIDENCE for k in names])

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - 0.2, rf, width=0.4, label="reference (old chart)", color="#7f9fc4")
    ax.bar(x + 0.2, rep, width=0.4, label="reproduced", color="#c47f7f")
    for i in np.where(~good)[0]:
        ax.axvspan(i - 0.5, i + 0.5, color="0.9", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Acceptance Rate")
    ax.set_xlabel("Clusters")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")

    mae = float(np.abs(rep[good] - rf[good]).mean())
    pear = float(np.corrcoef(rep[good], rf[good])[0, 1])
    title = args.title or "Acceptance Rate per Cluster (first 1 token) — reproduction vs reference"
    ax.set_title(title)
    txt = (f"reference: mean={rf.mean():.3f}, std={rf.std():.3f}\n"
           f"reproduced: mean={rep.mean():.3f}, std={rep.std():.3f}\n"
           f"MAE={mae:.3f}  pearson r={pear:.3f}")
    if meta:
        txt += (f"\npos={meta.get('pos', meta.get('offset'))} T_target={meta.get('t_target')} "
                f"T_draft={meta.get('t_draft')} top_p={meta.get('top_p')} top_k={meta.get('top_k')} "
                f"metric={meta.get('metric')}")
    ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.5"), va="bottom")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)
    print(txt)


if __name__ == "__main__":
    main()
