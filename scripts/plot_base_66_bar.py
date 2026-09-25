"""Bar chart: base (untrained) drafter metrics on all 66 Flan clusters (model mode).

Reads eval_results/model_mode_v2/ and produces a 4-panel bar chart
(overlap_area, top1_match, topk_overlap, kl) with one bar per cluster,
sorted by overlap_area descending.

Output: results/bar_base_66_clusters.pdf
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── design tokens ────────────────────────────────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8983"
RULE      = "#e3e2dd"
BAR_COLOR = "#2a78d6"
BAR_KL    = "#e05252"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": RULE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
})

# ── load data ────────────────────────────────────────────────────────────────
DATA_DIR = Path("eval_results/model_mode_v2")
files = sorted(glob.glob(str(DATA_DIR / "eval__Lite-Mistral-150M-v2-Instruct__*.json")))

rows = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    r = d["results"]
    cluster = Path(f).stem.split("__")[-1].replace("_10templates", "")
    oa = np.nanmean([v for v in r["single__overlap_area__full_target"] if v is not None])
    t1 = np.nanmean([v for v in r["single__top1_match__full_target"] if v is not None])
    tk = np.nanmean([v for v in r["single__topk_overlap__full_target"] if v is not None])
    kl = np.nanmean([v for v in r["single__kl__full_target"] if v is not None])
    rows.append((cluster, oa, t1, tk, kl))

# sort by overlap_area descending
rows.sort(key=lambda x: -x[1])

names = [r[0] for r in rows]
oa_vals = np.array([r[1] for r in rows])
t1_vals = np.array([r[2] for r in rows])
tk_vals = np.array([r[3] for r in rows])
kl_vals = np.array([r[4] for r in rows])

N = len(names)
y = np.arange(N)

# ── horizontal bar: overlap_area only, clusters readable on Y axis ───────────
fig, ax = plt.subplots(figsize=(10, 18))

# Reverse so highest is at the top
y_plot = y[::-1]

vmin, vmax = float(oa_vals.min()), float(oa_vals.max())
margin = (vmax - vmin) * 0.25

ax.barh(y_plot, oa_vals, color=BAR_COLOR, alpha=0.85, height=0.75, edgecolor="none")

mean_val = float(np.mean(oa_vals))
ax.axvline(mean_val, color=INK_MUTED, ls="--", lw=1.2, zorder=5)
ax.text(mean_val, N + 0.3, f"mean = {mean_val:.4f}", ha="center",
        fontsize=8.5, color=INK_MUTED)

# Value labels on each bar
for i in range(N):
    ax.text(oa_vals[i] + 0.0005, y_plot[i], f"{oa_vals[i]:.4f}",
            va="center", ha="left", fontsize=6.5, color=INK_2)

ax.set_xlim(max(0, vmin - margin), vmax + margin * 1.5)
ax.set_yticks(y_plot)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Overlap Area (1 − TVD),  higher = better", fontsize=10)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(RULE)
ax.spines["bottom"].set_color(RULE)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="x")

ax.set_title(
    "Base drafter (untrained) — Overlap Area per cluster\n"
    "Target: TurboSparse-Mistral-Instruct (7B)  •  66 Flan clusters, model mode",
    fontsize=11, weight="bold", color=INK, loc="left", pad=12,
)

fig.tight_layout()

OUT = Path("results/bar_base_66_clusters.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Written: {OUT}")
print(f"  clusters: {N}")
print(f"  overlap_area: mean={float(oa_vals.mean()):.4f}, min={float(oa_vals.min()):.4f}, max={float(oa_vals.max()):.4f}")
