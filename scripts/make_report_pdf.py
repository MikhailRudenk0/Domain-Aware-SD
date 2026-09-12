"""Short results-only PDF for the domain-drafter training report.

Renders results/REPORT_domain_drafters_short.pdf — a 2-page A4 summary of
results/REPORT_domain_drafters.md, keeping only the result tables (§4.4, §5.2)
plus training curves. Numbers below are transcribed from that report.

Usage (from the project root):
    conda run -n domain_sd python scripts/make_report_pdf.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np

# ---- design tokens (light surface) ----
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8983"
RULE      = "#e3e2dd"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
BLUE_RAMP = ["#ffffff", "#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4"]  # sequential, light->dark

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

MODELS = ["Understanding", "Text Reformulation", "Mixed (U+T)"]
COLORS = [S1, S2, S3]

# ---- data ----
curves = {
    "Understanding":      dict(ep=[2.0,2.5,3.0,3.5,4.0,4.5],
                               loss=[1.282,1.232,1.213,1.202,1.196,1.193],
                               acc=[64.76,64.83,64.93,64.99,65.02,65.05]),
    "Text Reformulation": dict(ep=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                               loss=[2.198,2.165,2.156,2.152,2.151,2.151,2.151,2.151,2.151,2.151],
                               acc=[53.77,54.15,54.28,54.33,54.33,54.35,54.34,54.35,54.34,54.34]),
    "Mixed (U+T)":        dict(ep=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                               loss=[1.865,1.818,1.799,1.791,1.789,1.789,1.788,1.788,1.788,1.788],
                               acc=[56.23,56.51,56.59,56.63,56.64,56.64,56.65,56.64,56.64,56.64]),
}

train_summary = [
    ["Understanding",      "21", "395,280", "1.193", "65.05%", "~3.5", "4.3 h"],
    ["Text Reformulation", "11", "327,330", "2.151", "54.34%", "~2.0", "4.2 h"],
    ["Mixed (U+T)",        "32", "722,610", "1.788", "56.64%", "~2.5", "12.6 h"],
]
train_cols = ["Drafter", "Clusters", "Train samples", "Final eval_loss",
              "Final top-1 acc", "Plateau epoch", "Train time"]

EVAL_DOMAINS = ["Understanding", "Text Reform.", "Mixed (U+T)"]
overlap = np.array([[0.7558, 0.6676, 0.7009],
                    [0.7091, 0.7026, 0.7046],
                    [0.7510, 0.6997, 0.7191]])
top1    = np.array([[0.6887, 0.5297, 0.5875],
                    [0.6239, 0.5897, 0.5997],
                    [0.6895, 0.5848, 0.6223]])
kl      = np.array([[0.5602, 1.0404, 0.8679],
                    [0.8165, 0.8200, 0.8370],
                    [0.5590, 0.8353, 0.7372]])


def shade(v, vmin, vmax, invert=False):
    """Map a value onto the light end of the blue sequential ramp."""
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    if invert:
        t = 1.0 - t
    idx = min(len(BLUE_RAMP) - 1, int(round(t * (len(BLUE_RAMP) - 1))))
    return BLUE_RAMP[idx]


def draw_table(ax, x0, y0, w, row_h, cols, rows, col_w, cell_bg=None,
               bold_cells=None):
    """Plain table: header rule + zebra-free rows, left col left-aligned."""
    bold_cells = bold_cells or set()
    n_rows = len(rows)
    xs = [x0]
    for cw in col_w:
        xs.append(xs[-1] + cw * w)
    # header
    y_head = y0
    for j, c in enumerate(cols):
        ha = "left" if j == 0 else "center"
        xt = xs[j] + 0.012 * w if j == 0 else (xs[j] + xs[j + 1]) / 2
        ax.text(xt, y_head + row_h * 0.34, c, ha=ha, va="center",
                fontsize=7.6, color=INK_MUTED, weight="bold")
    ax.plot([x0, x0 + w], [y_head, y_head], color=INK, lw=1.0, zorder=3)
    # rows
    for i, row in enumerate(rows):
        yt = y_head - (i + 1) * row_h
        for j, val in enumerate(row):
            if cell_bg is not None and j > 0:
                bg = cell_bg(i, j - 1)
                if bg:
                    ax.add_patch(Rectangle((xs[j] + 0.002, yt + 0.002),
                                           xs[j + 1] - xs[j] - 0.004, row_h - 0.004,
                                           facecolor=bg, edgecolor="none", zorder=0))
            ha = "left" if j == 0 else "center"
            xt = xs[j] + 0.012 * w if j == 0 else (xs[j] + xs[j + 1]) / 2
            is_bold = (i, j - 1) in bold_cells and j > 0
            ax.text(xt, yt + row_h / 2, val, ha=ha, va="center",
                    fontsize=8.4, color=INK if (is_bold or j == 0) else INK_2,
                    weight="bold" if is_bold or j == 0 else "normal", zorder=2)
        ax.plot([x0, x0 + w], [yt, yt], color=RULE, lw=0.6, zorder=1)
    return y_head - n_rows * row_h


def matrix_block(ax, y_top, title, note, M, fmt="{:.4f}", invert=False):
    ax.text(0.0, y_top, title, fontsize=10.5, weight="bold", color=INK, va="bottom")
    ax.text(1.0, y_top + 0.004, note, fontsize=7.8, color=INK_MUTED,
            va="bottom", ha="right")
    rows = [[MODELS[i]] + [fmt.format(M[i, j]) for j in range(3)] for i in range(3)]
    cols = ["Drafter  \\  Eval domain"] + EVAL_DOMAINS
    vmin, vmax = M.min(), M.max()
    best = set()
    for j in range(3):
        i_best = int(np.argmin(M[:, j]) if invert else np.argmax(M[:, j]))
        best.add((i_best, j))
    return draw_table(
        ax, 0.0, y_top - 0.018, 1.0, 0.030, cols, rows,
        col_w=[0.34, 0.22, 0.22, 0.22],
        cell_bg=lambda i, j: shade(M[i, j], vmin, vmax, invert=invert),
        bold_cells=best,
    )


def page_ax(fig):
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.90])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.grid(False)
    return ax


OUT = "results/REPORT_domain_drafters_short.pdf"
with PdfPages(OUT) as pdf:
    # ---------------- page 1: results tables ----------------
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = page_ax(fig)

    ax.text(0.0, 0.985, "Domain-Aware Speculative Decoding", fontsize=19,
            weight="bold", color=INK, va="top")
    ax.text(0.0, 0.950, "Domain drafters — results summary", fontsize=11.5,
            color=INK_2, va="top")
    ax.plot([0, 1], [0.928, 0.928], color=INK, lw=1.2)
    ax.text(0.0, 0.916,
            "Target: TurboSparse-Mistral-Instruct (7B)   •   Drafter: Lite-Mistral-150M-v2-Instruct (156M)   •   "
            "Loss: 0.5·CE + 0.5·KL (T=1.0)",
            fontsize=7.8, color=INK_MUTED, va="top")

    # training summary
    ax.text(0.0, 0.876, "Training summary", fontsize=12, weight="bold",
            color=INK, va="bottom")
    y = draw_table(ax, 0.0, 0.858, 1.0, 0.030, train_cols, train_summary,
                   col_w=[0.235, 0.105, 0.155, 0.145, 0.135, 0.13, 0.095])

    # AR matrices
    y = matrix_block(ax, y - 0.055, "Overlap area (acceptance-rate proxy)",
                     "higher is better", overlap)
    y = matrix_block(ax, y - 0.055, "Top-1 match", "higher is better", top1)
    y = matrix_block(ax, y - 0.055, "KL divergence", "lower is better", kl,
                     fmt="{:.4f}", invert=True)

    # deltas
    y -= 0.058
    ax.text(0.0, y, "Domain drafter vs. mixed baseline (overlap area)",
            fontsize=12, weight="bold", color=INK, va="bottom")
    y -= 0.020
    deltas = [
        ("Understanding domain", "0.7558", "0.7510", "+0.0048", S1),
        ("Text Reformulation domain", "0.7026", "0.6997", "+0.0029", S2),
    ]
    for name, dom, mix, d, col in deltas:
        y -= 0.030
        ax.add_patch(Rectangle((0.0, y), 0.006, 0.024, facecolor=col,
                               edgecolor="none"))
        ax.text(0.022, y + 0.012, name, fontsize=9, color=INK, va="center",
                weight="bold")
        ax.text(0.50, y + 0.012, f"domain {dom}   vs   mixed {mix}",
                fontsize=8.6, color=INK_2, va="center", ha="center")
        ax.text(1.0, y + 0.012, d, fontsize=9.4, color=INK, va="center",
                ha="right", weight="bold")

    y -= 0.048
    ax.text(0.0, y,
            "Each domain-specific drafter leads the mixed baseline on its own domain; the mixed model\n"
            "stays the best general-purpose fallback and wins on the combined domain (0.7191).",
            fontsize=8.6, color=INK_2, va="top", linespacing=1.5)

    ax.text(0.0, 0.005, "Metric: overlap_area = 1 − TVD between drafter and target top-10 distributions, "
                        "teacher-forced on 11,900 validation samples.",
            fontsize=7, color=INK_MUTED, va="bottom")
    pdf.savefig(fig); plt.close(fig)

    # ---------------- page 2: loss / accuracy curves ----------------
    fig = plt.figure(figsize=(8.27, 11.69))
    head = fig.add_axes([0.07, 0.93, 0.86, 0.05]); head.axis("off")
    head.set_xlim(0, 1); head.set_ylim(0, 1); head.grid(False)
    head.text(0.0, 0.75, "Training curves", fontsize=19, weight="bold",
              color=INK, va="top")
    head.text(0.0, 0.20, "Validation metrics, evaluated twice per epoch",
              fontsize=10.5, color=INK_2, va="top")
    head.plot([0, 1], [-0.05, -0.05], color=INK, lw=1.2, clip_on=False)

    ax1 = fig.add_axes([0.12, 0.520, 0.62, 0.30])
    ax2 = fig.add_axes([0.12, 0.135, 0.62, 0.30])

    for name, col in zip(MODELS, COLORS):
        c = curves[name]
        ax1.plot(c["ep"], c["loss"], color=col, lw=2.0, marker="o", ms=4.5,
                 mec=SURFACE, mew=1.4, label=name, clip_on=False, zorder=3)
        ax2.plot(c["ep"], c["acc"], color=col, lw=2.0, marker="o", ms=4.5,
                 mec=SURFACE, mew=1.4, label=name, clip_on=False, zorder=3)

    for ax_, ttl, sub in ((ax1, "Evaluation loss", "0.5·CE + 0.5·KL, lower is better"),
                          (ax2, "Top-1 accuracy", "argmax agreement with target, higher is better")):
        ax_.set_xlabel("Epoch", fontsize=9)
        ax_.set_xlim(0.28, 5.15)
        ax_.set_xticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        ax_.tick_params(labelsize=8.5, length=0)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
        ax_.spines["left"].set_color(RULE)
        ax_.spines["bottom"].set_color(RULE)
        ax_.set_axisbelow(True)
        ax_.set_title(ttl, fontsize=12, weight="bold", color=INK, loc="left", pad=18)
        ax_.text(0.0, 1.015, sub, transform=ax_.transAxes, fontsize=8.2,
                 color=INK_MUTED, va="bottom")

    ax1.set_ylabel("eval_loss", fontsize=9)
    ax1.set_ylim(1.10, 2.28)
    ax2.set_ylabel("top-1 accuracy, %", fontsize=9)
    ax2.set_ylim(53.2, 65.8)

    # direct labels at the end of each line
    for name, col in zip(MODELS, COLORS):
        c = curves[name]
        ax1.text(c["ep"][-1] + 0.08, c["loss"][-1], name, color=INK,
                 fontsize=8.4, va="center", ha="left", weight="bold")
        ax2.text(c["ep"][-1] + 0.08, c["acc"][-1], name, color=INK,
                 fontsize=8.4, va="center", ha="left", weight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
               loc="upper left", bbox_to_anchor=(0.07, 0.905),
               labelcolor=INK_2, handlelength=1.8, columnspacing=2.2)

    foot = fig.add_axes([0.07, 0.025, 0.86, 0.045]); foot.axis("off")
    foot.set_xlim(0, 1); foot.set_ylim(0, 1); foot.grid(False)
    foot.text(0.0, 1.0,
              "Understanding was launched for 25 epochs and stopped early at epoch 4.5 on plateau; its curve therefore\n"
              "starts at the first retained checkpoint (epoch 2.0). All three models plateau within 2–3.5 epochs.",
              fontsize=8, color=INK_MUTED, va="top", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)

print("written:", OUT)
