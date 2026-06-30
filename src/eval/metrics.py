"""
Per-position similarity metrics between a draft and a target distribution.

All metrics are pure functions on numpy arrays. The runner extracts the right
inputs (full vs top-K) and dispatches to the registered metric.

Metric contract
---------------

Each metric is registered in ``METRICS`` and accepts a single ``MetricInputs``
struct. Not every field is used by every metric; the runner is responsible for
filling whatever the chosen metrics need (see ``required_fields`` below).

Two operating regimes:

* **Full**  — ``draft_full`` and ``target_full`` are populated; both span the
  same vocabulary support (caller must align them).
* **Top-K** — ``target_topk_ids`` / ``target_topk_probs`` and
  ``draft_at_target_topk`` are populated; the comparison runs over those K
  token ids only. Target probs are assumed renormalized within the K support
  (so they sum to 1); ``draft_at_target_topk`` is renormalized the same way.

In both regimes:

* ``draft_argmax_id`` / ``draft_topk_ids`` come from the full draft
  distribution and are used by ``top1_match`` / ``topk_overlap``.
* ``target_topk_ids`` is always populated (in full mode, it is the top-K of
  the full target distribution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

_EPS = 1e-12


@dataclass
class MetricInputs:
    # Aligned distributions (same support, summing to 1):
    #   - full mode:  both [V] over the full vocab
    #   - top-K mode: both [K], renormalized within target's top-K
    draft_aligned: Optional[np.ndarray] = None
    target_aligned: Optional[np.ndarray] = None

    # argmax / top-K of the full draft distribution
    draft_argmax_id: Optional[int] = None
    draft_topk_ids: Optional[np.ndarray] = None  # [K] — same K as target_topk_ids

    # Target top-K ids (in full mode, top-K of the full target distribution)
    target_topk_ids: Optional[np.ndarray] = None  # [K]


def overlap_area(inp: MetricInputs) -> float:
    """1 - TVD (total variation distance) between aligned distributions.

    Equal to the expected acceptance probability of an ideal speculative
    decoder when draft and target are perfectly comparable distributions.
    """
    d = inp.draft_aligned
    t = inp.target_aligned
    return float(np.minimum(d, t).sum())


def top1_match(inp: MetricInputs) -> float:
    """1.0 if draft's argmax token id matches target's argmax id, else 0.0."""
    target_argmax = int(inp.target_topk_ids[0])
    return float(int(inp.draft_argmax_id) == target_argmax)


def topk_overlap(inp: MetricInputs) -> float:
    """Fraction of target's top-K ids that appear in draft's top-K."""
    d_ids = set(int(x) for x in inp.draft_topk_ids.tolist())
    t_ids = inp.target_topk_ids.tolist()
    if not t_ids:
        return 0.0
    hits = sum(1 for tid in t_ids if int(tid) in d_ids)
    return hits / len(t_ids)


def kl(inp: MetricInputs) -> float:
    """KL(target || draft). Aligned support; eps-floor on the draft."""
    d = inp.draft_aligned
    t = inp.target_aligned
    mask = t > 0
    if not mask.any():
        return 0.0
    return float((t[mask] * (np.log(t[mask] + _EPS) - np.log(d[mask] + _EPS))).sum())


METRICS: Dict[str, Callable[[MetricInputs], float]] = {
    "overlap_area": overlap_area,
    "top1_match": top1_match,
    "topk_overlap": topk_overlap,
    "kl": kl,
}


# Which fields each metric reads. Used by the runner to skip work.
REQUIRED_FIELDS: Dict[str, tuple] = {
    "overlap_area": ("draft_aligned", "target_aligned"),
    "top1_match": ("draft_argmax_id", "target_topk_ids"),
    "topk_overlap": ("draft_topk_ids", "target_topk_ids"),
    "kl": ("draft_aligned", "target_aligned"),
}


def get_metric(name: str) -> Callable[[MetricInputs], float]:
    if name not in METRICS:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
    return METRICS[name]
