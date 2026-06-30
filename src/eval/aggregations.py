"""
Aggregation strategies for combining draft distributions from multiple models.

Each aggregation takes a list of distributions (each shape [D]) and returns
a single distribution of the same shape, normalized to sum to 1.

The ``best`` strategy is special: it is *not* implemented here because it
operates at the metric level rather than the distribution level (per-position
max of the metric across models). The runner handles it directly.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

_EPS = 1e-12


def _normalize(x: np.ndarray) -> np.ndarray:
    s = x.sum()
    if s <= 0:
        # uniform fallback if everything collapsed (defensive)
        return np.full_like(x, 1.0 / len(x))
    return x / s


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def arithmetic_average(dists: List[np.ndarray]) -> np.ndarray:
    """Element-wise mean of probability vectors."""
    stacked = np.stack(dists, axis=0)
    avg = stacked.mean(axis=0)
    return _normalize(avg)


def geometric_average(dists: List[np.ndarray]) -> np.ndarray:
    """exp(mean(log p)) then renormalize. Equivalent to a softmax over the
    average of log-probabilities."""
    stacked = np.stack(dists, axis=0).clip(min=_EPS)
    log_avg = np.log(stacked).mean(axis=0)
    g = np.exp(log_avg - log_avg.max())  # max-subtract for numerical stability
    return _normalize(g)


def softmax_sum(dists: List[np.ndarray]) -> np.ndarray:
    """softmax(Σ_k p_k) — literal sum of probability vectors, then softmax.

    NOTE: by spec, the softmax is applied to *probabilities* (not logits).
    This is a sharpening transform: it concentrates mass on positions where
    multiple models agree by piling up probability mass.
    """
    s = np.stack(dists, axis=0).sum(axis=0)
    return _softmax(s)


def softmax_hadamard(dists: List[np.ndarray]) -> np.ndarray:
    """softmax(Π_k p_k) — literal element-wise product, then softmax.

    NOTE: by spec, the softmax is applied to *probabilities* (not logits).
    The product collapses fast when any model assigns near-zero probability,
    so the softmax of those tiny numbers re-spreads mass.
    """
    p = np.stack(dists, axis=0).prod(axis=0)
    return _softmax(p)


def most_confident(dists: List[np.ndarray]) -> np.ndarray:
    """Return the distribution whose maximum probability is the largest.

    Tie-break: the first one in the list.
    """
    best_idx = int(np.argmax([float(d.max()) for d in dists]))
    return _normalize(dists[best_idx].copy())


AGGREGATIONS: Dict[str, Callable[[List[np.ndarray]], np.ndarray]] = {
    "arithmetic_average": arithmetic_average,
    "geometric_average": geometric_average,
    "softmax_sum": softmax_sum,
    "softmax_hadamard": softmax_hadamard,
    "most_confident": most_confident,
}

# ``best`` is a sentinel name that the runner recognizes and handles itself.
BEST_SENTINEL = "best"


def get_aggregation(name: str) -> Callable[[List[np.ndarray]], np.ndarray]:
    if name == BEST_SENTINEL:
        raise ValueError(
            f"'{BEST_SENTINEL}' is handled by the runner, not as a distribution "
            "aggregation. Don't call get_aggregation('best')."
        )
    if name not in AGGREGATIONS:
        raise KeyError(
            f"Unknown aggregation '{name}'. Available: {sorted(AGGREGATIONS)} "
            f"(plus '{BEST_SENTINEL}' handled by the runner)."
        )
    return AGGREGATIONS[name]
