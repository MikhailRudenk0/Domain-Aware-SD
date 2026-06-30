import numpy as np
import pytest

from src.eval.metrics import (
    METRICS,
    MetricInputs,
    get_metric,
    kl,
    overlap_area,
    top1_match,
    topk_overlap,
)


# ── overlap_area ──────────────────────────────────────────────────────────────


def test_overlap_area_identical_dists_is_one():
    p = np.array([0.5, 0.3, 0.2])
    inp = MetricInputs(draft_aligned=p, target_aligned=p.copy())
    assert overlap_area(inp) == pytest.approx(1.0)


def test_overlap_area_disjoint_supports_is_zero():
    d = np.array([1.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 0.0])
    inp = MetricInputs(draft_aligned=d, target_aligned=t)
    assert overlap_area(inp) == pytest.approx(0.0)


def test_overlap_area_partial_overlap():
    # min by entry: [0.3, 0.2, 0.1] → sum 0.6
    d = np.array([0.7, 0.2, 0.1])
    t = np.array([0.3, 0.5, 0.2])
    inp = MetricInputs(draft_aligned=d, target_aligned=t)
    assert overlap_area(inp) == pytest.approx(0.6)


# ── top1_match ────────────────────────────────────────────────────────────────


def test_top1_match_hit():
    inp = MetricInputs(draft_argmax_id=42, target_topk_ids=np.array([42, 7, 3]))
    assert top1_match(inp) == 1.0


def test_top1_match_miss():
    inp = MetricInputs(draft_argmax_id=42, target_topk_ids=np.array([5, 42, 3]))
    assert top1_match(inp) == 0.0  # target argmax is 5, not 42


# ── topk_overlap ──────────────────────────────────────────────────────────────


def test_topk_overlap_full():
    inp = MetricInputs(
        draft_topk_ids=np.array([1, 2, 3, 4, 5]),
        target_topk_ids=np.array([5, 4, 3, 2, 1]),
    )
    assert topk_overlap(inp) == pytest.approx(1.0)


def test_topk_overlap_partial():
    inp = MetricInputs(
        draft_topk_ids=np.array([1, 2, 3]),
        target_topk_ids=np.array([3, 4, 5]),
    )
    # only id=3 is shared, K=3 → 1/3
    assert topk_overlap(inp) == pytest.approx(1.0 / 3.0)


def test_topk_overlap_empty_target():
    inp = MetricInputs(
        draft_topk_ids=np.array([1, 2, 3]),
        target_topk_ids=np.array([], dtype=int),
    )
    assert topk_overlap(inp) == 0.0


# ── kl ────────────────────────────────────────────────────────────────────────


def test_kl_identical_is_zero():
    p = np.array([0.5, 0.3, 0.2])
    inp = MetricInputs(draft_aligned=p, target_aligned=p.copy())
    assert kl(inp) == pytest.approx(0.0, abs=1e-9)


def test_kl_handles_zero_target():
    # KL(target || draft) — positions where target is 0 contribute nothing.
    t = np.array([0.5, 0.5, 0.0])
    d = np.array([0.3, 0.3, 0.4])
    inp = MetricInputs(draft_aligned=d, target_aligned=t)
    expected = 0.5 * (np.log(0.5) - np.log(0.3)) * 2  # symmetric two terms
    assert kl(inp) == pytest.approx(expected, rel=1e-6)


def test_kl_all_zero_target_is_zero():
    t = np.array([0.0, 0.0, 0.0])
    d = np.array([0.5, 0.3, 0.2])
    inp = MetricInputs(draft_aligned=d, target_aligned=t)
    assert kl(inp) == 0.0


# ── registry ──────────────────────────────────────────────────────────────────


def test_registry_has_all_expected_metrics():
    expected = {"overlap_area", "top1_match", "topk_overlap", "kl"}
    assert expected == set(METRICS.keys())


def test_get_metric_unknown_raises():
    with pytest.raises(KeyError):
        get_metric("nonsense")
