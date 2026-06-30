import numpy as np
import pytest

from src.eval.aggregations import (
    AGGREGATIONS,
    BEST_SENTINEL,
    arithmetic_average,
    geometric_average,
    get_aggregation,
    most_confident,
    softmax_hadamard,
    softmax_sum,
)


def _assert_valid_dist(p: np.ndarray):
    assert p.shape[0] > 0
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0, atol=1e-6)


# ── arithmetic_average ────────────────────────────────────────────────────────


def test_arithmetic_average_of_identical_dists_equals_the_dist():
    p = np.array([0.5, 0.3, 0.2])
    out = arithmetic_average([p, p.copy(), p.copy()])
    _assert_valid_dist(out)
    assert np.allclose(out, p)


def test_arithmetic_average_two_dists():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    out = arithmetic_average([a, b])
    assert np.allclose(out, [0.5, 0.0, 0.5])


# ── geometric_average ─────────────────────────────────────────────────────────


def test_geometric_average_of_identical_dists():
    p = np.array([0.5, 0.3, 0.2])
    out = geometric_average([p, p.copy()])
    _assert_valid_dist(out)
    assert np.allclose(out, p, atol=1e-6)


def test_geometric_average_zero_handling():
    # A model assigning 0 should not crash and should heavily down-weight that bin.
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.5, 0.0, 0.5])
    out = geometric_average([a, b])
    _assert_valid_dist(out)
    # Position 0 is the only one where both have nonzero mass → it should dominate.
    assert out[0] > out[1] and out[0] > out[2]


# ── softmax_sum ───────────────────────────────────────────────────────────────


def test_softmax_sum_returns_valid_distribution():
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.1, 0.7, 0.2])
    out = softmax_sum([a, b])
    _assert_valid_dist(out)


def test_softmax_sum_concentrates_on_agreement():
    # Both models put highest mass on index 0 → softmax_sum should too.
    a = np.array([0.9, 0.05, 0.05])
    b = np.array([0.9, 0.05, 0.05])
    out = softmax_sum([a, b])
    assert int(np.argmax(out)) == 0


# ── softmax_hadamard ──────────────────────────────────────────────────────────


def test_softmax_hadamard_returns_valid_distribution():
    a = np.array([0.6, 0.3, 0.1])
    b = np.array([0.1, 0.7, 0.2])
    out = softmax_hadamard([a, b])
    _assert_valid_dist(out)


def test_softmax_hadamard_zero_at_position_does_not_break():
    # Element-wise product becomes 0 at position 2 → softmax handles it fine.
    a = np.array([0.5, 0.4, 0.1])
    b = np.array([0.5, 0.5, 0.0])
    out = softmax_hadamard([a, b])
    _assert_valid_dist(out)


# ── most_confident ────────────────────────────────────────────────────────────


def test_most_confident_picks_sharpest():
    flat = np.array([0.4, 0.3, 0.3])
    sharp = np.array([0.9, 0.05, 0.05])
    medium = np.array([0.6, 0.2, 0.2])
    out = most_confident([flat, sharp, medium])
    assert np.allclose(out, sharp)


def test_most_confident_tie_breaks_first():
    a = np.array([0.7, 0.2, 0.1])
    b = np.array([0.7, 0.1, 0.2])
    out = most_confident([a, b])
    assert np.allclose(out, a)


# ── registry ──────────────────────────────────────────────────────────────────


def test_registry_has_all_expected():
    expected = {
        "arithmetic_average",
        "geometric_average",
        "softmax_sum",
        "softmax_hadamard",
        "most_confident",
    }
    assert expected == set(AGGREGATIONS.keys())


def test_get_aggregation_rejects_best():
    with pytest.raises(ValueError):
        get_aggregation(BEST_SENTINEL)


def test_get_aggregation_unknown_raises():
    with pytest.raises(KeyError):
        get_aggregation("nonsense")
