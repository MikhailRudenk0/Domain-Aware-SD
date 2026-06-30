"""
End-to-end smoke test for the eval loop.

No real models are loaded. Stub draft runners produce deterministic
per-position outputs; the synthetic dataset is built in tmp_path. We verify:

* Loop runs without exceptions.
* Output dict has the expected per-(agg, metric, mode) keys.
* Per-position arrays have length n_positions.
* Skipped & special-target counters move correctly.
* Single-model + multi-model + best aggregation all produce sensible numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from src.data import SpecDecDataset
from src.eval.draft_runner import DraftPositionInfo
from src.eval.evaluator import evaluate_dataset
from src.eval.target_provider import DatasetTargetProvider


# ── Stub runner ───────────────────────────────────────────────────────────────


@dataclass
class StubDraftRunner:
    """Returns deterministic per-position draft info regardless of input.

    Useful for exercising the loop without loading real models.
    """

    vocab_size: int = 32000
    # For each position, returns:
    #   own_topk_ids = [10, 11, 12, ..., 10+K-1]
    #   own_topk_probs = uniform 1/K
    #   prob_at_target_topk = uniform 1/K_target
    K: int = 10

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_starts: List[int],
        trunk_lens: List[int],
        n_positions: int,
        target_topk_ids_per_sample: List[List[np.ndarray]],
    ) -> List[List[DraftPositionInfo]]:
        out: List[List[DraftPositionInfo]] = []
        for b in range(input_ids.shape[0]):
            valid = min(n_positions, trunk_lens[b], len(target_topk_ids_per_sample[b]))
            sample: List[DraftPositionInfo] = []
            for p in range(valid):
                target_ids = target_topk_ids_per_sample[b][p]
                Kt = len(target_ids)
                # Own top-K: ids 10..10+K-1, uniform probs
                own_ids = np.arange(10, 10 + self.K, dtype=np.int64)
                own_probs = np.full(self.K, 1.0 / self.K, dtype=np.float32)
                # Uniform probs at target's top-K
                prob_at_target = np.full(Kt, 1.0 / max(Kt, 1), dtype=np.float32) if Kt > 0 else np.zeros(0, dtype=np.float32)
                sample.append(
                    DraftPositionInfo(
                        argmax_id=int(own_ids[0]),
                        own_topk_ids=own_ids,
                        own_topk_probs=own_probs,
                        prob_at_target_topk=prob_at_target,
                    )
                )
            out.append(sample)
        return out


@dataclass
class PerfectDraftRunner:
    """Returns draft info that matches the target perfectly — overlap_area = 1.0."""

    vocab_size: int = 32000

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_starts: List[int],
        trunk_lens: List[int],
        n_positions: int,
        target_topk_ids_per_sample: List[List[np.ndarray]],
    ) -> List[List[DraftPositionInfo]]:
        out: List[List[DraftPositionInfo]] = []
        for b in range(input_ids.shape[0]):
            valid = min(n_positions, trunk_lens[b], len(target_topk_ids_per_sample[b]))
            sample: List[DraftPositionInfo] = []
            for p in range(valid):
                target_ids = target_topk_ids_per_sample[b][p]
                # Draft's "full vocab" view: own_topk == target_topk; probs all match.
                own_ids = target_ids.astype(np.int64).copy()
                # Use uniform probs over target's ids (any valid dist; the matching
                # behavior of overlap_area relies on prob_at_target_topk).
                own_probs = np.full(len(own_ids), 1.0 / max(len(own_ids), 1), dtype=np.float32)
                # Match target probs exactly at target's top-K ids
                # (this drives overlap_area to 1.0 after renormalization).
                # We use uniform here — target is also renormalized to uniform
                # in this test by construction. See _make_fixture_dataset below.
                prob_at_target = np.full(len(own_ids), 1.0 / max(len(own_ids), 1), dtype=np.float32)
                sample.append(
                    DraftPositionInfo(
                        argmax_id=int(own_ids[0]) if len(own_ids) > 0 else 0,
                        own_topk_ids=own_ids,
                        own_topk_probs=own_probs,
                        prob_at_target_topk=prob_at_target,
                    )
                )
            out.append(sample)
        return out


# ── Fixture dataset ───────────────────────────────────────────────────────────


def _make_fixture_dataset(tmp_path: Path, n_records: int = 4, trunk_len: int = 6) -> Path:
    """Write a small JSONL with uniform top-K (so renormalized target == uniform)."""
    path = tmp_path / "fixture.jsonl"
    K = 10
    with path.open("w") as f:
        for i in range(n_records):
            top10_ids = [
                list(range(100, 100 + K)) for _ in range(trunk_len)
            ]
            top10_probs = [
                [round(1.0 / K, 3)] * K for _ in range(trunk_len)
            ]
            rec = {
                "cluster": "fixture_cluster",
                "prompt": f"prompt number {i} with some words",
                "reference": "ref",
                "trunk": [200 + i, 201, 202, 203, 204, 205][:trunk_len],
                "top10_ids": top10_ids,
                "top10_probs": top10_probs,
            }
            f.write(json.dumps(rec) + "\n")
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_smoke_single_model(tokenizer, tmp_path: Path):
    """End-to-end: 1 stub draft, dataset target, all 4 metrics."""
    ds_path = _make_fixture_dataset(tmp_path)
    dataset = SpecDecDataset(
        ds_path, tokenizer, mode="distillation",
        max_length=512, max_gen_length=10,
    )
    n_positions = 5
    result = evaluate_dataset(
        dataset=dataset,
        draft_runners=[StubDraftRunner(K=10)],
        target_provider=DatasetTargetProvider(),
        n_positions=n_positions,
        batch_size=2,
        pad_token_id=tokenizer.pad_token_id,
        metrics=["overlap_area", "top1_match", "topk_overlap", "kl"],
        aggregations=[],  # ignored for single-model
        mode_label_str="top10_from_dataset",
        draft_vocab_size=32000,
    )

    assert result.n_samples_total == 4
    # Every position should have all 4 samples accounted for
    assert result.n_samples_per_position[:n_positions] == [4] * n_positions
    assert result.n_skipped_per_position[:n_positions] == [0] * n_positions

    # Keys should be single__{metric}__top10_from_dataset
    expected_keys = {
        "single__overlap_area__top10_from_dataset",
        "single__top1_match__top10_from_dataset",
        "single__topk_overlap__top10_from_dataset",
        "single__kl__top10_from_dataset",
    }
    assert set(result.results.keys()) == expected_keys

    # Arrays have correct length
    for k, arr in result.results.items():
        assert len(arr) == n_positions
        for v in arr[:n_positions]:
            assert v is not None


def test_smoke_perfect_draft_overlap_one(tokenizer, tmp_path: Path):
    """Perfect-match stub draft should give overlap_area == 1.0 at every position."""
    ds_path = _make_fixture_dataset(tmp_path)
    dataset = SpecDecDataset(
        ds_path, tokenizer, mode="distillation",
        max_length=512, max_gen_length=10,
    )
    n_positions = 5
    result = evaluate_dataset(
        dataset=dataset,
        draft_runners=[PerfectDraftRunner()],
        target_provider=DatasetTargetProvider(),
        n_positions=n_positions,
        batch_size=2,
        pad_token_id=tokenizer.pad_token_id,
        metrics=["overlap_area"],
        aggregations=[],
        mode_label_str="top10_from_dataset",
        draft_vocab_size=32000,
    )
    vals = result.results["single__overlap_area__top10_from_dataset"]
    for v in vals[:n_positions]:
        assert v == pytest.approx(1.0)


def test_smoke_multi_model_aggregations(tokenizer, tmp_path: Path):
    """Two stub drafts, all aggregations × metrics, verify keys and shapes."""
    ds_path = _make_fixture_dataset(tmp_path)
    dataset = SpecDecDataset(
        ds_path, tokenizer, mode="distillation",
        max_length=512, max_gen_length=10,
    )
    n_positions = 5
    aggregations = [
        "best",
        "arithmetic_average",
        "geometric_average",
        "softmax_sum",
        "softmax_hadamard",
        "most_confident",
    ]
    metrics = ["overlap_area", "top1_match"]
    result = evaluate_dataset(
        dataset=dataset,
        draft_runners=[StubDraftRunner(K=10), PerfectDraftRunner()],
        target_provider=DatasetTargetProvider(),
        n_positions=n_positions,
        batch_size=2,
        pad_token_id=tokenizer.pad_token_id,
        metrics=metrics,
        aggregations=aggregations,
        mode_label_str="top10_from_dataset",
        draft_vocab_size=32000,
    )

    # Should have len(aggregations) * len(metrics) keys
    assert len(result.results) == len(aggregations) * len(metrics)
    for agg in aggregations:
        for metric in metrics:
            key = f"{agg}__{metric}__top10_from_dataset"
            assert key in result.results
            arr = result.results[key]
            assert len(arr) == n_positions

    # "best" overlap_area should be >= any single model's, since one of the
    # models (PerfectDraftRunner) hits overlap_area == 1.0.
    best_vals = result.results["best__overlap_area__top10_from_dataset"]
    for v in best_vals[:n_positions]:
        assert v == pytest.approx(1.0)


def test_skipped_positions_are_masked(tokenizer, tmp_path: Path):
    """Records with all-zero top10_probs at some positions should mask those
    positions out of the average and bump the skipped counter."""
    path = tmp_path / "with_skips.jsonl"
    K = 10
    with path.open("w") as f:
        # 2 records; position 1 is skipped in both (empty top10).
        for i in range(2):
            top10_ids = [
                list(range(100, 100 + K)),   # pos 0 — valid
                [0] * K,                     # pos 1 — skipped
                list(range(100, 100 + K)),   # pos 2 — valid
            ]
            top10_probs = [
                [round(1.0 / K, 3)] * K,
                [0.0] * K,
                [round(1.0 / K, 3)] * K,
            ]
            rec = {
                "cluster": "c",
                "prompt": f"p {i}",
                "reference": "r",
                "trunk": [200, 201, 202],
                "top10_ids": top10_ids,
                "top10_probs": top10_probs,
            }
            f.write(json.dumps(rec) + "\n")

    dataset = SpecDecDataset(
        path, tokenizer, mode="distillation",
        max_length=512, max_gen_length=3,
    )
    result = evaluate_dataset(
        dataset=dataset,
        draft_runners=[StubDraftRunner(K=10)],
        target_provider=DatasetTargetProvider(),
        n_positions=3,
        batch_size=2,
        pad_token_id=tokenizer.pad_token_id,
        metrics=["overlap_area"],
        aggregations=[],
        mode_label_str="top10_from_dataset",
        draft_vocab_size=32000,
    )
    assert result.n_skipped_per_position[:3] == [0, 2, 0]
    assert result.n_samples_per_position[:3] == [2, 0, 2]
