"""
Core evaluation loop. Pure-Python orchestration — no Hydra dependency, no
model-loading side effects. The entry point ``evaluate_dataset`` takes
already-loaded draft runners + a target provider and produces the per-position
result arrays.

This split makes the loop directly testable with stub runners/providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.eval.aggregations import AGGREGATIONS, BEST_SENTINEL, get_aggregation
from src.eval.draft_runner import DraftPositionInfo, DraftRunner
from src.eval.metrics import METRICS, MetricInputs, get_metric
from src.eval.output_writer import result_key
from src.eval.target_provider import (
    DatasetTargetProvider,
    ModelTargetProvider,
    TargetInfo,
)


# ── batch collation ───────────────────────────────────────────────────────────


@dataclass
class CollatedBatch:
    input_ids: torch.Tensor          # [B, L_max]
    attention_mask: torch.Tensor     # [B, L_max]
    gen_starts: List[int]            # length B
    trunk_lens: List[int]            # length B
    samples: List[dict]              # the raw sample dicts (with top10_ids/probs)


def collate_batch(samples: List[dict], pad_token_id: int) -> CollatedBatch:
    L_max = max(int(s["input_ids"].shape[0]) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, L_max), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L_max), dtype=torch.long)
    gen_starts: List[int] = []
    trunk_lens: List[int] = []
    for i, s in enumerate(samples):
        L = int(s["input_ids"].shape[0])
        input_ids[i, :L] = s["input_ids"]
        attention_mask[i, :L] = 1
        g = int(s["gen_start"])
        gen_starts.append(g)
        trunk_lens.append(L - g)
    return CollatedBatch(input_ids, attention_mask, gen_starts, trunk_lens, samples)


# ── metric inputs builders ────────────────────────────────────────────────────


def _build_inputs_single(
    di: DraftPositionInfo, target_info: TargetInfo
) -> MetricInputs:
    """Build MetricInputs from a single draft's per-position info."""
    K_target = len(target_info.topk_ids)
    draft_at_target = di.prob_at_target_topk
    s = float(draft_at_target.sum())
    if s > 0:
        draft_aligned = draft_at_target / s
    else:
        draft_aligned = np.full(K_target, 1.0 / max(K_target, 1), dtype=np.float32)

    K_metric = K_target
    return MetricInputs(
        draft_aligned=draft_aligned,
        target_aligned=target_info.topk_probs,
        draft_argmax_id=di.argmax_id,
        draft_topk_ids=di.own_topk_ids[:K_metric],
        target_topk_ids=target_info.topk_ids,
    )


def _aggregate_over_union(
    agg_name: str,
    draft_infos: List[DraftPositionInfo],
    target_info: TargetInfo,
) -> MetricInputs:
    """Aggregate M models' distributions on the union support
    (target_topk_ids ∪ each model's own top-K) and build MetricInputs.
    """
    all_ids = set(int(x) for x in target_info.topk_ids.tolist())
    for di in draft_infos:
        all_ids.update(int(x) for x in di.own_topk_ids.tolist())
    support_ids = np.array(sorted(all_ids), dtype=np.int64)
    S = len(support_ids)
    id_to_idx = {int(tid): i for i, tid in enumerate(support_ids.tolist())}

    per_model_dist: List[np.ndarray] = []
    for di in draft_infos:
        v = np.zeros(S, dtype=np.float32)
        for tid, prob in zip(di.own_topk_ids.tolist(), di.own_topk_probs.tolist()):
            v[id_to_idx[int(tid)]] = float(prob)
        for tid, prob in zip(target_info.topk_ids.tolist(), di.prob_at_target_topk.tolist()):
            idx = id_to_idx[int(tid)]
            if v[idx] == 0.0:
                v[idx] = float(prob)
        per_model_dist.append(v)

    aggregator = get_aggregation(agg_name)
    agg_dist = aggregator(per_model_dist)  # [S], sums to 1

    # Extract draft_aligned at target's top-K ids and renormalize
    K_target = len(target_info.topk_ids)
    draft_at_target = np.array(
        [agg_dist[id_to_idx[int(tid)]] for tid in target_info.topk_ids.tolist()],
        dtype=np.float32,
    )
    s = float(draft_at_target.sum())
    if s > 0:
        draft_aligned = draft_at_target / s
    else:
        draft_aligned = np.full(K_target, 1.0 / max(K_target, 1), dtype=np.float32)

    # Aggregated argmax / top-K over the union support
    argmax_idx = int(np.argmax(agg_dist))
    draft_argmax_id = int(support_ids[argmax_idx])
    topk_idx = np.argsort(-agg_dist)[:K_target]
    draft_topk_ids = support_ids[topk_idx]

    return MetricInputs(
        draft_aligned=draft_aligned,
        target_aligned=target_info.topk_probs,
        draft_argmax_id=draft_argmax_id,
        draft_topk_ids=draft_topk_ids,
        target_topk_ids=target_info.topk_ids,
    )


def compute_value(
    aggregation: str,
    metric: str,
    draft_infos: List[DraftPositionInfo],
    target_info: TargetInfo,
) -> float:
    metric_fn = get_metric(metric)
    if aggregation == "single":
        return float(metric_fn(_build_inputs_single(draft_infos[0], target_info)))
    if aggregation == BEST_SENTINEL:
        # Per-model metric value, take max
        values = [float(metric_fn(_build_inputs_single(di, target_info))) for di in draft_infos]
        return max(values)
    # Distribution-level aggregation on the union support
    inputs = _aggregate_over_union(aggregation, draft_infos, target_info)
    return float(metric_fn(inputs))


# ── accumulators ──────────────────────────────────────────────────────────────


class PositionAccumulator:
    """Per-position running sum and count; emits per-position averages."""

    def __init__(self, n_positions: int) -> None:
        self._sums = np.zeros(n_positions, dtype=np.float64)
        self._counts = np.zeros(n_positions, dtype=np.int64)

    def add(self, position: int, value: float) -> None:
        self._sums[position] += value
        self._counts[position] += 1

    def averages(self) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for s, c in zip(self._sums.tolist(), self._counts.tolist()):
            out.append(s / c if c > 0 else None)
        return out


# ── main loop ─────────────────────────────────────────────────────────────────


@dataclass
class EvalLoopResult:
    results: Dict[str, List[Optional[float]]]
    n_samples_total: int
    n_samples_per_position: List[int]
    n_skipped_per_position: List[int]
    n_special_target_per_position: List[int]


def evaluate_dataset(
    dataset: Dataset,
    draft_runners: List[DraftRunner],
    target_provider,  # DatasetTargetProvider | ModelTargetProvider
    n_positions: int,
    batch_size: int,
    pad_token_id: int,
    metrics: Sequence[str],
    aggregations: Sequence[str],
    mode_label_str: str,
    draft_vocab_size: int,
    max_samples: Optional[int] = None,
) -> EvalLoopResult:
    """Run the full eval loop on one dataset.

    ``aggregations`` semantics:
    * If a single draft model is supplied, this list is ignored and a "single"
      aggregation is used.
    * Otherwise, one accumulator is built per (aggregation, metric).
    """
    M = len(draft_runners)
    assert M >= 1
    aggs_effective: List[str] = ["single"] if M == 1 else list(aggregations)

    # Per-(agg, metric) accumulators
    accumulators: Dict[str, PositionAccumulator] = {}
    for agg in aggs_effective:
        for metric in metrics:
            key = result_key(agg, metric, mode_label_str)
            accumulators[key] = PositionAccumulator(n_positions)

    n_samples_total = 0
    n_samples_per_position = np.zeros(n_positions, dtype=np.int64)
    n_skipped_per_position = np.zeros(n_positions, dtype=np.int64)
    n_special_target_per_position = np.zeros(n_positions, dtype=np.int64)

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    # Iterate batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        samples = [dataset[i] for i in range(batch_start, batch_end)]
        batch = collate_batch(samples, pad_token_id)

        # Get target infos per (sample, position), truncated to n_positions
        if isinstance(target_provider, DatasetTargetProvider):
            target_infos_per_sample: List[List[TargetInfo]] = [
                target_provider.get_for_sample(s)[:n_positions] for s in samples
            ]
        else:  # ModelTargetProvider
            target_infos_per_sample = target_provider.get_for_batch(
                batch.input_ids,
                batch.attention_mask,
                batch.gen_starts,
                batch.trunk_lens,
                n_positions,
            )

        # target_topk_ids per (sample, position) for draft lookup
        target_topk_ids_per_sample = [
            [info.topk_ids for info in sample_infos]
            for sample_infos in target_infos_per_sample
        ]

        # Forward each draft model
        draft_infos_per_model: List[List[List[DraftPositionInfo]]] = []
        for runner in draft_runners:
            infos = runner.forward_batch(
                batch.input_ids,
                batch.attention_mask,
                batch.gen_starts,
                batch.trunk_lens,
                n_positions,
                target_topk_ids_per_sample,
            )
            draft_infos_per_model.append(infos)

        # Per (sample, position): bookkeeping + metric accumulation
        for b in range(len(samples)):
            n_samples_total += 1
            valid_positions = min(n_positions, batch.trunk_lens[b], len(target_infos_per_sample[b]))
            for p in range(valid_positions):
                ti = target_infos_per_sample[b][p]
                if ti.is_skipped:
                    n_skipped_per_position[p] += 1
                    continue
                target_argmax = int(ti.topk_ids[0])
                if target_argmax >= draft_vocab_size:
                    n_special_target_per_position[p] += 1
                    continue
                n_samples_per_position[p] += 1

                draft_infos_here = [
                    draft_infos_per_model[m][b][p] for m in range(M)
                ]
                for agg in aggs_effective:
                    for metric in metrics:
                        val = compute_value(agg, metric, draft_infos_here, ti)
                        key = result_key(agg, metric, mode_label_str)
                        accumulators[key].add(p, val)

    results = {k: acc.averages() for k, acc in accumulators.items()}
    return EvalLoopResult(
        results=results,
        n_samples_total=n_samples_total,
        n_samples_per_position=n_samples_per_position.tolist(),
        n_skipped_per_position=n_skipped_per_position.tolist(),
        n_special_target_per_position=n_special_target_per_position.tolist(),
    )
