"""
Build and write the evaluation result JSON.

Field naming convention for per-position arrays:
    <aggregation>__<metric>__<mode>

* ``aggregation``: "single" when there is only one draft model, otherwise the
  name of a registered aggregation ("best", "arithmetic_average", ...).
* ``metric``: name of a registered metric ("overlap_area", "top1_match", ...).
* ``mode``: "top{K}_from_dataset" (target source = dataset) or "full_target"
  (target source = model).

Output file naming:
    eval__<draft_tag>__<dataset_basename>.json

Where ``draft_tag`` is the single draft name when len == 1, otherwise
``multi-<hash6>`` over the sorted list (the full list lives in metadata).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── naming ────────────────────────────────────────────────────────────────────


def mode_label(target_source: str, top_k_from_dataset: Optional[int]) -> str:
    if target_source == "dataset":
        if top_k_from_dataset is None:
            raise ValueError("top_k_from_dataset must be set when target_source='dataset'")
        return f"top{top_k_from_dataset}_from_dataset"
    if target_source == "model":
        return "full_target"
    raise ValueError(f"Unknown target_source: {target_source!r}")


def result_key(aggregation: str, metric: str, mode: str) -> str:
    return f"{aggregation}__{metric}__{mode}"


def draft_tag(draft_models: List[str]) -> str:
    if len(draft_models) == 1:
        return draft_models[0]
    payload = ",".join(sorted(draft_models)).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:6]
    return f"multi-{h}"


def output_filename(draft_models: List[str], dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    return f"eval__{draft_tag(draft_models)}__{stem}.json"


# ── result container ──────────────────────────────────────────────────────────


@dataclass
class EvalMetadata:
    draft_models: List[str]
    target_model: str
    target_source: str
    dataset: str
    n_positions: int
    n_samples_total: int
    n_samples_per_position: List[int]
    n_skipped_per_position: List[int]
    n_special_target_per_position: List[int]
    top_k_from_dataset: Optional[int]
    target_renormalized: bool
    metrics: List[str]
    aggregations: List[str]
    config_snapshot: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")


def build_payload(
    metadata: EvalMetadata,
    results: Dict[str, List[float]],
) -> Dict[str, Any]:
    return {
        "metadata": asdict(metadata),
        "results": results,
    }


def write_eval_json(
    output_dir: Path,
    draft_models: List[str],
    dataset_path: str,
    metadata: EvalMetadata,
    results: Dict[str, List[float]],
) -> Path:
    """Write the JSON payload and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_filename(draft_models, dataset_path)
    payload = build_payload(metadata, results)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path
