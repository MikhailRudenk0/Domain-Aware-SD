"""
MLflow logging helpers for the eval pipeline.

Logs:
* All flattened config keys as params.
* Per-position arrays as timeseries metrics with ``step`` = position 0..N-1.
* Scalar summaries: mean_all / mean_first10 / mean_last10 per key.
* Artifacts: the result JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(d, (list, tuple)):
        # MLflow params don't accept lists; serialize as comma-joined.
        out[prefix] = ",".join(str(x) for x in d)
    else:
        out[prefix] = d
    return out


def _safe_log_param(key: str, value: Any) -> None:
    # MLflow params have a 500-char value limit; truncate if needed.
    s = "" if value is None else str(value)
    if len(s) > 490:
        s = s[:490] + "...<trunc>"
    try:
        mlflow.log_param(key, s)
    except mlflow.exceptions.MlflowException:
        # ignore duplicate-key errors etc.
        pass


def start_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: Optional[str],
) -> mlflow.ActiveRun:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params_from_config(config: Dict[str, Any]) -> None:
    for k, v in _flatten(config).items():
        _safe_log_param(k, v)


def log_results(
    results: Dict[str, List[float]],
    n_samples_per_position: Optional[List[int]] = None,
) -> None:
    """For each ``<agg>__<metric>__<mode>`` key, log a timeseries (step = position)
    plus mean_all / mean_first10 / mean_last10 scalars."""
    for key, arr in results.items():
        if not arr:
            continue
        for i, val in enumerate(arr):
            if val is None:
                continue
            try:
                mlflow.log_metric(key, float(val), step=i)
            except (ValueError, TypeError):
                pass
        valid = [v for v in arr if v is not None]
        if not valid:
            continue
        mlflow.log_metric(f"{key}__mean_all", float(sum(valid) / len(valid)))
        first10 = valid[:10]
        last10 = valid[-10:]
        mlflow.log_metric(f"{key}__mean_first10", float(sum(first10) / len(first10)))
        mlflow.log_metric(f"{key}__mean_last10", float(sum(last10) / len(last10)))

    if n_samples_per_position is not None:
        for i, n in enumerate(n_samples_per_position):
            mlflow.log_metric("n_samples_per_position", float(n), step=i)


def log_artifact(path: Path) -> None:
    mlflow.log_artifact(str(path))
