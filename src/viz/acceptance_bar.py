"""
Acceptance-rate bar chart across datasets and models.

For each eval JSON in ``input_dir``, pull the value of a chosen metric at a
chosen position (or averaged over a range of positions), then draw one grouped
bar chart:

* x-axis: dataset names (basename of the source dataset file).
* y-axis: the metric value.
* one bar per (dataset, series). Series = draft-model label with an optional
  ``/<aggregation>`` suffix when a multi-draft file contributes several rows.

Bars are overlaid with decreasing widths (widest drawn first / behind), so
several series stack cleanly at each x tick — matches the "sft (general) vs
baseline" comparison layout.

Run from the project root:
    python src/viz/acceptance_bar.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


PositionSpec = Union[int, List[int]]


# ── helpers ───────────────────────────────────────────────────────────────────


def _resolve_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def discover_eval_files(input_dir: Path) -> List[Path]:
    """Return all ``*.json`` files in ``input_dir`` (recursive), sorted."""
    return sorted(input_dir.rglob("*.json"))


def load_eval_json(path: Path) -> Optional[dict]:
    """Return parsed JSON if it looks like an eval-pipeline output, else None."""
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    md = data.get("metadata")
    if not isinstance(md, dict):
        return None
    if "draft_models" not in md or "dataset" not in md:
        return None
    if "results" not in data or not isinstance(data["results"], dict):
        return None
    return data


def parse_result_key(key: str) -> Optional[Tuple[str, str, str]]:
    """Split a result key ``<agg>__<metric>__<mode>``. Returns None if malformed."""
    parts = key.split("__")
    if len(parts) < 3:
        return None
    agg = parts[0]
    mode = parts[-1]
    metric = "__".join(parts[1:-1])
    return agg, metric, mode


def compute_position_value(
    arr: List[Optional[float]], position: PositionSpec
) -> Optional[float]:
    """Extract a scalar from a per-position array.

    * ``position`` is an int: return that entry (None if OOB or masked).
    * ``position`` is ``[start, end]``: return the mean over the inclusive
      range, skipping None entries. Return None if no non-None values.
    """
    if isinstance(position, int):
        if 0 <= position < len(arr) and arr[position] is not None:
            return float(arr[position])
        return None
    if isinstance(position, (list, tuple)) and len(position) == 2:
        start, end = int(position[0]), int(position[1])
        window = arr[start: end + 1]
        vals = [float(v) for v in window if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    raise ValueError(
        f"position must be int or [start, end] list, got {position!r}"
    )


def compute_position_samples(
    arr: List[int], position: PositionSpec
) -> Optional[int]:
    """Return a representative sample count for the position(s)."""
    if isinstance(position, int):
        return int(arr[position]) if 0 <= position < len(arr) else None
    if isinstance(position, (list, tuple)) and len(position) == 2:
        start, end = int(position[0]), int(position[1])
        window = arr[start: end + 1]
        if not window:
            return None
        return int(sum(window) / len(window))
    return None


def auto_series_label(draft_models: List[str], agg: str) -> str:
    """Auto-derive a legend label from the eval file's draft models + aggregation."""
    base = draft_models[0] if len(draft_models) == 1 else "+".join(draft_models)
    return base if agg == "single" else f"{base}/{agg}"


def _tag_key(draft_models: List[str]) -> str:
    return draft_models[0] if len(draft_models) == 1 else "+".join(draft_models)


def apply_label_override(
    auto_label: str, draft_models: List[str], overrides: Dict[str, str]
) -> str:
    """Replace the base part of ``auto_label`` using ``overrides[draft_tag]``.

    The ``/<agg>`` suffix (if any) is preserved unless the override itself
    contains a ``/`` (in which case the override is used verbatim).
    """
    tag = _tag_key(draft_models)
    if tag not in overrides:
        return auto_label
    replacement = overrides[tag]
    if "/" in replacement:
        return replacement
    if "/" in auto_label:
        _, suffix = auto_label.split("/", 1)
        return f"{replacement}/{suffix}"
    return replacement


# ── row assembly ──────────────────────────────────────────────────────────────


@dataclass
class Row:
    dataset: str          # dataset basename (from metadata.dataset stem)
    series: str           # legend label
    value: float          # metric value at position(s)
    n_samples: int        # samples that contributed
    source_path: Path


def build_rows(
    files: List[Path],
    metric: str,
    position: PositionSpec,
    aggregations: Optional[List[str]],
    labels_override: Dict[str, str],
) -> List[Row]:
    rows: List[Row] = []
    for path in files:
        data = load_eval_json(path)
        if data is None:
            continue
        md = data["metadata"]
        drafts = list(md["draft_models"])
        dataset_basename = Path(md["dataset"]).stem
        n_samples_arr = md.get("n_samples_per_position", [])

        for key, arr in data["results"].items():
            parsed = parse_result_key(key)
            if parsed is None:
                continue
            agg, m, _mode = parsed
            if m != metric:
                continue
            if aggregations and agg not in aggregations:
                continue
            value = compute_position_value(arr, position)
            if value is None:
                continue
            n_samples = compute_position_samples(n_samples_arr, position) or 0
            series = apply_label_override(
                auto_series_label(drafts, agg), drafts, labels_override
            )
            rows.append(
                Row(
                    dataset=dataset_basename,
                    series=series,
                    value=value,
                    n_samples=n_samples,
                    source_path=path,
                )
            )
    return rows


# ── sorting ───────────────────────────────────────────────────────────────────


def sort_datasets(rows: List[Row], sort_by: str) -> List[str]:
    by_ds: Dict[str, List[float]] = {}
    for r in rows:
        by_ds.setdefault(r.dataset, []).append(r.value)
    means = {ds: sum(v) / len(v) for ds, v in by_ds.items()}
    all_ds = list(by_ds.keys())
    if sort_by == "value_asc":
        return sorted(all_ds, key=lambda d: means[d])
    if sort_by == "value_desc":
        return sorted(all_ds, key=lambda d: -means[d])
    if sort_by == "name_asc":
        return sorted(all_ds)
    if sort_by == "name_desc":
        return sorted(all_ds, reverse=True)
    raise ValueError(f"unknown sort_by={sort_by!r}")


def order_series(rows: List[Row], explicit: List[str]) -> List[str]:
    """Return the ordered list of unique series names to plot.

    * If ``explicit`` is empty, use alphabetical order.
    * Otherwise, honor ``explicit`` for the listed series and append any
      others alphabetically.
    """
    seen = sorted({r.series for r in rows})
    if not explicit:
        return seen
    listed = [s for s in explicit if s in seen]
    remaining = [s for s in seen if s not in listed]
    return listed + remaining


# ── plot ──────────────────────────────────────────────────────────────────────


def pad_widths(widths: List[float], n: int) -> List[float]:
    """Extend a widths list to ``n`` entries by geometric shrinkage (×0.7)."""
    out = list(widths) if widths else [0.5]
    while len(out) < n:
        out.append(max(out[-1] * 0.7, 0.03))
    return out[:n]


def _auto_output_filename(
    metric: str, position: PositionSpec, aggregations: Optional[List[str]]
) -> str:
    if isinstance(position, int):
        pos_tag = f"pos{position}"
    else:
        pos_tag = f"pos{position[0]}-{position[1]}"
    parts = ["bar", metric, pos_tag]
    if aggregations:
        parts.append("+".join(aggregations))
    return "__".join(parts) + ".png"


def plot(
    rows: List[Row],
    ordered_datasets: List[str],
    ordered_series: List[str],
    output_path: Path,
    metric: str,
    position: PositionSpec,
    widths: List[float],
    alpha: float,
    annotate: str,
    info_box: str,
    title: Optional[str],
    fig_w: float,
    fig_h: float,
    dpi: int,
) -> Path:
    # value_map[series][dataset] = (value, n_samples)
    value_map: Dict[str, Dict[str, Tuple[float, int]]] = {s: {} for s in ordered_series}
    for r in rows:
        if r.series in value_map:
            value_map[r.series][r.dataset] = (r.value, r.n_samples)

    x = np.arange(len(ordered_datasets))
    widths_padded = pad_widths(widths, len(ordered_series))

    # Draw widest first (behind), narrowest last (in front)
    series_with_widths = list(zip(ordered_series, widths_padded))
    plot_order = sorted(series_with_widths, key=lambda p: -p[1])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    for series, w in plot_order:
        vals = [value_map[series].get(ds, (np.nan, 0))[0] for ds in ordered_datasets]
        ax.bar(x, np.array(vals, dtype=float), width=w, label=series, alpha=alpha)

    if annotate == "value":
        for series, _w in plot_order:
            for i, ds in enumerate(ordered_datasets):
                cell = value_map[series].get(ds)
                if cell is None:
                    continue
                ax.text(
                    i, cell[0], f"{cell[0]:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    elif annotate == "sample_count":
        for series, _w in plot_order:
            for i, ds in enumerate(ordered_datasets):
                cell = value_map[series].get(ds)
                if cell is None:
                    continue
                ax.text(
                    i, cell[0], f"{cell[1]}",
                    ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_datasets, rotation=90, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset")

    if title is None:
        if isinstance(position, int):
            pos_desc = f"position {position}"
        else:
            pos_desc = f"positions {position[0]}..{position[1]}"
        title = f"{metric} per dataset ({pos_desc})"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)

    if info_box == "summary":
        lines: List[str] = []
        for series in ordered_series:
            vals = [
                value_map[series][ds][0]
                for ds in ordered_datasets
                if ds in value_map[series]
            ]
            if vals:
                arr = np.array(vals)
                lines.append(
                    f"{series}: mean={arr.mean():.3f}, std={arr.std():.3f} "
                    f"(n_datasets={len(vals)})"
                )
        if lines:
            ax.text(
                0.005, -0.30, "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7, family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white", edgecolor="gray", alpha=0.85,
                ),
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Hydra entry ───────────────────────────────────────────────────────────────


def _normalize_position(p: Any) -> PositionSpec:
    if isinstance(p, int):
        return p
    if hasattr(p, "__iter__") and not isinstance(p, (str, bytes)):
        lst = list(p)
        if len(lst) != 2:
            raise ValueError(f"position range must have exactly 2 entries, got {lst}")
        return [int(lst[0]), int(lst[1])]
    raise ValueError(f"position must be int or [start, end], got {p!r}")


def _run(cfg) -> None:
    input_dir = _resolve_path(str(cfg.input_dir))
    if not input_dir.is_dir():
        sys.exit(f"input_dir does not exist or is not a directory: {input_dir}")

    files = discover_eval_files(input_dir)
    aggregations = list(cfg.aggregations) if cfg.aggregations else None
    labels_override = dict(cfg.labels) if cfg.labels else {}
    position = _normalize_position(cfg.position)

    rows = build_rows(
        files=files,
        metric=str(cfg.metric),
        position=position,
        aggregations=aggregations,
        labels_override=labels_override,
    )
    if not rows:
        sys.exit(
            f"no matching results in {input_dir} "
            f"for metric={cfg.metric}, position={position}"
        )

    ordered_datasets = sort_datasets(rows, str(cfg.sort_by))
    ordered_series = order_series(
        rows, list(cfg.series_order) if cfg.series_order else []
    )

    output_dir = _resolve_path(str(cfg.output.dir))
    filename = cfg.output.filename or _auto_output_filename(
        cfg.metric, position, aggregations
    )
    output_path = output_dir / filename

    plot(
        rows=rows,
        ordered_datasets=ordered_datasets,
        ordered_series=ordered_series,
        output_path=output_path,
        metric=str(cfg.metric),
        position=position,
        widths=list(cfg.bars.widths),
        alpha=float(cfg.bars.alpha),
        annotate=str(cfg.annotate),
        info_box=str(cfg.info_box),
        title=cfg.title,
        fig_w=float(cfg.figure.width),
        fig_h=float(cfg.figure.height),
        dpi=int(cfg.figure.dpi),
    )
    print(f"wrote {output_path}")
    print(f"  datasets: {len(ordered_datasets)}")
    print(f"  series:   {ordered_series}")


def main() -> None:
    import hydra

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="viz_acceptance_bar",
    )
    def _hydra_wrapper(cfg) -> None:
        _run(cfg)

    _hydra_wrapper()


if __name__ == "__main__":
    main()
