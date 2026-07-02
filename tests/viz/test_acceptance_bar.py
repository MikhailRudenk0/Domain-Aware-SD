import json
from pathlib import Path

import pytest

from src.viz.acceptance_bar import (
    Row,
    apply_label_override,
    auto_series_label,
    build_rows,
    compute_position_samples,
    compute_position_value,
    discover_eval_files,
    load_eval_json,
    order_series,
    pad_widths,
    parse_result_key,
    sort_datasets,
)


# ── fixture writer ────────────────────────────────────────────────────────────


def _write_eval_json(
    path: Path,
    draft_models,
    dataset,
    results,
    n_samples_per_position=None,
):
    n_positions = len(next(iter(results.values())))
    n_samples_per_position = n_samples_per_position or [10] * n_positions
    payload = {
        "metadata": {
            "draft_models": list(draft_models),
            "target_model": "TurboSparse-Mistral-Instruct",
            "target_source": "dataset",
            "dataset": dataset,
            "n_positions": n_positions,
            "n_samples_total": 10,
            "n_samples_per_position": n_samples_per_position,
            "n_skipped_per_position": [0] * n_positions,
            "n_special_target_per_position": [0] * n_positions,
            "n_samples_with_oob_in_trunk": 0,
            "top_k_from_dataset": 10,
            "target_renormalized": True,
            "metrics": ["overlap_area"],
            "aggregations": ["single"],
            "config_snapshot": {},
        },
        "results": results,
    }
    path.write_text(json.dumps(payload))


# ── parse_result_key ──────────────────────────────────────────────────────────


def test_parse_result_key_basic():
    assert parse_result_key("single__overlap_area__top10_from_dataset") == (
        "single", "overlap_area", "top10_from_dataset",
    )
    assert parse_result_key("best__kl__full_target") == ("best", "kl", "full_target")


def test_parse_result_key_metric_with_underscore():
    assert parse_result_key("single__topk_overlap__top10_from_dataset") == (
        "single", "topk_overlap", "top10_from_dataset",
    )
    assert parse_result_key("single__top1_match__full_target") == (
        "single", "top1_match", "full_target",
    )


def test_parse_result_key_malformed():
    assert parse_result_key("no__parts") is None
    assert parse_result_key("only_one_part") is None


# ── compute_position_value ────────────────────────────────────────────────────


def test_compute_position_value_int():
    arr = [0.5, 0.7, None, 0.3]
    assert compute_position_value(arr, 0) == 0.5
    assert compute_position_value(arr, 1) == 0.7
    assert compute_position_value(arr, 2) is None  # None value
    assert compute_position_value(arr, 5) is None  # out of range


def test_compute_position_value_range():
    arr = [0.5, 0.7, None, 0.3]
    assert compute_position_value(arr, [0, 1]) == pytest.approx(0.6)
    # None skipped in average
    assert compute_position_value(arr, [1, 3]) == pytest.approx(0.5)


def test_compute_position_value_all_none():
    assert compute_position_value([None, None], [0, 1]) is None


def test_compute_position_value_bad_input():
    with pytest.raises(ValueError):
        compute_position_value([1.0], "0")


# ── compute_position_samples ──────────────────────────────────────────────────


def test_compute_position_samples_int():
    assert compute_position_samples([10, 20, 30], 0) == 10
    assert compute_position_samples([10, 20, 30], 5) is None


def test_compute_position_samples_range():
    # mean of [10, 20, 30]
    assert compute_position_samples([10, 20, 30], [0, 2]) == 20


# ── label logic ───────────────────────────────────────────────────────────────


def test_auto_series_label():
    assert auto_series_label(["a"], "single") == "a"
    assert auto_series_label(["a"], "best") == "a/best"
    assert auto_series_label(["a", "b"], "best") == "a+b/best"
    assert auto_series_label(["a", "b"], "single") == "a+b"


def test_apply_label_override_basic():
    assert apply_label_override("m1", ["m1"], {"m1": "baseline"}) == "baseline"
    assert apply_label_override("m1", ["m1"], {}) == "m1"


def test_apply_label_override_preserves_agg_suffix():
    # override doesn't contain "/" → suffix "/best" is preserved
    assert apply_label_override("m1/best", ["m1"], {"m1": "baseline"}) == "baseline/best"


def test_apply_label_override_verbatim_when_slash():
    # override contains "/" → used verbatim
    assert apply_label_override("m1/best", ["m1"], {"m1": "MyModel/x"}) == "MyModel/x"


def test_apply_label_override_multi_draft():
    assert apply_label_override("a+b", ["a", "b"], {"a+b": "ensemble"}) == "ensemble"


# ── discovery / loading ───────────────────────────────────────────────────────


def test_discover_and_load(tmp_path):
    _write_eval_json(
        tmp_path / "eval__m1__ds1.json",
        draft_models=["m1"], dataset="ds1.jsonl",
        results={"single__overlap_area__top10_from_dataset": [0.7, 0.8, 0.6]},
    )
    (tmp_path / "not_eval.json").write_text('{"foo": "bar"}')
    (tmp_path / "notes.txt").write_text("random")

    files = discover_eval_files(tmp_path)
    assert len(files) == 2  # only *.json files
    loaded = [load_eval_json(f) for f in files]
    assert sum(1 for x in loaded if x is not None) == 1


def test_load_eval_json_rejects_bad_json(tmp_path):
    (tmp_path / "junk.json").write_text("{ not json")
    assert load_eval_json(tmp_path / "junk.json") is None


# ── build_rows ────────────────────────────────────────────────────────────────


def test_build_rows_two_models_one_dataset(tmp_path):
    _write_eval_json(
        tmp_path / "eval__m1__aeslc.json",
        draft_models=["m1"], dataset="aeslc_10templates.jsonl",
        results={
            "single__overlap_area__top10_from_dataset": [0.7, 0.8, 0.6],
            "single__top1_match__top10_from_dataset": [1.0, 0.0, 1.0],
        },
    )
    _write_eval_json(
        tmp_path / "eval__m2__aeslc.json",
        draft_models=["m2"], dataset="aeslc_10templates.jsonl",
        results={"single__overlap_area__top10_from_dataset": [0.9, 0.85, 0.8]},
    )
    files = discover_eval_files(tmp_path)
    rows = build_rows(files, "overlap_area", 0, None, {})
    assert len(rows) == 2
    assert {r.series for r in rows} == {"m1", "m2"}
    assert all(r.dataset == "aeslc_10templates" for r in rows)
    m1_value = next(r.value for r in rows if r.series == "m1")
    m2_value = next(r.value for r in rows if r.series == "m2")
    assert m1_value == pytest.approx(0.7)
    assert m2_value == pytest.approx(0.9)


def test_build_rows_range_position(tmp_path):
    _write_eval_json(
        tmp_path / "eval__m1__aeslc.json",
        draft_models=["m1"], dataset="aeslc.jsonl",
        results={"single__overlap_area__top10_from_dataset": [0.7, 0.9, 0.6, 0.4]},
    )
    files = discover_eval_files(tmp_path)
    rows = build_rows(files, "overlap_area", [0, 1], None, {})
    assert len(rows) == 1
    assert rows[0].value == pytest.approx(0.8)  # (0.7 + 0.9) / 2


def test_build_rows_aggregation_filter(tmp_path):
    _write_eval_json(
        tmp_path / "eval__multi__aeslc.json",
        draft_models=["a", "b"], dataset="aeslc.jsonl",
        results={
            "best__overlap_area__top10_from_dataset": [0.9],
            "arithmetic_average__overlap_area__top10_from_dataset": [0.7],
        },
    )
    files = discover_eval_files(tmp_path)
    rows_all = build_rows(files, "overlap_area", 0, None, {})
    assert len(rows_all) == 2
    rows_best = build_rows(files, "overlap_area", 0, ["best"], {})
    assert len(rows_best) == 1
    assert rows_best[0].series == "a+b/best"


def test_build_rows_label_override(tmp_path):
    _write_eval_json(
        tmp_path / "eval__m1__aeslc.json",
        draft_models=["m1"], dataset="aeslc.jsonl",
        results={"single__overlap_area__top10_from_dataset": [0.7]},
    )
    files = discover_eval_files(tmp_path)
    rows = build_rows(files, "overlap_area", 0, None, {"m1": "baseline"})
    assert rows[0].series == "baseline"


def test_build_rows_metric_mismatch_returns_empty(tmp_path):
    _write_eval_json(
        tmp_path / "eval__m1__aeslc.json",
        draft_models=["m1"], dataset="aeslc.jsonl",
        results={"single__overlap_area__top10_from_dataset": [0.7]},
    )
    files = discover_eval_files(tmp_path)
    rows = build_rows(files, "kl", 0, None, {})
    assert rows == []


# ── sorting ───────────────────────────────────────────────────────────────────


def _r(dataset: str, series: str, value: float) -> Row:
    return Row(dataset, series, value, 10, Path("/tmp/x.json"))


def test_sort_datasets_value_asc():
    rows = [_r("ds1", "m", 0.9), _r("ds2", "m", 0.5), _r("ds3", "m", 0.7)]
    assert sort_datasets(rows, "value_asc") == ["ds2", "ds3", "ds1"]


def test_sort_datasets_value_desc():
    rows = [_r("ds1", "m", 0.9), _r("ds2", "m", 0.5), _r("ds3", "m", 0.7)]
    assert sort_datasets(rows, "value_desc") == ["ds1", "ds3", "ds2"]


def test_sort_datasets_by_name():
    rows = [_r("b", "m", 0.5), _r("a", "m", 0.7), _r("c", "m", 0.3)]
    assert sort_datasets(rows, "name_asc") == ["a", "b", "c"]
    assert sort_datasets(rows, "name_desc") == ["c", "b", "a"]


def test_sort_datasets_averages_across_series():
    # ds1 has one series at 0.9; ds2 has two series averaged to 0.6
    rows = [
        _r("ds1", "m1", 0.9),
        _r("ds2", "m1", 0.5),
        _r("ds2", "m2", 0.7),
    ]
    assert sort_datasets(rows, "value_asc") == ["ds2", "ds1"]


def test_sort_datasets_unknown_raises():
    with pytest.raises(ValueError):
        sort_datasets([_r("a", "m", 0.5)], "bogus")


# ── order_series ──────────────────────────────────────────────────────────────


def test_order_series_auto_alphabetical():
    rows = [_r("d", "b", 0.5), _r("d", "a", 0.5), _r("d", "c", 0.5)]
    assert order_series(rows, []) == ["a", "b", "c"]


def test_order_series_explicit_honored():
    rows = [_r("d", "b", 0.5), _r("d", "a", 0.5), _r("d", "c", 0.5)]
    assert order_series(rows, ["c", "a"]) == ["c", "a", "b"]


def test_order_series_explicit_with_unknown_ignored():
    rows = [_r("d", "b", 0.5), _r("d", "a", 0.5)]
    # "z" is not in rows → dropped from explicit list
    assert order_series(rows, ["z", "a"]) == ["a", "b"]


# ── pad_widths ────────────────────────────────────────────────────────────────


def test_pad_widths_no_padding():
    assert pad_widths([0.8, 0.5], 2) == [0.8, 0.5]


def test_pad_widths_geometric_shrink():
    out = pad_widths([0.8, 0.5], 4)
    assert out[0] == 0.8
    assert out[1] == 0.5
    assert out[2] == pytest.approx(0.5 * 0.7)
    assert out[3] == pytest.approx(0.5 * 0.7 * 0.7)


def test_pad_widths_truncation():
    assert pad_widths([0.8, 0.5, 0.3], 2) == [0.8, 0.5]


def test_pad_widths_from_empty():
    out = pad_widths([], 3)
    assert len(out) == 3
    assert all(w > 0 for w in out)
