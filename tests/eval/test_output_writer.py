import json
from pathlib import Path

import pytest

from src.eval.output_writer import (
    EvalMetadata,
    draft_tag,
    mode_label,
    output_filename,
    result_key,
    write_eval_json,
)


def test_mode_label_dataset():
    assert mode_label("dataset", 10) == "top10_from_dataset"
    assert mode_label("dataset", 5) == "top5_from_dataset"


def test_mode_label_model():
    assert mode_label("model", None) == "full_target"


def test_mode_label_dataset_requires_k():
    with pytest.raises(ValueError):
        mode_label("dataset", None)


def test_mode_label_unknown_raises():
    with pytest.raises(ValueError):
        mode_label("nonsense", None)


def test_result_key_format():
    assert (
        result_key("arithmetic_average", "overlap_area", "top10_from_dataset")
        == "arithmetic_average__overlap_area__top10_from_dataset"
    )


def test_draft_tag_single():
    assert draft_tag(["my-model"]) == "my-model"


def test_draft_tag_multi_is_deterministic_and_order_independent():
    a = draft_tag(["a", "b", "c"])
    b = draft_tag(["c", "b", "a"])
    assert a == b
    assert a.startswith("multi-")
    assert len(a.split("multi-")[1]) == 6


def test_output_filename():
    assert (
        output_filename(["tiny-mixtral"], "data/synthetic/v1/aeslc_10templates.jsonl")
        == "eval__tiny-mixtral__aeslc_10templates.json"
    )


def test_write_eval_json_roundtrip(tmp_path: Path):
    meta = EvalMetadata(
        draft_models=["tiny-mixtral"],
        target_model="TurboSparse",
        target_source="dataset",
        dataset="data/synthetic/v1/foo.jsonl",
        n_positions=3,
        n_samples_total=5,
        n_samples_per_position=[5, 5, 4],
        n_skipped_per_position=[0, 0, 1],
        n_special_target_per_position=[0, 0, 0],
        n_samples_with_oob_in_trunk=0,
        top_k_from_dataset=10,
        target_renormalized=True,
        metrics=["overlap_area"],
        aggregations=["single"],
        config_snapshot={"k": "v"},
    )
    results = {"single__overlap_area__top10_from_dataset": [0.9, 0.8, 0.7]}

    path = write_eval_json(
        tmp_path, ["tiny-mixtral"], "data/synthetic/v1/foo.jsonl", meta, results
    )

    assert path.name == "eval__tiny-mixtral__foo.json"
    data = json.loads(path.read_text())
    assert data["metadata"]["draft_models"] == ["tiny-mixtral"]
    assert data["metadata"]["n_positions"] == 3
    assert data["metadata"]["timestamp"].endswith("Z")
    assert data["results"]["single__overlap_area__top10_from_dataset"] == [0.9, 0.8, 0.7]
