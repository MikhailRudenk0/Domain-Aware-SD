"""
Round-trip tests for the storage formats.

Take real validation records, write them through both writers (JSONL and NPZ),
read them back via the dataset reader, and confirm the logical fields survive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data import SpecDecDataset
from src.data.utils import _load_npz, build_index, detect_format, walk_data_files


def _load_records_from_jsonl(path: Path, limit: int = 5) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            records.append(json.loads(line))
    return records


def test_walk_data_files_picks_up_jsonl_and_npz(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("")
    (tmp_path / "b.npz").write_bytes(b"")
    (tmp_path / "c.txt").write_text("")
    paths = walk_data_files(tmp_path)
    suffixes = {p.suffix for p in paths}
    assert suffixes == {".jsonl", ".npz"}


def test_detect_format_synthetic(subset_paths) -> None:
    rec = _load_records_from_jsonl(subset_paths[0], limit=1)[0]
    assert detect_format(rec) == "synthetic"


def test_detect_format_flan() -> None:
    assert detect_format({"inputs": "x", "targets": "y"}) == "flan"


def test_detect_format_plain() -> None:
    assert detect_format({"text": "hello"}) == "plain"


def test_detect_format_unknown() -> None:
    with pytest.raises(ValueError):
        detect_format({"foo": "bar"})


def test_legacy_top10_normalized_on_load(tmp_path: Path) -> None:
    """Old-schema records must be silently converted to the new parallel arrays."""
    legacy = {
        "cluster": "demo",
        "prompt": "P",
        "trunk": [10, 20],
        "top10": [
            [{"token_id": 10, "prob": 0.6, "token": "a"},
             {"token_id": 11, "prob": 0.3, "token": "b"}],
            [{"token_id": 20, "prob": 0.5, "token": "c"}],
        ],
    }
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps(legacy) + "\n")
    records, _ = build_index([p])
    assert len(records) == 1
    r = records[0]
    assert "top10" not in r
    assert r["top10_ids"] == [[10, 11], [20]]
    assert r["top10_probs"] == [[0.6, 0.3], [0.5]]


def test_npz_roundtrip_from_real_records(subset_paths, tmp_path: Path, tokenizer) -> None:
    """
    Take ~5 records from a real cluster, round-trip through write_npz, then
    confirm the reader gets back trunks + top10 in the same shape and values.
    """
    np = pytest.importorskip("numpy")
    from src.generate_synthetic_data import write_npz

    source = subset_paths[0]
    records = _load_records_from_jsonl(source, limit=5)
    # `write_npz` requires every record to have a `cluster` and the unified field set
    for r in records:
        r.setdefault("reference", "")

    out = tmp_path / "demo.npz"
    write_npz(records, out)
    assert out.exists() and out.stat().st_size > 0

    loaded = _load_npz(out)
    assert len(loaded) == len(records)
    for orig, back in zip(records, loaded):
        assert back["trunk"] == orig["trunk"]
        assert back["prompt"] == orig["prompt"]
        # Each non-empty row should match in ids; probs are quantized to 0.001
        for o_ids, b_ids in zip(orig["top10_ids"], back["top10_ids"]):
            if not o_ids:
                assert b_ids == []
                continue
            assert o_ids == b_ids
        for o_probs, b_probs in zip(orig["top10_probs"], back["top10_probs"]):
            if not o_probs:
                assert b_probs == []
                continue
            assert len(o_probs) == len(b_probs)
            for op, bp in zip(o_probs, b_probs):
                assert abs(op - bp) <= 1e-3  # uint16 quantization tolerance


def test_dataset_loads_npz_file(subset_paths, tmp_path: Path, tokenizer) -> None:
    """SpecDecDataset should accept an .npz file and produce normal items."""
    np = pytest.importorskip("numpy")
    from src.generate_synthetic_data import write_npz

    records = _load_records_from_jsonl(subset_paths[0], limit=4)
    for r in records:
        r.setdefault("reference", "")
    npz_path = tmp_path / "demo.npz"
    write_npz(records, npz_path)

    ds = SpecDecDataset(npz_path, tokenizer, mode="distillation")
    assert len(ds) == 4
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["top10_ids"].size(1) == 10


def test_dataset_loads_jsonl_and_npz_together(subset_paths, tmp_path: Path, tokenizer) -> None:
    """Walking a directory must mix .jsonl and .npz transparently."""
    np = pytest.importorskip("numpy")
    from src.generate_synthetic_data import write_npz

    # Stage a tiny mixed directory: one .jsonl (a copy of a real file's first 3 records)
    # and one .npz built from the next file's first 3 records.
    src_jsonl = subset_paths[0]
    src_for_npz = subset_paths[1]

    j = tmp_path / "from_jsonl.jsonl"
    with open(src_jsonl) as fin, open(j, "w") as fout:
        for i, line in enumerate(fin):
            if i >= 3:
                break
            fout.write(line)

    records = _load_records_from_jsonl(src_for_npz, limit=3)
    for r in records:
        r.setdefault("reference", "")
    write_npz(records, tmp_path / "from_npz.npz")

    ds = SpecDecDataset.from_dir(tmp_path, tokenizer=tokenizer, mode="distillation")
    assert len(ds) == 6
    # Two distinct clusters present
    assert len(set(ds.cluster_names())) == 2
