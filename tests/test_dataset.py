"""
Behavior tests for SpecDecDataset over the validation set.

Most tests use the small `subset_dataset` fixture (3 clusters) for speed.
One marker-gated test loads the full 66-cluster set to catch issues that
only show up at scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

from src.data import SpecDecDataset


# ─── construction ───────────────────────────────────────────────────────────


def test_subset_dataset_loads(subset_dataset: SpecDecDataset) -> None:
    assert len(subset_dataset) > 0


def test_subset_dataset_cluster_count(subset_dataset: SpecDecDataset) -> None:
    assert len(subset_dataset.cluster_names()) == 3


def test_subset_cluster_names_unique(subset_dataset: SpecDecDataset) -> None:
    names = subset_dataset.cluster_names()
    assert len(set(names)) == len(names)


@pytest.mark.slow
def test_full_dataset_loads(validation_dir: Path, tokenizer) -> None:
    ds = SpecDecDataset.from_dir(validation_dir, tokenizer=tokenizer, mode="distillation")
    assert len(ds.cluster_names()) == 66
    assert len(ds) > 0


def test_from_dir_finds_files(validation_dir: Path, tokenizer) -> None:
    """from_dir should accept the validation root and pull at least one cluster."""
    ds = SpecDecDataset.from_dir(validation_dir, tokenizer=tokenizer, mode="distillation")
    assert len(ds) > 0


# ─── __getitem__ contract ───────────────────────────────────────────────────


def test_getitem_distillation_keys(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    required = {"input_ids", "attention_mask", "labels", "gen_start", "cluster",
                "top10_ids", "top10_probs"}
    assert required.issubset(item.keys()), f"missing keys: {required - set(item.keys())}"


def test_getitem_tensor_lengths_match(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    L = item["input_ids"].size(0)
    assert item["attention_mask"].size(0) == L
    assert item["labels"].size(0) == L
    assert item["attention_mask"].sum().item() == L  # all ones


def test_getitem_dtypes(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["top10_ids"].dtype == torch.long
    assert item["top10_probs"].dtype == torch.float32


def test_top10_shapes_consistent(subset_dataset: SpecDecDataset) -> None:
    """top10_ids and top10_probs must agree on shape; second dim is always 10."""
    for idx in range(min(20, len(subset_dataset))):
        item = subset_dataset[idx]
        assert item["top10_ids"].shape == item["top10_probs"].shape
        assert item["top10_ids"].size(1) == 10


def test_top10_gen_len_matches_labels(subset_dataset: SpecDecDataset) -> None:
    """top10 rows should match the number of generated (non -100) label positions."""
    for idx in range(min(10, len(subset_dataset))):
        item = subset_dataset[idx]
        non_pad = (item["labels"] != -100).sum().item()
        assert item["top10_ids"].size(0) == non_pad, (
            f"item {idx}: top10 has {item['top10_ids'].size(0)} rows but "
            f"{non_pad} trunk-labelled positions"
        )


def test_labels_mask_prompt_positions(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    gs = item["gen_start"]
    if gs > 0:
        assert (item["labels"][:gs] == -100).all(), "prompt positions should be masked"
    assert (item["labels"][gs:] != -100).all(), "trunk positions should not be masked"


def test_labels_match_trunk(subset_dataset: SpecDecDataset) -> None:
    """labels[gen_start:] should equal input_ids[gen_start:] (i.e. the trunk)."""
    item = subset_dataset[0]
    gs = item["gen_start"]
    assert torch.equal(item["labels"][gs:], item["input_ids"][gs:])


def test_gen_start_in_range(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    assert 0 <= item["gen_start"] <= item["input_ids"].size(0)


def test_cluster_field_is_string(subset_dataset: SpecDecDataset) -> None:
    item = subset_dataset[0]
    assert isinstance(item["cluster"], str)
    assert item["cluster"] in subset_dataset.cluster_names()


# ─── standard mode ──────────────────────────────────────────────────────────


def test_standard_mode_skips_top10(subset_paths: List[Path], tokenizer) -> None:
    ds = SpecDecDataset(subset_paths, tokenizer, mode="standard")
    item = ds[0]
    assert "top10_ids" not in item
    assert "top10_probs" not in item
    # Other keys still present
    for k in ("input_ids", "attention_mask", "labels", "gen_start", "cluster"):
        assert k in item


# ─── cluster API ────────────────────────────────────────────────────────────


def test_get_cluster_subset(subset_dataset: SpecDecDataset) -> None:
    for name in subset_dataset.cluster_names():
        sub = subset_dataset.get_cluster_subset(name)
        assert len(sub) > 0
        # Spot-check: every sample in the subset has matching cluster
        for j in range(min(3, len(sub))):
            assert sub[j]["cluster"] == name


def test_get_cluster_subset_unknown_raises(subset_dataset: SpecDecDataset) -> None:
    with pytest.raises(KeyError):
        subset_dataset.get_cluster_subset("__no_such_cluster__")


def test_cluster_counts_sum_to_total(subset_dataset: SpecDecDataset) -> None:
    stats = subset_dataset.stats()
    assert sum(stats["cluster_counts"].values()) == len(subset_dataset)


# ─── split() ────────────────────────────────────────────────────────────────


def test_split_total_preserved(subset_dataset: SpecDecDataset) -> None:
    train, val, test = subset_dataset.split((0.8, 0.1, 0.1), seed=42)
    assert len(train) + len(val) + len(test) == len(subset_dataset)


def test_split_no_overlap(subset_dataset: SpecDecDataset) -> None:
    train, val, test = subset_dataset.split((0.8, 0.1, 0.1), seed=42)
    train_idx = set(train.indices)
    val_idx = set(val.indices)
    test_idx = set(test.indices)
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)


def test_split_is_deterministic(subset_dataset: SpecDecDataset) -> None:
    a = subset_dataset.split((0.8, 0.1, 0.1), seed=42)
    b = subset_dataset.split((0.8, 0.1, 0.1), seed=42)
    assert a[0].indices == b[0].indices
    assert a[1].indices == b[1].indices
    assert a[2].indices == b[2].indices


def test_split_ratios_bad_sum_raises(subset_dataset: SpecDecDataset) -> None:
    with pytest.raises(ValueError):
        subset_dataset.split((0.5, 0.4, 0.4))


def test_split_stratified(subset_dataset: SpecDecDataset) -> None:
    """Each split should contain samples from every cluster."""
    train, val, test = subset_dataset.split((0.8, 0.1, 0.1), seed=42)
    for sub in (train, val, test):
        clusters_present = {subset_dataset[i]["cluster"] for i in sub.indices}
        assert clusters_present == set(subset_dataset.cluster_names())


# ─── stats ──────────────────────────────────────────────────────────────────


def test_stats_keys(subset_dataset: SpecDecDataset) -> None:
    stats = subset_dataset.stats()
    for k in ("total_samples", "num_clusters", "cluster_counts",
              "gen_len", "mean_top1_prob"):
        assert k in stats, f"missing stats key: {k}"


def test_stats_top1_in_unit_interval(subset_dataset: SpecDecDataset) -> None:
    stats = subset_dataset.stats()
    assert 0.0 <= stats["mean_top1_prob"] <= 1.0


def test_stats_gen_len_bounds(subset_dataset: SpecDecDataset) -> None:
    stats = subset_dataset.stats()
    g = stats["gen_len"]
    assert g["min"] >= 1
    assert g["min"] <= g["mean"] <= g["max"]


# ─── filters ────────────────────────────────────────────────────────────────


def test_clusters_filter(subset_paths: List[Path], tokenizer) -> None:
    name = "wsc_10templates"
    ds = SpecDecDataset(subset_paths, tokenizer, mode="distillation",
                        clusters_filter=[name])
    assert ds.cluster_names() == [name]


def test_min_top1_prob_filter_keeps_subset(subset_paths: List[Path], tokenizer) -> None:
    base = SpecDecDataset(subset_paths, tokenizer, mode="distillation")
    # Use the dataset's own stats to set a reasonable threshold so we don't drop everything.
    threshold = base.stats()["mean_top1_prob"]
    filtered = SpecDecDataset(subset_paths, tokenizer, mode="distillation",
                              min_top1_prob=threshold)
    assert len(filtered) <= len(base)


def test_min_top1_prob_filter_extreme(subset_paths: List[Path], tokenizer) -> None:
    """A threshold > 1 must drop everything; a threshold < 0 must keep everything."""
    base = SpecDecDataset(subset_paths, tokenizer, mode="distillation")
    drop_all = SpecDecDataset(subset_paths, tokenizer, mode="distillation",
                              min_top1_prob=1.5)
    keep_all = SpecDecDataset(subset_paths, tokenizer, mode="distillation",
                              min_top1_prob=-1.0)
    assert len(drop_all) == 0
    assert len(keep_all) == len(base)
