"""Tests for DistillationCollator over real validation samples."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import DistillationCollator, SpecDecDataset


def test_collator_basic_shapes(subset_dataset: SpecDecDataset, tokenizer) -> None:
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([subset_dataset[i] for i in range(4)])
    B, L = batch["input_ids"].shape
    assert B == 4
    assert batch["attention_mask"].shape == (B, L)
    assert batch["labels"].shape == (B, L)
    assert "top10_ids" in batch and "top10_probs" in batch
    assert batch["top10_ids"].size(0) == B
    assert batch["top10_ids"].size(2) == 10
    assert batch["top10_probs"].shape == batch["top10_ids"].shape


def test_collator_pads_to_max_in_batch(subset_dataset: SpecDecDataset, tokenizer) -> None:
    """Padded length must equal the longest sample in the batch."""
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    items = [subset_dataset[i] for i in range(8)]
    batch = collator(items)
    expected_L = max(it["input_ids"].size(0) for it in items)
    assert batch["input_ids"].size(1) == expected_L


def test_collator_label_padding_is_minus_100(subset_dataset: SpecDecDataset, tokenizer) -> None:
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    items = [subset_dataset[i] for i in range(4)]
    batch = collator(items)
    # Any position beyond an item's original length must be -100 in labels.
    for i, it in enumerate(items):
        L_i = it["input_ids"].size(0)
        if batch["labels"].size(1) > L_i:
            assert (batch["labels"][i, L_i:] == -100).all()


def test_collator_attention_mask_zero_on_pad(subset_dataset: SpecDecDataset, tokenizer) -> None:
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    items = [subset_dataset[i] for i in range(4)]
    batch = collator(items)
    for i, it in enumerate(items):
        L_i = it["input_ids"].size(0)
        if batch["attention_mask"].size(1) > L_i:
            assert (batch["attention_mask"][i, L_i:] == 0).all()


def test_collator_handles_mixed_with_without_top10(tokenizer, subset_paths: List[Path]) -> None:
    """Distillation-mode item + standard-mode item in the same batch."""
    distill = SpecDecDataset(subset_paths, tokenizer, mode="distillation")
    standard = SpecDecDataset(subset_paths, tokenizer, mode="standard")
    collator = DistillationCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([distill[0], standard[0]])
    # The collator emits distillation tensors when ANY item carries them.
    assert "top10_ids" in batch
    # The standard-mode slot (index 1) must be all-zero.
    assert (batch["top10_ids"][1] == 0).all()
    assert (batch["top10_probs"][1] == 0).all()


def test_dataloader_integration(subset_dataset: SpecDecDataset, tokenizer) -> None:
    """The collator must plug into a real DataLoader and yield batched tensors."""
    loader = DataLoader(
        subset_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=DistillationCollator(pad_token_id=tokenizer.pad_token_id),
    )
    batch = next(iter(loader))
    assert batch["input_ids"].size(0) == 4
    assert isinstance(batch["cluster"], list) and len(batch["cluster"]) == 4
    assert isinstance(batch["gen_start"], list) and len(batch["gen_start"]) == 4
