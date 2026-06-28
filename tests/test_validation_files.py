"""
File-level invariants for the downloaded validation set.

These tests look at the raw .jsonl files only — they don't touch the
SpecDecDataset / tokenizer layers, so they fail loudly if the files
themselves are malformed (truncated lines, missing fields, drift in schema).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

EXPECTED_FILE_COUNT = 66
REQUIRED_FIELDS = {"cluster", "prompt", "trunk", "top10_ids", "top10_probs"}
TARGET_VOCAB_SIZE = 32064  # TurboSparse-Mistral-Instruct
K = 10                     # top-K width fixed at generation time


def test_expected_file_count(all_jsonl_paths: List[Path]) -> None:
    assert len(all_jsonl_paths) == EXPECTED_FILE_COUNT, (
        f"expected {EXPECTED_FILE_COUNT} validation files, got {len(all_jsonl_paths)}"
    )


def test_filenames_unique(all_jsonl_paths: List[Path]) -> None:
    names = [p.name for p in all_jsonl_paths]
    assert len(set(names)) == len(names), "duplicate filenames in validation dir"


def test_files_are_non_empty(all_jsonl_paths: List[Path]) -> None:
    empty = [p for p in all_jsonl_paths if p.stat().st_size == 0]
    assert not empty, f"empty files: {empty}"


def test_every_first_record_has_required_fields(first_record_per_file) -> None:
    for path, rec in first_record_per_file:
        missing = REQUIRED_FIELDS - set(rec.keys())
        assert not missing, f"{path.name}: missing fields {missing}"


def test_no_legacy_top10_field(first_record_per_file) -> None:
    """We migrated away from the list-of-dicts schema — make sure validation is clean."""
    for path, rec in first_record_per_file:
        assert "top10" not in rec, f"{path.name}: legacy 'top10' field still present"


def test_cluster_field_matches_filename(first_record_per_file) -> None:
    """`cluster` should equal the filename stem (without trailing '_train' etc.)."""
    for path, rec in first_record_per_file:
        expected = path.stem
        for suffix in ("_train", "_test", "_validation"):
            if expected.endswith(suffix):
                expected = expected[: -len(suffix)]
                break
        assert rec["cluster"] == expected, (
            f"{path.name}: cluster='{rec['cluster']}' but expected '{expected}'"
        )


@pytest.mark.parametrize("sample_n", [5])
def test_arrays_aligned_per_record(all_jsonl_paths: List[Path], sample_n: int) -> None:
    """
    For up to `sample_n` records per file, the parallel arrays must be aligned:
      len(top10_ids) == len(top10_probs) == len(trunk)
      every non-empty row has the same length in ids and probs
      probs are in [0, 1] and ids fit in target vocab
    """
    for path in all_jsonl_paths:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_n:
                    break
                rec = json.loads(line)
                trunk = rec["trunk"]
                ids = rec["top10_ids"]
                probs = rec["top10_probs"]
                assert len(ids) == len(trunk), f"{path.name}#{i}: top10_ids/trunk length mismatch"
                assert len(probs) == len(trunk), f"{path.name}#{i}: top10_probs/trunk length mismatch"
                for j, (id_row, prob_row) in enumerate(zip(ids, probs)):
                    assert len(id_row) == len(prob_row), (
                        f"{path.name}#{i} row {j}: id/prob row length mismatch"
                    )
                    if id_row:
                        assert all(0 <= t < TARGET_VOCAB_SIZE for t in id_row), (
                            f"{path.name}#{i} row {j}: token id out of vocab range"
                        )
                        assert all(0.0 <= p <= 1.0 for p in prob_row), (
                            f"{path.name}#{i} row {j}: prob outside [0, 1]"
                        )
                        # probs sorted descending (top-1 first)
                        assert prob_row == sorted(prob_row, reverse=True), (
                            f"{path.name}#{i} row {j}: probs not sorted descending"
                        )


def test_trunk_token_ids_within_vocab(first_record_per_file) -> None:
    for path, rec in first_record_per_file:
        out_of_range = [t for t in rec["trunk"] if not 0 <= t < TARGET_VOCAB_SIZE]
        assert not out_of_range, (
            f"{path.name}: trunk has {len(out_of_range)} token IDs outside vocab"
        )


def test_top_k_prob_mass_reasonable(first_record_per_file) -> None:
    """
    Per-position invariant on the *distribution itself*: the sum of the
    top-K probabilities at any non-empty row should be in (0, 1]. We
    further check that the mean row-sum across each record is > 0.1 —
    if it were near zero, the 3-d.p. rounding would have stripped all mass
    or the writer would be storing logprobs instead of probs.
    """
    for path, rec in first_record_per_file:
        row_sums: list[float] = []
        for prob_row in rec["top10_probs"]:
            if not prob_row:
                continue
            s = sum(prob_row)
            assert 0.0 < s <= 1.001, (
                f"{path.name}: top-K row sum {s:.4f} outside (0, 1]"
            )
            row_sums.append(s)
        if not row_sums:
            continue
        mean_sum = sum(row_sums) / len(row_sums)
        assert mean_sum > 0.10, (
            f"{path.name}: mean top-K row sum is {mean_sum:.3f} — "
            f"distribution mass looks wrong"
        )
