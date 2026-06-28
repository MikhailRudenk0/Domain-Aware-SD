"""
Shared pytest fixtures for the data-pipeline test suite.

Adds the repo root to sys.path so `from src.data import ...` works without
installing the project.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation" / "v1"
TOKENIZER_DIR = PROJECT_ROOT / "tiny-mixtral"

# A small subset for "fast" tests — picked to be small files (wsc is the smallest).
SUBSET_NAMES = [
    "wsc_10templates.jsonl",
    "aeslc_10templates.jsonl",
    "yelp_polarity_reviews_10templates.jsonl",
]


def _require_validation_data() -> None:
    if not VALIDATION_DIR.exists():
        pytest.skip(
            f"validation data not present at {VALIDATION_DIR} — "
            f"download with the s3 helper before running these tests"
        )


@pytest.fixture(scope="session")
def validation_dir() -> Path:
    _require_validation_data()
    return VALIDATION_DIR


@pytest.fixture(scope="session")
def all_jsonl_paths(validation_dir: Path) -> List[Path]:
    paths = sorted(validation_dir.glob("*.jsonl"))
    if not paths:
        pytest.skip(f"no .jsonl files found in {validation_dir}")
    return paths


@pytest.fixture(scope="session")
def subset_paths(validation_dir: Path) -> List[Path]:
    paths = [validation_dir / n for n in SUBSET_NAMES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"subset files missing: {[str(p) for p in missing]}")
    return paths


@pytest.fixture(scope="session")
def tokenizer():
    """Load the tiny-mixtral tokenizer (shares Mistral SentencePiece with target)."""
    if not TOKENIZER_DIR.exists():
        pytest.skip(f"tokenizer not found at {TOKENIZER_DIR}")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def subset_dataset(subset_paths, tokenizer):
    """Dataset built from the 3-cluster subset (fast)."""
    from src.data import SpecDecDataset
    return SpecDecDataset(subset_paths, tokenizer, mode="distillation")


@pytest.fixture(scope="session")
def first_record_per_file(all_jsonl_paths: List[Path]) -> List[tuple[Path, dict]]:
    """First record of every cluster — cheap header-style schema sanity."""
    out: list[tuple[Path, dict]] = []
    for p in all_jsonl_paths:
        with open(p, encoding="utf-8") as f:
            line = f.readline().strip()
        if not line:
            pytest.fail(f"{p.name}: first line is empty")
        out.append((p, json.loads(line)))
    return out
