from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union


def walk_jsonl(root: Union[str, Path]) -> List[Path]:
    """Recursively collect all *.jsonl files under root, sorted."""
    return sorted(Path(root).rglob("*.jsonl"))


def detect_format(record: dict) -> str:
    """
    Detect record format from field names.
    Returns 'synthetic', 'flan', or 'plain'.
    """
    if "trunk" in record and "top10" in record:
        return "synthetic"
    if "inputs" in record and "targets" in record:
        return "flan"
    if "text" in record:
        return "plain"
    raise ValueError(f"Unrecognized record format. Keys: {list(record.keys())}")


def build_index(paths: List[Path]) -> Tuple[List[dict], List[str]]:
    """
    Load all records from a list of JSONL paths into memory.
    Returns (records, cluster_labels).
    Cluster label comes from record['cluster'] if present, else from the file stem.
    """
    records: List[dict] = []
    cluster_labels: List[str] = []

    for path in paths:
        path = Path(path)
        default_cluster = path.stem
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
                cluster_labels.append(record.get("cluster", default_cluster))

    return records, cluster_labels
