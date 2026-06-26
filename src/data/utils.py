from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union


def walk_data_files(root: Union[str, Path]) -> List[Path]:
    """Recursively collect all *.jsonl and *.npz files under root, sorted."""
    root = Path(root)
    return sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.npz")))


# Back-compat alias — older callers imported walk_jsonl directly.
def walk_jsonl(root: Union[str, Path]) -> List[Path]:
    return walk_data_files(root)


def detect_format(record: dict) -> str:
    """
    Detect record format from field names.
    Returns 'synthetic', 'flan', or 'plain'.
    """
    if "trunk" in record and ("top10_ids" in record or "top10" in record):
        return "synthetic"
    if "inputs" in record and "targets" in record:
        return "flan"
    if "text" in record:
        return "plain"
    raise ValueError(f"Unrecognized record format. Keys: {list(record.keys())}")


def _normalize_top10(record: dict) -> None:
    """
    In-place: convert legacy top10 (list of list of dicts) to the new schema
    (parallel top10_ids / top10_probs arrays). No-op if already normalized.
    Drops the "token" field — it's unused downstream.
    """
    if "top10_ids" in record:
        return
    legacy = record.pop("top10", None)
    if legacy is None:
        return
    ids: list[list[int]] = []
    probs: list[list[float]] = []
    for pos in legacy:
        if not pos:
            ids.append([])
            probs.append([])
            continue
        ids.append([int(e["token_id"]) for e in pos])
        probs.append([float(e["prob"]) for e in pos])
    record["top10_ids"] = ids
    record["top10_probs"] = probs


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    default_cluster = path.stem
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "cluster" not in record:
                record["cluster"] = default_cluster
            _normalize_top10(record)
            records.append(record)
    return records


def _load_npz(path: Path) -> List[dict]:
    """
    Materialize npz-packed cluster into a list of record dicts compatible
    with the JSONL path. top10_ids / top10_probs are reconstructed from the
    concatenated arrays + per-sample trunk_lens.
    """
    import numpy as np

    archive = np.load(path, allow_pickle=True)
    default_cluster = str(archive["cluster"]) if "cluster" in archive.files else path.stem
    prompts = archive["prompts"]
    references = archive["references"] if "references" in archive.files else None
    trunk_lens = archive["trunk_lens"].astype(int)
    trunk_ids = archive["trunk_ids"]
    top10_ids_all = archive["top10_ids"]
    top10_probs_q = archive["top10_probs_q"]
    top10_mask = archive["top10_mask"]
    k = int(archive["top_k"]) if "top_k" in archive.files else top10_ids_all.shape[1]

    records: List[dict] = []
    offset = 0
    for i, n in enumerate(trunk_lens):
        n = int(n)
        sample_trunk = trunk_ids[offset : offset + n].astype(int).tolist()

        if k > 0 and n > 0:
            ids_slice = top10_ids_all[offset : offset + n]
            probs_slice = top10_probs_q[offset : offset + n]
            mask_slice = top10_mask[offset : offset + n]
            ids_rows: list[list[int]] = []
            probs_rows: list[list[float]] = []
            for j in range(n):
                if mask_slice[j]:
                    ids_rows.append(ids_slice[j].astype(int).tolist())
                    probs_rows.append((probs_slice[j].astype(float) / 1000.0).tolist())
                else:
                    ids_rows.append([])
                    probs_rows.append([])
        else:
            ids_rows = [[] for _ in range(n)]
            probs_rows = [[] for _ in range(n)]

        records.append({
            "cluster": default_cluster,
            "prompt": str(prompts[i]),
            "reference": str(references[i]) if references is not None else "",
            "trunk": sample_trunk,
            "top10_ids": ids_rows,
            "top10_probs": probs_rows,
        })
        offset += n

    return records


def build_index(paths: List[Path]) -> Tuple[List[dict], List[str]]:
    """
    Load all records from a list of JSONL or NPZ paths into memory.
    Returns (records, cluster_labels).
    Cluster label comes from record['cluster'] if present, else from the file stem.
    """
    records: List[dict] = []
    cluster_labels: List[str] = []

    for path in paths:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            loaded = _load_npz(path)
        else:
            loaded = _load_jsonl(path)
        for record in loaded:
            records.append(record)
            cluster_labels.append(record.get("cluster", path.stem))

    return records, cluster_labels
