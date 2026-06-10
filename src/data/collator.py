from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class DistillationCollator:
    """
    Collate function for SpecDecDataset batches.

    Pads variable-length sequences (input_ids, attention_mask, labels) and,
    when present, top-10 distillation tensors (top10_ids, top10_probs) to the
    batch maximums. Missing distillation fields (standard-mode records) are
    silently skipped.
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [item["labels"] for item in batch],
                batch_first=True,
                padding_value=-100,
            ),
            "gen_start": [item["gen_start"] for item in batch],
            "cluster": [item["cluster"] for item in batch],
        }

        if "top10_ids" in batch[0]:
            max_gen_len = max(item["top10_ids"].size(0) for item in batch)
            n = 10  # always 10 candidates per position

            ids_padded = torch.zeros(len(batch), max_gen_len, n, dtype=torch.long)
            probs_padded = torch.zeros(len(batch), max_gen_len, n, dtype=torch.float32)

            for i, item in enumerate(batch):
                gl = item["top10_ids"].size(0)
                ids_padded[i, :gl] = item["top10_ids"]
                probs_padded[i, :gl] = item["top10_probs"]

            result["top10_ids"] = ids_padded     # [B, max_gen_len, 10]
            result["top10_probs"] = probs_padded  # [B, max_gen_len, 10]

        return result
