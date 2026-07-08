from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutputs:
    loss: torch.Tensor
    ce_loss: Optional[torch.Tensor] = None
    kd_loss: Optional[torch.Tensor] = None


class DistillationLoss(nn.Module):
    """
    Drafter training loss with three modes:
      ce            — standard next-token cross-entropy against trunk labels.
      distillation  — soft cross-entropy against the target's top-K distribution
                      (-sum_i p_i * log q_i, computed only at slots the target
                      captured, and only for token IDs the drafter can predict).
      mixed         — ce_weight * ce + (1 - ce_weight) * kd.

    Positions with an all-zero top10_probs row are masked out (they mean either
    padding, "normal mode", or a skipped high-confidence step in generation).
    Individual top-K slots whose ID exceeds the drafter vocab (target has 64
    extra chat special tokens) are also masked out.
    """

    def __init__(
        self,
        mode: str = "distillation",
        ce_weight: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"ce", "distillation", "mixed"}:
            raise ValueError(f"Unknown loss mode: {mode!r}")
        self.mode = mode
        self.ce_weight = float(ce_weight)
        self.temperature = float(temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        top10_ids: Optional[torch.Tensor] = None,
        top10_probs: Optional[torch.Tensor] = None,
        gen_starts: Optional[List[int]] = None,
    ) -> LossOutputs:
        has_kd_inputs = (
            top10_ids is not None
            and top10_probs is not None
            and gen_starts is not None
        )

        ce = None
        kd = None

        if self.mode in {"ce", "mixed"} or not has_kd_inputs:
            ce = self._ce_loss(logits, labels)

        if self.mode in {"distillation", "mixed"} and has_kd_inputs:
            kd = self._kd_loss(logits, top10_ids, top10_probs, gen_starts)

        if self.mode == "ce":
            loss = ce
        elif self.mode == "distillation":
            loss = kd if kd is not None else ce
        else:
            if kd is None:
                loss = ce
            else:
                loss = self.ce_weight * ce + (1.0 - self.ce_weight) * kd

        return LossOutputs(loss=loss, ce_loss=ce, kd_loss=kd)

    def _ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        vocab = shifted_logits.size(-1)
        return F.cross_entropy(
            shifted_logits.view(-1, vocab),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

    def _kd_loss(
        self,
        logits: torch.Tensor,
        top10_ids: torch.Tensor,
        top10_probs: torch.Tensor,
        gen_starts: List[int],
    ) -> torch.Tensor:
        B, L, V = logits.shape
        G = top10_ids.size(1)
        device = logits.device

        # logits[gen_start - 1 + j] predicts trunk[j] (HF applies internal shift).
        gs = torch.tensor(gen_starts, device=device, dtype=torch.long).unsqueeze(1)  # [B, 1]
        pos = torch.arange(G, device=device, dtype=torch.long).unsqueeze(0)          # [1, G]
        idx = (gs - 1 + pos).clamp_(0, L - 1)                                        # [B, G]

        gen_logits = logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, V))           # [B, G, V]
        log_probs = F.log_softmax(gen_logits / self.temperature, dim=-1)             # [B, G, V]

        # Slots whose target-vocab ID is beyond the drafter vocab are masked out.
        valid_id_mask = top10_ids < V                                                # [B, G, K]
        safe_ids = top10_ids.clamp(max=V - 1)
        gathered_log_probs = log_probs.gather(-1, safe_ids)                          # [B, G, K]

        contributions = top10_probs * gathered_log_probs * valid_id_mask.to(top10_probs.dtype)
        per_pos = -contributions.sum(-1)                                             # [B, G]

        pos_mask = (top10_probs.sum(-1) > 0).to(per_pos.dtype)                       # [B, G]
        denom = pos_mask.sum().clamp(min=1)
        kd = (per_pos * pos_mask).sum() / denom

        if self.temperature != 1.0:
            kd = kd * (self.temperature ** 2)

        return kd
