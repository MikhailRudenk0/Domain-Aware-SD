"""Small torch-related helpers shared by draft / target inference paths."""

from __future__ import annotations

import torch


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    """Map a config dtype string to a torch.dtype, honoring device constraints.

    MPS (Apple Silicon) does not support bfloat16; we silently fall back to
    float16 with a one-line note so the eval pipeline can run on Mac for
    small local tests.
    """
    if device.startswith("mps") and dtype_str == "bfloat16":
        print("[eval] device=mps does not support bfloat16; using float16 instead")
        return torch.float16
    return getattr(torch, dtype_str)
