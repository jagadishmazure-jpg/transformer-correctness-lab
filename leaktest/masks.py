"""Mask builders. Two boolean conventions exist in PyTorch:

* ``F.scaled_dot_product_attention`` bool ``attn_mask``: True = ATTEND.
* ``nn.MultiheadAttention`` bool ``attn_mask``: True = BLOCK.

Mixing them up silently inverts the mask.
"""

from __future__ import annotations

import torch


def sdpa_causal(T: int, device=None) -> torch.Tensor:
    """Correct SDPA convention: True where key j <= query i (attend)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def mha_causal(T: int, device=None) -> torch.Tensor:
    """Correct nn.MultiheadAttention convention: True where j > i (block)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)


def additive_from_attend(attend: torch.Tensor, neg: float = float("-inf"), dtype=torch.float32) -> torch.Tensor:
    """Turn an attend-mask into an additive float mask (0 = attend, ``neg`` = block)."""
    out = torch.zeros(attend.shape, dtype=dtype, device=attend.device)
    return out.masked_fill(~attend, neg)


def packed_causal(doc_ids: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal attend-mask for packed sequences. doc_ids: (B, T) -> (B, 1, T, T)."""
    T = doc_ids.shape[-1]
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    causal = sdpa_causal(T, doc_ids.device)
    return (same_doc & causal)[:, None]


def padded_causal(key_padding: torch.Tensor) -> torch.Tensor:
    """Causal attend-mask that also hides pad keys. key_padding: (B, T) True = pad.

    A query row that would be fully masked (a left-pad position) is allowed to
    attend to itself so softmax never sees an all-masked row (no NaNs). Outputs
    at pad positions are meaningless and must be ignored by the caller.
    """
    _B, T = key_padding.shape
    attend = sdpa_causal(T, key_padding.device)[None] & ~key_padding[:, None, :]
    eye = torch.eye(T, dtype=torch.bool, device=key_padding.device)[None]
    empty = ~attend.any(-1, keepdim=True)
    attend = attend | (empty & eye)
    return attend[:, None]
