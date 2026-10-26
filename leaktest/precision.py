"""Half-precision mask pitfalls, demonstrated on raw attention scores."""

from __future__ import annotations

import torch


def softmax_masked(scores, attend, mask_value, dtype, upcast=False):
    s = scores.to(dtype)
    s = s.masked_fill(~attend, mask_value)
    if upcast:
        s = s.float()
    return s.softmax(-1)


def fully_masked_row_demo(dtype=torch.float16):
    """A row with every key masked: -inf -> NaN, finite large negative -> uniform weights
    over the masked (future/pad) keys, i.e. a silent leak."""
    scores = torch.randn(1, 4, 4)
    attend = torch.ones(4, 4, dtype=torch.bool).tril()
    attend[0] = False  # e.g. a left-pad query row with every key blocked
    inf = softmax_masked(scores, attend, float("-inf"), dtype)
    finite = softmax_masked(scores, attend, torch.finfo(dtype).min, dtype)
    return {
        "neg_inf_row0": inf[0, 0].float().tolist(),
        "neg_inf_has_nan": bool(torch.isnan(inf).any()),
        "finite_row0": finite[0, 0].float().tolist(),
        "finite_leaks_to_future": bool(finite[0, 0, 1:].sum() > 0),
    }


def finite_mask_too_small_demo(dtype=torch.float16):
    """An additive 'large negative' constant that is not large enough relative to the logits leaks."""
    scores = torch.tensor([[0.0, 30.0]])  # future key has a big logit
    attend = torch.tensor([[True, False]])
    # additive masks: score + mask_value (common in hand-written attention)
    leaky = (scores.to(dtype) + torch.where(attend, 0.0, -20.0).to(dtype)).softmax(-1)
    safe = (scores.float() + torch.where(attend, 0.0, float("-inf"))).softmax(-1)
    return {"mask_-20_future_weight": float(leaky[0, 1]), "neg_inf_fp32_future_weight": float(safe[0, 1])}


def fp16_min_overflow_demo():
    """Adding finfo(fp16).min to a negative score overflows to -inf in fp16."""
    v = torch.tensor([-100.0], dtype=torch.float16) + torch.finfo(torch.float16).min
    return {"fp16_min_plus_neg100": float(v[0]), "is_inf": bool(torch.isinf(v).any())}
