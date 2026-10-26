"""Module 4: position encodings and their failure modes."""

from __future__ import annotations

import torch


def sinusoidal(T: int, d: int, offset: int = 0) -> torch.Tensor:
    pos = torch.arange(offset, offset + T, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def rope_angles(positions: torch.Tensor, d: int, base: float = 10000.0, ntk_scale: float = 1.0, interp: float = 1.0):
    """Angles for RoPE. ``interp`` < 1 = linear position interpolation; ``ntk_scale`` > 1 = NTK-aware base scaling."""
    base = base * ntk_scale ** (d / (d - 2)) if ntk_scale != 1.0 else base
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    return (positions.double()[:, None] * interp) * inv[None]


def apply_rope(x: torch.Tensor, positions: torch.Tensor, **kw) -> torch.Tensor:
    """x: (..., T, d). Rotates consecutive pairs by position-dependent angles."""
    ang = rope_angles(positions, x.shape[-1], **kw).to(x.dtype)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    return out


def rope_score(q, k, i: int, j: int) -> float:
    qi = apply_rope(q[None], torch.tensor([i]))[0]
    kj = apply_rope(k[None], torch.tensor([j]))[0]
    return float(qi @ kj)


def alibi_bias(T: int, n_heads: int) -> torch.Tensor:
    """(H, T, T) linear distance penalty, slopes 2^(-8h/H)."""
    slopes = torch.tensor([2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)])
    dist = torch.arange(T)[None, :] - torch.arange(T)[:, None]  # j - i (<=0 in causal region)
    return slopes[:, None, None] * dist.clamp(max=0)[None].float()


def position_ids_from_padding(pad: torch.Tensor, bug: bool = False) -> torch.Tensor:
    """Left padding: real tokens must start at position 0. bug=True uses arange (pads consume ids)."""
    if bug:
        return torch.arange(pad.shape[1]).expand_as(pad)
    keep = (~pad).long()
    return (keep.cumsum(1) - 1).clamp_min(0) * keep


def learned_position_cliff(train_len: int = 16, test_len: int = 32, d: int = 16, steps: int = 300, seed: int = 0):
    """Train a learned absolute embedding table on a toy 'predict your position parity' task
    for positions < train_len only; positions >= train_len stay at their random init.
    Returns accuracy inside vs beyond the trained length (the position cliff)."""
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(test_len, d)
    head = torch.nn.Linear(d, 2)
    opt = torch.optim.Adam([*emb.parameters(), *head.parameters()], lr=0.05)
    pos = torch.arange(train_len)
    y = pos % 2
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(head(emb(pos)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        allp = torch.arange(test_len)
        pred = head(emb(allp)).argmax(-1)
        ok = (pred == allp % 2).float()
    return {"acc_in_range": ok[:train_len].mean().item(), "acc_beyond": ok[train_len:].mean().item()}


def sinusoid_extrapolates(T: int = 64, d: int = 16) -> float:
    """Sinusoidal encodings are defined for any position; returns max |pe| (always <= 1)."""
    return float(sinusoidal(T, d).abs().max())


def max_rope_angle(T: int, d: int, interp: float = 1.0, ntk_scale: float = 1.0) -> float:
    """Largest angle used by the lowest-frequency pair (what interpolation/NTK keep in range)."""
    return float(rope_angles(torch.tensor([T - 1]), d, interp=interp, ntk_scale=ntk_scale)[0, -1])

