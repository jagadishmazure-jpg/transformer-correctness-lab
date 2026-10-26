"""The leak checks. Model-agnostic: anything mapping (B,T) tokens -> (B,T,...) outputs works."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class LeakReport:
    check: str
    leaked: bool
    first_leak_position: int | None = None
    max_abs_diff: float = 0.0
    nan_detected: bool = False
    positions_checked: int = 0
    details: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "LEAK" if self.leaked else "PASS"
        s = f"[{status}] {self.check}: positions_checked={self.positions_checked} max_abs_diff={self.max_abs_diff:.3e}"
        if self.first_leak_position is not None:
            s += f" first_leak_position={self.first_leak_position}"
        if self.nan_detected:
            s += " nan_detected=True"
        return s


@torch.no_grad()
def perturbation_leak_check(
    fn: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    vocab_size: int,
    atol: float = 1e-5,
    trials: int = 2,
    seed: int = 0,
) -> LeakReport:
    """For each t: replace tokens at positions > t with random tokens and require
    outputs at positions <= t to be unchanged (within ``atol``)."""
    g = torch.Generator().manual_seed(seed)
    base = fn(tokens)
    T = tokens.shape[1]
    rep = LeakReport("perturbation", False, positions_checked=T - 1)
    if torch.isnan(base).any():
        rep.nan_detected = True
        rep.leaked = True
        rep.details.append("NaN in baseline outputs (fully-masked attention row?)")
    for t in range(T - 1):
        for _ in range(trials):
            pert = tokens.clone()
            pert[:, t + 1 :] = torch.randint(0, vocab_size, pert[:, t + 1 :].shape, generator=g)
            out = fn(pert)
            diff = (out[:, : t + 1] - base[:, : t + 1]).abs().nan_to_num(nan=float("inf")).max().item()
            rep.max_abs_diff = max(rep.max_abs_diff, diff)
            if diff > atol and rep.first_leak_position is None:
                rep.leaked = True
                rep.first_leak_position = t
                rep.details.append(f"output at position <= {t} changed by {diff:.3e} when tokens > {t} changed")
    return rep


def gradient_leak_check(model, tokens: torch.Tensor, atol: float = 1e-7) -> LeakReport:
    """d output_t / d input_embedding_{>t} must be exactly zero for a causal model."""
    T = tokens.shape[1]
    rep = LeakReport("gradient", False, positions_checked=T - 1)
    x = model.embed(tokens).detach().requires_grad_(True)
    logits, _ = model.forward_embeds(x)
    if torch.isnan(logits).any():
        rep.nan_detected = rep.leaked = True
        rep.details.append("NaN in outputs")
    for t in range(T - 1):
        (grad,) = torch.autograd.grad(logits[:, t].sum(), x, retain_graph=True)
        g = grad[:, t + 1 :].abs().nan_to_num(nan=float("inf")).max().item()
        rep.max_abs_diff = max(rep.max_abs_diff, g)
        if g > atol and rep.first_leak_position is None:
            rep.leaked = True
            rep.first_leak_position = t
            rep.details.append(f"grad of output[{t}] wrt inputs > {t} is {g:.3e}")
    return rep


@torch.no_grad()
def kv_cache_consistency(model, tokens: torch.Tensor, atol: float = 1e-5) -> LeakReport:
    """Incremental (KV-cache) decoding can never see the future; full forward must match it."""
    full = model(tokens)
    inc = model.incremental_logits(tokens)
    diff = (full - inc).abs().nan_to_num(nan=float("inf"))
    per_pos = diff.amax(dim=(0, 2))
    bad = (per_pos > atol).nonzero()
    rep = LeakReport("kv_cache_vs_full", bool(len(bad)), positions_checked=tokens.shape[1],
                     max_abs_diff=per_pos.max().item(), nan_detected=bool(torch.isnan(full).any()))
    if len(bad):
        rep.first_leak_position = int(bad[0])
    return rep
