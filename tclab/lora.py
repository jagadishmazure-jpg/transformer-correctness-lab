"""Module 5: LoRA linear layer and its correctness properties."""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = x W^T + b + (alpha / r) * x A^T B^T, with A ~ small random, B = 0 at init.

    Supports several named adapters, switching, merge and unmerge."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, bug: str | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.bug = r, alpha, bug
        self.adapters = nn.ModuleDict()
        self.active: str | None = None
        self.merged = False
        self.add_adapter("default")

    @property
    def scaling(self) -> float:
        return self.alpha if self.bug == "no_scaling" else self.alpha / self.r

    def add_adapter(self, name: str):
        A = nn.Parameter(torch.randn(self.r, self.base.in_features) * 0.01)
        B = nn.Parameter(torch.randn(self.base.out_features, self.r) * 0.01 if self.bug == "random_b_init"
                         else torch.zeros(self.base.out_features, self.r))
        self.adapters[name] = nn.ParameterDict({"A": A, "B": B})
        self.active = name

    def delta(self, name: str | None = None) -> torch.Tensor:
        a = self.adapters[name or self.active]
        return self.scaling * (a["B"] @ a["A"])

    def forward(self, x):
        y = self.base(x)
        if self.merged or self.active is None:
            return y
        a = self.adapters[self.active]
        return y + self.scaling * (x @ a["A"].T @ a["B"].T)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta()
            self.merged = False


def apply_lora(model: nn.Module, target: str = "qkv", r: int = 4, alpha: float = 8.0) -> list[LoRALinear]:
    """Replace every nn.Linear whose attribute name == ``target`` with a LoRALinear; freeze the rest."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if name == target and isinstance(child, nn.Linear):
                lw = LoRALinear(child, r, alpha)
                setattr(mod, name, lw)
                wrapped.append(lw)
    return wrapped
