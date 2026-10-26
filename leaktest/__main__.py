"""CLI: python -m leaktest --mask buggy_transpose"""

from __future__ import annotations

import argparse
import sys

import torch

from leaktest.checks import gradient_leak_check, kv_cache_consistency, perturbation_leak_check
from leaktest.model import CORRECT_MODES, MASK_MODES, build


def run(mode: str, seq_len: int, seed: int) -> bool:
    model = build(mode, seed=seed)
    torch.manual_seed(seed)
    tokens = torch.randint(0, model.cfg.vocab_size, (2, seq_len))
    print(f"mask mode : {mode}\n            {MASK_MODES[mode]}")
    reports = [perturbation_leak_check(model, tokens, model.cfg.vocab_size)]
    if mode not in ("mha_correct", "buggy_mha_inverted"):
        reports.append(gradient_leak_check(model, tokens))
        # the KV-cache path uses the SDPA projections, so it only applies to SDPA modes
        reports.append(kv_cache_consistency(model, tokens))
    for r in reports:
        print("  " + r.summary())
        for d in r.details[:2]:
            print("      - " + d)
    leaked = any(r.leaked for r in reports)
    expected = mode not in CORRECT_MODES
    print(f"  verdict  : {'FUTURE-TOKEN LEAK DETECTED' if leaked else 'causal (no leak)'}\n")
    return leaked == expected


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="leaktest", description="Future-token (causal mask) leak test")
    p.add_argument("--mask", default="all", choices=["all", *MASK_MODES])
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    modes = list(MASK_MODES) if a.mask == "all" else [a.mask]
    results = {m: run(m, a.seq_len, a.seed) for m in modes}
    if a.mask == "all":
        ok = all(results.values())
        print(f"summary: {sum(results.values())}/{len(modes)} modes classified as expected")
        return 0 if ok else 1
    # single mode: exit 1 when a leak is found (usable as a CI gate)
    return 0 if modes[0] in CORRECT_MODES else 1


if __name__ == "__main__":
    sys.exit(main())
