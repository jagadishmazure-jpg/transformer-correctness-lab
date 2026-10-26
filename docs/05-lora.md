# Module 5: LoRA

`tclab.lora.LoRALinear` wraps a frozen `nn.Linear`: `y = base(x) + (alpha / r) * x Aᵀ Bᵀ`, with A small random and **B = 0** at init.

| Property | Test | Injected bug |
|---|---|---|
| Zero-init B gives exactly the base output | `torch.equal` | `random_b_init` |
| Base weights unchanged after training steps; adapter weights did change | 3 SGD steps | |
| Merge (W += ΔW) and unmerge give the same outputs; unmerge restores W | relative diff ~1e-7 | |
| Scaling is alpha / r | explicit formula | `no_scaling` (uses alpha) |
| Multiple adapters: switching restores each adapter's output; no adapter = base | | |
| Only adapter parameters are trainable after `apply_lora` | | |
| LoRA on attention Q/K/V does not break the causal mask (B set to large random values) | module 1 perturbation check | |
| A buggy model stays detected after LoRA is added | | |

## Real output: `python -m tclab lora`
<!-- OUTPUT:lora.txt -->
```text
[lora]
  OK  zero-init B => identical to base, diff: 0.0
  OK  frozen base weight unchanged after 3 steps: True
  OK  merge / unmerge equivalence (relative diff): 1.415e-07 / 7.076e-08
  OK  LoRA on 2 qkv layers keeps causal mask: [PASS] perturbation: positions_checked=9 max_abs_diff=0.000e+00

RESULT: lora=pass
```
<!-- /OUTPUT -->

Code: `tclab/lora.py`. Tests: `tests/test_m5_lora.py`.
