# Module 1: Causal mask leak

**Contract:** in a decoder, output at position *t* depends only on tokens 0..t.
**Symptom of a leak:** training loss suspiciously low, generation poor, because at train time the model reads the answer.

## Mask modes (`leaktest/model.py`)
| Mode | Implementation | Expected |
|---|---|---|
| `correct` | SDPA, bool `tril` (True = attend) | pass |
| `fused` | SDPA `is_causal=True` (fused-kernel path) | pass |
| `mha_correct` | `nn.MultiheadAttention`, bool `triu(1)` (True = block) | pass |
| `additive_fp32` | hand-written additive `-inf` mask, softmax in fp32 | pass |
| `buggy_transpose` | upper triangle instead of lower | leak |
| `buggy_inverted_bool` | MHA-style mask handed to SDPA | leak (sees only the future) |
| `buggy_mha_inverted` | SDPA-style mask handed to MHA | leak + NaN on fully-blocked row |
| `buggy_off_by_one` | `tril(diagonal=1)` | leak (sees next token) |
| `no_mask` | none | leak |

## Three independent checks (`leaktest/checks.py`)
1. **Perturbation:** for each *t*, randomise tokens > t (2 trials) and require outputs at <= t unchanged within `atol=1e-5`. Reports the first leaking position. NaNs count as leaks.
2. **Gradient:** `d logits[t] / d embeddings[>t]` must be exactly zero. Catches leaks too small for a perturbation tolerance.
3. **KV-cache vs full forward:** decoding one token at a time through a cache cannot see the future, so a correct full forward must match it (diff ~5e-7 from float rounding).

## Padding, packing, precision
- `masks.padded_causal(pad)`: causal AND key-not-pad, with fully-masked rows (left-pad queries) allowed to see themselves so softmax never produces NaN. Tests: changing pad ids leaves real-token logits unchanged for left and right padding, and a padded batch row equals the unpadded single sequence. Without the key mask, real tokens read pads (test `test_padding_without_key_mask_leaks`).
- `masks.packed_causal(doc_ids)`: block-diagonal causal mask for packed training. Changing document A must not change document B's outputs; with a plain causal mask it does (cross-document leak test).
- `precision.py`: in fp16 and bf16, a fully-masked row with `-inf` gives NaN; with a finite "large negative" it gives uniform weights over masked (future/pad) keys, a silent leak. An additive mask of -20 against a logit of 30 leaks nearly all weight to the future; adding `finfo(fp16).min` to a score overflows to `-inf`. **Recommendation:** use fused `is_causal` kernels, or `-inf` masks with softmax upcast to fp32, and explicitly handle fully-masked rows.

## Real output: `python -m tclab leak`
<!-- OUTPUT:leak.txt -->
```text
mask mode : correct
            SDPA with bool tril mask (True = attend) - correct
  [PASS] perturbation: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] gradient: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] kv_cache_vs_full: positions_checked=12 max_abs_diff=4.768e-07
  verdict  : causal (no leak)

mask mode : fused
            SDPA with is_causal=True (fused kernel path) - correct
  [PASS] perturbation: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] gradient: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] kv_cache_vs_full: positions_checked=12 max_abs_diff=4.768e-07
  verdict  : causal (no leak)

mask mode : mha_correct
            nn.MultiheadAttention with bool triu(1) mask (True = block) - correct
  [PASS] perturbation: positions_checked=11 max_abs_diff=0.000e+00
  verdict  : causal (no leak)

mask mode : additive_fp32
            additive -inf mask, softmax upcast to fp32 - correct
  [PASS] perturbation: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] gradient: positions_checked=11 max_abs_diff=0.000e+00
  [PASS] kv_cache_vs_full: positions_checked=12 max_abs_diff=3.576e-07
  verdict  : causal (no leak)

mask mode : buggy_transpose
            BUG: upper triangle instead of lower (mask transposed)
  [LEAK] perturbation: positions_checked=11 max_abs_diff=4.662e-01 first_leak_position=0
      - output at position <= 0 changed by 1.029e-01 when tokens > 0 changed
  [LEAK] gradient: positions_checked=11 max_abs_diff=2.779e-01 first_leak_position=0
      - grad of output[0] wrt inputs > 0 is 8.438e-02
  [LEAK] kv_cache_vs_full: positions_checked=12 max_abs_diff=5.781e-01 first_leak_position=0
  verdict  : FUTURE-TOKEN LEAK DETECTED

mask mode : buggy_inverted_bool
            BUG: MHA-style mask (True = block) passed to SDPA (True = attend)
  [LEAK] perturbation: positions_checked=11 max_abs_diff=9.708e-01 first_leak_position=0
      - output at position <= 0 changed by 1.137e-01 when tokens > 0 changed
  [LEAK] gradient: positions_checked=11 max_abs_diff=4.236e-01 first_leak_position=0
      - grad of output[0] wrt inputs > 0 is 9.015e-02
  [LEAK] kv_cache_vs_full: positions_checked=12 max_abs_diff=5.351e-01 first_leak_position=0
  verdict  : FUTURE-TOKEN LEAK DETECTED

mask mode : buggy_mha_inverted
            BUG: SDPA-style mask (True = attend) passed to nn.MultiheadAttention
  [LEAK] perturbation: positions_checked=11 max_abs_diff=inf first_leak_position=0 nan_detected=True
      - NaN in baseline outputs (fully-masked attention row?)
      - output at position <= 0 changed by inf when tokens > 0 changed
  verdict  : FUTURE-TOKEN LEAK DETECTED

mask mode : buggy_off_by_one
            BUG: tril(diagonal=1) lets each token see the next one
  [LEAK] perturbation: positions_checked=11 max_abs_diff=4.649e-01 first_leak_position=0
      - output at position <= 0 changed by 4.649e-01 when tokens > 0 changed
  [LEAK] gradient: positions_checked=11 max_abs_diff=2.799e-01 first_leak_position=0
      - grad of output[0] wrt inputs > 0 is 2.799e-01
  [LEAK] kv_cache_vs_full: positions_checked=12 max_abs_diff=3.379e-01 first_leak_position=0
  verdict  : FUTURE-TOKEN LEAK DETECTED

mask mode : no_mask
            BUG: no mask at all (bidirectional attention)
  [LEAK] perturbation: positions_checked=11 max_abs_diff=2.727e-01 first_leak_position=0
      - output at position <= 0 changed by 9.672e-02 when tokens > 0 changed
  [LEAK] gradient: positions_checked=11 max_abs_diff=8.730e-02 first_leak_position=0
      - grad of output[0] wrt inputs > 0 is 8.730e-02
  [LEAK] kv_cache_vs_full: positions_checked=12 max_abs_diff=5.762e-01 first_leak_position=0
  verdict  : FUTURE-TOKEN LEAK DETECTED

summary: 9/9 modes classified as expected

RESULT: leak=pass
```
<!-- /OUTPUT -->

## Real output: inverted boolean convention
<!-- OUTPUT:leak_inverted.txt -->
```text
mask mode : buggy_inverted_bool
            BUG: MHA-style mask (True = block) passed to SDPA (True = attend)
  [LEAK] perturbation: positions_checked=11 max_abs_diff=9.708e-01 first_leak_position=0
      - output at position <= 0 changed by 1.137e-01 when tokens > 0 changed
  [LEAK] gradient: positions_checked=11 max_abs_diff=4.236e-01 first_leak_position=0
      - grad of output[0] wrt inputs > 0 is 9.015e-02
  [LEAK] kv_cache_vs_full: positions_checked=12 max_abs_diff=5.351e-01 first_leak_position=0
  verdict  : FUTURE-TOKEN LEAK DETECTED
```
<!-- /OUTPUT -->

## Real output: precision demos
<!-- OUTPUT:precision.txt -->
```text
fully_masked_row fp16: {'neg_inf_row0': [nan, nan, nan, nan], 'neg_inf_has_nan': True, 'finite_row0': [0.25, 0.25, 0.25, 0.25], 'finite_leaks_to_future': True}
fully_masked_row bf16: {'neg_inf_row0': [nan, nan, nan, nan], 'neg_inf_has_nan': True, 'finite_row0': [0.25, 0.25, 0.25, 0.25], 'finite_leaks_to_future': True}
finite_mask_too_small fp16: {'mask_-20_future_weight': 1.0, 'neg_inf_fp32_future_weight': 0.0}
fp16_min_overflow: {'fp16_min_plus_neg100': -inf, 'is_inf': True}
```
<!-- /OUTPUT -->

Tests: `tests/test_m1_causal_leak.py`, `tests/test_m1_precision.py`.
