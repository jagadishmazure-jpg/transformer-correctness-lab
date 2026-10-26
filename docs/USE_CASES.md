# Use cases

Each entry: what goes wrong, how the tests catch it, and where to run it.

### 1. Reviewing custom attention code in pretraining
**What goes wrong:** a hand-written mask uses `triu` instead of `tril`, `diagonal=1`, or forgets the mask in one code path. Loss drops suspiciously fast; generation is poor.
**How it is caught:** perturbation check finds the first leaking position; gradient check confirms with exact zeros/non-zeros.
**Run:** `python -m leaktest --mask buggy_transpose`, `--mask buggy_off_by_one`, `--mask no_mask`; `tests/test_m1_causal_leak.py::test_buggy_masks_detected_by_perturbation`.

### 2. Switching attention backends and boolean convention flips
**What goes wrong:** moving between `nn.MultiheadAttention` (bool True = block), `F.scaled_dot_product_attention` (bool True = attend) and fused `is_causal=True` kernels. Reusing a mask across backends inverts it.
**How it is caught:** the same checks run against every backend: `correct`, `fused` and `mha_correct` pass; `buggy_inverted_bool` and `buggy_mha_inverted` fail (the latter also produces NaN on the fully-blocked row).
**Run:** `python -m leaktest --mask all`.

### 3. fp16 / bf16 mixed-precision mask bugs and NaNs
**What goes wrong:** fully-masked rows become NaN with `-inf`, or silently attend uniformly to masked keys with a finite large-negative value; a too-small additive mask leaks; `finfo(fp16).min` plus a score overflows.
**How it is caught:** `leaktest/precision.py` demos and `tests/test_m1_precision.py` (fp16 and bf16); the fused causal model is leak-free in bf16 with zero tolerance.
**Fix:** fused `is_causal` kernels, or `-inf` with fp32 softmax and explicit handling of empty rows.

### 4. Padding + causal mask (left/right padding, variable-length batches)
**What goes wrong:** the key-padding mask is forgotten, so real tokens read pads; left-padded rows become fully masked (NaN); position ids count pads.
**How it is caught:** changing pad ids must not change real-token logits; a padded row must equal the unpadded single sequence; no NaNs.
**Run:** `tests/test_m1_causal_leak.py::test_padding_plus_causal[left|right]`, `test_padding_without_key_mask_leaks`, `tests/test_m4_positions.py::test_left_padding_position_ids`.

### 5. KV-cache incremental decoding vs full forward
**What goes wrong:** training (full forward) and serving (KV cache) disagree because of a mask leak, a wrong cache offset or a wrong RoPE position.
**How it is caught:** `kv_cache_consistency` compares token-by-token decoding against the full forward (diff ~5e-7 when correct, 0.3-0.6 on every buggy mask); RoPE offset test in module 4.
**Run:** `tests/test_m1_causal_leak.py::test_kv_cache_matches_full_forward_only_when_causal`.

### 6. Sequence packing with document boundaries
**What goes wrong:** several documents are packed into one training row with a plain causal mask, so document B attends to document A (cross-document leakage).
**How it is caught:** change document A, require document B's outputs unchanged. Block-diagonal mask passes; plain causal fails.
**Run:** `test_packed_sequences_block_diagonal_no_cross_doc_leak`, `test_packed_sequences_plain_causal_leaks_across_documents`.

### 7. Fine-tuning / LoRA adapter changes
**What goes wrong:** adapters on Q/K/V, a merge step or adapter switching accidentally alters base behaviour or the mask path.
**How it is caught:** module 5 equivalence tests plus re-running the module 1 leak test on the adapted model.
**Run:** `python -m tclab lora`, `tests/test_m5_lora.py`.

### 8. Model export / quantization regressions (ONNX, int8)
**What goes wrong:** exporters sometimes bake in a fixed sequence length, drop or re-implement the mask, or quantization changes a mask constant so it no longer fully blocks.
**How it would be caught:** the checks take any callable `fn(tokens) -> outputs`. Wrap the exported runtime (e.g. an ONNX Runtime session) as `fn` and run `perturbation_leak_check(fn, tokens, vocab)`, plus a full-vs-exported diff with a looser tolerance for int8.
**Status:** pattern only, not implemented here (no extra runtime dependencies).

### 9. CI regression gate for model repos
**What goes wrong:** a refactor reintroduces a leak months later.
**How it is caught:** `python -m leaktest --mask <mode>` exits 1 on a leak; `python -m tclab all` exits 1 if any module fails; this repo's own CI runs everything on each push. See [implementation-guide.md](implementation-guide.md).

### 10. Interview and teaching demo
**Run:** `python -m tclab all` prints a correct-vs-bug report for every module in a few seconds on a laptop. Talking points: [interview-points.md](interview-points.md).
