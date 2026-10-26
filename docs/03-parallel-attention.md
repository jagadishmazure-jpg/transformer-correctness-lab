# Module 3: Parallel attention, simulated on CPU

Large-context training splits work across devices. Every split must give **the same numbers** as plain attention.

## Blockwise / ring attention with online softmax
`tclab.parallel.blockwise_attention` processes the sequence in chunks. For each query chunk it visits key chunks one by one (in ring attention, each would arrive from a neighbouring device), keeping a running max `m`, running normaliser `l` and accumulator `acc`. When a new chunk raises the max, the old `l` and `acc` are rescaled by `exp(m_old - m_new)`. This is the same trick FlashAttention uses inside a GPU.

Injected bugs:
- `no_rescale`: forgets the rescale; results drift badly (max diff ~6-7 here).
- `local_causal`: builds the causal mask from chunk-local indices, so it is wrong across chunk boundaries (max diff ~15).

Chunk sizes 1, 4, 6 (uneven) and 20 (one chunk) all match full attention within 1e-4, causal and non-causal. A separate test confirms the causal blockwise version has no future leak.

## Tensor-parallel heads
Megatron-style: split heads across 2/4/8 simulated devices (column-parallel Q/K/V, row-parallel output projection), then sum the partial outputs (the all-reduce). Must equal the single-device result.

## Batch invariance
A sample's logits must not depend on what else is in the batch. Checked for batch sizes 1, 2 and 8 (exact match on CPU here). On GPUs, some kernels pick different reduction orders per batch shape, so expect tiny, tolerance-level differences there.

## Work vs span (why transformers train in parallel)
Per layer, attention does O(T²) work but every position is computed at once (span 1). An RNN does O(T) work but needs T sequential steps. Training uses the full causal mask to compute all T next-token predictions in one pass; that is exactly why a mask bug leaks during training but not during step-by-step generation.

## Real output: `python -m tclab parallel`
<!-- OUTPUT:parallel.txt -->
```text
[parallel] blockwise/ring attention (chunk=6) vs full attention
  OK  causal=False bug=None         max_diff: 1.907e-06
  OK  causal=False bug=no_rescale   max_diff: 7.478e+00
  OK  causal=True  bug=None         max_diff: 1.431e-06
  OK  causal=True  bug=no_rescale   max_diff: 6.141e+00
  OK  causal=True  bug=local_causal max_diff: 1.504e+01
  OK  tensor-parallel heads shards=2 max_diff: 5.960e-07
  OK  tensor-parallel heads shards=4 max_diff: 7.153e-07
  OK  batch invariance (1 vs 8 samples) max_diff: 0.000e+00
  info work/span for T=1024: {'attention_work': 1048576, 'attention_span': 1, 'rnn_work': 1024, 'rnn_span': 1024}

RESULT: parallel=pass
```
<!-- /OUTPUT -->

Code: `tclab/parallel.py`. Tests: `tests/test_m3_parallel.py`.
