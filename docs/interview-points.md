# Interview talking points

## Module 1: causal mask leak
- "Loss is suspiciously low but generation is bad": first suspect a future-token leak. Prove it by perturbing tokens after *t* and checking outputs up to *t* do not move.
- Two PyTorch boolean conventions: `nn.MultiheadAttention` True = block, SDPA True = attend. Mixing them inverts the mask.
- Three independent proofs: perturbation, zero gradient, KV-cache equals full forward.
- fp16: `-inf` on a fully-masked row gives NaN; a finite large-negative value silently leaks. Use fused causal kernels or fp32 softmax.
- Packing needs a block-diagonal mask, or documents leak into each other.

## Module 2: encoders
- Encoders are bidirectional; the contract is about padding, not the future.
- Mask pad *keys* (columns), not pad *queries* (rows).
- Mean pooling must divide by the real-token count.
- In seq2seq: causal decoder self-attention, padding-aware cross-attention.

## Module 3: parallel
- Online softmax: keep a running max and normaliser, and rescale when the max changes. That is the core of FlashAttention and ring attention.
- Causal masks in chunked attention must use global positions.
- Tensor parallelism: column-parallel QKV, row-parallel output, all-reduce sum.
- Training is parallel over positions (span 1, O(T²) work); RNNs need T sequential steps.

## Module 4: positions
- Learned absolute embeddings fall off a cliff past the trained length.
- RoPE makes q·k depend only on i - j. A KV-cache bug is using the wrong absolute position for new tokens.
- Interpolation and NTK scaling keep RoPE angles inside the trained range for longer contexts.
- ALiBi: a linear distance penalty per head and no embeddings.
- Left padding needs position ids computed from the attention mask.

## Module 5: LoRA
- B = 0 at init, so training starts exactly from the base model.
- Scale is alpha / r. Merging adds the delta to W for zero-latency serving, and unmerging restores it.
- Only the adapters train. Re-run the causal contract tests after adding adapters.

## General
- An invariance test that never fails is worthless, so each check here is shown to fail on an injected bug.
- Tiny models are enough for wiring bugs, which do not depend on scale.
