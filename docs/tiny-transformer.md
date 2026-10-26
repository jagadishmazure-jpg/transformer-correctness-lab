# What is a "tiny transformer"?

`leaktest/model.py` defines `TinyGPT`, the same decoder-only architecture used by GPT-style LLMs, just small:

| Part | What it does | Size here |
|---|---|---|
| Token embedding | token id -> vector | 64-token vocab, 32 dims |
| Position embedding | position -> vector (learned absolute) | 64 positions |
| Causal self-attention | each position mixes information from itself and earlier positions only; the **mask** enforces "earlier only" | 4 heads |
| MLP | per-position feed-forward (32 -> 128 -> 32, GELU) | |
| LayerNorm + residuals | pre-norm blocks, as in modern decoders | |
| Blocks | attention + MLP, stacked | 2 layers |
| LM head | vector -> next-token logits | 64 outputs |

It is randomly initialised and never trained; the tests do not need it to be good at language. They need it to obey a **contract**: the output at position *t* must depend only on tokens 0..t.

**Why tiny is enough.** A mask bug is a wrong line of indexing code (`tril` vs `triu`, `True` meaning attend vs block, a missing key-padding term). The same line runs in a 2-layer and an 80-layer model, so the bug is scale-independent. A tiny model makes the test run in milliseconds on a laptop CPU, so it can run on every commit for free. With random weights any leak shows up as a clear numeric change (around 0.1-1.0 here), not a subtle quality drift.

**What tiny does not prove:** kernel-specific numerics on GPU, behaviour at very long context, or anything about model quality. See the README limitations.
