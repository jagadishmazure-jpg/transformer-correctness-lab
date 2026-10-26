# ADR 0002: Black-box invariance tests plus injected bugs

**Status:** accepted
**Decision:** state each contract as "output X must not change when input Y changes," and check it with a tolerance. Add a second method where possible (gradients, KV cache, equivalent parallel form). Every check ships with at least one injected bug that it must catch.
**Consequences:** the checks work on any callable model, including exported runtimes. Perturbation is probabilistic, so the gradient check covers leaks that fall below tolerance.
