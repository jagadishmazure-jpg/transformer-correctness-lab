# ADR 0003: Keep `leaktest` as its own package

**Status:** accepted
**Context:** the causal leak test started as a standalone project with the CLI `python -m leaktest`. It later grew into a five-module lab.
**Decision:** keep `leaktest` (module 1) importable and runnable on its own so it is easy to vendor into another repo. Modules 2-5 live in `tclab`, and `python -m tclab all` runs everything.
**Consequences:** two top-level packages, but adopting just the leak test means copying one small folder.
