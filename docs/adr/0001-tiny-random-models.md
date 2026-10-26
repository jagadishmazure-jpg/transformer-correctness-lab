# ADR 0001: Tiny, randomly initialised models

**Status:** accepted
**Context:** we need tests that run on every commit, on CPU, for free.
**Decision:** use 1-2 layer, 32-dim models with fixed seeds and no training, except a small toy for the position cliff.
**Consequences:** each test takes milliseconds, and wiring bugs show up as large numeric changes. This setup says nothing about model quality or GPU kernel numerics, and the README states that.
