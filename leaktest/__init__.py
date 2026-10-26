"""Future-token (causal mask) leak tests for decoder-only transformers."""

from leaktest.checks import LeakReport, gradient_leak_check, perturbation_leak_check
from leaktest.model import MASK_MODES, TinyGPT, TinyGPTConfig

__all__ = [
    "MASK_MODES",
    "LeakReport",
    "TinyGPT",
    "TinyGPTConfig",
    "gradient_leak_check",
    "perturbation_leak_check",
]
__version__ = "0.1.0"
