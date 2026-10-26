"""Reference attention used across modules (fp32, explicit, easy to read)."""

from __future__ import annotations


def attention(q, k, v, attend=None, bias=None):
    """q,k,v: (..., T, d). attend: bool (True = attend), broadcastable to (..., Tq, Tk)."""
    s = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    if bias is not None:
        s = s + bias
    if attend is not None:
        s = s.masked_fill(~attend, float("-inf"))
    return s.softmax(-1) @ v
