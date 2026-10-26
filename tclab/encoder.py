"""Module 2: encoder (BERT-style) and encoder-decoder masking correctness."""

from __future__ import annotations

import torch
from torch import nn

from tclab.attention import attention


class TinyEncoder(nn.Module):
    """Bidirectional self-attention encoder. ``bug``:
    * None             - correct key-padding mask
    * "no_pad_mask"    - pad keys are attended to
    * "query_mask"     - masks pad *queries* instead of pad keys (common confusion)
    """

    def __init__(self, vocab=50, d=32, layers=2, max_len=64, bug=None, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.bug = bug
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers))
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.eval()

    def forward(self, ids, pad):  # pad: (B,T) True = padding
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))
        if self.bug == "no_pad_mask":
            attend = None
        elif self.bug == "query_mask":
            # masks pad QUERY rows instead of pad KEY columns: real tokens still read pads
            eye = torch.eye(ids.shape[1], dtype=torch.bool)[None]
            attend = (~pad)[:, :, None] | eye
        else:
            attend = (~pad)[:, None, :]  # (B,1,T) key mask
        for qkv, mlp, ln in zip(self.qkv, self.mlp, self.ln):
            q, k, v = qkv(ln(x)).chunk(3, -1)
            x = x + attention(q, k, v, attend)
            x = x + mlp(x)
        return x


def pool(h, pad, how="mean", bug=False):
    """[CLS] or mean pooling. bug=True averages over pad positions too."""
    if how == "cls":
        return h[:, 0]
    if bug:
        return h.mean(1)
    keep = (~pad).unsqueeze(-1).float()
    return (h * keep).sum(1) / keep.sum(1).clamp_min(1)


class TinySeq2Seq(nn.Module):
    """One encoder layer + one decoder layer with causal self-attn and cross-attn. ``bug``:
    * None                 - correct
    * "no_causal_decoder"  - decoder self-attention is bidirectional (sees future targets)
    * "cross_ignores_pad"  - cross-attention attends to encoder pad positions
    """

    def __init__(self, vocab=50, d=32, max_len=64, bug=None, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.bug = bug
        self.enc = TinyEncoder(vocab, d, 1, max_len, seed=seed)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.self_qkv, self.cross_q, self.cross_kv = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.head = nn.Linear(d, vocab)
        self.eval()

    def forward(self, src, src_pad, tgt):
        mem = self.enc(src, src_pad)
        T = tgt.shape[1]
        y = self.tok(tgt) + self.pos(torch.arange(T))
        q, k, v = self.self_qkv(y).chunk(3, -1)
        causal = None if self.bug == "no_causal_decoder" else torch.ones(T, T, dtype=torch.bool).tril()
        y = y + attention(q, k, v, causal)
        ck, cv = self.cross_kv(mem).chunk(2, -1)
        cross = None if self.bug == "cross_ignores_pad" else (~src_pad)[:, None, :]
        y = y + attention(self.cross_q(y), ck, cv, cross)
        return self.head(y)
