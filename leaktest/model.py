"""A tiny GPT-style decoder (embeddings -> N x [causal self-attention + MLP] -> LM head).

Same architecture family as large decoder-only LLMs, just small enough to run
on a laptop CPU in milliseconds. The mask bugs it demonstrates are
scale-independent: the same wrong line breaks a 2-layer and a 70-layer model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from leaktest import masks as M

# mode -> human description
MASK_MODES: dict[str, str] = {
    "correct": "SDPA with bool tril mask (True = attend) - correct",
    "fused": "SDPA with is_causal=True (fused kernel path) - correct",
    "mha_correct": "nn.MultiheadAttention with bool triu(1) mask (True = block) - correct",
    "additive_fp32": "additive -inf mask, softmax upcast to fp32 - correct",
    "buggy_transpose": "BUG: upper triangle instead of lower (mask transposed)",
    "buggy_inverted_bool": "BUG: MHA-style mask (True = block) passed to SDPA (True = attend)",
    "buggy_mha_inverted": "BUG: SDPA-style mask (True = attend) passed to nn.MultiheadAttention",
    "buggy_off_by_one": "BUG: tril(diagonal=1) lets each token see the next one",
    "no_mask": "BUG: no mask at all (bidirectional attention)",
}
CORRECT_MODES = ("correct", "fused", "mha_correct", "additive_fp32")
BUGGY_MODES = tuple(m for m in MASK_MODES if m not in CORRECT_MODES)


@dataclass
class TinyGPTConfig:
    vocab_size: int = 64
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 64
    mask_mode: str = "correct"


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        if cfg.mask_mode not in MASK_MODES:
            raise ValueError(f"unknown mask_mode {cfg.mask_mode!r}; choose from {list(MASK_MODES)}")
        self.cfg = cfg
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        # used only by the nn.MultiheadAttention modes
        self.mha = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)

    def _heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, -1).transpose(1, 2)

    def forward(self, x, attend_mask=None, past_kv=None):
        """attend_mask: optional (B,1,T,T) bool, True = attend (overrides mode; SDPA path).
        past_kv: optional (k, v) cache for incremental decoding. Returns (y, (k, v))."""
        B, T, C = x.shape
        mode = self.cfg.mask_mode
        if mode in ("mha_correct", "buggy_mha_inverted") and attend_mask is None and past_kv is None:
            m = M.mha_causal(T, x.device) if mode == "mha_correct" else M.sdpa_causal(T, x.device)
            y, _ = self.mha(x, x, x, attn_mask=m, need_weights=False)
            return y, None

        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        S = k.shape[2]  # total key length

        if attend_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attend_mask)
        elif past_kv is not None:
            # query i (absolute S-T+i) may attend keys 0..S-T+i
            attend = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attend)
        elif mode == "fused":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mode == "correct":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=M.sdpa_causal(T, x.device))
        elif mode == "additive_fp32":
            att = (q.float() @ k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            att = att + M.additive_from_attend(M.sdpa_causal(T, x.device))
            y = (att.softmax(-1) @ v.float()).to(q.dtype)
        elif mode == "buggy_transpose":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=M.sdpa_causal(T, x.device).T)
        elif mode == "buggy_inverted_bool":
            # last row of triu(1) is all False -> fully masked -> NaN; others see only the future
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=M.mha_causal(T, x.device))
        elif mode == "buggy_off_by_one":
            m = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        elif mode == "no_mask":
            y = F.scaled_dot_product_attention(q, k, v)
        else:  # pragma: no cover
            raise AssertionError(mode)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x, attend_mask=None, past_kv=None):
        a, kv = self.attn(self.ln1(x), attend_mask, past_kv)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or TinyGPTConfig(**kw)
        c = self.cfg
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def embed(self, idx, start_pos: int = 0, position_ids=None):
        T = idx.shape[1]
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + T, device=idx.device)
        return self.tok(idx) + self.pos(position_ids)

    def forward_embeds(self, x, attend_mask=None, past=None):
        new_past = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, attend_mask, None if past is None else past[i])
            new_past.append(kv)
        return self.head(self.ln_f(x)), new_past

    def forward(self, idx, attend_mask=None, position_ids=None):
        logits, _ = self.forward_embeds(self.embed(idx, position_ids=position_ids), attend_mask)
        return logits

    @torch.no_grad()
    def incremental_logits(self, idx):
        """Feed tokens one at a time through a KV cache; returns (B, T, V) logits."""
        past, outs = None, []
        for t in range(idx.shape[1]):
            logits, past = self.forward_embeds(self.embed(idx[:, t : t + 1], start_pos=t), past=past)
            outs.append(logits)
        return torch.cat(outs, dim=1)


def build(mask_mode: str = "correct", seed: int = 0, **kw) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(TinyGPTConfig(mask_mode=mask_mode, **kw)).eval()
