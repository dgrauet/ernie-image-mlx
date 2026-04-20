"""ErnieAttention — single-stream self-attention with qk_layernorm + triple-axis RoPE.

MLX port of `ErnieImageSingleStreamAttnProcessor` + `ErnieImageAttention`
(diffusers 0.36 `transformer_ernie_image.py`).

Forward order (matches reference exactly):
    1. independent Q/K/V Linear projections (bias=False)
    2. unflatten last dim to (heads, head_dim)
    3. RMSNorm on Q and K (per-head-dim, affine) if qk_layernorm
    4. rotary position embedding (Megatron non-interleaved)
    5. SDPA (fast path)
    6. flatten heads, output Linear (bias=False)

Shapes: input `[B, N, hidden_size]` → output `[B, N, hidden_size]`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ernie_image_core_mlx.model.config import ErnieImageConfig
from ernie_image_core_mlx.model.rope import apply_rotary_emb


class ErnieAttention(nn.Module):
    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.inner_dim = cfg.hidden_size  # == num_heads * head_dim

        # Independent Q/K/V. Names match the reference for straightforward weight mapping
        # (PT `to_q.weight` → MLX `to_q.weight`, etc).
        self.to_q = nn.Linear(cfg.hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(cfg.hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(cfg.hidden_size, self.inner_dim, bias=False)
        self.to_out_0 = nn.Linear(self.inner_dim, cfg.hidden_size, bias=False)

        if cfg.qk_layernorm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=cfg.eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=cfg.eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.to_q(x).reshape(B, N, H, D)
        k = self.to_k(x).reshape(B, N, H, D)
        v = self.to_v(x).reshape(B, N, H, D)

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # SDPA wants (B, H, N, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / (D**0.5)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=attention_mask
        )
        # back to (B, N, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, H * D)
        return self.to_out_0(out)
