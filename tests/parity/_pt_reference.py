"""Self-contained PyTorch reference for ERNIE-Image attention + RoPE.

Vendored from `diffusers/models/transformers/transformer_ernie_image.py`
(huggingface/diffusers @ main, unreleased as of diffusers 0.37.1).

Kept minimal: only the classes used by the MLX parity tests, with internals
inlined so the file has no dependencies beyond `torch` (no `diffusers.*`).

Upstream is Apache 2.0 — see NOTICE at bottom.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE — verbatim from the reference
# ---------------------------------------------------------------------------

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            dim=-1,
        )
        emb = emb.unsqueeze(2)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Megatron non-interleaved rotate_half RoPE."""
    rot_dim = freqs_cis.shape[-1]
    x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
    cos_ = torch.cos(freqs_cis).to(x.dtype)
    sin_ = torch.sin(freqs_cis).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)


# ---------------------------------------------------------------------------
# Attention — stripped of diffusers wrapper; math is identical
# ---------------------------------------------------------------------------

class ErnieImageAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        bias: bool = False,
        out_bias: bool = False,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, query_dim, bias=out_bias)]
        )

        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.unflatten(-1, (self.heads, -1))
        k = k.unflatten(-1, (self.heads, -1))
        v = v.unflatten(-1, (self.heads, -1))

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        dtype = q.dtype
        q, k = q.to(dtype), k.to(dtype)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        out = out.permute(0, 2, 1, 3).flatten(2, 3).to(dtype)
        return self.to_out[0](out)


# ---------------------------------------------------------------------------
# Feed-forward (GeGLU), shared-AdaLN block, final AdaLN-continuous norm
# ---------------------------------------------------------------------------


class ErnieImageFeedForward(nn.Module):
    """GeGLU: linear_fc2( up_proj(x) * gelu(gate_proj(x)) )."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    """One of the 36 transformer blocks. AdaLN modulation is SHARED across all blocks
    (the caller passes the same temb 6-tuple into every block)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            bias=False,
            out_bias=False,
            qk_norm="rms_norm" if qk_layernorm else "",
            eps=eps,
            elementwise_affine=True,
        )
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def forward(
        self,
        x: torch.Tensor,  # [S, B, H]
        rotary_pos_emb: torch.Tensor,
        temb,  # 6-tuple of [S, B, H]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb
        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
        x_bsh = x.permute(1, 0, 2)  # [S, B, H] -> [B, S, H]
        attn_out = self.self_attention(
            hidden_states=x_bsh,
            image_rotary_emb=rotary_pos_emb,
            attention_mask=attention_mask,
        )
        attn_out = attn_out.permute(1, 0, 2)  # [B, S, H] -> [S, B, H]
        x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)
        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
        return residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)


class ErnieImageAdaLNContinuous(nn.Module):
    """Final LayerNorm + conditioning-driven affine. Used once after the stack."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)  # bias=True default

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        # reference: x is [S, B, H], conditioning is [B, H]
        # scale, shift are each [B, H]; unsqueeze(0) broadcasts over S
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


# ---------------------------------------------------------------------------
# Top-level transformer — stripped of diffusers Model/ConfigMixin
# ---------------------------------------------------------------------------


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()


class ErnieImageTransformer2DModel(nn.Module):
    """Inference-only stripped clone of the reference class.

    Uses `diffusers.models.embeddings.{Timesteps, TimestepEmbedding}` which are
    stable public classes across recent diffusers versions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
        ffn_hidden_size: int,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim=(32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        # Import lazily — diffusers may not be installed in some environments.
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = (
            nn.Linear(text_in_dim, hidden_size, bias=False)
            if text_in_dim != hidden_size
            else None
        )
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = ErnieImageEmbedND3(
            dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(
                    hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm=qk_layernorm
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()
        x = torch.cat([img_sbh, text_sbh], dim=0)
        S = x.shape[0]

        text_ids = (
            torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32)
                    .view(1, Tmax, 1)
                    .expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
            if Tmax > 0
            else torch.zeros((B, 0, 3), device=device)
        )
        grid_yx = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(Hp, device=device, dtype=torch.float32),
                    torch.arange(Wp, device=device, dtype=torch.float32),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
        )
        image_ids = torch.cat(
            [
                text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),
                grid_yx.view(1, N_img, 2).expand(B, -1, -1),
            ],
            dim=-1,
        )
        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        valid_text = (
            torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
            if Tmax > 0
            else torch.zeros((B, 0), device=device, dtype=torch.bool)
        )
        attention_mask = torch.cat(
            [torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text], dim=1
        )[:, None, None, :]

        sample = self.time_proj(timestep).to(dtype=dtype)
        c = self.time_embedding(sample)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(0).expand(S, -1, -1).contiguous()
            for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]
        for layer in self.layers:
            temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            x = layer(x, rotary_pos_emb, temb, attention_mask)

        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
        return (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )


# ---------------------------------------------------------------------------
# NOTICE
#
# This file contains code derived from huggingface/diffusers
# (src/diffusers/models/transformers/transformer_ernie_image.py), copyright
# 2025 Baidu ERNIE-Image Team and The HuggingFace Team, licensed under the
# Apache License, Version 2.0. See http://www.apache.org/licenses/LICENSE-2.0
# ---------------------------------------------------------------------------
