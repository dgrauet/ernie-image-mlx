"""ErnieImageTransformer2DModel — 8B single-stream DiT — MLX port.

Reference (vendored for tests/parity): `tests/parity/_pt_reference.py`.
Upstream source: `diffusers/models/transformers/transformer_ernie_image.py`.

MLX convention: batch-first `[B, S, H]` throughout. The reference uses
`[S, B, H]` internally but only as an artifact of diffusers' batch-agnostic
`Attention` wrapper. The math is identical — we just skip the permutes.

Shared-AdaLN pattern: the 6-tuple `temb = (shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp)` is produced ONCE from the timestep embedding
and fed to every block. Each block keeps its own RMSNorm / attention /
FFN weights, but the modulation is global — saves params, stabilizes training.

Patch embedding: production config is `patch_size=1`, which makes the reference
Conv2d(kernel=1) equivalent to a pointwise Linear. We use Linear here and squeeze
the PT conv weight `(O, I, 1, 1) → (O, I)` at load time. Support for
`patch_size > 1` would swap this for MLX Conv2d (channels-last).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_arsenal.diffusion import TimestepEmbedding, get_timestep_embedding

from ernie_image_core_mlx.model.attention import ErnieAttention
from ernie_image_core_mlx.model.config import ErnieImageConfig
from ernie_image_core_mlx.model.rope import ErnieImageEmbedND3

TembTuple = tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ErnieImageFeedForward(nn.Module):
    """GeGLU: `linear_fc2( up_proj(x) * gelu(gate_proj(x)) )`. Note: GELU on the
    *gate* branch, not on the up branch — matches the reference, not SwiGLU."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.up_proj(x) * nn.gelu(self.gate_proj(x)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    """One DiT block. Expects AdaLN modulation values supplied by the caller."""

    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(cfg.hidden_size, eps=cfg.eps)
        self.self_attention = ErnieAttention(cfg)
        self.adaLN_mlp_ln = nn.RMSNorm(cfg.hidden_size, eps=cfg.eps)
        self.mlp = ErnieImageFeedForward(cfg.hidden_size, cfg.ffn_hidden_size)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        temb: TembTuple,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = x
        h = self.adaLN_sa_ln(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attention(h, freqs_cis=freqs_cis, attention_mask=attention_mask)
        x = residual + gate_msa * h

        residual = x
        h = self.adaLN_mlp_ln(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return residual + gate_mlp * h


class ErnieImageAdaLNContinuous(nn.Module):
    """Final LayerNorm (no affine) followed by conditioning-driven shift/scale."""

    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps, affine=False)
        self.linear = nn.Linear(cfg.hidden_size, cfg.hidden_size * 2, bias=True)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        """Args:
            x: `[B, S, H]`
            conditioning: `[B, H]`
        """
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        x = self.norm(x)
        # broadcast [B, H] → [B, 1, H] over the S dim
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class ErnieImagePatchEmbedLinear(nn.Module):
    """Pointwise patch embed for `patch_size=1`. Equivalent to the reference
    `Conv2d(in, out, kernel=1, stride=1)` since kernel=stride=1 collapse to a
    per-pixel Linear over channels.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        if patch_size != 1:
            raise NotImplementedError(
                f"patch_size={patch_size} not supported yet — use Conv2d-based variant"
            )
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Args:
            x: `[B, C, H, W]` (channels-second — matches PT input convention)
        Returns:
            `[B, H*W, hidden_size]`
        """
        B, C, H, W = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(B, H * W, C)
        return self.proj(x)


class SharedAdaLNModulation(nn.Module):
    """SiLU → Linear(hidden_size → 6*hidden_size). Output is chunked into the
    6-tuple (shift/scale/gate × MSA/MLP) fed to every DiT block."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def __call__(self, c: mx.array) -> mx.array:
        return self.linear(nn.silu(c))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ErnieImageTransformer2DModel(nn.Module):
    """The 8B single-stream DiT.

    Forward inputs:
        hidden_states:     `[B, C_in=128, H, W]` latent (from VAE encoder).
        timestep:          `[B]` or scalar int/float — diffusion timestep.
        text_bth:          `[B, Tmax, text_in_dim=3072]` Mistral3 last hidden state.
        text_lens:         `[B]` int — valid text tokens per batch (padding tolerant).

    Output: `[B, C_out=128, H, W]` predicted velocity (flow-match target).
    """

    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.cfg = cfg

        self.x_embedder = ErnieImagePatchEmbedLinear(
            cfg.in_channels, cfg.hidden_size, cfg.patch_size
        )
        # text_proj only exists when text_in_dim != hidden_size (matches reference).
        self.text_proj = (
            nn.Linear(cfg.text_in_dim, cfg.hidden_size, bias=False)
            if cfg.text_in_dim != cfg.hidden_size
            else None
        )
        # Time stack: sinusoidal proj (param-free) + MLP (silu between 2 Linears).
        self.time_embedding = TimestepEmbedding(cfg.hidden_size, cfg.hidden_size)
        self.pos_embed = ErnieImageEmbedND3(
            head_dim=cfg.head_dim,
            theta=cfg.rope_theta,
            axes_dim=cfg.rope_axes_dim,
        )
        self.adaLN_modulation = SharedAdaLNModulation(cfg.hidden_size)

        self.layers = [ErnieImageSharedAdaLNBlock(cfg) for _ in range(cfg.num_layers)]
        self.final_norm = ErnieImageAdaLNContinuous(cfg)
        self.final_linear = nn.Linear(
            cfg.hidden_size,
            cfg.patch_size * cfg.patch_size * cfg.out_channels,
            bias=True,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
    ) -> mx.array:
        cfg = self.cfg
        p = cfg.patch_size
        B, C_in, H, W = hidden_states.shape
        Hp, Wp = H // p, W // p
        N_img = Hp * Wp

        # ---- embed image patches ----
        img_bnh = self.x_embedder(hidden_states)  # [B, N_img, H_dim]

        # ---- embed text ----
        if self.text_proj is not None:
            text_bnh = self.text_proj(text_bth)
        else:
            text_bnh = text_bth
        Tmax = text_bnh.shape[1]

        # ---- concat [img, text] on sequence axis (image FIRST — matches reference) ----
        x = mx.concatenate([img_bnh, text_bnh], axis=1)  # [B, N_img+Tmax, H_dim]

        # ---- build RoPE position ids ----
        # image_ids: (text_len_offset, y, x) per image token
        text_lens_f = text_lens.astype(mx.float32)
        text_lens_col = text_lens_f.reshape(B, 1, 1)
        text_lens_exp = mx.broadcast_to(text_lens_col, (B, N_img, 1))

        grid_y = mx.arange(Hp, dtype=mx.float32)
        grid_x = mx.arange(Wp, dtype=mx.float32)
        yy, xx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        grid_yx = mx.stack([yy, xx], axis=-1).reshape(N_img, 2)[None, :, :]
        grid_yx_exp = mx.broadcast_to(grid_yx, (B, N_img, 2))
        image_ids = mx.concatenate([text_lens_exp, grid_yx_exp], axis=-1)  # [B, N_img, 3]

        # text_ids: (text_token_position, 0, 0) — axis 0 is the text position axis
        if Tmax > 0:
            text_pos = mx.arange(Tmax, dtype=mx.float32).reshape(1, Tmax, 1)
            text_pos = mx.broadcast_to(text_pos, (B, Tmax, 1))
            text_zeros = mx.zeros((B, Tmax, 2), dtype=mx.float32)
            text_ids = mx.concatenate([text_pos, text_zeros], axis=-1)
            ids_all = mx.concatenate([image_ids, text_ids], axis=1)
        else:
            ids_all = image_ids

        freqs_cis = self.pos_embed(ids_all)  # [B, N_total, 1, head_dim]

        # ---- attention mask: True=attend, False=padding ----
        if Tmax > 0:
            text_valid = (
                mx.arange(Tmax, dtype=mx.int32).reshape(1, Tmax)
                < text_lens.astype(mx.int32).reshape(B, 1)
            )
            img_ones = mx.ones((B, N_img), dtype=mx.bool_)
            attn_mask = mx.concatenate([img_ones, text_valid], axis=1)
        else:
            attn_mask = mx.ones((B, N_img), dtype=mx.bool_)
        attn_mask = attn_mask[:, None, None, :]  # [B, 1, 1, N_total]

        # ---- timestep + shared AdaLN modulation ----
        t_sin = get_timestep_embedding(
            timestep.astype(mx.float32),
            cfg.hidden_size,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.0,
        )
        c = self.time_embedding(t_sin.astype(x.dtype))  # [B, H_dim]
        six = mx.split(self.adaLN_modulation(c), 6, axis=-1)  # 6 × [B, H_dim]
        temb: TembTuple = tuple(t[:, None, :] for t in six)  # 6 × [B, 1, H_dim]

        # ---- transformer stack ----
        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, temb=temb, attention_mask=attn_mask)

        # ---- final norm + unpatchify ----
        x = self.final_norm(x, c)
        patches = self.final_linear(x)[:, :N_img, :]  # [B, N_img, p*p*out_ch]

        out = patches.reshape(B, Hp, Wp, p, p, cfg.out_channels)
        out = out.transpose(0, 5, 1, 3, 2, 4)  # [B, out_ch, Hp, p, Wp, p]
        return out.reshape(B, cfg.out_channels, H, W)
