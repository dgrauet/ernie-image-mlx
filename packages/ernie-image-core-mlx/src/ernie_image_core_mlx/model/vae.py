"""AutoencoderKLFlux2 — Flux 2 VAE used by ERNIE-Image.

Reference: `diffusers/models/autoencoders/autoencoder_kl_flux2.py` + the shared
Encoder/Decoder in `diffusers/models/autoencoders/vae.py` + ResnetBlock2D /
Down/UpEncoderBlock2D / UNetMidBlock2D in `diffusers/models/*`. Module names
match diffusers exactly so weight loading needs only conv layout transposes.

MLX layout: channels-last `(B, H, W, C)` throughout. Convert at the VAE
boundary — the DiT speaks channels-second, the pipeline handles the transpose.

Single-head attention path (`mid_block.attentions.0`): `heads=1`, `head_dim=channels`,
GroupNorm on the channel axis before QKV, residual after projection.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ernie_image_core_mlx.model.config import ErnieImageVaeConfig


def _silu(x: mx.array) -> mx.array:
    return nn.silu(x)


# ---------------------------------------------------------------------------
# ResnetBlock2D — GroupNorm → SiLU → Conv2d(3×3) → GN → SiLU → Conv2d(3×3) + skip
# ---------------------------------------------------------------------------


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        """`x: (B, H, W, C_in)` → `(B, H, W, C_out)`."""
        h = self.norm1(x)
        h = _silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _silu(h)
        h = self.conv2(h)
        skip = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return skip + h


# ---------------------------------------------------------------------------
# Down/Up samplers — 3×3 Conv2d (with stride 2 for down, preceded by nearest ×2 for up)
# ---------------------------------------------------------------------------


class Downsample2D(nn.Module):
    """Stride-2 Conv2d(3×3). Matches diffusers `Downsample2D(padding=0)` BUT with
    explicit asymmetric padding — see `__call__`. This mirrors the diffusers
    behavior `downsample_padding=0`."""

    def __init__(self, channels: int):
        super().__init__()
        # PT uses kernel=3, stride=2, padding=0 AND F.pad (0,1,0,1) externally.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # Asymmetric right/bottom zero-pad to emulate `downsample_padding=0` + stride-2 trick.
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor ×2 upsample, then Conv2d(3×3)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Nearest ×2 along H and W. (B, H, W, C) → (B, 2H, 2W, C).
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return self.conv(x)


# ---------------------------------------------------------------------------
# DownEncoderBlock2D / UpDecoderBlock2D — N resnets + optional sampler
# ---------------------------------------------------------------------------


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(c_in, out_channels, groups=groups, eps=eps))
        # Match diffusers' ModuleList with one downsampler — list of len 1 or 0.
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                x = d(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_upsample: bool = True,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(c_in, out_channels, groups=groups, eps=eps))
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            for u in self.upsamplers:
                x = u(x)
        return x


# ---------------------------------------------------------------------------
# Mid-block attention — single-head, GroupNorm on channels, residual
# ---------------------------------------------------------------------------


class VAEAttention(nn.Module):
    """Single-head (`heads=1`, `head_dim=channels`) self-attention with GroupNorm
    pre-projection and a residual connection. Submodule names match
    `diffusers.models.attention_processor.Attention` exactly so weight loading
    is one-to-one.
    """

    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels, pytorch_compatible=True, eps=eps)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        # `to_out` is a ModuleList in diffusers: [Linear, Dropout]. Match the naming.
        self.to_out = [nn.Linear(channels, channels, bias=True)]

    def __call__(self, x: mx.array) -> mx.array:
        """`x: (B, H, W, C)` → `(B, H, W, C)`."""
        B, H, W, C = x.shape
        residual = x
        h = self.group_norm(x)
        h = h.reshape(B, H * W, C)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        scale = 1.0 / math.sqrt(C)
        scores = mx.matmul(q, k.transpose(0, 2, 1)) * scale
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        h = mx.matmul(probs, v)
        h = self.to_out[0](h)
        h = h.reshape(B, H, W, C)
        return residual + h


class VAEMidBlock(nn.Module):
    """UNetMidBlock2D minus time-embedding plumbing (VAE has `temb=None`)."""

    def __init__(
        self,
        channels: int,
        groups: int = 32,
        eps: float = 1e-6,
        add_attention: bool = True,
    ):
        super().__init__()
        # Matches diffusers naming: mid_block.resnets.[0|1], mid_block.attentions.[0].
        self.resnets = [
            ResnetBlock2D(channels, channels, groups=groups, eps=eps),
            ResnetBlock2D(channels, channels, groups=groups, eps=eps),
        ]
        self.attentions = (
            [VAEAttention(channels, groups=groups, eps=eps)] if add_attention else []
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        if not self.attentions:
            # No attention: pair each remaining resnet with a no-op pre-step.
            for r in self.resnets[1:]:
                x = r(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, cfg: ErnieImageVaeConfig, double_z: bool = True):
        super().__init__()
        self.cfg = cfg
        c0 = cfg.block_out_channels[0]
        self.conv_in = nn.Conv2d(cfg.in_channels, c0, kernel_size=3, padding=1)

        self.down_blocks = []
        c_prev = c0
        for i, c_out in enumerate(cfg.block_out_channels):
            is_last = i == len(cfg.block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    c_prev,
                    c_out,
                    num_layers=cfg.layers_per_block,
                    add_downsample=not is_last,
                    groups=cfg.norm_num_groups,
                )
            )
            c_prev = c_out

        self.mid_block = VAEMidBlock(
            c_prev, groups=cfg.norm_num_groups, add_attention=cfg.mid_block_add_attention
        )

        self.conv_norm_out = nn.GroupNorm(cfg.norm_num_groups, c_prev, pytorch_compatible=True, eps=1e-6)
        out_ch = 2 * cfg.latent_channels if double_z else cfg.latent_channels
        self.conv_out = nn.Conv2d(c_prev, out_ch, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """`x: (B, H, W, 3)` → `(B, H/8, W/8, 2*latent_ch)`."""
        h = self.conv_in(x)
        for b in self.down_blocks:
            h = b(h)
        h = self.mid_block(h)
        h = self.conv_norm_out(h)
        h = _silu(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, cfg: ErnieImageVaeConfig):
        super().__init__()
        self.cfg = cfg
        c_top = cfg.block_out_channels[-1]
        self.conv_in = nn.Conv2d(cfg.latent_channels, c_top, kernel_size=3, padding=1)

        self.mid_block = VAEMidBlock(
            c_top, groups=cfg.norm_num_groups, add_attention=cfg.mid_block_add_attention
        )

        # Decoder iterates the reversed channel list; diffusers uses `layers_per_block + 1`
        # resnets per up-block (one more than the encoder).
        self.up_blocks = []
        reversed_channels = list(reversed(cfg.block_out_channels))
        c_prev = reversed_channels[0]
        for i, c_out in enumerate(reversed_channels):
            is_last = i == len(reversed_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    c_prev,
                    c_out,
                    num_layers=cfg.layers_per_block + 1,
                    add_upsample=not is_last,
                    groups=cfg.norm_num_groups,
                )
            )
            c_prev = c_out

        self.conv_norm_out = nn.GroupNorm(
            cfg.norm_num_groups, cfg.block_out_channels[0], pytorch_compatible=True, eps=1e-6
        )
        self.conv_out = nn.Conv2d(cfg.block_out_channels[0], cfg.out_channels, kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        """`z: (B, H, W, latent_ch)` → `(B, 8H, 8W, 3)`."""
        h = self.conv_in(z)
        h = self.mid_block(h)
        for b in self.up_blocks:
            h = b(h)
        h = self.conv_norm_out(h)
        h = _silu(h)
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# Latent normalization (top-level BatchNorm2d, affine=False)
# ---------------------------------------------------------------------------


class _LatentBN(nn.Module):
    """BatchNorm2d with `affine=False` — only `running_mean` / `running_var` are
    kept. Matches the submodule path `bn.*` in the reference checkpoint.

    Note: the reference *pipeline* hard-codes `eps=1e-5` for the inverse step,
    while the VAE's declared `batch_norm_eps=1e-4` would only apply if the BN
    were ever run forward (which it isn't at inference). We store the declared
    value on `self.eps` but use a fixed `1e-5` in the inverse path to match
    the reference denoising loop byte-for-byte.
    """

    # The reference pipeline uses this exact epsilon — keep it constant.
    _INVERSE_EPS = 1e-5

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))

    def apply_forward(self, x: mx.array) -> mx.array:
        """Apply `(x - running_mean) / sqrt(running_var + eps)`. `x` is channels-last."""
        return (x - self.running_mean) / mx.sqrt(self.running_var + self.eps)

    def apply_inverse(self, x: mx.array) -> mx.array:
        """Inverse: `x * sqrt(running_var + 1e-5) + running_mean`. Used by the
        pipeline to map DiT-normalized latents back into VAE-native space."""
        return x * mx.sqrt(self.running_var + self._INVERSE_EPS) + self.running_mean


# ---------------------------------------------------------------------------
# Top-level VAE — channels-last in/out
# ---------------------------------------------------------------------------


class AutoencoderKLFlux2(nn.Module):
    """Flux 2 VAE. `(B, H, W, 3)` ↔ `(B, H/8, W/8, 32)` latent.

    The reference carries an extra top-level `BatchNorm2d` (affine=False,
    track_running_stats=True) over `latent_channels × prod(patch)` channels used
    to normalize DiT-space latents. Keep its `running_mean` / `running_var` at
    `self.bn.*` so the pipeline can apply the inverse before `decode`.
    """

    def __init__(self, cfg: ErnieImageVaeConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, double_z=True)
        self.decoder = Decoder(cfg)
        self.quant_conv = (
            nn.Conv2d(2 * cfg.latent_channels, 2 * cfg.latent_channels, kernel_size=1)
            if cfg.use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)
            if cfg.use_post_quant_conv
            else None
        )

        # Latent normalization stats (`bn.*` keys in the reference checkpoint).
        # The reference model carries a BatchNorm2d(affine=False, track_running_stats=True)
        # over (patch²·latent_ch) channels — same channel count as the DiT input.
        # The pipeline calls `self.bn.apply_inverse(...)` on DiT-space latents
        # before pixel-shuffle + `vae.decode`.
        bn_channels = math.prod(cfg.patch_size) * cfg.latent_channels
        self.bn = _LatentBN(bn_channels, eps=cfg.batch_norm_eps)

    def encode(self, x: mx.array) -> mx.array:
        """Encode `(B, H, W, 3)` → `(B, H/8, W/8, 2*latent_ch=64)` moments (mean+logvar)."""
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return h

    def decode(self, z: mx.array) -> mx.array:
        """Decode `(B, H/8, W/8, latent_ch=32)` → `(B, H, W, 3)`."""
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)

    def __call__(self, x: mx.array) -> mx.array:
        """Full round-trip using the posterior mode (deterministic)."""
        moments = self.encode(x)
        mean, _logvar = mx.split(moments, 2, axis=-1)
        return self.decode(mean)
