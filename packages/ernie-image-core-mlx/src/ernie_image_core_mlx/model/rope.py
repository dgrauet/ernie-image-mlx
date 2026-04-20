"""Triple-axis RoPE for ERNIE-Image DiT — Megatron non-interleaved variant.

Direct MLX port of `rope` + `ErnieImageEmbedND3` + the `apply_rotary_emb` closure
in `diffusers.models.transformers.transformer_ernie_image` (v0.36).

Axis layout:
    axis 0: text/temporal position offset (32 channels)
    axis 1: image grid y             (48 channels)
    axis 2: image grid x             (48 channels)
Sum = 128 = head_dim.

Angle layout in `freqs_cis`: each per-axis base angle θ is duplicated — final
shape is `[B, N, 1, head_dim]` with values `[θ0,θ0,θ1,θ1,...,θ_{hd/2-1},θ_{hd/2-1}]`.
This is Megatron's `_apply_rotary_pos_emb_bshd` with `rotary_interleaved=False`
and DIFFERS from the LLaMA convention where the second-half shares angles with
the first. `mx.fast.rope` does not cover this — roll by hand.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _axis_angles(positions: mx.array, dim: int, theta: float) -> mx.array:
    """Return per-position angles for one RoPE axis.

    Mirrors `rope()` in the reference: `omega = 1 / theta^(2k/dim)` for k in [0, dim/2),
    then `angles[n, k] = positions[n] * omega[k]`.

    Args:
        positions: float array, shape `[..., N]`.
        dim: channels allocated to this axis (must be even).
        theta: RoPE base (256 for ERNIE-Image — do NOT change).

    Returns:
        array, shape `[..., N, dim // 2]`.
    """
    assert dim % 2 == 0, f"axis dim must be even, got {dim}"
    # scale[k] = 2k / dim, k in [0, dim/2)
    k = mx.arange(0, dim, 2, dtype=mx.float32)
    scale = k / dim
    omega = 1.0 / (theta**scale)  # [dim/2]
    # einsum "...n,d->...nd"
    return positions.astype(mx.float32)[..., None] * omega


class ErnieImageEmbedND3(nn.Module):
    """Build RoPE `freqs_cis` for text + image tokens jointly.

    Input `ids` has shape `[B, N, 3]` — (text_pos_or_len, y, x) per token.
    Output `freqs_cis` has shape `[B, N, 1, head_dim]` and is broadcast along the
    heads axis at attention time.
    """

    def __init__(self, head_dim: int, theta: float, axes_dim: tuple[int, int, int]):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.axes_dim = tuple(axes_dim)
        assert sum(self.axes_dim) == head_dim, f"axes_dim {self.axes_dim} must sum to head_dim {head_dim}"

    def __call__(self, ids: mx.array) -> mx.array:
        per_axis = [_axis_angles(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(per_axis, axis=-1)  # [B, N, head_dim/2]
        emb = emb[:, :, None, :]  # [B, N, 1, head_dim/2]
        # Duplicate each angle along the last dim: [θ0,θ0,θ1,θ1,...].
        # `stack([emb, emb], -1).reshape(..., -1)` in torch == `repeat_interleave(2)`.
        emb = mx.stack([emb, emb], axis=-1)  # [B, N, 1, head_dim/2, 2]
        shape = emb.shape
        return emb.reshape(*shape[:-2], shape[-2] * shape[-1])  # [B, N, 1, head_dim]


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Megatron non-interleaved rotate_half RoPE.

    Args:
        x: `[B, N, H, D]` where D == head_dim (or >= freqs_cis.shape[-1] for
           the `x_pass` unrotated tail).
        freqs_cis: `[B, N, 1, rot_dim]`, unbounded angles (cos/sin applied here).

    Returns:
        Rotated tensor, same shape as `x`.
    """
    rot_dim = freqs_cis.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    cos_ = mx.cos(freqs_cis).astype(x.dtype)
    sin_ = mx.sin(freqs_cis).astype(x.dtype)

    # rotate_half(x) = concat(-x2, x1) where x1, x2 = chunk(x, 2, dim=-1)
    d_half = rot_dim // 2
    x1 = x_rot[..., :d_half]
    x2 = x_rot[..., d_half:]
    x_rotated = mx.concatenate([-x2, x1], axis=-1)

    out_rot = x_rot * cos_ + x_rotated * sin_
    if x_pass.shape[-1] == 0:
        return out_rot
    return mx.concatenate([out_rot, x_pass], axis=-1)
