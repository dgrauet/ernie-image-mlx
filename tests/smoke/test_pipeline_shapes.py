"""Smoke test for pipeline helper math. No weights, no tokenizer."""

from __future__ import annotations

import numpy as np


def test_downsample_factor():
    from ernie_image_core_mlx.model.config import ErnieImageVaeConfig
    from ernie_image_core_mlx.pipelines.ernie_image import _downsample_factor

    # Production config — 4 block_out_channels, patch=(2,2) → ×16 total.
    assert _downsample_factor(ErnieImageVaeConfig()) == 16

    # 3 blocks (2 strides) + patch 2 → ×8
    cfg_small = ErnieImageVaeConfig(
        block_out_channels=(64, 128, 256),
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
    )
    assert _downsample_factor(cfg_small) == 8


def test_dit_to_vae_unpack_shape():
    import mlx.core as mx

    from ernie_image_core_mlx.pipelines.ernie_image import _unpack_dit_to_vae

    B, C_dit, H, W = 2, 128, 4, 4
    patch = 2
    z = mx.array(np.random.standard_normal((B, C_dit, H, W)).astype("float32"))
    out = _unpack_dit_to_vae(z, patch)

    # 128 channels pack 32 × 2×2 patches → unpack to 32 ch, ×2 spatial.
    assert out.shape == (B, H * patch, W * patch, C_dit // (patch * patch))
