"""Cheap, weight-free sanity checks on config + construction.

Runs without model weights, without torch. Meant to stay green from day one.
"""

from __future__ import annotations

import pytest


def test_default_transformer_config_consistent():
    from ernie_image_core_mlx.model.config import ErnieImageConfig

    cfg = ErnieImageConfig()
    assert cfg.head_dim == 128
    assert sum(cfg.rope_axes_dim) == cfg.head_dim
    assert cfg.num_layers == 36
    assert cfg.rope_theta == 256.0  # ERNIE-Image value — NOT the common 10000
    assert cfg.qk_layernorm is True


def test_vae_config_downsample_math():
    from ernie_image_core_mlx.model.config import ErnieImageVaeConfig

    cfg = ErnieImageVaeConfig()
    # 4 down blocks → stride 8, + 2×2 patch → total ×16 downsample
    assert len(cfg.block_out_channels) == 4
    assert cfg.patch_size == (2, 2)
    # latent_ch * patch_area packs to DiT's in_channels
    assert cfg.latent_channels * cfg.patch_size[0] * cfg.patch_size[1] == 128


def test_pipeline_variant_presets():
    from ernie_image_core_mlx.model.config import ErniePipelineConfig

    sft = ErniePipelineConfig.for_variant("sft")
    turbo = ErniePipelineConfig.for_variant("turbo")
    assert sft.num_inference_steps == 50 and sft.guidance_scale == pytest.approx(5.0)
    assert turbo.num_inference_steps == 8 and turbo.guidance_scale == pytest.approx(1.0)

    with pytest.raises(ValueError):
        ErniePipelineConfig.for_variant("lightning")


def test_transformer_module_constructs():
    """Weight-free instantiation — catches trivial shape/init bugs."""
    from ernie_image_core_mlx.model.config import ErnieImageConfig
    from ernie_image_core_mlx.model.transformer import ErnieImageTransformer2DModel

    cfg = ErnieImageConfig(num_layers=2)
    model = ErnieImageTransformer2DModel(cfg)
    assert len(model.layers) == 2
