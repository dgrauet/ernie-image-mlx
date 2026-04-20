"""End-to-end smoke test for VAE + latent BN + pixel shuffle chain.

Exercises the exact sequence the pipeline runs after the DiT:
    (B, 128, H, W)          # DiT output, channels-second
  → transpose → (B, H, W, 128)
  → bn.apply_inverse
  → pixel_shuffle(2) → (B, 2H, 2W, 32)
  → vae.decode → (B, 16H, 16W, 3)

No PyTorch, no weights — just verifies the forward chain doesn't crash and
produces the expected shapes.
"""

from __future__ import annotations

import numpy as np


def test_vae_decode_chain_shape():
    import mlx.core as mx
    from mlx_arsenal.spatial import pixel_shuffle

    from ernie_image_core_mlx.model.config import ErnieImageVaeConfig
    from ernie_image_core_mlx.model.vae import AutoencoderKLFlux2

    cfg = ErnieImageVaeConfig(
        block_out_channels=(16, 32, 64, 64),
        norm_num_groups=8,
        latent_channels=8,
    )
    vae = AutoencoderKLFlux2(cfg)

    # Simulated DiT output: (B, in_channels=128 with our defaults, H_lat, W_lat).
    # With latent_channels=8 + patch=(2,2) → DiT channels = 8 * 4 = 32.
    dit_channels = cfg.latent_channels * cfg.patch_size[0] * cfg.patch_size[1]
    B, H_lat, W_lat = 1, 2, 2
    latents_ch_second = mx.array(
        np.random.default_rng(0).standard_normal((B, dit_channels, H_lat, W_lat)).astype("float32")
    )

    # Pipeline chain
    nhwc = latents_ch_second.transpose(0, 2, 3, 1)
    normed = vae.bn.apply_inverse(nhwc)
    vae_latents = pixel_shuffle(normed, upscale_factor=cfg.patch_size[0])
    image = vae.decode(vae_latents)

    # VAE downsample = 2 * 2 * 2 = 8 (3 stride-2 stages for 4 blocks).
    # patch=2 multiplies H_lat by 2 → VAE-space latent 4×4 → decode ×8 → 32×32.
    expected_H = H_lat * cfg.patch_size[0] * (2 ** (len(cfg.block_out_channels) - 1))
    assert image.shape == (B, expected_H, expected_H, cfg.out_channels)
    assert np.isfinite(np.array(image)).all()


def test_vae_bn_inverse_matches_reference_formula():
    """The pipeline's BN inverse uses a hard-coded eps=1e-5 (matches the
    reference ERNIE-Image pipeline byte-for-byte, not the VAE's trained
    `batch_norm_eps=1e-4`). Test the inverse directly against the formula."""
    import mlx.core as mx

    from ernie_image_core_mlx.model.vae import _LatentBN

    C = 16
    bn = _LatentBN(C, eps=1e-4)  # stored eps, not used by inverse
    rng = np.random.default_rng(11)
    mean_np = rng.standard_normal(C).astype("float32")
    var_np = (rng.standard_normal(C) ** 2 + 0.5).astype("float32")
    bn.running_mean = mx.array(mean_np)
    bn.running_var = mx.array(var_np)

    x_np = rng.standard_normal((2, 4, 4, C)).astype("float32")
    expected = x_np * np.sqrt(var_np + 1e-5) + mean_np

    got = np.array(bn.apply_inverse(mx.array(x_np)))
    err = float(np.max(np.abs(got - expected)))
    assert err < 1e-5, f"BN inverse diverges from reference formula: err={err:.3e}"
