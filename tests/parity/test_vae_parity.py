"""PT vs MLX parity for AutoencoderKLFlux2 building blocks.

Uses diffusers' shared `Encoder` / `Decoder` / `ResnetBlock2D` as the oracle
(these are stable across 0.37+). No vendoring needed — only the top-level
`AutoencoderKLFlux2` is new in the unreleased diffusers main, and we don't
need the wrapper's forward for parity (we test the pieces).

Layout mapping at the PT/MLX boundary:
    PT: (B, C, H, W)  — channels-second
    MLX: (B, H, W, C) — channels-last

Weight layout:
    PT Conv2d weight (O, I, H, W) → MLX Conv2d weight (O, H, W, I)  — transpose(0,2,3,1)
    PT Linear/GroupNorm weights: identity

Thresholds (fp32):
    Single ResnetBlock2D:     < 1e-5
    Attention midblock:       < 5e-5
    Full Encoder/Decoder:     < 1e-3
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.parity


def _pt_conv2d_to_mlx(w: np.ndarray) -> np.ndarray:
    return w.transpose(0, 2, 3, 1)


def _nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    return x.transpose(0, 2, 3, 1)


def _nhwc_to_nchw(x: np.ndarray) -> np.ndarray:
    return x.transpose(0, 3, 1, 2)


def _update_mx(mx_module, flat):
    import mlx.utils as mx_utils

    mx_module.update(mx_utils.tree_unflatten(flat))


def _translate_state(pt_state: dict) -> list[tuple[str, np.ndarray]]:
    """Translate a PT VAE submodule state_dict to MLX: conv weights get transposed."""
    out = []
    for k, v in pt_state.items():
        arr = v.detach().float().numpy()
        if arr.ndim == 4 and k.endswith(".weight"):
            arr = _pt_conv2d_to_mlx(arr)
        out.append((k, arr))
    return out


# ---------------------------------------------------------------------------
# ResnetBlock2D
# ---------------------------------------------------------------------------


def test_resnet_block_parity():
    import mlx.core as mx

    from diffusers.models.resnet import ResnetBlock2D as PtResnet

    from ernie_image_core_mlx.model.vae import ResnetBlock2D as MxResnet

    C_in, C_out = 16, 32
    groups = 8

    torch.manual_seed(0)
    pt = PtResnet(
        in_channels=C_in,
        out_channels=C_out,
        groups=groups,
        eps=1e-6,
        non_linearity="silu",
        temb_channels=None,
    )
    pt.train(False)

    mxm = MxResnet(C_in, C_out, groups=groups, eps=1e-6)
    flat = [(k, mx.array(v)) for k, v in _translate_state(pt.state_dict())]
    _update_mx(mxm, flat)

    rng = np.random.default_rng(0)
    x_pt = rng.standard_normal((2, C_in, 8, 8)).astype(np.float32)
    x_mx = _nchw_to_nhwc(x_pt)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_pt), None).float().numpy()
    mx_out = _nhwc_to_nchw(np.array(mxm(mx.array(x_mx))))

    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-5, f"resnet parity FAIL: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# VAE attention (single head, mid-block)
# ---------------------------------------------------------------------------


def test_vae_attention_parity():
    import mlx.core as mx

    from diffusers.models.attention_processor import Attention, AttnProcessor

    from ernie_image_core_mlx.model.vae import VAEAttention

    C = 32
    groups = 8

    torch.manual_seed(1)
    pt_attn = Attention(
        query_dim=C,
        heads=1,
        dim_head=C,
        bias=True,
        out_bias=True,
        norm_num_groups=groups,
        eps=1e-6,
        residual_connection=True,
        _from_deprecated_attn_block=True,
        processor=AttnProcessor(),
    )
    pt_attn.train(False)

    mx_attn = VAEAttention(C, groups=groups, eps=1e-6)
    pt_sd = {k: v.detach().float().numpy() for k, v in pt_attn.state_dict().items()}
    _update_mx(mx_attn, [(k, mx.array(v)) for k, v in pt_sd.items()])

    B, H, W = 2, 4, 4
    rng = np.random.default_rng(2)
    x_pt = rng.standard_normal((B, C, H, W)).astype(np.float32)
    x_mx = _nchw_to_nhwc(x_pt)

    with torch.no_grad():
        pt_out = pt_attn(torch.from_numpy(x_pt)).float().numpy()
    mx_out = _nhwc_to_nchw(np.array(mx_attn(mx.array(x_mx))))

    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 5e-5, f"attention parity FAIL: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# Full Encoder + Decoder
# ---------------------------------------------------------------------------


def test_encoder_parity():
    import mlx.core as mx

    from diffusers.models.autoencoders.vae import Encoder as PtEncoder

    from ernie_image_core_mlx.model.config import ErnieImageVaeConfig
    from ernie_image_core_mlx.model.vae import Encoder as MxEncoder

    cfg = ErnieImageVaeConfig(
        block_out_channels=(16, 32, 64, 64),
        norm_num_groups=8,
        latent_channels=8,
    )

    torch.manual_seed(3)
    pt = PtEncoder(
        in_channels=cfg.in_channels,
        out_channels=cfg.latent_channels,
        down_block_types=cfg.down_block_types,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
        norm_num_groups=cfg.norm_num_groups,
        act_fn=cfg.act_fn,
        double_z=True,
        mid_block_add_attention=cfg.mid_block_add_attention,
    )
    pt.train(False)

    mxm = MxEncoder(cfg, double_z=True)
    flat = [(k, mx.array(v)) for k, v in _translate_state(pt.state_dict())]
    _update_mx(mxm, flat)

    rng = np.random.default_rng(4)
    x_pt = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)
    x_mx = _nchw_to_nhwc(x_pt)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_pt)).float().numpy()
    mx_out = _nhwc_to_nchw(np.array(mxm(mx.array(x_mx))))

    assert pt_out.shape == mx_out.shape
    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-3, f"encoder parity FAIL: max_abs={max_abs:.3e}"


def test_decoder_parity():
    import mlx.core as mx

    from diffusers.models.autoencoders.vae import Decoder as PtDecoder

    from ernie_image_core_mlx.model.config import ErnieImageVaeConfig
    from ernie_image_core_mlx.model.vae import Decoder as MxDecoder

    cfg = ErnieImageVaeConfig(
        block_out_channels=(16, 32, 64, 64),
        norm_num_groups=8,
        latent_channels=8,
    )

    torch.manual_seed(5)
    pt = PtDecoder(
        in_channels=cfg.latent_channels,
        out_channels=cfg.out_channels,
        up_block_types=cfg.up_block_types,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
        norm_num_groups=cfg.norm_num_groups,
        act_fn=cfg.act_fn,
        mid_block_add_attention=cfg.mid_block_add_attention,
    )
    pt.train(False)

    mxm = MxDecoder(cfg)
    flat = [(k, mx.array(v)) for k, v in _translate_state(pt.state_dict())]
    _update_mx(mxm, flat)

    rng = np.random.default_rng(6)
    z_pt = rng.standard_normal((1, cfg.latent_channels, 4, 4)).astype(np.float32)
    z_mx = _nchw_to_nhwc(z_pt)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(z_pt)).float().numpy()
    mx_out = _nhwc_to_nchw(np.array(mxm(mx.array(z_mx))))

    assert pt_out.shape == mx_out.shape
    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-3, f"decoder parity FAIL: max_abs={max_abs:.3e}"
