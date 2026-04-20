"""Full-model PT vs MLX parity for ErnieImageTransformer2DModel.

Small config (256 hidden, 4 heads, 2 layers, 32-channel latent) — keeps runtime
reasonable while exercising the full forward path: patch embed + text proj +
time embedding + shared AdaLN + 2-block stack + final norm + unpatchify.

Threshold is looser than single-block parity: stacking 2 layers + AdaLN
modulation + timestep embedding all compound error.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.parity


def _copy_pt_state(pt_module) -> dict[str, np.ndarray]:
    return {k: v.detach().float().numpy() for k, v in pt_module.state_dict().items()}


def _update_mx(mx_module, flat: list[tuple[str, object]]) -> None:
    import mlx.utils as mx_utils

    mx_module.update(mx_utils.tree_unflatten(flat))


def _pt_to_mx_rename(key: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Rename a PT key + optionally reshape its value to match the MLX model."""
    # Patch embed: Conv2d (O, I, 1, 1) → Linear (O, I)
    if key == "x_embedder.proj.weight":
        return key, value.squeeze((2, 3))

    # Attention output projection: ModuleList[0] → MLX flat name
    key = key.replace(".self_attention.to_out.0.weight", ".self_attention.to_out_0.weight")

    # TimestepEmbedding: diffusers names `linear_1`/`linear_2`, mlx-arsenal uses `linear1`/`linear2`
    key = key.replace("time_embedding.linear_1.", "time_embedding.linear1.")
    key = key.replace("time_embedding.linear_2.", "time_embedding.linear2.")

    # AdaLN-modulation: PT Sequential[1] → MLX `linear` attribute
    key = key.replace("adaLN_modulation.1.", "adaLN_modulation.linear.")

    return key, value


def test_full_model_parity():
    import mlx.core as mx

    from ernie_image_core_mlx.model.config import ErnieImageConfig
    from ernie_image_core_mlx.model.transformer import (
        ErnieImageTransformer2DModel as MxModel,
    )
    from tests.parity._pt_reference import ErnieImageTransformer2DModel as PtModel

    cfg = ErnieImageConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_layers=2,
        ffn_hidden_size=512,
        in_channels=32,
        out_channels=32,
        patch_size=1,
        text_in_dim=3072,
        rope_axes_dim=(16, 24, 24),
        rope_theta=256.0,
        qk_layernorm=True,
    )
    assert cfg.head_dim == 64 and sum(cfg.rope_axes_dim) == 64

    torch.manual_seed(7)
    pt_model = PtModel(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_layers=cfg.num_layers,
        ffn_hidden_size=cfg.ffn_hidden_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        patch_size=cfg.patch_size,
        text_in_dim=cfg.text_in_dim,
        rope_theta=int(cfg.rope_theta),
        rope_axes_dim=cfg.rope_axes_dim,
        eps=cfg.eps,
        qk_layernorm=cfg.qk_layernorm,
    )
    pt_model.train(False)

    mx_model = MxModel(cfg)

    pt_sd = _copy_pt_state(pt_model)
    flat = []
    for pt_k, pt_v in pt_sd.items():
        mx_k, mx_v = _pt_to_mx_rename(pt_k, pt_v)
        flat.append((mx_k, mx.array(mx_v)))
    _update_mx(mx_model, flat)

    B, Hl, Wl = 2, 8, 8
    Tmax = 16
    rng = np.random.default_rng(777)
    x_np = rng.standard_normal((B, cfg.in_channels, Hl, Wl)).astype(np.float32)
    text_np = rng.standard_normal((B, Tmax, cfg.text_in_dim)).astype(np.float32)
    text_lens_np = np.array([12, 16], dtype=np.int64)
    timestep_np = np.array([500.0, 750.0], dtype=np.float32)

    with torch.no_grad():
        pt_out = (
            pt_model(
                torch.from_numpy(x_np),
                torch.from_numpy(timestep_np),
                torch.from_numpy(text_np),
                torch.from_numpy(text_lens_np),
            )
            .float()
            .numpy()
        )

    mx_out = np.array(
        mx_model(
            mx.array(x_np),
            mx.array(timestep_np),
            mx.array(text_np),
            mx.array(text_lens_np),
        )
    )

    assert pt_out.shape == mx_out.shape == (B, cfg.out_channels, Hl, Wl)
    diff = np.abs(pt_out - mx_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # Observed ~3e-6 on Apple Silicon fp32; 1e-4 leaves headroom for M1/M2/M3 variation.
    assert max_abs < 1e-4, f"full-model parity FAIL: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
