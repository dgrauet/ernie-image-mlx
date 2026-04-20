"""PT vs MLX parity for FFN, shared-AdaLN block, and AdaLN-continuous.

Attention parity already locked at `test_attention_parity.py`. Here we close
the loop on the rest of a DiT block so the full 36-block stack can be assembled
with confidence.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.parity


def _copy_pt_state(pt_module) -> dict[str, np.ndarray]:
    """Return a {name: np.ndarray} mirror of a PyTorch module state_dict."""
    return {k: v.detach().float().numpy() for k, v in pt_module.state_dict().items()}


def _update_mx(mx_module, flat: list[tuple[str, object]]) -> None:
    import mlx.core as mx
    import mlx.utils as mx_utils

    mx_module.update(mx_utils.tree_unflatten(flat))
    mx.eval(mx_module.parameters())


# ---------------------------------------------------------------------------


def test_ffn_parity():
    import mlx.core as mx

    from ernie_image_core_mlx.model.transformer import ErnieImageFeedForward as MxFFN
    from tests.parity._pt_reference import ErnieImageFeedForward as PtFFN

    H, FFN = 256, 512
    torch.manual_seed(0)
    pt = PtFFN(H, FFN)
    pt.train(False)
    mxm = MxFFN(H, FFN)

    pt_sd = _copy_pt_state(pt)
    _update_mx(mxm, [(k, mx.array(v)) for k, v in pt_sd.items()])

    rng = np.random.default_rng(10)
    x_np = rng.standard_normal((2, 64, H)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np)).float().numpy()
    mx_out = np.array(mxm(mx.array(x_np)))

    assert pt_out.shape == mx_out.shape
    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-5, f"FFN parity FAIL: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------


def test_adaln_continuous_parity():
    import mlx.core as mx

    from ernie_image_core_mlx.model.config import ErnieImageConfig
    from ernie_image_core_mlx.model.transformer import ErnieImageAdaLNContinuous as MxFinal
    from tests.parity._pt_reference import ErnieImageAdaLNContinuous as PtFinal

    cfg = ErnieImageConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_layers=1,
        ffn_hidden_size=512,
        rope_axes_dim=(16, 24, 24),
    )
    torch.manual_seed(0)
    pt = PtFinal(cfg.hidden_size, eps=cfg.eps)
    pt.train(False)
    mxm = MxFinal(cfg)

    _update_mx(mxm, [(k, mx.array(v)) for k, v in _copy_pt_state(pt).items()])

    B, S = 2, 40
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((B, S, cfg.hidden_size)).astype(np.float32)
    c_np = rng.standard_normal((B, cfg.hidden_size)).astype(np.float32)

    with torch.no_grad():
        pt_out_sbh = pt(
            torch.from_numpy(x_np).permute(1, 0, 2),
            torch.from_numpy(c_np),
        )
        pt_out = pt_out_sbh.permute(1, 0, 2).float().numpy()

    mx_out = np.array(mxm(mx.array(x_np), mx.array(c_np)))

    assert pt_out.shape == mx_out.shape
    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-5, f"AdaLN-continuous parity FAIL: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------


def _pt_to_mx_block_rename(key: str) -> str:
    return key.replace("self_attention.to_out.0.weight", "self_attention.to_out_0.weight")


def test_shared_adaln_block_parity():
    import mlx.core as mx

    from ernie_image_core_mlx.model.config import ErnieImageConfig
    from ernie_image_core_mlx.model.rope import ErnieImageEmbedND3
    from ernie_image_core_mlx.model.transformer import ErnieImageSharedAdaLNBlock as MxBlock
    from tests.parity._pt_reference import (
        ErnieImageEmbedND3 as PtEmbed,
    )
    from tests.parity._pt_reference import (
        ErnieImageSharedAdaLNBlock as PtBlock,
    )

    cfg = ErnieImageConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_layers=1,
        ffn_hidden_size=512,
        rope_axes_dim=(16, 24, 24),
        rope_theta=256.0,
        qk_layernorm=True,
    )
    assert cfg.head_dim == 64 and sum(cfg.rope_axes_dim) == 64

    torch.manual_seed(42)
    pt = PtBlock(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        ffn_hidden_size=cfg.ffn_hidden_size,
        eps=cfg.eps,
        qk_layernorm=True,
    )
    pt.train(False)
    mxm = MxBlock(cfg)

    pt_sd = _copy_pt_state(pt)
    flat = [(_pt_to_mx_block_rename(k), mx.array(v)) for k, v in pt_sd.items()]
    _update_mx(mxm, flat)

    B, S = 2, 48
    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((B, S, cfg.hidden_size)).astype(np.float32)
    ids_np = rng.integers(0, 32, size=(B, S, 3)).astype(np.float32)

    pt_embed = PtEmbed(dim=cfg.head_dim, theta=cfg.rope_theta, axes_dim=cfg.rope_axes_dim)
    mx_embed = ErnieImageEmbedND3(head_dim=cfg.head_dim, theta=cfg.rope_theta, axes_dim=cfg.rope_axes_dim)
    pt_freqs = pt_embed(torch.from_numpy(ids_np))
    mx_freqs = mx_embed(mx.array(ids_np))

    # temb — 6-tuple. PT expects [S, B, H]; MLX block expects broadcastable [B, 1, H].
    temb_vals = [rng.standard_normal((B, cfg.hidden_size)).astype(np.float32) for _ in range(6)]
    pt_temb = [torch.from_numpy(v).unsqueeze(0).expand(S, -1, -1).contiguous() for v in temb_vals]
    mx_temb = tuple(mx.array(v)[:, None, :] for v in temb_vals)

    with torch.no_grad():
        pt_out_sbh = pt(
            torch.from_numpy(x_np).permute(1, 0, 2),
            pt_freqs,
            pt_temb,
            attention_mask=None,
        )
        pt_out = pt_out_sbh.permute(1, 0, 2).float().numpy()

    mx_out = np.array(mxm(mx.array(x_np), freqs_cis=mx_freqs, temb=mx_temb))

    assert pt_out.shape == mx_out.shape == (B, S, cfg.hidden_size)
    diff = np.abs(pt_out - mx_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # AdaLN + FFN compound on top of attention — looser threshold than pure attention.
    assert max_abs < 5e-3, f"block parity FAIL: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
