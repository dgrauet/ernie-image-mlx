"""PT vs MLX parity for ErnieAttention + triple-axis RoPE.

Gate: needs `[parity]` extra (torch + diffusers ≥ 0.36).
Runs on the attention block in isolation — no weights needed, PT state_dict
is mirrored into MLX at test time.

Target thresholds (fp32):
    single attention pass: max_abs < 1e-4
    RoPE alone: max_abs < 1e-5
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# RoPE parity — verifies the Megatron non-interleaved variant matches exactly.
# ---------------------------------------------------------------------------


def test_rope_embed_matches_reference():
    import mlx.core as mx

    from ernie_image_core_mlx.model.rope import ErnieImageEmbedND3 as MxEmbed
    from tests.parity._pt_reference import ErnieImageEmbedND3 as PtEmbed

    head_dim = 128
    axes = (32, 48, 48)
    theta = 256

    pt_embed = PtEmbed(dim=head_dim, theta=theta, axes_dim=axes)
    mx_embed = MxEmbed(head_dim=head_dim, theta=theta, axes_dim=axes)

    rng = np.random.default_rng(0)
    ids_np = rng.integers(0, 64, size=(2, 160, 3)).astype(np.float32)

    pt_out = pt_embed(torch.from_numpy(ids_np)).detach().float().numpy()
    mx_out = np.array(mx_embed(mx.array(ids_np)))

    assert pt_out.shape == mx_out.shape == (2, 160, 1, head_dim)
    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-5, f"rope embed diverged: max_abs={max_abs:.3e}"


def test_apply_rotary_matches_reference():
    import mlx.core as mx

    from ernie_image_core_mlx.model.rope import apply_rotary_emb as mx_apply

    B, N, H, D = 2, 64, 4, 128
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((B, N, H, D)).astype(np.float32)
    freqs_np = rng.standard_normal((B, N, 1, D)).astype(np.float32)

    # Reference implementation, copied verbatim from diffusers
    def pt_apply(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        rot_dim = freqs_cis.shape[-1]
        x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
        cos_ = torch.cos(freqs_cis).to(x.dtype)
        sin_ = torch.sin(freqs_cis).to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

    pt_out = pt_apply(torch.from_numpy(x_np), torch.from_numpy(freqs_np)).float().numpy()
    mx_out = np.array(mx_apply(mx.array(x_np), mx.array(freqs_np)))

    max_abs = float(np.max(np.abs(pt_out - mx_out)))
    assert max_abs < 1e-5, f"apply_rotary_emb diverged: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# Full attention parity — PT reference built from the diffusers Attention wrapper
# ---------------------------------------------------------------------------


def _build_pt_attention(hidden_size: int, num_heads: int, eps: float):
    from tests.parity._pt_reference import ErnieImageAttention

    return ErnieImageAttention(
        query_dim=hidden_size,
        heads=num_heads,
        dim_head=hidden_size // num_heads,
        bias=False,
        out_bias=False,
        qk_norm="rms_norm",
        eps=eps,
        elementwise_affine=True,
    )


def _pt_to_mx_rename(key: str) -> str:
    """Map PT keys to MLX equivalents."""
    if key == "to_out.0.weight":
        return "to_out_0.weight"
    return key


def test_attention_forward_parity():
    import mlx.core as mx
    import mlx.utils as mx_utils

    from ernie_image_core_mlx.model.attention import ErnieAttention
    from ernie_image_core_mlx.model.config import ErnieImageConfig
    from ernie_image_core_mlx.model.rope import ErnieImageEmbedND3

    # Small config (real is 4096/32) — keeps tests fast while exercising identical code paths.
    cfg = ErnieImageConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_layers=1,
        ffn_hidden_size=512,
        rope_axes_dim=(16, 24, 24),  # sums to 64 = head_dim
        rope_theta=256.0,
        qk_layernorm=True,
    )
    assert cfg.head_dim == 64 and sum(cfg.rope_axes_dim) == 64

    torch.manual_seed(0)
    pt_attn = _build_pt_attention(cfg.hidden_size, cfg.num_attention_heads, cfg.eps)
    pt_attn.train(False)  # drop dropout etc.
    pt_sd = {k: v.detach().clone() for k, v in pt_attn.state_dict().items()}

    mx_attn = ErnieAttention(cfg)
    flat = [(_pt_to_mx_rename(k), mx.array(v.float().numpy())) for k, v in pt_sd.items()]
    mx_attn.update(mx_utils.tree_unflatten(flat))
    mx.eval(mx_attn.parameters())

    B, N = 2, 80
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, N, cfg.hidden_size)).astype(np.float32)
    ids_np = rng.integers(0, 32, size=(B, N, 3)).astype(np.float32)

    from tests.parity._pt_reference import ErnieImageEmbedND3 as PtEmbed

    pt_embed = PtEmbed(dim=cfg.head_dim, theta=cfg.rope_theta, axes_dim=cfg.rope_axes_dim)
    mx_embed = ErnieImageEmbedND3(head_dim=cfg.head_dim, theta=cfg.rope_theta, axes_dim=cfg.rope_axes_dim)

    pt_freqs = pt_embed(torch.from_numpy(ids_np))
    mx_freqs = mx_embed(mx.array(ids_np))

    with torch.no_grad():
        pt_out = (
            pt_attn(
                hidden_states=torch.from_numpy(x_np),
                image_rotary_emb=pt_freqs,
            )
            .float()
            .numpy()
        )

    mx_out = np.array(mx_attn(mx.array(x_np), freqs_cis=mx_freqs))

    assert pt_out.shape == mx_out.shape == (B, N, cfg.hidden_size)
    diff = np.abs(pt_out - mx_out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    assert max_abs < 1e-4, f"attention parity FAIL: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
