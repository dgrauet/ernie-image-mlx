"""Smoke test for the Mistral3 text-encoder wrapper.

No weights, no PT reference — just verifies that mlx-lm is reachable, the
config plumbing is correct, and the output shape matches the DiT's expectation.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_mistral3_text_encoder_shape():
    pytest.importorskip("mlx_lm")
    import mlx.core as mx

    from ernie_image_core_mlx.model.config import Mistral3TextConfig
    from ernie_image_core_mlx.text_encoders.mistral3 import Mistral3TextEncoder

    cfg = Mistral3TextConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,
    )
    enc = Mistral3TextEncoder(cfg)

    rng = np.random.default_rng(0)
    input_ids = mx.array(rng.integers(0, cfg.vocab_size, size=(2, 16)))
    out_np = np.array(enc.encode(input_ids))

    assert out_np.shape == (2, 16, cfg.hidden_size)
    assert np.isfinite(out_np).all()
