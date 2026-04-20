"""Mistral3 text encoder wrapper — delegates to mlx-lm.

ERNIE-Image's `text_encoder/config.json` declares `Mistral3Model` (multimodal),
but for t2i we only need the text path. mlx-lm ships a `mistral3` module that
wraps `ministral3` under the hood and provides a `sanitize` method which drops
`vision_tower.*` / `multi_modal_projector.*` keys at load time — exactly what
we need.

Our wrapper exposes a single `encode(input_ids)` method that returns the DiT's
expected input: `[B, L, text_in_dim=3072]` — the last hidden state from the
language-model stack (post final RMSNorm, pre lm_head).
"""

from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn

from ernie_image_core_mlx.model.config import Mistral3TextConfig


def _mistral3_text_config_dict(cfg: Mistral3TextConfig) -> dict:
    """Shape our dataclass into the dict mlx-lm expects inside `text_config`."""
    rope_parameters = {
        "rope_type": cfg.rope_type,
        "rope_theta": cfg.rope_theta,
        "factor": cfg.rope_factor,
        "beta_fast": cfg.rope_beta_fast,
        "beta_slow": cfg.rope_beta_slow,
        # mlx-lm's YaRN path also reads these three — provide safe defaults matching
        # the baidu/ERNIE-Image text_encoder/config.json (even when unused).
        "llama_4_scaling_beta": 0.1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": cfg.rope_original_max_position_embeddings,
    }
    return {
        "model_type": "ministral3",
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "rope_parameters": rope_parameters,
    }


class Mistral3TextEncoder(nn.Module):
    """Thin adapter that exposes `.encode(input_ids) -> (B, L, text_in_dim)`."""

    def __init__(self, cfg: Mistral3TextConfig):
        super().__init__()
        from mlx_lm.models import mistral3

        self.cfg = cfg
        args = mistral3.ModelArgs(
            model_type="mistral3",
            text_config=_mistral3_text_config_dict(cfg),
        )
        self.model = mistral3.Model(args)

    def sanitize(self, weights: dict) -> dict:
        """Strip vision_tower / multi_modal_projector. Delegates to mlx-lm."""
        return self.model.sanitize(weights)

    def encode(self, input_ids: mx.array) -> mx.array:
        """Return `hidden_states[-2]` in transformers terms.

        HF's `hidden_states` tuple is indexed by the INPUT to each layer:
            hs[i] = input to layer i = output of layer (i-1) for i > 0.
        With N_layers = 26 there are 27 entries. `hs[-1]` is the post-final-norm
        output; `hs[-2]` is the INPUT to the last layer, i.e. the output of
        the first 25 layers. The reference ERNIE-Image `encode_prompt` uses
        `outputs.hidden_states[-2]` — so we stop ONE layer short of the full stack.

        Shape: `(B, L, cfg.hidden_size)` = `(B, L, 3072)` at production scale.
        """
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.ministral3 import _get_llama_4_attn_scale

        lm = self.model.language_model.model
        h = lm.embed_tokens(input_ids)
        cache = [None] * len(lm.layers)

        fa_mask = create_attention_mask(h, cache[lm.fa_idx]) if lm.fa_idx is not None else None
        swa_mask = None
        if lm.swa_idx is not None:
            swa_mask = create_attention_mask(h, cache[lm.swa_idx], window_size=lm.sliding_window)

        attn_scale = _get_llama_4_attn_scale(
            input_ids.shape[1],
            0,
            lm.args.rope_parameters["llama_4_scaling_beta"],
            lm.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Apply the first N-1 layers; stop before the last one to match hs[-2].
        for layer, c in zip(lm.layers[:-1], cache[:-1]):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, attn_scale, mask, cache=c)

        return h

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.encode(input_ids)
