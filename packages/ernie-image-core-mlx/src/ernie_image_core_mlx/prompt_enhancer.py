"""Prompt Enhancer — Ministral3ForCausalLM wrapper for prompt expansion.

The reference `diffusers.ErnieImagePipeline` runs a 3B Ministral3 CausalLM
before the text encoder. It takes the user prompt, wraps it as JSON
``{"prompt": ..., "width": W, "height": H}`` and pushes that through a chat
template (``pe/chat_template.jinja``) that bakes in a Chinese system prompt:

    You are a professional text-to-image prompt enhancement assistant.
    You will receive a short image description from the user and the target
    generation resolution. Please expand it into a rich, detailed visual
    description to help the text-to-image model generate high-quality
    images. Output only the enhanced description, without any explanation
    or prefix.

Generation params (from ``pipeline_ernie_image.py``):
    temperature=0.6, top_p=0.95, max_new_tokens=2048

Output: slice off the prompt tokens and ``tokenizer.decode(..., skip_special_tokens=True)``.

Backbone-wise the PE is architecturally identical to the text encoder
(same 26-layer Ministral3, same GQA 32/8, same YaRN RoPE) — only the forward
differs: full stack + tied ``lm_head`` for autoregressive decode, whereas the
text encoder stops at ``hidden_states[-2]``. We therefore instantiate
``mlx_lm.models.ministral3.Model`` directly (it IS the CausalLM), and route
generation through ``mlx_lm.generate.generate_step`` which owns the KV cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ernie_image_core_mlx.loader.weights import resolve_weights_dir
from ernie_image_core_mlx.model.config import Mistral3TextConfig

DEFAULT_PE_REPO_ID = "dgrauet/ernie-image-pe-mlx-q4"


def _ministral3_args_dict(cfg: Mistral3TextConfig) -> dict:
    """Shape the ``Mistral3TextConfig`` into the dict mlx-lm's ``ministral3.ModelArgs`` expects."""
    rope_parameters = {
        "rope_type": cfg.rope_type,
        "rope_theta": cfg.rope_theta,
        "factor": cfg.rope_factor,
        "beta_fast": cfg.rope_beta_fast,
        "beta_slow": cfg.rope_beta_slow,
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


def _pe_config_from_json(blob: dict) -> Mistral3TextConfig:
    rope = blob.get("rope_parameters", {})
    return Mistral3TextConfig(
        hidden_size=blob["hidden_size"],
        num_hidden_layers=blob["num_hidden_layers"],
        num_attention_heads=blob["num_attention_heads"],
        num_key_value_heads=blob["num_key_value_heads"],
        head_dim=blob["head_dim"],
        intermediate_size=blob["intermediate_size"],
        vocab_size=blob["vocab_size"],
        max_position_embeddings=blob.get("max_position_embeddings", 262144),
        rms_norm_eps=blob.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=blob.get("tie_word_embeddings", True),
        rope_type=rope.get("rope_type", "yarn"),
        rope_theta=rope.get("rope_theta", 1_000_000.0),
        rope_factor=rope.get("factor", 16.0),
        rope_beta_fast=rope.get("beta_fast", 32.0),
        rope_beta_slow=rope.get("beta_slow", 1.0),
        rope_original_max_position_embeddings=rope.get("original_max_position_embeddings", 16384),
    )


class PromptEnhancer(nn.Module):
    """Ministral3 CausalLM wrapper that expands short prompts via chat template."""

    def __init__(self, cfg: Mistral3TextConfig, tokenizer):
        super().__init__()
        from mlx_lm.models import ministral3

        self.cfg = cfg
        self.tokenizer = tokenizer
        args = ministral3.ModelArgs(**_ministral3_args_dict(cfg))
        self.model = ministral3.Model(args)

    def sanitize(self, weights: dict) -> dict:
        return self.model.sanitize(weights)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_PE_REPO_ID,
        *,
        local_dir: str | None = None,
    ) -> PromptEnhancer:
        """Load a converted MLX PE directory produced by ``mlx-forge convert ernie-image-pe``.

        Expected layout (mirrors the image-side recipe):
            <dir>/
            ├── pe.safetensors            # Ministral3ForCausalLM weights (possibly quantized)
            ├── pe_config.json            # HF config.json copied verbatim
            ├── chat_template.jinja       # from baidu's pe/
            ├── tokenizer.json            # from baidu's pe_tokenizer/
            ├── tokenizer_config.json     # same
            └── quantize_config.json      # optional, written when --quantize was used
        """
        weights_dir = resolve_weights_dir(repo_id=repo_id, local_dir=local_dir)

        # Accept two layouts for flexibility: the MLX recipe output (flat, `pe_config.json`)
        # and a raw HF checkpoint subfolder (nested `pe/config.json`).
        cfg_path = weights_dir / "pe_config.json"
        safetensors_path = weights_dir / "pe.safetensors"
        tokenizer_source = weights_dir
        if not cfg_path.exists() and (weights_dir / "pe" / "config.json").exists():
            pe_sub = weights_dir / "pe"
            cfg_path = pe_sub / "config.json"
            safetensors_path = pe_sub / "model.safetensors"
            tokenizer_source = pe_sub

        with open(cfg_path) as f:
            blob = json.load(f)
        cfg = _pe_config_from_json(blob)

        tokenizer = cls._load_tokenizer(tokenizer_source)
        pe = cls(cfg, tokenizer)

        raw = mx.load(str(safetensors_path))
        # Strip the recipe's "pe." prefix when present; raw HF checkpoints already
        # ship keys like `model.layers.*` / `lm_head.weight`.
        prefix = "pe."
        stripped = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in raw.items()}

        # Handle optional quantization. Match the text_encoder's scope — whole
        # stack — since every Linear in the PE is a transformer block projection.
        q_cfg_path = weights_dir / "quantize_config.json"
        if q_cfg_path.exists() and any(k.endswith(".scales") for k in stripped):
            with open(q_cfg_path) as f:
                q_blob = json.load(f)
            q = q_blob.get("quantization", q_blob)
            skip = set(q.get("skip_components", []))
            if "pe" not in skip:
                nn.quantize(
                    pe.model,
                    group_size=int(q.get("group_size", 64)),
                    bits=int(q.get("bits", 8)),
                )

        sanitized = pe.sanitize(stripped)
        from mlx.utils import tree_unflatten

        pe.model.update(tree_unflatten(list(sanitized.items())))
        return pe

    @staticmethod
    def _load_tokenizer(source_dir: Path):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(source_dir))
        # Baidu's pe_tokenizer_config.json declares pad=`<pad>` but the Pixtral
        # tokenizer ships with pad_token=None. Inject to match the PE pipeline
        # expectations (pad_token_id=11 per pe/config.json).
        if tok.pad_token is None:
            tok.pad_token = "<pad>"
        return tok

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _apply_chat_template(self, prompt: str, width: int, height: int) -> str:
        """Reference does ``json.dumps({"prompt": p, "width": W, "height": H}, ensure_ascii=False)``
        then wraps it in ``[INST]...[/INST]``. The Jinja template prepends BOS and
        the Chinese system prompt automatically.
        """
        user_content = json.dumps({"prompt": prompt, "width": width, "height": height}, ensure_ascii=False)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=False,
        )

    def enhance(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        """Expand ``prompt`` into a richer visual description using the PE CausalLM.

        Matches reference sampling (T=0.6, top_p=0.95, max=2048). Pass ``seed``
        for reproducibility — the reference is non-deterministic by default.
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)

        templated = self._apply_chat_template(prompt, width, height)
        # ``apply_chat_template`` returns the full formatted string including BOS;
        # encode WITHOUT adding another BOS.
        input_ids = self.tokenizer.encode(templated, add_special_tokens=False)
        prompt_arr = mx.array(input_ids, dtype=mx.int32)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        for token, _ in generate_step(prompt_arr, self.model, max_tokens=max_new_tokens, sampler=sampler):
            # ``generate_step`` may yield either an mx.array scalar or a Python
            # int depending on mlx-lm version — normalize both to int.
            tok_id = token.item() if hasattr(token, "item") else int(token)
            if tok_id == eos_id:
                break
            generated.append(tok_id)

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.enhance(prompt, **kwargs)
