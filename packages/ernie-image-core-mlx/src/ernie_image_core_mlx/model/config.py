"""Dataclass configs that mirror ERNIE-Image's `config.json` files verbatim.

Defaults come from the reference checkpoint — do NOT override without cross-checking
against the oracle JSONs. Wrong defaults silently ruin inference (see CLAUDE.md trap list).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ErnieImageConfig:
    """Mirror of `baidu/ERNIE-Image/transformer/config.json`."""

    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_layers: int = 36
    ffn_hidden_size: int = 12288
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    qk_layernorm: bool = True
    rope_axes_dim: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 256.0
    text_in_dim: int = 3072
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def __post_init__(self) -> None:
        assert sum(self.rope_axes_dim) == self.head_dim, (
            f"rope_axes_dim {self.rope_axes_dim} must sum to head_dim {self.head_dim}"
        )


@dataclass
class ErnieImageVaeConfig:
    """Mirror of `baidu/ERNIE-Image/vae/config.json` (AutoencoderKLFlux2)."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    layers_per_block: int = 2
    patch_size: tuple[int, int] = (2, 2)
    norm_num_groups: int = 32
    act_fn: str = "silu"
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    force_upcast: bool = True
    mid_block_add_attention: bool = True
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    sample_size: int = 1024


@dataclass
class Mistral3TextConfig:
    """Text-only subset of `baidu/ERNIE-Image/text_encoder/config.json`.

    Only the text path matters for t2i — the Pixtral vision tower is loaded
    but never called.
    """

    hidden_size: int = 3072
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9216
    vocab_size: int = 131072
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    rope_type: str = "yarn"
    rope_theta: float = 1_000_000.0
    rope_factor: float = 16.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_position_embeddings: int = 16_384


@dataclass
class ErniePipelineConfig:
    """Tuning knobs for the full pipeline. Variant-specific defaults set via
    `ErniePipelineConfig.for_variant('turbo' | 'sft')`."""

    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    shift: float = 3.0  # flow-match dynamic shift placeholder — verify from scheduler_config.json
    variant: str = "sft"  # "sft" or "turbo"

    scheduler_kwargs: dict = field(default_factory=dict)

    @classmethod
    def for_variant(cls, variant: str) -> ErniePipelineConfig:
        if variant == "turbo":
            return cls(num_inference_steps=8, guidance_scale=1.0, variant="turbo")
        if variant == "sft":
            return cls(num_inference_steps=50, guidance_scale=5.0, variant="sft")
        raise ValueError(f"Unknown variant {variant!r} (expected 'sft' or 'turbo')")
