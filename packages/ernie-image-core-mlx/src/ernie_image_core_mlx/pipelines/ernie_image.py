"""ErnieImagePipeline — text → image orchestration.

One pipeline class, two checkpoints:
  - baidu/ERNIE-Image        → SFT, ~50 steps, guidance ≈ 5
  - baidu/ERNIE-Image-Turbo  → 8-step distilled, guidance = 1.0

Components assumed to exist on the weights directory (see the `mlx-forge
ernie-image` recipe):
    transformer.safetensors        — ErnieImageTransformer2DModel
    text_encoder.safetensors       — Mistral3 text path (vision_tower dropped)
    vae.safetensors                — AutoencoderKLFlux2 (channels-last convs)
    transformer_config.json        — mirrors diffusers transformer config
    vae_config.json                — mirrors diffusers VAE config
    text_encoder_config.json       — Mistral3 config (text_config + vision_config)
    scheduler_scheduler_config.json — flow-match scheduler params

The Prompt Enhancer (`Ministral3ForCausalLM` → `pe/` subfolder) is skipped in
v0. Pass a pre-expanded prompt for best results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler
from mlx_arsenal.spatial import pixel_shuffle

from ernie_image_core_mlx.loader.weights import (
    resolve_weights_dir,
)
from ernie_image_core_mlx.model.config import (
    ErnieImageConfig,
    ErnieImageVaeConfig,
    ErniePipelineConfig,
    Mistral3TextConfig,
)
from ernie_image_core_mlx.model.transformer import ErnieImageTransformer2DModel
from ernie_image_core_mlx.model.vae import AutoencoderKLFlux2
from ernie_image_core_mlx.text_encoders.mistral3 import Mistral3TextEncoder


@dataclass
class PipelineOutput:
    images: list  # list[PIL.Image.Image]; typed loosely to avoid a hard Pillow dep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _config_from_transformer_json(blob: dict) -> ErnieImageConfig:
    return ErnieImageConfig(
        hidden_size=blob["hidden_size"],
        num_attention_heads=blob["num_attention_heads"],
        num_layers=blob["num_layers"],
        ffn_hidden_size=blob["ffn_hidden_size"],
        in_channels=blob["in_channels"],
        out_channels=blob["out_channels"],
        patch_size=blob.get("patch_size", 1),
        qk_layernorm=blob.get("qk_layernorm", True),
        rope_axes_dim=tuple(blob["rope_axes_dim"]),
        rope_theta=float(blob["rope_theta"]),
        text_in_dim=blob["text_in_dim"],
        eps=blob.get("eps", 1e-6),
    )


def _config_from_vae_json(blob: dict) -> ErnieImageVaeConfig:
    return ErnieImageVaeConfig(
        in_channels=blob.get("in_channels", 3),
        out_channels=blob.get("out_channels", 3),
        latent_channels=blob["latent_channels"],
        block_out_channels=tuple(blob["block_out_channels"]),
        down_block_types=tuple(blob["down_block_types"]),
        up_block_types=tuple(blob["up_block_types"]),
        layers_per_block=blob.get("layers_per_block", 2),
        patch_size=tuple(blob.get("patch_size", (2, 2))),
        norm_num_groups=blob.get("norm_num_groups", 32),
        use_quant_conv=blob.get("use_quant_conv", True),
        use_post_quant_conv=blob.get("use_post_quant_conv", True),
        force_upcast=blob.get("force_upcast", True),
        mid_block_add_attention=blob.get("mid_block_add_attention", True),
    )


def _config_from_text_encoder_json(blob: dict) -> Mistral3TextConfig:
    text = blob["text_config"]
    rope = text.get("rope_parameters", {})
    return Mistral3TextConfig(
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        intermediate_size=text["intermediate_size"],
        vocab_size=text["vocab_size"],
        max_position_embeddings=text.get("max_position_embeddings", 262144),
        rms_norm_eps=text.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=text.get("tie_word_embeddings", True),
        rope_type=rope.get("rope_type", "yarn"),
        rope_theta=rope.get("rope_theta", 1_000_000.0),
        rope_factor=rope.get("factor", 16.0),
        rope_beta_fast=rope.get("beta_fast", 32.0),
        rope_beta_slow=rope.get("beta_slow", 1.0),
        rope_original_max_position_embeddings=rope.get("original_max_position_embeddings", 16384),
    )


def _downsample_factor(vae_cfg: ErnieImageVaeConfig) -> int:
    """Total image→DiT-latent downsample.

    Example: `block_out_channels=(128,256,512,512)` gives 4 down-blocks with
    `add_downsample=True` on all but the last → 3 stride-2 stages → ×8.
    Combined with `patch_size=2` pixel-unshuffle → ×16 overall.
    """
    num_down = len(vae_cfg.block_out_channels) - 1
    vae_stride = 2**num_down
    patch = vae_cfg.patch_size[0]
    return vae_stride * patch


def _unpack_dit_to_vae(z_dit_ch_second: mx.array, patch: int) -> mx.array:
    """DiT output `(B, 128, H, W)` → VAE input `(B, H*patch, W*patch, 32)` via pixel shuffle."""
    z_ch_last = z_dit_ch_second.transpose(0, 2, 3, 1)
    return pixel_shuffle(z_ch_last, upscale_factor=patch)


def _pad_concat_text(text_hiddens: list[mx.array]) -> mx.array:
    """Right-pad text embeddings to a common length and concat along the batch.

    Input: list of `(B_i, T_i, H)` arrays (typically uncond + cond, each B=1).
    Output: `(sum B_i, Tmax, H)` with zeros in the padding region.
    """
    Tmax = max(t.shape[1] for t in text_hiddens)
    H = text_hiddens[0].shape[-1]
    padded = []
    for t in text_hiddens:
        B_i, T_i, _ = t.shape
        if T_i == Tmax:
            padded.append(t)
            continue
        pad = mx.zeros((B_i, Tmax - T_i, H), dtype=t.dtype)
        padded.append(mx.concatenate([t, pad], axis=1))
    return mx.concatenate(padded, axis=0)


def _tensor_to_pil_image(x: mx.array):
    """`(3, H, W)` float in [-1, 1] → PIL.Image RGB."""
    import numpy as np
    from PIL import Image

    arr = np.array(x)
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    arr = (arr.transpose(1, 2, 0) * 255.0).astype("uint8")
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ErnieImagePipeline:
    def __init__(
        self,
        transformer: ErnieImageTransformer2DModel,
        vae: AutoencoderKLFlux2,
        text_encoder: Mistral3TextEncoder,
        scheduler: FlowMatchEulerDiscreteScheduler,
        config: ErniePipelineConfig,
        *,
        tokenizer=None,
        weights_dir: Path | None = None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.config = config
        self.tokenizer = tokenizer
        self.weights_dir = weights_dir

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "baidu/ERNIE-Image-Turbo",
        *,
        variant: str | None = None,
        local_dir: str | None = None,
    ) -> ErnieImagePipeline:
        variant = variant or ("turbo" if "turbo" in repo_id.lower() else "sft")
        weights_dir = resolve_weights_dir(repo_id=repo_id, local_dir=local_dir)

        tf_cfg = _config_from_transformer_json(_load_json(weights_dir / "transformer_config.json"))
        vae_cfg = _config_from_vae_json(_load_json(weights_dir / "vae_config.json"))
        text_cfg = _config_from_text_encoder_json(_load_json(weights_dir / "text_encoder_config.json"))

        transformer = ErnieImageTransformer2DModel(tf_cfg)
        vae = AutoencoderKLFlux2(vae_cfg)
        text_encoder = Mistral3TextEncoder(text_cfg)

        import mlx.nn as mx_nn
        from mlx.utils import tree_unflatten

        def _load_stripped(path: Path, prefix: str) -> dict:
            """Load a safetensors file produced by `mlx-forge convert ernie-image`
            and strip the `<component>.` prefix the recipe adds to every key."""
            raw = mx.load(str(path))
            p = f"{prefix}."
            stripped: dict = {}
            for k, v in raw.items():
                if k.startswith(p):
                    stripped[k[len(p) :]] = v
                else:
                    stripped[k] = v
            return stripped

        def _has_quantized_keys(weights: dict) -> bool:
            """Safetensors produced with `--quantize` ship `.scales` / `.biases`
            alongside the regular `.weight` for every quantized Linear."""
            return any(k.endswith(".scales") for k in weights)

        tf_w = _load_stripped(weights_dir / "transformer.safetensors", "transformer")
        vae_w = _load_stripped(weights_dir / "vae.safetensors", "vae")
        te_w = _load_stripped(weights_dir / "text_encoder.safetensors", "text_encoder")

        # PyTorch BatchNorm ships a scalar `num_batches_tracked` — we don't model it.
        vae_w.pop("bn.num_batches_tracked", None)

        # If the recipe quantized Linear weights, swap `nn.Linear` → `nn.QuantizedLinear`
        # BEFORE calling `update`, so the `.scales` / `.biases` keys land in the right
        # places. Read group_size/bits from `quantize_config.json` (written by the recipe).
        q_cfg_path = weights_dir / "quantize_config.json"
        if q_cfg_path.exists():
            q_blob = _load_json(q_cfg_path)["quantization"]
            group_size = int(q_blob.get("group_size", 64))
            bits = int(q_blob.get("bits", 8))
            skip = set(q_blob.get("skip_components", []))
            if "transformer" not in skip and _has_quantized_keys(tf_w):
                # Match the recipe's predicate: only block Linears ≥ 256×256 are quantized.
                def _tf_class_predicate(_path: str, m) -> bool:
                    if not isinstance(m, mx_nn.Linear):
                        return False
                    # Skip projections outside the block stack (they're not quantized at conversion time).
                    if not _path.startswith("layers."):
                        return False
                    w = m.weight
                    return w.shape[0] >= 256 and w.shape[1] >= 256

                mx_nn.quantize(
                    transformer,
                    group_size=group_size,
                    bits=bits,
                    class_predicate=_tf_class_predicate,
                )
            if "text_encoder" not in skip and _has_quantized_keys(te_w):
                # mlx-lm models handle `nn.quantize` across the whole text tower.
                mx_nn.quantize(text_encoder, group_size=group_size, bits=bits)

        transformer.update(tree_unflatten(list(tf_w.items())))
        vae.update(tree_unflatten(list(vae_w.items())))

        # Text-encoder path:
        #   1. saved keys look like `language_model.<rest>` after prefix strip
        #   2. mlx-lm's `mistral3.Model.sanitize` expects that exact shape and
        #      returns the same layout (it just drops vision_* subtrees)
        #   3. our MLX wrapper nests the mistral3.Model at `self.model`, so we
        #      prepend `model.` AFTER sanitize before calling `.update`
        te_w_sanitized = text_encoder.sanitize(te_w)
        te_w_nested = {f"model.{k}": v for k, v in te_w_sanitized.items()}
        text_encoder.update(tree_unflatten(list(te_w_nested.items())))

        sched_cfg_path = weights_dir / "scheduler_scheduler_config.json"
        shift = 3.0
        if sched_cfg_path.exists():
            sched_blob = _load_json(sched_cfg_path)
            shift = float(sched_blob.get("shift", shift))
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)

        pipe_cfg = ErniePipelineConfig.for_variant(variant)
        pipe_cfg.shift = shift

        tokenizer = cls._try_load_tokenizer(weights_dir)

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            config=pipe_cfg,
            tokenizer=tokenizer,
            weights_dir=weights_dir,
        )

    @staticmethod
    def _try_load_tokenizer(weights_dir: Path):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None
        try:
            tok = AutoTokenizer.from_pretrained(str(weights_dir))
        except Exception:
            return None
        # Baidu's tokenizer_config.json declares `pad_token: "<pad>"` but the
        # community Pixtral tokenizer ships with pad_token=None. Inject it so
        # batched encoding with padding works end-to-end.
        if tok.pad_token is None:
            tok.pad_token = "<pad>"
        return tok

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _tokenize(self, prompts: list[str]) -> tuple[mx.array, mx.array]:
        """Tokenize a batch of prompts with BOS prepend + BOS-fallback.

        The community Pixtral tokenizer does NOT auto-prepend `<s>` even with
        `add_special_tokens=True`. Every other token matches HF Ministral3's
        forward exactly, but the first token's state diverges catastrophically
        after layer 2 (max_abs ~6 vs ~0.05 on the rest). Prepending BOS moves
        the divergence onto the BOS token itself, where the DiT has been
        trained to ignore it anyway, and leaves every content token in parity
        with the PyTorch reference.
        """
        tok = self.tokenizer(
            prompts,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=512,
        )
        raw_ids = tok["input_ids"]
        bos = self.tokenizer.bos_token_id or 1
        per_prompt = [[bos] + r if (not r or r[0] != bos) else r for r in raw_ids]
        if not any(per_prompt):
            per_prompt = [[bos] for _ in per_prompt]

        max_len = max(len(r) for r in per_prompt)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = [r + [pad_id] * (max_len - len(r)) for r in per_prompt]
        mask = [[1] * len(r) + [0] * (max_len - len(r)) for r in per_prompt]

        input_ids = mx.array(ids).astype(mx.int32)
        attn_mask = mx.array(mask).astype(mx.int32)
        lens = attn_mask.sum(axis=1)
        return input_ids, lens

    def _encode_prompts(
        self, prompt: str | list[str], negative_prompt: str | None
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        """Returns (cond_hidden, cond_lens, uncond_hidden|None, uncond_lens|None)."""
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available. Install `transformers` and ensure the "
                "weights_dir contains tokenizer.json (bundled by the ernie-image recipe)."
            )
        if isinstance(prompt, str):
            prompt = [prompt]

        cond_ids, cond_lens = self._tokenize(prompt)
        cond_hidden = self.text_encoder.encode(cond_ids)

        uncond_hidden = uncond_lens = None
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            uncond_ids, uncond_lens = self._tokenize(negative_prompt)
            uncond_hidden = self.text_encoder.encode(uncond_ids)

        return cond_hidden, cond_lens, uncond_hidden, uncond_lens

    def __call__(
        self,
        prompt: str | list[str],
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = "",
        seed: int | None = None,
    ) -> PipelineOutput:
        cfg = self.config
        tf_cfg: ErnieImageConfig = self.transformer.cfg
        vae_cfg: ErnieImageVaeConfig = self.vae.cfg

        steps = num_inference_steps or cfg.num_inference_steps
        cfg_scale = guidance_scale if guidance_scale is not None else cfg.guidance_scale
        use_cfg = cfg_scale > 1.0 and negative_prompt is not None

        cond_hidden, cond_lens, uncond_hidden, uncond_lens = self._encode_prompts(
            prompt, negative_prompt if use_cfg else None
        )
        B = cond_hidden.shape[0]

        down = _downsample_factor(vae_cfg)
        assert height % down == 0 and width % down == 0, (
            f"height/width must be multiples of {down} (got {height}x{width})"
        )
        H_lat, W_lat = height // down, width // down

        if seed is not None:
            mx.random.seed(seed)
        # Match the transformer's working dtype (bfloat16 post-conversion) so the
        # DiT doesn't have to upcast every forward. Timesteps also use this dtype.
        dtype = cond_hidden.dtype
        latents = mx.random.normal((B, tf_cfg.in_channels, H_lat, W_lat)).astype(dtype)

        # Reference pipeline (`diffusers.pipelines.ernie_image`) uses a custom
        # linear sigma schedule `linspace(1, 0, N+1)[:-1]` instead of the
        # scheduler default.
        sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]
        self.scheduler.set_timesteps(steps, sigmas=sigmas)

        # mlx-arsenal's FlowMatchEulerDiscreteScheduler appends `1.0` as its
        # terminal sigma; diffusers appends `0.0`. With `1.0` the final Euler
        # step would re-noise instead of landing at the clean latent. Override.
        fixed_sigmas = np.array(self.scheduler.sigmas.tolist(), dtype=np.float32)
        fixed_sigmas[-1] = 0.0
        self.scheduler.sigmas = mx.array(fixed_sigmas)

        # When CFG is on, the reference batches `[uncond, cond]` into a single
        # transformer pass. We match that for compute efficiency.
        if use_cfg:
            text_bth = _pad_concat_text([uncond_hidden, cond_hidden])
            text_lens_cat = mx.concatenate([uncond_lens, cond_lens], axis=0)
        else:
            text_bth = cond_hidden
            text_lens_cat = cond_lens

        for t in self.scheduler.timesteps:
            t_scalar = mx.array([float(t)], dtype=dtype)
            if use_cfg:
                latent_in = mx.concatenate([latents, latents], axis=0)
                t_batch = mx.broadcast_to(t_scalar, (B * 2,))
            else:
                latent_in = latents
                t_batch = mx.broadcast_to(t_scalar, (B,))

            pred = self.transformer(
                latent_in,
                timestep=t_batch,
                text_bth=text_bth,
                text_lens=text_lens_cat,
            )

            if use_cfg:
                pred_uncond, pred_cond = mx.split(pred, 2, axis=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            # Scheduler keeps sigmas in fp32; step() promotes our bf16 latent
            # to fp32. Cast back so subsequent DiT passes see the same dtype
            # as the first.
            latents = self.scheduler.step(pred, t_scalar, latents).astype(dtype)

        # DiT output (channels-second) → apply latent BN inverse on the 128-channel
        # axis → transpose to channels-last → pixel-shuffle unpack → VAE decode.
        # `force_upcast=True` in the VAE config means the reference runs the VAE
        # in fp32 even when the DiT is in bfloat16 — match that here.
        latents_nhwc = latents.transpose(0, 2, 3, 1).astype(mx.float32)
        latents_nhwc = self.vae.bn.apply_inverse(latents_nhwc)

        patch = vae_cfg.patch_size[0]
        vae_latents = pixel_shuffle(latents_nhwc, upscale_factor=patch)

        image = self.vae.decode(vae_latents)

        image_chfirst = image.transpose(0, 3, 1, 2)
        pil_images = [_tensor_to_pil_image(image_chfirst[i]) for i in range(B)]
        return PipelineOutput(images=pil_images)
