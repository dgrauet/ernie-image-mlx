# CLAUDE.md — ernie-image-mlx

## Project overview

Pure MLX port of Baidu ERNIE-Image (8B text-to-image DiT) for Apple-Silicon inference.

Reference:
- HF: https://huggingface.co/baidu/ERNIE-Image
- HF (Turbo): https://huggingface.co/baidu/ERNIE-Image-Turbo
- Repo: https://github.com/baidu/ERNIE-Image (only contains an `infer_demo.py`)
- Diffusers classes (v0.36.0+): `ErnieImagePipeline`, `ErnieImageTransformer2DModel`, `AutoencoderKLFlux2`

## Port plan (follows `mlx-porting` skill)

All 7 steps done. End-to-end inference produces clean images — sample in `docs/example_dog.png`.

1. **[done]** Step 1 — Read reference. `diffusers/main` sources vendored in `tests/parity/_pt_reference.py`.
2. **[done]** Step 2 — Scaffold.
3. **[done]** Step 3a — DiT. 36-layer stack + shared AdaLN + GeGLU FFN + triple-axis RoPE + qk_layernorm. Full-model fp32 parity max_abs 3.1e-6; single block at full scale (hidden=4096, 32 heads) vs real PT weights max_abs 3.5e-5.
4. **[done]** Step 3b — Text encoder wrapper delegating to `mlx_lm.models.mistral3` (text path only; vision tower dropped via `model.sanitize`). Returns `hidden_states[-2]` = output of layer `N-1` before the final RMSNorm, matching HF convention exactly.
5. **[done]** Step 3c — VAE `AutoencoderKLFlux2`. All modules ported; encoder max_abs 1.7e-6, decoder 6.7e-6.
6. **[done]** Step 4 — `mlx-forge ernie-image` recipe. Six variants (fp16/int8/int4 × SFT/Turbo) convert cleanly and load end-to-end.
7. **[done]** Step 5 — 23 tests green. 11 parity (rope, attention, ffn, AdaLN-continuous, full block, full model, ResnetBlock2D, VAE attention, VAE encoder, VAE decoder, VAE round-trip) + 12 smoke.
8. **[done]** Step 6 — Pipeline end-to-end validated with `一只黑白相间的中华田园犬` on Turbo 8-step: clean dog image rendered in 45 s.
9. **[done]** Step 7 — int4/int8 quantization via `mlx-forge convert --quantize --bits {4,8}`. Loader auto-swaps `nn.Linear` → `nn.QuantizedLinear` before `update`, class-predicate scoped to `layers.*` weights ≥ 256×256.

## Bug cascade that was burned through

Eight distinct bugs surfaced between scaffold and working output, in the order they were isolated:

1. **Scheduler sigmas ascending by default** — `set_timesteps(N)` without `sigmas=` gave a noising schedule. Fix: pass `sigmas=linspace(1.0, 0.0, N+1)[:-1]` explicitly.
2. **Terminal sigma mismatch** — `mlx_arsenal.FlowMatchEulerDiscreteScheduler` appends `1.0` after `set_timesteps`; `diffusers` appends `0.0`. The last Euler step re-noises instead of landing on clean. Override in the pipeline.
3. **BN inverse epsilon** — the VAE's stored `batch_norm_eps=1e-4` is a training-time value; the reference *pipeline* hard-codes `1e-5` for the inverse. Hardcoded `_LatentBN._INVERSE_EPS = 1e-5`.
4. **Text-encoder off-by-one** — HF's `outputs.hidden_states[-2]` is the INPUT to the last layer, i.e. the output after `N-1` layers. My wrapper was applying all N. Fix: `for layer in lm.layers[:-1]`.
5. **Dtype leak scheduler → DiT** — `scheduler.step()` multiplies bf16 latents by an fp32 scalar, promoting the result to fp32. The next DiT forward then mixed fp32 input with bf16 weights. Cast back to model dtype after each step.
6. **Tekken tokenizer silently drops BOS** — `pixtral-12b` (and Mistral 3 / Ministral 3) do NOT auto-prepend `<s>` even with `add_special_tokens=True`. Without BOS, token-0 diverges from HF by ~100× starting at layer 2. Explicit prepend in `_tokenize`.
7. **`mlx_arsenal.pixel_shuffle` channel-axis order wrong** ← **root cause of the checkerboard.** The implementation reshaped `(B, H, W, r, r, oc)` but PT's convention is `(B, H, W, oc, r, r)`. Output channels were scrambled, the VAE decoder saw OOD latents, all spatial patterns emerged as checker. Fixed in mlx-arsenal commit `726fa74` and a PT-parity regression test (`TestPixelShufflePyTorchParity`) added to prevent re-occurrence.
8. **pip editable shadow** — site-packages had an older `mlx-arsenal==0.2.1` pinned, while `mlx_arsenal.__file__` resolved to the local editable path. `inspect.getsource` showed the fix, tests passed, but the pipeline still ran the stale code. `pip uninstall && pip install -e .` resolved it.

The first six were caught by the three-test diagnostic in the mlx-porting skill's pitfall #7. #7 itself was the new trap this port surfaced and contributed back to the skill. #8 became a new feedback memory (`feedback_pip_editable_install_stale.md`).

## Remaining optional work

- **Upstream diffusers** — `ErnieImageTransformer2DModel` landed on diffusers `main` but isn't in a numbered release yet. When it ships, replace the vendored `tests/parity/_pt_reference.py` classes with the upstream import.

## Prompt Enhancer — done

- Wrapper: `prompt_enhancer.py` → delegates to `mlx_lm.models.ministral3.Model` (the full Ministral3ForCausalLM; same 26-layer backbone as the text encoder + tied lm_head). Chat template + JSON-wrapped payload + `mlx_lm.generate_step` with `temperature=0.6, top_p=0.95, max_new_tokens=2048` — matches diffusers verbatim.
- Recipe: separate `mlx-forge convert ernie-image-pe` (not bundled with image variants — PE is identical across Turbo/SFT so hosting once as `dgrauet/ernie-image-pe-mlx[-q4]` avoids ~7 GB duplication per image repo).
- Quantization: int4 via `--quantize --bits 4` → ~1.8 GB (from 7 GB fp16). Quantizes both block Linears AND `embed_tokens` (MLX's `QuantizedEmbedding.as_linear` handles the tied lm_head path natively). Skipping `embed_tokens` would leave ~768 MB of unquantized vocab table on disk.
- Recipe drops the redundant `lm_head.weight` at conversion time (`tie_word_embeddings=true` → mlx-lm's sanitize drops it at load, but dropping on disk saves ~768 MB fp16).
- Pipeline integration: `ErnieImagePipeline.from_pretrained(..., pe_repo_id="dgrauet/ernie-image-pe-mlx-q4")`, runtime `pipe(..., use_pe=True)`. `PipelineOutput.revised_prompts` exposes the expansion so CLI can print it.
- CLI: `--no-pe`, `--pe-repo-id`, `--pe-local-dir`, `--pe-seed`. Revised prompt printed to stdout when PE ran.

## Configs (oracle — do not deviate)

### Transformer (`transformer/config.json`)

```json
{
  "_class_name": "ErnieImageTransformer2DModel",
  "eps": 1e-06,
  "ffn_hidden_size": 12288,
  "hidden_size": 4096,
  "in_channels": 128,
  "num_attention_heads": 32,
  "num_layers": 36,
  "out_channels": 128,
  "patch_size": 1,
  "qk_layernorm": true,
  "rope_axes_dim": [32, 48, 48],
  "rope_theta": 256,
  "text_in_dim": 3072
}
```

- head_dim = 4096 / 32 = **128** — and 32 + 48 + 48 = 128 → triple-axis RoPE covers the full head, not a partial slice.
- `rope_theta=256` is unusually small (common is 10000). Probably intentional for image-domain RoPE — match exactly, do not "correct" it.
- `qk_layernorm=true` → add `RMSNorm(head_dim)` on Q and K after projection, before attention. Use `mx.fast.rms_norm`.
- `text_in_dim=3072` matches Mistral3 text hidden_size — text embeds fed directly, no extra projection assumed (verify in source).

### VAE (`vae/config.json`)

```json
{
  "_class_name": "AutoencoderKLFlux2",
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
  "latent_channels": 32,
  "patch_size": [2, 2],
  "norm_num_groups": 32,
  "act_fn": "silu",
  "use_quant_conv": true,
  "use_post_quant_conv": true,
  "force_upcast": true,
  "mid_block_add_attention": true
}
```

- Image → latent stride = 8 (4 down blocks × 2) × 2 (patch) = **×16 downsample** → 1024 image → 64×64 latent
- Latent packs 4 patch positions into 32 ch → DiT sees 32×4 = **128 channels** at patch_size=1 ✅
- `force_upcast=true` — run VAE in fp32 even when DiT runs fp16. Cast inputs/outputs.
- `mid_block_add_attention=true` — single self-attention block at the bottleneck.
- No existing MLX port of Flux2 VAE. Port from diffusers source.

### Text encoder (`text_encoder/config.json`)

```json
{
  "architectures": ["Mistral3Model"],
  "text_config": {"hidden_size": 3072, "num_hidden_layers": 26, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "rope_parameters": {"rope_type": "yarn", ...}},
  "vision_config": {"hidden_size": 1024, "num_hidden_layers": 24, ...}
}
```

- Multimodal model. For pure t2i we want text path only — skip the Pixtral vision tower entirely at load time (don't load its safetensors keys).
- GQA: 32 Q heads / 8 KV heads — use `mx.fast.scaled_dot_product_attention(... , n_kv_heads=8)`.
- YaRN RoPE with `original_max_position_embeddings=16384`, `factor=16`, `beta_fast=32`, `beta_slow=1` → use mlx-lm's YaRN helper or port equivalent.
- `mlx-lm` may already provide a compatible Mistral class — investigate before writing from scratch.

## Reading-time trap checklist (resolved from diffusers 0.36 source)

Source: `diffusers/models/transformers/transformer_ernie_image.py` (430 lines).

- [x] **Defaults** — verified against `config.json` above.
- [x] **Attention head-dim misnomer** — N/A, uses `num_attention_heads` correctly (hidden_size/heads = head_dim).
- [x] **QKV interleaving** — SAFE: `to_q / to_k / to_v` are independent Linears, reshape is standard `unflatten(-1, (heads, head_dim))`. No interleaving trap.
- [x] **Weight layout** — DiT is Linear-only except `x_embedder` which is Conv2d(kernel=patch_size=1) = pointwise conv. At patch_size=1 we can represent it as `Linear(in_channels → hidden_size)` (mathematically equivalent, no transpose needed). VAE still has conv transposes — handle at recipe time.
- [x] **Norms** — DiT uses RMSNorm everywhere (`adaLN_sa_ln`, `adaLN_mlp_ln`, `norm_q`, `norm_k`), all with `eps=1e-6` and affine weights. Final norm is an AdaLN-continuous LayerNorm (`elementwise_affine=False`) with a separate linear modulation. VAE is GroupNorm(32, eps=1e-6). Mistral3 is RMSNorm.
- [x] **Flags verified:**
    - `qk_layernorm=True` → RMSNorm on Q and K before RoPE (actually AFTER unflatten, BEFORE rope — important order).
    - `adaLN_modulation` is SHARED across all 36 blocks (one `SiLU → Linear(hidden, 6*hidden)` fed by the timestep embedding; outputs broadcast to every token).
    - FFN is **GeGLU**: `linear_fc2(up_proj(x) * gelu(gate_proj(x)))` — NOT SwiGLU.
    - `rope_theta=256` (unusual — DO NOT bump to 10000).
    - `freqs_cis` layout is Megatron-style: angles duplicated as `[θ0,θ0,θ1,θ1,...]` over head_dim, rotate_half splits in two halves. `mx.fast.rope` does NOT match — implement manually (see `model/rope.py`).
    - Position IDs: image token = `(text_len, y, x)`, text token = `(text_pos, 0, 0)` — triple-axis RoPE covers all 128 channels for image, but only first 32 for text (axes 2/3 are zero → cos=1, sin=0 → identity).
    - Concatenation order inside the transformer: `cat([img_tokens, text_tokens])` along sequence — image FIRST. Attention mask has ones for image, bool-valid for text.

## Reference source location

- Transformer: vendored in `tests/parity/_pt_reference.py` (from huggingface/diffusers main, Apache 2.0 per the NOTICE at bottom of that file).
- Key classes: `ErnieImageTransformer2DModel`, `ErnieImageSharedAdaLNBlock`, `ErnieImageSingleStreamAttnProcessor`, `ErnieImageEmbedND3`, `ErnieImageFeedForward`, `ErnieImageAdaLNContinuous`

## Layout

```
packages/ernie-image-core-mlx/src/ernie_image_core_mlx/
├── model/
│   ├── transformer.py     # ErnieImageTransformer2DModel (36 layers, single-stream DiT)
│   ├── attention.py       # MHA + qk_layernorm + triple-axis RoPE
│   ├── rope.py            # rope_axes_dim [32,48,48] decomposition
│   ├── vae.py             # AutoencoderKLFlux2
│   └── config.py          # Dataclass configs (matches diffusers config.json)
├── text_encoders/
│   └── mistral3.py        # Mistral3Model text-only wrapper (delegates to mlx-lm where possible)
├── pipelines/
│   └── ernie_image.py     # ErnieImagePipeline — handles SFT 50-step + Turbo 8-step via n_steps arg
├── loader/
│   └── weights.py         # from_pretrained + HF snapshot_download + env-var override
└── utils/
```

## Conventions

- `mx.fast.*` primitives over hand-rolled ops (SDPA, RoPE, RMSNorm).
- `mlx-arsenal` before hand-rolling: `FlowMatchEulerDiscreteScheduler`, `FourierEmbedder`, `get_timestep_embedding`, `classifier_free_guidance`.
- Single `from_pretrained(repo_id)` entrypoint; `ERNIE_IMAGE_MLX_WEIGHTS_DIR` env override for local dev.
- Parity tests generate input on one side (numpy), inject on both (PT seed ≠ MLX seed).
- Materialize weights (`mx.eval( *tensors )`) before any `mx.save_safetensors` in the conversion recipe.
