"""Smoke tests for the Prompt Enhancer — no weights, no network, no torch.

Full-pipeline tests live under the converted-model path (dev-local only). These
tests guard the pure-Python helpers: config shaping, chat-template JSON payload,
CLI plumbing. They run in CI against a fresh venv without ever touching the 7 GB
PE checkpoint.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ernie_image_core_mlx.model.config import Mistral3TextConfig
from ernie_image_core_mlx.prompt_enhancer import (
    DEFAULT_PE_REPO_ID,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    PromptEnhancer,
    _default_system_prompt,
    _detect_language_name,
    _ministral3_args_dict,
    _pe_config_from_json,
    _prefill_for,
)


def test_default_pe_repo_id_is_int4():
    """The int4 variant is the sensible default — fp16 is 7 GB and the PE is
    already stateless (no image-quality loss from aggressive quantization)."""
    assert DEFAULT_PE_REPO_ID == "dgrauet/ernie-image-pe-mlx-q4"


def test_ministral3_args_dict_shape():
    """The dict must carry every field mlx-lm's ``ministral3.ModelArgs`` expects,
    otherwise model instantiation fails with a cryptic dataclass error."""
    cfg = Mistral3TextConfig()
    d = _ministral3_args_dict(cfg)
    assert d["model_type"] == "ministral3"
    assert d["hidden_size"] == cfg.hidden_size
    assert d["num_hidden_layers"] == cfg.num_hidden_layers
    assert d["head_dim"] == cfg.head_dim
    assert d["tie_word_embeddings"] is True

    rp = d["rope_parameters"]
    # YaRN RoPE: all three scaling params + llama_4_scaling_beta must be present —
    # mlx-lm's ministral3 crashes at forward time if any are missing.
    assert rp["rope_type"] == "yarn"
    assert rp["factor"] == cfg.rope_factor
    assert rp["original_max_position_embeddings"] == cfg.rope_original_max_position_embeddings
    assert rp["llama_4_scaling_beta"] == 0.1


def test_pe_config_from_json_reads_real_pe_blob():
    """Feed the exact structure of baidu's ``pe/config.json`` (minus unused
    fields) and verify the dataclass extraction is correct."""
    blob = {
        "hidden_size": 3072,
        "num_hidden_layers": 26,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9216,
        "vocab_size": 131072,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1_000_000.0,
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 16384,
        },
    }
    cfg = _pe_config_from_json(blob)
    assert cfg.hidden_size == 3072
    assert cfg.num_hidden_layers == 26
    assert cfg.num_key_value_heads == 8
    assert cfg.rope_factor == 16.0
    assert cfg.rope_original_max_position_embeddings == 16384


class _FakeTokenizer:
    """Minimal tokenizer stand-in — the rewritten ``_apply_chat_template``
    builds the string manually, so we only need ``bos_token`` for it to work."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 11


def _extract_user_json(rendered: str) -> dict:
    """Pull the JSON payload out of the ``[INST]…[/INST]`` block."""
    start = rendered.index("[INST]") + len("[INST]")
    end = rendered.index("[/INST]", start)
    return json.loads(rendered[start:end])


def test_chat_template_wraps_prompt_as_json_payload():
    """Reference behavior (diffusers): the user prompt is serialized as
    JSON ``{"prompt": ..., "width": W, "height": H}`` with ``ensure_ascii=False``
    inside an ``[INST]...[/INST]`` block, preceded by the BOS-wrapped system
    prompt. We build this manually so we can swap the system prompt — but the
    payload shape must stay identical to what the PE was trained on."""
    pe = PromptEnhancer.__new__(PromptEnhancer)
    pe.tokenizer = _FakeTokenizer()

    rendered = pe._apply_chat_template("一只小狗", 1024, 768, system_prompt="SYS")

    assert rendered.startswith("<s>[SYSTEM_PROMPT]SYS[/SYSTEM_PROMPT][INST]")
    assert rendered.endswith("[/INST]")
    assert _extract_user_json(rendered) == {"prompt": "一只小狗", "width": 1024, "height": 768}


def test_apply_chat_template_preserves_non_ascii():
    """``ensure_ascii=False`` is not optional — the PE sees Chinese/emoji/
    diacritic characters directly at training time. Escaping to ``\\uXXXX``
    would change what the model sees at inference."""
    pe = PromptEnhancer.__new__(PromptEnhancer)
    pe.tokenizer = _FakeTokenizer()
    rendered = pe._apply_chat_template("café — 💡 — 中文", 512, 512, system_prompt="SYS")

    assert "café" in rendered
    assert "中文" in rendered
    assert "\\u" not in rendered


def test_default_system_prompt_names_target_language():
    """The PE never learned to introspect "the user's language"; empirically
    it only switches out of Chinese when an explicit language NAME is given.
    The template must therefore carry a ``{language}`` slot, and the rendered
    default must name that language in the instruction."""
    assert "{language}" in DEFAULT_SYSTEM_PROMPT_TEMPLATE

    rendered = _default_system_prompt("French")
    assert "French" in rendered
    assert "{language}" not in rendered


def test_detect_language_handles_non_latin_scripts():
    """Non-Latin scripts must be classified deterministically from a single
    codepoint scan — these are the anchors that make the auto-detection
    robust in the cases where it matters most (e.g. the PE was trained
    heavily on Chinese, so getting the Chinese → Chinese case right is
    essential)."""
    assert _detect_language_name("一只黑色的小猫") == "Chinese"
    assert _detect_language_name("黒い猫") == "Japanese"
    assert _detect_language_name("검은 고양이") == "Korean"
    assert _detect_language_name("чёрный кот") == "Russian"


def test_detect_language_latin_script_heuristic():
    """Latin-script detection is best-effort: English is the safe fallback
    for stopword-sparse prompts, but common function words should still
    steer common Romance languages correctly."""
    assert _detect_language_name("a small black cat on a rooftop at sunset") == "English"
    assert _detect_language_name("un petit chat noir sur un toit au coucher du soleil") == "French"
    assert _detect_language_name("un pequeño gato negro en un tejado al atardecer") == "Spanish"


def test_detect_language_falls_back_to_english_when_uncertain():
    """Two-word prompts with no stopwords shouldn't trip a false positive
    into French/Spanish/etc. — the fallback must be English so the model
    gets its best-shot default."""
    assert _detect_language_name("cute puppy") == "English"


def test_prefill_for_latin_languages_and_cjk_blank():
    """The prefill locks the PE's first assistant tokens onto the target
    language's script. For Latin languages we need a natural starter
    (``A`` / ``Une`` / ``Una`` / ``Ein``). CJK languages keep the model's
    native bias, so their prefill must be empty — a Latin starter there
    would fight Chinese/Japanese tokenization and corrupt the output."""
    assert _prefill_for("English").strip() == "A"
    assert _prefill_for("French").strip() == "Une"
    assert _prefill_for("Spanish").strip() == "Una"
    assert _prefill_for("German").strip() == "Ein"
    assert _prefill_for("Chinese") == ""
    assert _prefill_for("Japanese") == ""
    assert _prefill_for("Korean") == ""
    # Unknown language → blank (safer than guessing a wrong starter).
    assert _prefill_for("Klingon") == ""


def test_from_pretrained_missing_cfg_raises(tmp_path):
    """The loader accepts two layouts (flat ``pe_config.json`` from the recipe,
    or nested ``pe/config.json`` from a raw HF checkpoint). When NEITHER exists
    we should get a clear FileNotFoundError-family error, not a silent success."""
    import pytest

    # Create an empty dir + minimal weights stub so the existence check in
    # resolve_weights_dir passes but config loading fails.
    weights_dir = tmp_path / "empty_pe_dir"
    weights_dir.mkdir()

    with pytest.raises((FileNotFoundError, OSError)):
        PromptEnhancer.from_pretrained(repo_id="fake/repo", local_dir=str(weights_dir))


def test_load_tokenizer_injects_pad_when_missing():
    """The Pixtral/Tekken tokenizer family ships ``pad_token=None`` even when
    ``tokenizer_config.json`` names one. Without the fallback injection, batched
    encoding at generation time raises with an unhelpful `pad_token is None`
    error. Pin the fix."""
    tok = MagicMock()
    tok.pad_token = None
    mock_from_pretrained = MagicMock(return_value=tok)

    import transformers

    orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = mock_from_pretrained
    try:
        out = PromptEnhancer._load_tokenizer("/does/not/matter")
    finally:
        transformers.AutoTokenizer.from_pretrained = orig

    assert out is tok
    assert tok.pad_token == "<pad>"
