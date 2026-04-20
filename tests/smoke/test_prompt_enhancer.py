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
    PromptEnhancer,
    _ministral3_args_dict,
    _pe_config_from_json,
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
    """Minimal tokenizer stand-in capturing the ``apply_chat_template`` call so
    the chat-template wiring can be inspected without a real tokenizer on disk."""

    def __init__(self):
        self.captured: list[dict] = []
        self.pad_token = "<pad>"
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 11

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.captured.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "<fake-rendered-template>"


def test_chat_template_wraps_prompt_as_json_payload():
    """Reference behavior (diffusers): the user prompt must be serialized as
    JSON ``{"prompt": ..., "width": W, "height": H}`` with ``ensure_ascii=False``
    so Chinese / emoji pass through unescaped — and the message role must be
    'user' so the Jinja template emits the ``[INST]...[/INST]`` wrapping."""
    pe = PromptEnhancer.__new__(PromptEnhancer)
    pe.tokenizer = _FakeTokenizer()

    rendered = pe._apply_chat_template("一只小狗", 1024, 768)
    assert rendered == "<fake-rendered-template>"

    call = pe.tokenizer.captured[0]
    assert call["tokenize"] is False
    # The reference calls with ``add_generation_prompt=False`` — that relies on
    # the jinja template doing the right thing. Changing to True would
    # double-emit the [INST] block.
    assert call["add_generation_prompt"] is False

    msgs = call["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    payload = json.loads(msgs[0]["content"])
    assert payload == {"prompt": "一只小狗", "width": 1024, "height": 768}


def test_apply_chat_template_preserves_non_ascii():
    """``ensure_ascii=False`` is not optional — the PE was trained on Chinese
    system-prompt + Chinese/UTF-8 user content, so escaping Unicode into
    ``\\uXXXX`` changes what the model sees."""
    pe = PromptEnhancer.__new__(PromptEnhancer)
    pe.tokenizer = _FakeTokenizer()
    pe._apply_chat_template("café — 💡 — 中文", 512, 512)

    content = pe.tokenizer.captured[0]["messages"][0]["content"]
    # Direct Unicode, not escape sequences.
    assert "café" in content
    assert "中文" in content
    assert "\\u" not in content


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
