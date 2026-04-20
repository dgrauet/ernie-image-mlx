"""Loader resolution order — no network, no weights."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_env_var_override(tmp_path, monkeypatch):
    from ernie_image_core_mlx.loader.weights import (
        ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV,
        resolve_weights_dir,
    )

    monkeypatch.setenv(ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV, str(tmp_path))
    resolved = resolve_weights_dir(repo_id="baidu/ERNIE-Image-Turbo")
    assert resolved == tmp_path.resolve()


def test_explicit_local_dir_beats_env(tmp_path, monkeypatch):
    from ernie_image_core_mlx.loader.weights import (
        ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV,
        resolve_weights_dir,
    )

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV, str(tmp_path))
    resolved = resolve_weights_dir(repo_id="baidu/ERNIE-Image", local_dir=str(other))
    assert resolved == other.resolve()


def test_missing_local_dir_raises(monkeypatch):
    from ernie_image_core_mlx.loader.weights import resolve_weights_dir

    with pytest.raises(FileNotFoundError):
        resolve_weights_dir(repo_id="baidu/ERNIE-Image", local_dir="/does/not/exist/xyz")
