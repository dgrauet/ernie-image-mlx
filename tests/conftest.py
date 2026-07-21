"""Shared pytest fixtures and markers.

- `slow`: needs real model weights — runs when `ERNIE_IMAGE_MLX_WEIGHTS_DIR` is
  set OR the default Turbo int8 repo is already in the local HF cache.
- `parity`: needs the `[parity]` extra (torch + diffusers + transformers).
"""

from __future__ import annotations

import os

import pytest

from ernie_image_core_mlx.pipelines.ernie_image import DEFAULT_REPO_ID


def _default_repo_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=DEFAULT_REPO_ID, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def weights_dir() -> str | None:
    return os.environ.get("ERNIE_IMAGE_MLX_WEIGHTS_DIR")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ERNIE_IMAGE_MLX_WEIGHTS_DIR") or _default_repo_cached():
        return
    skip_slow = pytest.mark.skip(reason=f"set ERNIE_IMAGE_MLX_WEIGHTS_DIR or cache {DEFAULT_REPO_ID} to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
