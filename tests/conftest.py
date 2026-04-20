"""Shared pytest fixtures and markers.

- `slow`: needs model weights — skipped by default on CI without `ERNIE_IMAGE_MLX_WEIGHTS_DIR`.
- `parity`: needs the `[parity]` extra (torch + diffusers + transformers).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def weights_dir() -> str | None:
    return os.environ.get("ERNIE_IMAGE_MLX_WEIGHTS_DIR")


def pytest_collection_modifyitems(config, items):
    weights = os.environ.get("ERNIE_IMAGE_MLX_WEIGHTS_DIR")
    skip_slow = pytest.mark.skip(reason="set ERNIE_IMAGE_MLX_WEIGHTS_DIR to run")
    for item in items:
        if "slow" in item.keywords and not weights:
            item.add_marker(skip_slow)
