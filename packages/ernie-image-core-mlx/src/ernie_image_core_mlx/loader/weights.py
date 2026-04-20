"""HF hub snapshot + local-dir override for ERNIE-Image weights.

Resolution order (first match wins):
  1. explicit `local_dir` arg
  2. env var `ERNIE_IMAGE_MLX_WEIGHTS_DIR`
  3. huggingface_hub.snapshot_download(repo_id)

Materialization: any code path that calls `mx.save_safetensors` must first force
lazy tensors to evaluate — otherwise they serialize to zeros with no error. This
is handled at conversion time in the `mlx-forge` ernie-image recipe, not here;
runtime loading is already-materialized.
"""

from __future__ import annotations

import os
from pathlib import Path

ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV = "ERNIE_IMAGE_MLX_WEIGHTS_DIR"


def resolve_weights_dir(
    *,
    repo_id: str,
    local_dir: str | None = None,
) -> Path:
    if local_dir is not None:
        p = Path(local_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"local_dir does not exist: {p}")
        return p

    env_dir = os.environ.get(ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV)
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"{ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV}={env_dir} does not exist"
            )
        return p

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id))
