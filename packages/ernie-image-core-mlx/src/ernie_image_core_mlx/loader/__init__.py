"""Weight loading + HF hub integration."""

from ernie_image_core_mlx.loader.weights import (
    ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV,
    resolve_weights_dir,
)

__all__ = ["ERNIE_IMAGE_MLX_WEIGHTS_DIR_ENV", "resolve_weights_dir"]
