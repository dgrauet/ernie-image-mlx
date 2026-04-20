"""ernie-image-core-mlx — ERNIE-Image 8B text-to-image DiT ported to MLX."""

from ernie_image_core_mlx.model.config import ErnieImageConfig, ErnieImageVaeConfig
from ernie_image_core_mlx.pipelines.ernie_image import ErnieImagePipeline

__version__ = "0.2.0"
__all__ = ["ErnieImageConfig", "ErnieImagePipeline", "ErnieImageVaeConfig"]
