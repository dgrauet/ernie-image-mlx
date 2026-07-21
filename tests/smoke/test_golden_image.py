"""End-to-end golden-image regression test.

Runs the full pipeline (text encoder → DiT → BN inverse → VAE decode) on the
real Turbo int8 checkpoint with a fixed seed and compares the output against a
committed golden PNG. This is the only test that exercises the production-scale
config — layer-level parity (random weights, small configs) cannot catch bugs
that appear only at trained magnitudes (checkerboard trap, dtype leaks,
scheduler drift).

Marked `slow`: skipped unless weights are resolvable (see conftest.py).
Regenerate the golden after a *justified* behavior change with:

    uv run ernie-image-mlx generate -p "一只黑白相间的中华田园犬" \
        -o tests/golden/turbo_q8_dog_512_seed42.png \
        --height 512 --width 512 --seed 42 --no-pe
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent.parent / "golden" / "turbo_q8_dog_512_seed42.png"
PROMPT = "一只黑白相间的中华田园犬"
SEED = 42
# Same machine + same MLX version reproduce bit-exactly; the PSNR floor absorbs
# minor kernel-level drift across MLX releases. A real regression (wrong layer,
# checkerboard, dtype leak) lands far below 30 dB.
PSNR_FLOOR_DB = 35.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)


@pytest.mark.slow
def test_golden_image_turbo_q8() -> None:
    from PIL import Image

    from ernie_image_core_mlx import ErnieImagePipeline

    pipe = ErnieImagePipeline.from_pretrained(pe_repo_id=None)
    out = pipe(
        PROMPT,
        height=512,
        width=512,
        seed=SEED,
        negative_prompt=None,
    )
    got = np.asarray(out.images[0].convert("RGB"))
    ref = np.asarray(Image.open(GOLDEN).convert("RGB"))

    assert got.shape == ref.shape, f"shape drift: {got.shape} vs golden {ref.shape}"
    psnr = _psnr(got, ref)
    assert psnr >= PSNR_FLOOR_DB, (
        f"golden regression: PSNR {psnr:.1f} dB < {PSNR_FLOOR_DB} dB floor "
        f"(bit-exact reproduction expected on an unchanged MLX version)"
    )
