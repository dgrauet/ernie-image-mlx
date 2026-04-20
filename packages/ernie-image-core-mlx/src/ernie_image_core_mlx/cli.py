"""CLI for ernie-image-mlx.

Example:
    ernie-image-mlx generate -p "一只黑白相间的中华田园犬" -o dog.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_REPO_ID = "dgrauet/ernie-image-turbo-mlx-q8"
DEFAULT_PE_REPO_ID = "dgrauet/ernie-image-pe-mlx-q4"


def _build_parser() -> argparse.ArgumentParser:
    from ernie_image_core_mlx import __version__

    p = argparse.ArgumentParser(
        prog="ernie-image-mlx",
        description="Run Baidu ERNIE-Image on Apple Silicon via MLX.",
    )
    p.add_argument("--version", action="version", version=f"ernie-image-mlx {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate an image from a text prompt.")
    gen.add_argument(
        "-p",
        "--prompt",
        required=True,
        help=(
            "Text prompt. Any language works; the Prompt Enhancer expands it into a "
            "rich Chinese visual description before the text encoder (disable with --no-pe)."
        ),
    )
    gen.add_argument(
        "-o",
        "--output",
        default="output.png",
        help="Output image path (default: output.png).",
    )
    gen.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF repo id to load MLX weights from (default: {DEFAULT_REPO_ID}).",
    )
    gen.add_argument(
        "--local-dir",
        default=None,
        help="Already-converted MLX weights dir. Overrides --repo-id resolution.",
    )
    gen.add_argument(
        "--variant",
        choices=("turbo", "sft"),
        default=None,
        help="Force the pipeline variant. Inferred from --repo-id when omitted.",
    )
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Inference steps (default: 8 for turbo, 50 for sft).",
    )
    gen.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: 1.0 for turbo, ~5 for sft).",
    )
    gen.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt (default: empty string — CFG still runs when guidance > 1).",
    )
    gen.add_argument(
        "--no-cfg",
        action="store_true",
        help="Disable classifier-free guidance entirely (skip the uncond pass).",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility. Pass -1 to draw a random seed (printed so you can reuse it).",
    )
    gen.add_argument(
        "--no-pe",
        action="store_true",
        help="Disable the Prompt Enhancer (skip loading and skip prompt expansion).",
    )
    gen.add_argument(
        "--pe-repo-id",
        default=DEFAULT_PE_REPO_ID,
        help=f"HF repo id for the Prompt Enhancer MLX weights (default: {DEFAULT_PE_REPO_ID}).",
    )
    gen.add_argument(
        "--pe-local-dir",
        default=None,
        help="Already-converted PE weights dir. Overrides --pe-repo-id.",
    )
    gen.add_argument(
        "--pe-seed",
        type=int,
        default=None,
        help=(
            "Seed for PE sampling (separate from image seed). "
            "Pass -1 to draw random. Defaults to non-deterministic sampling to match diffusers."
        ),
    )
    return p


def _resolve_seed(raw: int | None) -> int | None:
    """Expand the `--seed -1` sentinel into a freshly-drawn seed.

    Printing the chosen value is the point: the run stays reproducible because the
    user can re-run with the exact integer we announce here.
    """
    if raw != -1:
        return raw
    import random

    seed = random.randint(0, 2**32 - 1)
    print(f"Random seed: {seed}")
    return seed


def _run_generate(args: argparse.Namespace) -> int:
    from ernie_image_core_mlx import ErnieImagePipeline

    # `--no-pe` sets `pe_repo_id=None` which tells the pipeline to skip the
    # Prompt Enhancer load entirely (saves ~3 s and 4 GB of RAM).
    pipe = ErnieImagePipeline.from_pretrained(
        args.repo_id,
        variant=args.variant,
        local_dir=args.local_dir,
        pe_repo_id=None if args.no_pe else args.pe_repo_id,
        pe_local_dir=args.pe_local_dir,
    )

    negative = None if args.no_cfg else args.negative_prompt
    seed = _resolve_seed(args.seed)
    pe_seed = _resolve_seed(args.pe_seed) if args.pe_seed is not None else None
    t0 = time.perf_counter()
    out = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        negative_prompt=negative,
        seed=seed,
        use_pe=not args.no_pe,
        pe_seed=pe_seed,
    )
    elapsed = time.perf_counter() - t0

    if out.revised_prompts:
        print("Revised prompt:")
        for p in out.revised_prompts:
            print(f"  {p}")

    out_path = Path(args.output).expanduser()
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(out_path)
    print(f"Saved: {out_path}  ({elapsed:.1f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
