"""CLI smoke tests — no weights, no network, no torch.

These guard the console-script wiring (`ernie-image-mlx ...`) and argument parsing.
If these fail, the CLI is broken at the entrypoint level, regardless of whether
the underlying pipeline works.
"""

from __future__ import annotations

import pytest

from ernie_image_core_mlx import __version__
from ernie_image_core_mlx.cli import DEFAULT_REPO_ID, _build_parser, _resolve_seed, main


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version_prints_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_generate_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])
    assert exc.value.code == 0


def test_generate_requires_prompt():
    # argparse exits with code 2 when required args are missing.
    with pytest.raises(SystemExit) as exc:
        main(["generate"])
    assert exc.value.code == 2


def test_parser_defaults_are_turbo_q8():
    """The default repo-id is the one advertised in the README — guard against
    a silent drift if someone bumps it without updating docs."""
    parser = _build_parser()
    args = parser.parse_args(["generate", "-p", "hi"])
    assert args.repo_id == DEFAULT_REPO_ID
    assert DEFAULT_REPO_ID == "dgrauet/ernie-image-turbo-mlx-q8"
    assert args.height == 1024 and args.width == 1024
    assert args.steps is None and args.guidance is None  # let pipeline pick per variant


def test_resolve_seed_passthrough():
    assert _resolve_seed(None) is None
    assert _resolve_seed(42) == 42


def test_resolve_seed_expands_sentinel(capsys):
    seed = _resolve_seed(-1)
    assert isinstance(seed, int) and 0 <= seed < 2**32
    # The echoed line is the whole point of the sentinel — without it the run is
    # not reproducible. Pin the format so a refactor doesn't drop it silently.
    assert f"Random seed: {seed}" in capsys.readouterr().out
