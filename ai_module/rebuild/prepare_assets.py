#!/usr/bin/env python3
"""Prepare GroundingDINO-Tiny, SAM2.1, Qwen3-VL, and DA3 local assets.

Runtime loading is local-only. This explicit preparation command is the only
place in the rebuilt module that contacts public model servers. The selected
repositories and the official SAM2.1 checkpoint do not require gated access.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REPOSITORIES = {
    "qwen4b": "Qwen/Qwen3-VL-4B-Instruct",
    "da3metric": "depth-anything/DA3METRIC-LARGE",
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
}

DEFAULT_DESTINATIONS = {
    "grounding_dino": "grounding_dino_tiny",
}
SAM21_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_base_plus.pt"
)


class AssetPreparationError(RuntimeError):
    """Raised when a requested model cannot be materialized locally."""


def _snapshot_download(
    *,
    repo_id: str,
    destination: Path,
    token: str | None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    # The runtime image sets HF_HUB_OFFLINE=1. Preparation is an explicit
    # operator action, so clear only this switch before importing the Hub client.
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise AssetPreparationError(
            "huggingface_hub is required to prepare model assets; install the "
            "rebuild requirements first"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(destination),
        "local_files_only": False,
    }
    if token:
        kwargs["token"] = token
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        raise AssetPreparationError(
            f"failed to download {repo_id} into {destination}: {exc}"
        ) from exc
    return destination


def _assert_required_files(model: str, destination: Path) -> None:
    required = {
        "qwen4b": (destination / "config.json", destination / "tokenizer_config.json"),
        "da3metric": (destination / "config.json", destination / "model.safetensors"),
        "grounding_dino": (
            destination / "config.json",
            destination / "preprocessor_config.json",
            destination / "tokenizer_config.json",
            destination / "model.safetensors",
        ),
    }[model]
    missing = [str(path) for path in required if not path.is_file()]
    if model in {"qwen4b", "grounding_dino"} and not any(
        destination.glob("*.safetensors")
    ):
        missing.append(str(destination / "*.safetensors"))
    if missing:
        raise AssetPreparationError(
            f"{model} snapshot is incomplete in {destination}; missing: "
            + ", ".join(missing)
        )


def prepare(
    *,
    output_dir: Path,
    models: Sequence[str],
    repositories: dict[str, str],
    token: str | None,
) -> dict[str, str]:
    if output_dir.exists() and not output_dir.is_dir():
        raise AssetPreparationError(f"--output-dir is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for model in models:
        destination = output_dir / DEFAULT_DESTINATIONS.get(model, model)
        if model == "sam21":
            destination.mkdir(parents=True, exist_ok=True)
            checkpoint = destination / "sam2.1_hiera_base_plus.pt"
            if not checkpoint.is_file():
                partial = checkpoint.with_suffix(".pt.part")
                try:
                    with urllib.request.urlopen(SAM21_URL, timeout=60) as response, partial.open("wb") as target:
                        shutil.copyfileobj(response, target)
                    partial.replace(checkpoint)
                except Exception as exc:
                    raise AssetPreparationError(f"failed to download SAM2.1: {exc}") from exc
            result[model] = str(checkpoint)
            print(f"prepared {model}: {checkpoint}")
            continue
        _snapshot_download(
            repo_id=repositories[model],
            destination=destination,
            token=token,
        )
        _assert_required_files(model, destination)
        result[model] = str(destination)
        print(f"prepared {model}: {destination}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "directory in which sam21/, qwen4b/, da3metric/, and grounding_dino_tiny/ "
            "are created"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*DEFAULT_REPOSITORIES, "sam21"),
        default=(*DEFAULT_REPOSITORIES, "sam21"),
        help="models to download (default: all available models)",
    )
    parser.add_argument("--qwen-repo", default=DEFAULT_REPOSITORIES["qwen4b"])
    parser.add_argument("--da3-repo", default=DEFAULT_REPOSITORIES["da3metric"])
    parser.add_argument(
        "--grounding-dino-repo",
        default=DEFAULT_REPOSITORIES["grounding_dino"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repositories = {
        "qwen4b": args.qwen_repo,
        "da3metric": args.da3_repo,
        "grounding_dino": args.grounding_dino_repo,
    }
    try:
        prepare(
            output_dir=args.output_dir.expanduser(),
            models=args.models,
            repositories=repositories,
            token=os.environ.get("HF_TOKEN"),
        )
    except AssetPreparationError as exc:
        print(f"asset preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())


__all__ = [
    "AssetPreparationError",
    "DEFAULT_DESTINATIONS",
    "DEFAULT_REPOSITORIES",
    "build_parser",
    "main",
    "prepare",
]
