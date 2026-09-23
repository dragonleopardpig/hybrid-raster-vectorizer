from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .renderer import load_spec, render_file


def _require(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise SystemExit(f"Required command not found: {command}")
    return executable


def _run_render(args: argparse.Namespace) -> None:
    spec_path = args.spec.resolve()
    output_path = args.output.resolve()
    render_file(spec_path, output_path)

    if args.outlined:
        inkscape = _require("inkscape")
        outlined_path = args.outlined.resolve()
        outlined_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                inkscape,
                str(output_path),
                "--export-type=svg",
                "--export-plain-svg",
                "--export-text-to-path",
                f"--export-filename={outlined_path}",
            ],
            check=True,
        )

    if args.preview:
        inkscape = _require("inkscape")
        preview_path = args.preview.resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [inkscape, str(output_path), f"--export-filename={preview_path}"],
            check=True,
        )


def _run_inspect(args: argparse.Namespace) -> None:
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    source_path = args.input.resolve() if args.input else spec_path.parent / spec["source_image"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    magick = _require("magick")
    results: dict[str, str] = {}

    for region in spec.get("ocr_regions", []):
        name = region["name"]
        x, y, width, height = region["crop"]
        crop_path = output_dir / f"{name}.png"
        subprocess.run(
            [
                magick,
                str(source_path),
                "-crop",
                f"{width}x{height}+{x}+{y}",
                "+repage",
                str(crop_path),
            ],
            check=True,
        )

        if region["engine"] == "formulaocr":
            command = [_require("formulaocr-offline"), "--no-classify", str(crop_path)]
        elif region["engine"] == "tesseract":
            command = [_require("tesseract"), str(crop_path), "stdout", "--psm", str(region.get("psm", 7))]
        else:
            raise SystemExit(f"Unsupported OCR engine: {region['engine']}")

        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        results[name] = completed.stdout.strip()

    print(json.dumps(results, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-vectorizer",
        description="OCR-assisted semantic reconstruction of raster scientific figures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render an SVG from a semantic JSON specification")
    render.add_argument("spec", type=Path)
    render.add_argument("-o", "--output", required=True, type=Path)
    render.add_argument("--outlined", type=Path, help="Also create an SVG with text converted to paths")
    render.add_argument("--preview", type=Path, help="Also create a PNG preview")
    render.set_defaults(handler=_run_render)

    inspect = subparsers.add_parser("inspect", help="Run configured OCR probes against the source image")
    inspect.add_argument("spec", type=Path)
    inspect.add_argument("--input", type=Path, help="Override the source image from the specification")
    inspect.add_argument("--output-dir", type=Path, default=Path("build/ocr"))
    inspect.set_defaults(handler=_run_inspect)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)
