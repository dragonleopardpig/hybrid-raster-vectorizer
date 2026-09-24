from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .convert import Options, convert
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


def _run_convert(args: argparse.Namespace) -> None:
    options = Options(
        deskew=not args.no_deskew,
        bezier_tolerance=args.bezier_tolerance,
        idealise=args.idealise,
        confidence_threshold=args.confidence,
        largest_label=args.largest_label,
        raster_fallback=args.raster_fallback,
        substitute_glyphs=args.substitute_glyphs,
        solve_alphabet=args.solve_alphabet,
        font_family=args.font,
        background=args.background,
        ensemble=args.ensemble,
        use_formula_ocr=not args.no_formula_ocr,
        verify=not args.no_verify,
    )
    document = convert(args.input.resolve(), options)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.to_svg(), encoding="utf-8")

    report_path = args.report.resolve() if args.report else output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(document.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = document.report
    print(f"wrote {output_path}")
    print(f"wrote {report_path}")
    print(f"  font        {summary['font']['chosen']}")
    for curve in summary["curves"]:
        print(
            f"  curve {curve['curve']}     {curve['segments']} bezier segments, "
            f"model {curve['analytic_model']} residual {curve['analytic_residual_px']}px"
        )
    for area in summary.get("areas", []):
        hatch = area["hatch"]
        if hatch:
            detail = f"hatched {hatch['angle_degrees']:.0f}deg at {hatch['spacing_px']:.1f}px"
        elif area["kind"] == "tint":
            detail = f"tinted at {area['opacity']:.0%}"
        else:
            detail = "filled"
        print(f"  area        {area['shape']} {area['box']}, {detail}")
    for series in summary.get("marker_series", []):
        fill = "filled" if series["filled"] else "hollow"
        print(f"  markers     {series['count']} x {fill} {series['shape']}, {series['size_px']:.0f}px")
    for line in summary.get("dashed_lines", []):
        print(
            f"  dashed      {line['from']} to {line['to']}, "
            f"{line['marks']} marks, {line['dash_px']:.0f}/{line['gap_px']:.0f}px"
        )
    for legend in summary.get("legends", []):
        tied = sum(1 for entry in legend["entries"] if entry["series"] is not None)
        print(
            f"  legend      {len(legend['entries'])} entries at {legend['box']}"
            f"{'' if legend['framed'] else ' (no frame)'}, {tied} tied to a marker series"
        )
    for tick in summary["ticks"]:
        print(f"  ticks       {tick['count']} on the {tick['orientation']} axis, spacing {tick['spacing_px']}px")
    for label in summary["labels"]:
        note = f"  <- {label['corrections']}" if label["corrections"] else ""
        others = label.get("other_readings") or []
        if others:
            note += f"  [read {label['reading_confidence']:.0%} of the time, {len(others)} other reading(s)]"
        ambiguous = label.get("ambiguous_glyphs") or []
        if ambiguous:
            note += f"  [{len(ambiguous)} unverified glyph(s); see the report]"
        print(f"  {label['id']:<10} {label['confidence']:.2f}  {label['text']}{note}")
    if "agreement" in summary:
        scores = summary["agreement"]
        print(
            f"  agreement   recall {scores['recall']:.3f}  precision {scores['precision']:.3f} "
            f"(within {scores['tolerance_px']}px)"
        )
    for entry in summary.get("not_labels", []):
        print(
            f"  not a label  {entry['box']} would need {entry['size_px']:.0f}px type: "
            f"{entry['reading'][:40]!r}"
        )
    if summary["needs_review"]:
        print(f"  review      {', '.join(summary['needs_review'])}")

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
        preview_path = args.preview.resolve()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = shutil.which("resvg")
        command = (
            [renderer, str(output_path), str(preview_path)]
            if renderer
            else [_require("inkscape"), str(output_path), f"--export-filename={preview_path}"]
        )
        subprocess.run(command, check=True)


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

    convert_parser = subparsers.add_parser(
        "convert", help="Automatically reconstruct a raster figure as semantic SVG"
    )
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("-o", "--output", required=True, type=Path)
    convert_parser.add_argument("--report", type=Path, help="Where to write the JSON report")
    convert_parser.add_argument("--outlined", type=Path, help="Also write an SVG with text as paths")
    convert_parser.add_argument("--preview", type=Path, help="Also write a PNG preview")
    convert_parser.add_argument("--font", help="Force a font family instead of matching one")
    convert_parser.add_argument(
        "--background",
        help="Paint a solid background (default: transparent, inheriting the page)",
    )
    convert_parser.add_argument(
        "--bezier-tolerance", type=float, default=0.25,
        help="Curve fit tolerance as a fraction of the pen width (default 0.25)",
    )
    convert_parser.add_argument(
        "--idealise", action="store_true",
        help="Redraw curves from the fitted analytic model instead of the traced ink",
    )
    convert_parser.add_argument(
        "--confidence", type=float, default=0.55, help="Below this, a label is flagged for review"
    )
    convert_parser.add_argument(
        "--raster-fallback", action="store_true",
        help="Embed the original pixels for labels below the confidence threshold",
    )
    convert_parser.add_argument(
        "--substitute-glyphs", action="store_true",
        help="Re-read confusable glyphs against installed fonts (measured unreliable)",
    )
    convert_parser.add_argument(
        "--solve-alphabet", action="store_true",
        help="Name every distinct shape at once against installed fonts (measured unreliable)",
    )
    convert_parser.add_argument(
        "--ensemble", type=int, default=1, metavar="N",
        help="Read each formula N ways and report how often they agree (slower)",
    )
    convert_parser.add_argument(
        "--largest-label", type=float, default=3.5, metavar="F",
        help="Refuse a reading needing type more than F times the page's text height",
    )
    convert_parser.add_argument("--no-deskew", action="store_true")
    convert_parser.add_argument("--no-formula-ocr", action="store_true")
    convert_parser.add_argument("--no-verify", action="store_true")
    convert_parser.set_defaults(handler=_run_convert)

    inspect = subparsers.add_parser("inspect", help="Run configured OCR probes against the source image")
    inspect.add_argument("spec", type=Path)
    inspect.add_argument("--input", type=Path, help="Override the source image from the specification")
    inspect.add_argument("--output-dir", type=Path, default=Path("build/ocr"))
    inspect.set_defaults(handler=_run_inspect)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)
