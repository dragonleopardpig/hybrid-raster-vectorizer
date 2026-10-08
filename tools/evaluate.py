"""Run the offline converter and Potrace on identical ink, then score annotations."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from hybrid_vectorizer.convert import Options, convert
from hybrid_vectorizer.evaluation import evaluate, unique_output_names
from hybrid_vectorizer.preprocess import load_page
from hybrid_vectorizer.refine import agreement, rasterise


ROOT = Path(__file__).resolve().parents[1]


def render_preview(svg_path: Path, png_path: Path) -> None:
    """Rasterise the conversion for eyeballing; never fail the run over it.

    With --reports-only the SVG may not exist (an earlier run kept only the
    report), and the record is already scored by the time we get here, so a
    missing file or a renderer failure is a note, not an exception.
    """
    renderer = shutil.which('resvg')
    if not renderer:
        return
    if not svg_path.exists():
        print(f'no preview: {svg_path} is absent', flush=True)
        return
    completed = subprocess.run([renderer, '--quiet', '--background', 'white', str(svg_path), str(png_path)],
                               check=False)
    if completed.returncode != 0:
        print(f'no preview: resvg exited {completed.returncode} on {svg_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build/benchmark')
    parser.add_argument('--references', type=Path, default=ROOT / 'tests/fixtures/reconstruction.json')
    parser.add_argument('--reports-only', action='store_true', help='Score conversions already in output-dir')
    arguments = parser.parse_args()
    directory = arguments.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    references = json.loads(arguments.references.read_text())
    summary = []
    sources = [source.resolve() for source in arguments.images]
    for source, name in zip(sources, unique_output_names(sources)):
        report_path = directory / f'{name}.report.json'
        if arguments.reports_only:
            report = json.loads(report_path.read_text())
            if Path(report['source']).resolve() != source:
                raise SystemExit(f'{report_path} describes a different source')
        else:
            document = convert(source, Options())
            report = document.report
            (directory / f'{name}.svg').write_text(document.to_svg())
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        page = load_page(source)
        record = {'source': str(source), 'hybrid': report.get('agreement'), 'quality': report.get('quality')}
        key = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        if key in references:
            record['semantic'] = evaluate(report, references[key], figure=key)
        potrace = shutil.which('potrace')
        if potrace:
            bitmap = directory / f'{name}.pbm'
            output = directory / f'{name}-potrace.svg'
            cv2.imwrite(str(bitmap), 255 - page.ink)
            subprocess.run([potrace, str(bitmap), '--svg', '-o', str(output)], check=True)
            rendered = rasterise(output.read_text(), page.width, page.height)
            if rendered is not None:
                record['potrace'] = agreement(page.ink, rendered)
        render_preview(directory / f'{name}.svg', directory / f'{name}.png')
        summary.append(record)
        (directory / 'benchmark.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        semantic = record.get('semantic', {})
        exact = semantic.get('exact_ocr_labels')
        count = semantic.get('expected_labels')
        print(f'{name}: IoU {report.get("agreement", {}).get("iou")}; '
              f'exact OCR labels {exact}/{count}; see {report_path}', flush=True)


if __name__ == '__main__':
    main()
