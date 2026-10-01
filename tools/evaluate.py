"""Run the offline converter and Potrace on identical ink, then score annotations."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2

from hybrid_vectorizer.convert import Options, convert
from hybrid_vectorizer.evaluation import evaluate
from hybrid_vectorizer.preprocess import load_page
from hybrid_vectorizer.refine import agreement, rasterise


ROOT = Path(__file__).resolve().parents[1]


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
    names = set()
    for source in arguments.images:
        source = source.resolve()
        name = source.parent.name if source.stem in {'raster', 'figure'} else source.stem
        if name in names:
            raise SystemExit(f'Duplicate output name {name!r}; evaluate these inputs separately')
        names.add(name)
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
            record['semantic'] = evaluate(report, references[key])
        potrace = shutil.which('potrace')
        if potrace:
            bitmap = directory / f'{name}.pbm'
            output = directory / f'{name}-potrace.svg'
            cv2.imwrite(str(bitmap), 255 - page.ink)
            subprocess.run([potrace, str(bitmap), '--svg', '-o', str(output)], check=True)
            rendered = rasterise(output.read_text(), page.width, page.height)
            if rendered is not None:
                record['potrace'] = agreement(page.ink, rendered)
        renderer = shutil.which('resvg')
        if renderer:
            subprocess.run([renderer, '--quiet', '--background', 'white',
                            str(directory / f'{name}.svg'), str(directory / f'{name}.png')], check=True)
        summary.append(record)
        (directory / 'benchmark.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        semantic = record.get('semantic', {})
        exact = semantic.get('exact_ocr_labels')
        count = semantic.get('expected_labels')
        print(f'{name}: IoU {report.get("agreement", {}).get("iou")}; '
              f'exact OCR labels {exact}/{count}; see {report_path}', flush=True)


if __name__ == '__main__':
    main()
