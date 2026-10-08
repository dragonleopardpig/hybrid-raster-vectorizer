"""Evaluate OCR and object counts against independent annotations.

A reference (``truth.json`` / ``tests/fixtures/reconstruction.json``) carries two
kinds of annotation:

* ``counts`` -- expected object counts. Only five names can be scored against a
  conversion report: ``marker_sets``, ``markers``, ``curves``, ``closed_curves``
  and ``curve_arrowheads``. Any other name (the repo's own examples use
  ``markers_per_set`` and ``points_per_set``) is carried through as
  ``unscorable`` rather than refused, so one unfamiliar key cannot abort a
  whole benchmark run.
* ``labels`` -- text annotations, each ``{"box": [x, y, w, h], "latex": ...}``
  or ``{"box": [x, y, w, h], "text": ...}`` with the box in deskewed page
  pixels. Boxes are matched to predicted labels by IoU, and each reading is
  scored by character edit distance.

A LaTeX truth is read through :func:`latex.to_text`, which writes a fraction
as ``(num)/(den)`` and scripts as ``_x`` / ``^x``. A Tesseract reading of the
same label arrives raw, so ``\\frac{\\lambda}{4}`` (``(λ)/(4)``) against a
correct ``λ/4`` used to cost four character errors. Each prediction is
therefore scored as the minimum distance over its reasonable spellings
(raw, and parsed as LaTeX when that parses cleanly) against the truth's
reasonable spellings (as written, and with the grouping parentheses around a
single token removed).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import latex as tex

SCORABLE_COUNTS = ('marker_sets', 'markers', 'curves', 'closed_curves', 'curve_arrowheads')

LABEL_SCHEMA = 'box = [x, y, w, h] in deskewed page pixels; latex or text'

# A pair of parentheses that wraps exactly one character is grouping that
# `latex.to_text` added around a fraction's numerator or denominator; a human
# (and Tesseract) writes `λ/4`, not `(λ)/(4)`.
_SINGLE_TOKEN_GROUP = re.compile(r'\(([^()])\)')


def normalise(text: str) -> str:
    return ''.join(text.split()).replace('−', '-').replace('₀', '_0')


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, character in enumerate(first, 1):
        current = [row]
        for column, other in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (character != other)))
        previous = current
    return previous[-1]


def box_iou(first: list, second: list) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / max(1, first[2] * first[3] + second[2] * second[3] - intersection)


def count_objects(report: dict) -> dict:
    """The five object counts a conversion report can be scored on."""
    return {
        'marker_sets': len(report.get('marker_series', [])),
        'markers': sum(series['count'] for series in report.get('marker_series', [])),
        'curves': len(report.get('curves', [])),
        'closed_curves': sum(curve.get('closed', False) for curve in report.get('curves', [])),
        'curve_arrowheads': sum(curve.get('arrowheads', 0) for curve in report.get('curves', [])),
    }


def score_counts(report: dict, expected_counts: dict) -> dict:
    """Score every expected count; names outside SCORABLE_COUNTS are reported, never refused."""
    actual_counts = count_objects(report)
    scored = {}
    for name, value in expected_counts.items():
        if name in actual_counts:
            scored[name] = {'expected': value, 'actual': actual_counts[name],
                            'correct': actual_counts[name] == value}
        else:
            scored[name] = {'expected': value, 'actual': None, 'correct': None, 'unscorable': True}
    return scored


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_reference_labels(labels: list, figure: str = 'reference') -> None:
    """Refuse a malformed label annotation before any of it is indexed.

    A truth file is hand-written, and a label missing its box or carrying both
    a `latex` and a `text` reading used to surface as a KeyError deep inside
    the matching loop. The error names the figure and the label's position so
    the author can find the line, and restates the schema.
    """
    for index, label in enumerate(labels):
        where = f'{figure}: label {index}'
        if not isinstance(label, dict):
            raise ValueError(f'{where} is not an object ({LABEL_SCHEMA})')
        box = label.get('box')
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_is_number(v) for v in box):
            raise ValueError(f'{where} needs a 4-number box, got {box!r} ({LABEL_SCHEMA})')
        readings = [key for key in ('latex', 'text') if key in label]
        if len(readings) != 1 or not isinstance(label[readings[0]], str):
            raise ValueError(f'{where} needs exactly one of latex/text as a string, '
                             f'got {sorted(label)} ({LABEL_SCHEMA})')


def _parses_cleanly(text: str) -> str | None:
    """`latex.to_text` of the reading when the LaTeX subset accepts all of it."""
    unknown: list[str] = []
    try:
        node = tex.parse(text, unknown=unknown)
    except Exception:  # noqa: BLE001 -- a reading is arbitrary OCR output
        return None
    if unknown:
        return None
    return tex.to_text(node)


def truth_spellings(label: dict) -> list[str]:
    """The normalised truth, first as written and then without single-token grouping.

    The first entry is the canonical form that reports print as `expected`.
    """
    if 'latex' in label:
        canonical = normalise(tex.to_text(tex.parse(label['latex'])))
    else:
        canonical = normalise(label['text'])
    spellings = [canonical]
    # Only a fraction's rendering adds grouping that a reading may fairly omit;
    # a literal '(a)' panel letter must keep its parentheses to score exact.
    if 'latex' in label and '\\frac' in label['latex']:
        ungrouped = _SINGLE_TOKEN_GROUP.sub(r'\1', canonical)
        if ungrouped != canonical:
            spellings.append(ungrouped)
    return spellings


def reading_spellings(candidate: dict) -> list[str]:
    """The normalised prediction under each spelling it could reasonably mean.

    A formula engine's reading is LaTeX and is only meaningful read through
    `latex.to_text`. Any other engine's reading is scored raw, and also as
    LaTeX when it parses cleanly, so a Tesseract `x_0` meets a truth `x_0`
    however the truth was written.
    """
    source = candidate.get('source_reading', '')
    if candidate.get('engine') == 'formulaocr':
        return [normalise(tex.to_text(tex.parse(source)))]
    spellings = [normalise(source)]
    parsed = _parses_cleanly(source)
    if parsed is not None and normalise(parsed) not in spellings:
        spellings.append(normalise(parsed))
    return spellings


def score_reading(label: dict, candidate: dict) -> dict:
    """Score one matched (or missing) prediction against its reference label."""
    truths = truth_spellings(label)
    readings = reading_spellings(candidate) if candidate else ['']
    best_distance = None
    best_reading = readings[0]
    for truth in truths:
        for reading in readings:
            distance = edit_distance(truth, reading)
            if best_distance is None or distance < best_distance:
                best_distance, best_reading = distance, reading
    return {'expected': truths[0], 'reading': best_reading, 'correct': best_distance == 0,
            'representation': candidate.get('representation'),
            'id': candidate.get('id'), 'character_errors': best_distance}


def evaluate(report: dict, reference: dict, figure: str = 'reference') -> dict:
    """Score a conversion report against one figure's annotations.

    `figure` only names the annotation set in validation errors.
    """
    expected = reference.get('labels', [])
    validate_reference_labels(expected, figure)
    result = {'object_counts': score_counts(report, reference.get('counts', {}))}
    if not expected:
        return result
    predicted = report.get('labels', [])
    scores = np.zeros((len(expected), len(predicted)))
    for row, label in enumerate(expected):
        for column, candidate in enumerate(predicted):
            if candidate.get('box'):
                scores[row, column] = box_iou(label['box'], candidate['box'])
    rows, columns = linear_sum_assignment(-scores)
    matches = {int(row): int(column) for row, column in zip(rows, columns) if scores[row, column] >= 0.15}
    details = []
    for index, label in enumerate(expected):
        candidate = predicted[matches[index]] if index in matches else {}
        details.append(score_reading(label, candidate))
    characters = sum(len(label['expected']) for label in details)
    result.update({
        'expected_labels': len(expected),
        'matched_labels': len(matches),
        'exact_ocr_labels': sum(label['correct'] for label in details),
        'character_error_rate': sum(label['character_errors'] for label in details) / max(1, characters),
        'unmatched_predictions': len(predicted) - len(matches),
        'representations': dict(Counter(label.get('representation', 'unknown') for label in predicted)),
        'labels': details,
    })
    return result


def unique_output_names(sources: list[Path]) -> list[str]:
    """One output stem per input, unique even when several inputs share a file name.

    `a/x.png` and `b/x.png` both used to want the name `x`; the old answer was
    to refuse and advise separate runs, which silently overwrote each other's
    `x.report.json`. A generic stem (`raster`, `figure`) is replaced by its
    directory; names that still collide gain one more parent directory each
    round (`a-x`, `b-x`), and inputs that are the very same path end with a
    numeric suffix.
    """
    sources = [Path(source) for source in sources]
    # Which path pieces each name is built from, innermost last: the stem, or
    # the directory for a generic stem. Collisions prepend the next directory up.
    pieces = []
    for source in sources:
        chain = [part for part in source.parts[:-1] if part not in ('/', '')]
        if source.stem in {'raster', 'figure'} and chain:
            pieces.append([chain[:-1], [chain[-1]]])
        else:
            pieces.append([chain, [source.stem]])
    while True:
        names = ['-'.join(used) for _, used in pieces]
        duplicates = {name for name, count in Counter(names).items() if count > 1}
        if not duplicates:
            return names
        grew = False
        for name, (remaining, used) in zip(names, pieces):
            if name in duplicates and remaining:
                used.insert(0, remaining.pop())
                grew = True
        if not grew:
            seen: Counter = Counter()
            for index, name in enumerate(names):
                if name in duplicates:
                    seen[name] += 1
                    names[index] = f'{name}-{seen[name]}'
            return names
