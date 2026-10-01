"""Evaluate OCR and object counts against independent annotations."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import latex as tex


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


def evaluate(report: dict, reference: dict) -> dict:
    actual_counts = {
        'marker_sets': len(report.get('marker_series', [])),
        'markers': sum(series['count'] for series in report.get('marker_series', [])),
        'curves': len(report.get('curves', [])),
        'closed_curves': sum(curve.get('closed', False) for curve in report.get('curves', [])),
        'curve_arrowheads': sum(curve.get('arrowheads', 0) for curve in report.get('curves', [])),
    }
    result = {'object_counts': {
        name: {'expected': value, 'actual': actual_counts[name], 'correct': actual_counts[name] == value}
        for name, value in reference.get('counts', {}).items()
    }}
    expected = reference.get('labels', [])
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
    errors = 0
    characters = 0
    details = []
    for index, label in enumerate(expected):
        truth = normalise(tex.to_text(tex.parse(label['latex'])) if 'latex' in label else label['text'])
        candidate = predicted[matches[index]] if index in matches else {}
        source = candidate.get('source_reading', candidate.get('text', ''))
        reading = normalise(tex.to_text(tex.parse(source)) if candidate.get('engine') == 'formulaocr' else source)
        distance = edit_distance(truth, reading)
        errors += distance
        characters += len(truth)
        details.append({'expected': truth, 'reading': reading, 'correct': truth == reading,
                        'representation': candidate.get('representation'),
                        'id': candidate.get('id'), 'character_errors': distance})
    result.update({
        'expected_labels': len(expected),
        'matched_labels': len(matches),
        'exact_ocr_labels': sum(label['correct'] for label in details),
        'character_error_rate': errors / max(1, characters),
        'unmatched_predictions': len(predicted) - len(matches),
        'representations': dict(Counter(label.get('representation', 'unknown') for label in predicted)),
        'labels': details,
    })
    return result
