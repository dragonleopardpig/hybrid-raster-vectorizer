"""Preserve uncertain ink as exact vector contours, including holes and dots."""

from __future__ import annotations

import numpy as np


def mask_path(mask: np.ndarray) -> str:
    solid = np.pad(mask > 0, 1)
    centre = solid[1:-1, 1:-1]
    boundaries = (
        (centre & ~solid[:-2, 1:-1], (0, 0), (1, 0)),
        (centre & ~solid[1:-1, 2:], (1, 0), (1, 1)),
        (centre & ~solid[2:, 1:-1], (1, 1), (0, 1)),
        (centre & ~solid[1:-1, :-2], (0, 1), (0, 0)),
    )
    edges = {}
    for boundary, start_offset, end_offset in boundaries:
        for row, column in np.argwhere(boundary):
            start = (int(column) + start_offset[0], int(row) + start_offset[1])
            end = (int(column) + end_offset[0], int(row) + end_offset[1])
            edges.setdefault(start, []).append(end)
    paths = []
    while edges:
        start = next(iter(edges))
        current = start
        previous_direction = None
        corners = []
        while True:
            candidates = edges[current]
            if previous_direction is None or len(candidates) == 1:
                following = candidates[0]
            else:
                def turn(candidate):
                    direction = (candidate[0] - current[0], candidate[1] - current[1])
                    return previous_direction[0] * direction[1] - previous_direction[1] * direction[0]
                following = max(candidates, key=turn)
            candidates.remove(following)
            if not candidates:
                del edges[current]
            direction = (following[0] - current[0], following[1] - current[1])
            if direction != previous_direction:
                corners.append(current)
            previous_direction = direction
            current = following
            if current == start:
                break
        paths.append(f'M{corners[0][0]} {corners[0][1]} ' +
                     ' '.join(f'L{column} {row}' for column, row in corners[1:]) + ' Z')
    return ' '.join(paths)
