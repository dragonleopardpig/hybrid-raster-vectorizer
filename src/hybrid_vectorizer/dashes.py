"""Recognising a broken line as one line rather than as a row of marks."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .components import Component


@dataclass
class DashedLine:
    """A line drawn as repeated marks, with the pattern that draws it."""

    start: tuple[float, float]
    end: tuple[float, float]
    dash: float
    gap: float
    stroke_width: float
    components: list[Component] = field(default_factory=list, repr=False)

    @property
    def length(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))


def _rect(component: Component) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Centre, axis, length and breadth of the smallest rectangle around a mark."""
    points = cv2.findNonZero(component.mask)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    centre = np.array([cx + component.x, cy + component.y], dtype=float)
    if w < h:
        w, h = h, w
        angle += 90.0
    radians = np.radians(angle)
    axis = np.array([np.cos(radians), np.sin(radians)], dtype=float)
    return centre, axis, float(w), float(h)


def is_dash(component: Component, stroke_width: float, page_size: tuple[int, int]) -> bool:
    """A short straight mark no wider than the pen: a dash, or a dot."""
    if component.area < 0.3 * stroke_width * stroke_width:
        return False
    _centre, _axis, long_side, short_side = _rect(component)
    if short_side > 2.5 * stroke_width:
        return False
    if long_side > 0.12 * min(page_size):
        return False
    filled = component.area / max(1.0, long_side * max(short_side, 1.0))
    return filled >= 0.5


@dataclass
class _Mark:
    component: Component
    centre: np.ndarray
    axis: np.ndarray
    length: float


def _collect(components: list[Component], stroke_width: float, page_size) -> list[_Mark]:
    marks: list[_Mark] = []
    for component in components:
        if not is_dash(component, stroke_width, page_size):
            continue
        centre, axis, long_side, _short = _rect(component)
        marks.append(_Mark(component=component, centre=centre, axis=axis, length=long_side))
    return marks


def _extend(
    seed: _Mark, direction: np.ndarray, available: list[_Mark], corridor: float, reach: float
) -> list[_Mark]:
    """Walk along a direction, taking the nearest mark still in the corridor."""
    chain = [seed]
    current = seed
    while True:
        best: tuple[float, _Mark] | None = None
        for mark in available:
            if any(mark is member for member in chain):
                continue
            offset = mark.centre - current.centre
            along = float(np.dot(offset, direction))
            across = abs(float(offset[0] * -direction[1] + offset[1] * direction[0]))
            if along <= 0 or along > reach or across > corridor:
                continue
            if best is None or along < best[0]:
                best = (along, mark)
        if best is None:
            return chain
        chain.append(best[1])
        current = best[1]


def find_dashed_lines(
    components: list[Component],
    stroke_width: float,
    page_size: tuple[int, int],
    *,
    minimum: int = 3,
    regularity: float = 0.20,
    length_spread: float = 2.5,
    breadth_spread: float = 1.6,
    widest_gap: float = 3.5,
) -> tuple[list[DashedLine], set[int]]:
    """Group collinear, evenly spaced, identical marks into the lines they draw.

    The marks are walked along their own direction rather than fitted globally,
    because a figure holds several broken lines at once and a global fit happily
    joins marks from two of them that happen to line up.

    A row of tick labels offers a fraction bar apiece: short, straight, thin,
    horizontal, and collinear. What separates them is the period. Measured on
    these figures, a real broken line spaces its marks to within 11-12% while a
    row of labels, centred on ticks and of differing widths, manages only 31-81%.
    Dash length is deliberately not used: the marks at each end of a line are
    clipped, so a true line varies its lengths about as much as a false one.
    """
    marks = _collect(components, stroke_width, page_size)
    if len(marks) < minimum:
        return [], set()

    corridor = max(2.0, 1.2 * stroke_width)
    lines: list[DashedLine] = []
    used: set[int] = set()

    for seed in sorted(marks, key=lambda mark: -mark.length):
        if id(seed.component) in used:
            continue
        available = [mark for mark in marks if id(mark.component) not in used]
        reach = max(6.0 * stroke_width, 4.0 * max(seed.length, 1.0))

        forward = _extend(seed, seed.axis, available, corridor, reach)
        backward = _extend(seed, -seed.axis, available, corridor, reach)
        chain = list(reversed(backward[1:])) + forward
        if len(chain) < minimum:
            continue

        centres = np.array([mark.centre for mark in chain])
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        if steps.size == 0 or steps.mean() <= 0:
            continue
        if steps.std() / steps.mean() > regularity:
            continue

        lengths = np.array([mark.length for mark in chain])
        if lengths.min() <= 0 or lengths.max() / lengths.min() > length_spread:
            continue
        breadths = np.array([max(_rect(mark.component)[3], 1.0) for mark in chain])
        if breadths.max() / breadths.min() > breadth_spread:
            continue

        dash = float(np.median(lengths))
        gap = float(max(1.0, steps.mean() - dash))
        if gap > widest_gap * dash:
            continue
        half = 0.5 * dash * seed.axis
        first, last = centres[0] - half, centres[-1] + half

        lines.append(
            DashedLine(
                start=(float(first[0]), float(first[1])),
                end=(float(last[0]), float(last[1])),
                dash=dash,
                gap=gap,
                stroke_width=float(np.median([
                    min(_rect(mark.component)[3], 2.5 * stroke_width) for mark in chain
                ])),
                components=[mark.component for mark in chain],
            )
        )
        used.update(id(mark.component) for mark in chain)

    return lines, used
