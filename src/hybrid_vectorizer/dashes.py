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


def _hemmed_in(centre: np.ndarray, axis: np.ndarray, length: float, ink: np.ndarray) -> bool:
    """Is there ink on both sides of this mark, across its own direction?

    A fraction bar is short, straight, thin and collinear with the next label's
    bar, and passes every test a dash does. What it has that a dash has not is a
    numerator above it and a denominator below.
    """
    height, width = ink.shape
    normal = np.array([-axis[1], axis[0]])
    for fraction in (0.25, 0.4, 0.6, 1.0, 1.5):
        reach = fraction * length
        sides = 0
        for sign in (-1.0, 1.0):
            point = centre + sign * reach * normal
            x, y = int(round(point[0])), int(round(point[1]))
            x0, x1 = max(0, x - 2), min(width, x + 3)
            y0, y1 = max(0, y - 2), min(height, y + 3)
            if x1 > x0 and y1 > y0 and np.any(ink[y0:y1, x0:x1]):
                sides += 1
        if sides == 2:
            return True
    return False


def _collect(
    components: list[Component],
    stroke_width: float,
    page_size,
    ink: np.ndarray | None = None,
) -> list[_Mark]:
    marks: list[_Mark] = []
    for component in components:
        if not is_dash(component, stroke_width, page_size):
            continue
        centre, axis, long_side, _short = _rect(component)
        if ink is not None and _hemmed_in(centre, axis, long_side, ink):
            continue
        marks.append(_Mark(component=component, centre=centre, axis=axis, length=long_side))
    return marks


def _period(steps: np.ndarray) -> tuple[float, float]:
    """The repeat of a run of marks, tolerating places where one is missing.

    A broken line passing behind something else loses a dash or two, leaving a
    double-length gap. Judged on the spread of raw steps that reads as
    irregular; judged as multiples of one repeat it is exactly as regular as the
    rest of the line.
    """
    if steps.size == 0:
        return 0.0, np.inf

    best = (0.0, np.inf)
    for candidate in np.unique(steps):
        if candidate <= 0:
            continue
        counts = np.maximum(1.0, np.round(steps / candidate))
        residual = float(np.mean(np.abs(steps - counts * candidate)) / candidate)
        if residual < best[1]:
            best = (float(candidate), residual)
    return best


def _breadth(mark: _Mark) -> float:
    return max(_rect(mark.component)[3], 1.0)


def _alike(mark: _Mark, anchor: _Mark, lengths: tuple[float, float], breadths: tuple[float, float]) -> bool:
    ratio = mark.length / max(anchor.length, 1e-6)
    if not (lengths[0] <= ratio <= lengths[1]):
        return False
    thickness = _breadth(mark) / _breadth(anchor)
    return breadths[0] <= thickness <= breadths[1]


def find_dashed_lines(
    components: list[Component],
    stroke_width: float,
    page_size: tuple[int, int],
    *,
    ink: np.ndarray | None = None,
    minimum: int = 3,
    regularity: float = 0.20,
    angle_tolerance: float = 12.0,
    length_ratio: tuple[float, float] = (0.5, 2.0),
    breadth_ratio: tuple[float, float] = (0.5, 2.0),
    widest_gap: float = 3.5,
) -> tuple[list[DashedLine], set[int]]:
    """Group collinear, evenly spaced, matching marks into the lines they draw.

    Every mark proposes the line through itself, and the marks that lie along it
    answer. Walking outward from a seed instead makes the result depend on which
    mark was picked first and on how far a step may reach: widening that reach by
    half took one figure from two lines to one, because a chain consumed marks
    another line needed.

    A row of tick labels offers a fraction bar apiece: short, straight, thin,
    horizontal and collinear. What separates them is the period, measured as
    multiples of one repeat so that a line passing behind something else, and
    losing a dash to it, still reads as regular.
    """
    marks = _collect(components, stroke_width, page_size, ink)
    if len(marks) < minimum:
        return [], set()

    corridor = max(2.0, 1.2 * stroke_width)
    proposals: list[tuple[float, float, float, list[_Mark]]] = []

    for anchor in marks:
        direction = anchor.axis
        normal = np.array([-direction[1], direction[0]])
        offset = float(np.dot(anchor.centre, normal))

        group: list[_Mark] = []
        for mark in marks:
            turn = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(mark.axis, direction))), 0.0, 1.0))))
            if turn > angle_tolerance:
                continue
            if abs(float(np.dot(mark.centre, normal)) - offset) > corridor:
                continue
            if not _alike(mark, anchor, length_ratio, breadth_ratio):
                continue
            group.append(mark)

        if len(group) < minimum:
            continue
        group.sort(key=lambda mark: float(np.dot(mark.centre, direction)))
        centres = np.array([mark.centre for mark in group])
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        period, residual = _period(steps)
        if period <= 0 or residual > regularity:
            continue

        span = float(np.linalg.norm(centres[-1] - centres[0]))
        proposals.append((span * len(group), period, span, group))

    proposals.sort(key=lambda item: -item[0])
    lines: list[DashedLine] = []
    used: set[int] = set()

    for _score, period, _span, group in proposals:
        remaining = [mark for mark in group if id(mark.component) not in used]
        if len(remaining) < minimum:
            continue
        centres = np.array([mark.centre for mark in remaining])
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        period, residual = _period(steps)
        if period <= 0 or residual > regularity:
            continue

        lengths = np.array([mark.length for mark in remaining])
        dash = float(np.median(lengths))
        gap = float(max(1.0, period - dash))
        if gap > widest_gap * dash:
            continue
        axis = remaining[0].axis
        half = 0.5 * dash * axis
        first, last = centres[0] - half, centres[-1] + half

        lines.append(
            DashedLine(
                start=(float(first[0]), float(first[1])),
                end=(float(last[0]), float(last[1])),
                dash=dash,
                gap=gap,
                stroke_width=float(np.median([_breadth(mark) for mark in remaining])),
                components=[mark.component for mark in remaining],
            )
        )
        used.update(id(mark.component) for mark in remaining)

    return lines, used
