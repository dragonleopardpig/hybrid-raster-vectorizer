"""Turning graphic ink into ordered centreline points."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .components import Component
from .preprocess import Page
from .primitives import Rule, TickSet


@dataclass
class Trace:
    """One stroke of a drawing, as points along its centreline."""

    points: np.ndarray = field(repr=False)
    stroke_width: float
    components: list[Component] = field(default_factory=list)
    method: str = "column"

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]


def furniture_mask(page: Page, rules: list[Rule], ticks: list[TickSet]) -> np.ndarray:
    """Every pixel already explained by an axis, an arrowhead or a tick."""
    mask = np.zeros_like(page.ink)
    height, width = mask.shape

    for rule in rules:
        low, high = rule.band(pad=1.5)
        start, end = int(np.floor(rule.start)), int(np.ceil(rule.end))
        if rule.orientation == "horizontal":
            mask[max(0, low) : min(height, high + 1), max(0, start) : min(width, end + 1)] = 255
        else:
            mask[max(0, start) : min(height, end + 1), max(0, low) : min(width, high + 1)] = 255

        for arrow in rule.arrows:
            reach = int(np.ceil(arrow.length)) + 2
            flare = int(np.ceil(arrow.width / 2.0)) + 2
            tip = rule.end if arrow.at_end else rule.start
            along = (int(tip) - reach, int(tip) + 1) if arrow.at_end else (int(tip), int(tip) + reach)
            across = (int(rule.position) - flare, int(rule.position) + flare + 1)
            if rule.orientation == "horizontal":
                mask[max(0, across[0]) : min(height, across[1]), max(0, along[0]) : min(width, along[1])] = 255
            else:
                mask[max(0, along[0]) : min(height, along[1]), max(0, across[0]) : min(width, across[1])] = 255

    for tick_set in ticks:
        half = max(2.0, tick_set.thickness) / 2.0 + 1.5
        near, far = int(np.floor(tick_set.near)), int(np.ceil(tick_set.far))
        for position in tick_set.positions:
            low, high = int(np.floor(position - half)), int(np.ceil(position + half))
            if tick_set.rule.orientation == "horizontal":
                mask[max(0, near) : min(height, far + 1), max(0, low) : min(width, high + 1)] = 255
            else:
                mask[max(0, low) : min(height, high + 1), max(0, near) : min(width, far + 1)] = 255

    return mask


def is_graphic(component: Component, text_height: float, stroke_width: float) -> bool:
    diagonal = float(np.hypot(component.width, component.height))
    thin_limit = max(3.0, 2.0 * stroke_width)
    return (
        diagonal > 3.0 * text_height
        and min(component.width, component.height) >= thin_limit
        and component.area >= 8.0 * stroke_width * stroke_width
    )


def _runs(column: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(column)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return np.split(indices, splits)


def _trace_axis(mask: np.ndarray, origin: tuple[int, int], *, by_column: bool, limit: float) -> np.ndarray | None:
    """Midpoint of the ink run in each column (or row); valid while single-valued."""
    working = mask if by_column else mask.T
    height, width = working.shape

    along: list[float] = []
    across: list[float] = []
    multi = 0
    for index in range(width):
        runs = _runs(working[:, index] > 0)
        if not runs:
            continue
        if len(runs) > 1:
            multi += 1
        run = max(runs, key=len)
        along.append(float(index))
        across.append(float(run.mean()))

    if not along or multi > limit * len(along):
        return None

    x_offset, y_offset = origin
    if by_column:
        return np.column_stack([np.asarray(along) + x_offset, np.asarray(across) + y_offset])
    return np.column_stack([np.asarray(across) + x_offset, np.asarray(along) + y_offset])


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace_skeleton(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray | None:
    """Longest path through the medial axis, for strokes that are not functions."""
    skeleton = skeletonize(mask > 0)
    pixels = {(int(y), int(x)) for y, x in zip(*np.nonzero(skeleton))}
    if len(pixels) < 2:
        return None

    def neighbours(node: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = node
        return [(y + dy, x + dx) for dy, dx in _NEIGHBOURS if (y + dy, x + dx) in pixels]

    def farthest(source: tuple[int, int]) -> tuple[tuple[int, int], dict]:
        previous = {source: None}
        queue = [source]
        last = source
        while queue:
            node = queue.pop(0)
            last = node
            for candidate in neighbours(node):
                if candidate not in previous:
                    previous[candidate] = node
                    queue.append(candidate)
        return last, previous

    endpoints = [node for node in pixels if len(neighbours(node)) == 1]
    seed = endpoints[0] if endpoints else next(iter(pixels))
    far, _ = farthest(seed)
    other, previous = farthest(far)

    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = other
    while node is not None:
        path.append(node)
        node = previous[node]
    if len(path) < 2:
        return None

    x_offset, y_offset = origin
    return np.array([[x + x_offset, y + y_offset] for y, x in path], dtype=float)


def component_stroke_width(component: Component) -> float:
    padded = cv2.copyMakeBorder(component.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    medial = skeletonize(padded > 0)
    samples = distance[medial]
    if samples.size == 0:
        return 1.0
    return float(2.0 * np.median(samples))


def trace_component(component: Component, *, multi_run_limit: float = 0.15) -> Trace | None:
    origin = (component.x, component.y)
    for by_column in (True, False):
        points = _trace_axis(component.mask, origin, by_column=by_column, limit=multi_run_limit)
        if points is not None and points.shape[0] >= 2:
            return Trace(
                points=points,
                stroke_width=component_stroke_width(component),
                components=[component],
                method="column" if by_column else "row",
            )

    points = _trace_skeleton(component.mask, origin)
    if points is None or points.shape[0] < 2:
        return None
    return Trace(
        points=points,
        stroke_width=component_stroke_width(component),
        components=[component],
        method="skeleton",
    )


def chain(traces: list[Trace], *, maximum_gap: float) -> list[Trace]:
    """Rejoin strokes an occluding axis or tick cut into pieces."""
    functional = [trace for trace in traces if trace.method == "column"]
    other = [trace for trace in traces if trace.method != "column"]
    functional.sort(key=lambda trace: float(trace.points[0][0]))

    chains: list[list[Trace]] = []
    for trace in functional:
        if chains:
            previous = chains[-1][-1]
            gap = float(trace.points[0][0] - previous.points[-1][0])
            rise = abs(float(trace.points[0][1] - previous.points[-1][1]))
            compatible = abs(trace.stroke_width - previous.stroke_width) <= 0.5 * max(
                trace.stroke_width, previous.stroke_width
            )
            if 0 <= gap <= maximum_gap and rise <= maximum_gap and compatible:
                chains[-1].append(trace)
                continue
        chains.append([trace])

    merged: list[Trace] = []
    for group in chains:
        if len(group) == 1:
            merged.append(group[0])
            continue
        points = np.vstack([trace.points for trace in group])
        merged.append(
            Trace(
                points=points[np.argsort(points[:, 0])],
                stroke_width=float(np.median([trace.stroke_width for trace in group])),
                components=[c for trace in group for c in trace.components],
                method="column",
            )
        )
    return merged + other


def extract_curves(
    page: Page,
    rules: list[Rule],
    ticks: list[TickSet],
    components: list[Component],
    text_height: float,
) -> tuple[list[Trace], list[Component]]:
    """Split ink into traced graphic strokes and the components left for text."""
    furniture = furniture_mask(page, rules, ticks)
    residual = cv2.bitwise_and(page.ink, cv2.bitwise_not(furniture))

    graphic_labels = {
        component.label for component in components if is_graphic(component, text_height, page.stroke_width)
    }
    pieces = []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(residual, 8)
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        mask = (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
        piece = Component(
            label=label,
            x=x,
            y=y,
            width=width,
            height=height,
            area=area,
            centroid=(float(x + width / 2.0), float(y + height / 2.0)),
            mask=mask,
        )
        pieces.append(piece)

    graphics = [piece for piece in pieces if is_graphic(piece, text_height, page.stroke_width)]
    leftovers = [piece for piece in pieces if piece not in graphics]

    traces = [trace for trace in (trace_component(piece) for piece in graphics) if trace is not None]
    maximum_gap = max(4.0 * page.stroke_width, 0.015 * page.width)
    return chain(traces, maximum_gap=maximum_gap), leftovers
