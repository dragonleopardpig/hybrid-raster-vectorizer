"""Turning graphic ink into ordered centreline points."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .components import Component, extract
from .preprocess import Page
from .primitives import Arrow, Rule, TickSet


@dataclass
class Trace:
    """One stroke of a drawing, as points along its centreline."""

    points: np.ndarray = field(repr=False)
    stroke_width: float
    components: list[Component] = field(default_factory=list)
    method: str = "column"
    dash: float = 0.0
    gap: float = 0.0

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


_NODE_KERNEL = np.ones((3, 3), np.float32)


def skeleton_nodes(mask: np.ndarray, *, middle: float = 0.0) -> tuple[int, int]:
    """Count where the medial axis ends and where it branches.

    With ``middle`` set, only the central share of the shape is counted, along
    its own long axis. An arrow branches at its heads and nowhere else, while a
    word branches all the way along, so ignoring the ends tells them apart.
    """
    spine = skeletonize(mask > 0).astype(np.float32)
    if not spine.any():
        return 0, 0
    counted = cv2.filter2D(spine, cv2.CV_32F, _NODE_KERNEL, borderType=cv2.BORDER_CONSTANT)
    neighbours = np.rint(counted - spine).astype(np.int32)
    live = spine > 0

    if middle > 0.0:
        ys, xs = np.nonzero(live)
        if xs.size == 0:
            return 0, 0
        points = np.column_stack([xs, ys]).astype(float)
        centred = points - points.mean(axis=0)
        _u, _spread, axes = np.linalg.svd(centred, full_matrices=False)
        along = centred @ axes[0]
        reach = float(np.max(np.abs(along))) or 1.0
        keep = np.abs(along) <= middle * reach
        inner = np.zeros_like(live)
        inner[ys[keep], xs[keep]] = True
        live = live & inner

    ends = int(np.count_nonzero(live & (neighbours == 1)))
    junctions = int(np.count_nonzero(live & (neighbours >= 3)))
    return ends, junctions


def looks_like_text(
    component: Component, text_height: float, *, minimum_nodes: int = 12, tallest: float = 2.5
) -> bool:
    """Distinguish a word whose letters touch from a line that was drawn.

    In heavy type a whole word can arrive as one component, wide enough to pass
    for a curve, and then it is traced as a squiggle and never read at all. A
    word branches all the way along; a line does not, and the branches an
    arrowhead or a ragged scan add are at the ends or few. Counted over the
    middle only, words on these figures score 18 and 49 while dimension arrows
    score 6 to 9.
    """
    if component.height > tallest * text_height:
        return False
    ends, junctions = skeleton_nodes(component.mask, middle=0.6)
    return ends + junctions >= minimum_nodes


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


def _across(ink: np.ndarray, point: np.ndarray, normal: np.ndarray, limit: int) -> int:
    """Width of the ink through a point, measured across the given direction."""
    height, width = ink.shape
    span = 0
    for sign in (-1.0, 1.0):
        for step in range(1, limit + 1):
            probe = point + sign * step * normal
            x, y = int(round(probe[0])), int(round(probe[1]))
            if not (0 <= x < width and 0 <= y < height) or ink[y, x] == 0:
                break
            span += 1
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= x < width and 0 <= y < height and ink[y, x] > 0:
        span += 1
    return span


def arrowheads_on(
    points: np.ndarray, ink: np.ndarray, stroke_width: float, *, look: int = 40
) -> tuple[Arrow | None, Arrow | None]:
    """Find a solid head at either end of a traced stroke.

    Only the long straight rules of a plot were checked for one, so the heads on
    a dimension line -- which is short, and often at an angle -- were traced
    through as if they were part of the line and then not drawn.
    """
    from .primitives import _find_arrow

    if points.shape[0] < 6:
        return None, None

    limit = max(6, int(6 * stroke_width))
    found: list[Arrow | None] = []
    for at_end in (False, True):
        ordered = points[::-1] if at_end else points
        step = min(look, ordered.shape[0] - 1)
        direction = ordered[step] - ordered[0]
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            found.append(None)
            continue
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])

        profile = np.array(
            [_across(ink, ordered[i], normal, limit) for i in range(step + 1)],
            dtype=np.int32,
        )
        # The stroke's own pen width is the body to compare against. Taking it
        # from the sampled window instead measures the head, which is most of
        # what that window covers, and then nothing can flare above it.
        arrow = _find_arrow(profile, stroke_width, at_end=False, search=step + 1)
        if arrow is not None:
            arrow.at_end = at_end
        found.append(arrow)
    return found[0], found[1]


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


def follow_dashed_curves(
    pieces: list[Component],
    stroke_width: float,
    maximum_gap: float,
    *,
    minimum: int = 4,
    forward: float = 0.55,
    regularity: float = 0.3,
    widest_gap: float = 3.5,
    shortest_run: float = 8.0,
) -> list[Trace]:
    """Join the marks of a broken line that bends.

    A straight one can be found by the line its marks share; a dashed curve has
    no such line. Each mark is followed instead, taking the nearest mark that
    lies ahead in the direction the last one was heading, which lets the run
    turn. The spacing still has to repeat, or a caption's letters would be
    followed just as happily.
    """
    from .dashes import _period

    traced = [
        trace
        for trace in (trace_component(piece) for piece in pieces)
        if trace is not None and trace.points.shape[0] >= 3
    ]
    if len(traced) < minimum:
        return []
    traced.sort(key=lambda trace: float(trace.points[0][0]))

    reach = 3.0 * maximum_gap
    used: set[int] = set()
    curves: list[Trace] = []

    for seed in traced:
        if id(seed) in used:
            continue
        run = [seed]
        used.add(id(seed))
        while True:
            last = run[-1].points
            tip = last[-1]
            back = last[max(0, last.shape[0] - 5)]
            heading = tip - back
            size = float(np.linalg.norm(heading))
            heading = heading / size if size > 1e-6 else None

            best: tuple[float, Trace] | None = None
            for candidate in traced:
                if id(candidate) in used:
                    continue
                step = candidate.points[0] - tip
                distance = float(np.linalg.norm(step))
                if distance <= 0 or distance > reach:
                    continue
                if heading is not None and float(np.dot(step / distance, heading)) < forward:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, candidate)
            if best is None:
                break
            used.add(id(best[1]))
            run.append(best[1])

        if len(run) < minimum:
            for trace in run[1:]:
                used.discard(id(trace))
            if len(run) > 1:
                used.discard(id(run[0]))
            continue

        centres = np.array(
            [trace.points.mean(axis=0) for trace in run], dtype=float
        )
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        period, residual = _period(steps)
        if period <= 0 or residual > regularity:
            for trace in run:
                used.discard(id(trace))
            continue

        points = np.vstack([trace.points for trace in run])
        spans = [float(np.linalg.norm(t.points[-1] - t.points[0])) for t in run]
        dash = max(1.0, float(np.median(spans)))
        gap = float(max(1.0, period - dash))
        reach_of_run = float(np.linalg.norm(centres[-1] - centres[0]))
        # A handful of marks a few dash-lengths apart, or strung barely further
        # than one mark, is a legend or a caption rather than a line.
        if gap > widest_gap * dash or reach_of_run < shortest_run * dash:
            for trace in run:
                used.discard(id(trace))
            continue

        curves.append(
            Trace(
                points=points,
                stroke_width=float(np.median([trace.stroke_width for trace in run])),
                components=[c for trace in run for c in trace.components],
                method="column",
                dash=dash,
                gap=gap,
            )
        )
    return curves


def partition(
    page: Page,
    working: np.ndarray,
    rules: list[Rule],
    ticks: list[TickSet],
    text_height: float,
) -> tuple[list, list, list[Trace], list[Component]]:
    """Split what is left into broken lines, frames, traced strokes and text."""
    from .dashes import find_dashed_lines, is_dash
    from .shapes import Frame, is_frame

    furniture = furniture_mask(page, rules, ticks)
    residual = cv2.bitwise_and(working, cv2.bitwise_not(furniture))
    pieces = extract(residual)

    # Broken lines go first: each of their marks is small enough to be taken for
    # a letter, and once grouped into a label the line cannot be recovered.
    dashed, claimed = find_dashed_lines(
        pieces, page.stroke_width, (page.width, page.height), ink=residual
    )
    pieces = [piece for piece in pieces if id(piece) not in claimed]

    # What is left of the broken lines does not run straight, so its marks
    # cannot be grouped by a shared line. They are followed instead: each mark
    # continues in the direction the last one was heading.
    maximum_gap = max(4.0 * page.stroke_width, 0.015 * page.width)
    spare = [
        piece
        for piece in pieces
        if is_dash(piece, page.stroke_width, (page.width, page.height))
    ]
    curved = follow_dashed_curves(spare, page.stroke_width, maximum_gap)
    if curved:
        taken = {id(c) for trace in curved for c in trace.components}
        pieces = [piece for piece in pieces if id(piece) not in taken]

    frames: list[Frame] = []
    traces: list[Trace] = list(curved)
    leftovers: list[Component] = []
    for component in pieces:
        if is_frame(component, page.stroke_width):
            frames.append(
                Frame(
                    component=component,
                    x=component.x, y=component.y,
                    width=component.width, height=component.height,
                )
            )
            continue
        graphic = is_graphic(component, text_height, page.stroke_width)
        if graphic and not looks_like_text(component, text_height):
            trace = trace_component(component)
            if trace is not None:
                traces.append(trace)
                continue
        leftovers.append(component)

    return dashed, frames, chain(traces, maximum_gap=maximum_gap), leftovers


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
    return dashed, frames, chain(traces, maximum_gap=maximum_gap), leftovers
