"""Detection of the straight-line furniture of a plot: axes, arrowheads, ticks."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .preprocess import Page


@dataclass
class Arrow:
    """A solid head at one end of a rule, reported as an SVG marker."""

    at_end: bool
    length: float
    width: float


@dataclass
class Rule:
    """A long straight line: a plot axis, a frame edge or a leader line."""

    orientation: str
    position: float
    start: float
    end: float
    thickness: float
    arrows: list[Arrow] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def length(self) -> float:
        return abs(self.end - self.start)

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.orientation == "horizontal":
            return (self.start, self.position), (self.end, self.position)
        return (self.position, self.start), (self.position, self.end)

    def band(self, pad: float = 1.0) -> tuple[int, int]:
        half = self.thickness / 2.0 + pad
        return int(np.floor(self.position - half)), int(np.ceil(self.position + half))


@dataclass
class TickSet:
    """Evenly spaced marks belonging to one rule."""

    rule: Rule
    positions: list[float]
    near: float
    far: float
    thickness: float
    spacing: float | None
    confidence: float = 1.0


def _transpose(rule_like: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rule_like.T)


def _profile_run(mask: np.ndarray, centre: int) -> np.ndarray:
    """For each column, the length of the vertical ink run through ``centre``."""
    height, width = mask.shape
    centre = int(np.clip(centre, 0, height - 1))
    solid = mask > 0
    lengths = np.zeros(width, dtype=np.int32)
    for column in range(width):
        if not solid[centre, column]:
            continue
        top = centre
        while top > 0 and solid[top - 1, column]:
            top -= 1
        bottom = centre
        while bottom < height - 1 and solid[bottom + 1, column]:
            bottom += 1
        lengths[column] = bottom - top + 1
    return lengths


def _find_arrow(
    profile: np.ndarray,
    body_thickness: float,
    *,
    at_end: bool,
    search: int,
    minimum_flare: float = 2.2,
    proportions: tuple[float, float] = (0.4, 3.0),
) -> Arrow | None:
    """Walk inwards from a tip while the perpendicular thickness keeps growing."""
    if profile.size == 0:
        return None
    order = profile[::-1] if at_end else profile
    limit = min(search, order.size)

    tip = 0
    while tip < limit and order[tip] == 0:
        tip += 1
    if tip >= limit:
        return None

    peak = order[tip]
    length = 1
    index = tip + 1
    slack = 0
    while index < limit:
        value = order[index]
        if value == 0:
            break
        if value >= peak:
            peak = value
            length = index - tip + 1
            slack = 0
        else:
            slack += 1
            if slack > max(2, int(0.25 * body_thickness)):
                break
        index += 1

    if peak < minimum_flare * max(1.0, body_thickness) or length < 3:
        return None

    # An arrowhead is about as long as it is wide. A long thin spike is where a
    # filled area was cut away, and a short very wide one is a crossing rule;
    # neither tapers to a point, and neither is an arrowhead.
    slimmest, stoutest = proportions
    if not (slimmest <= length / max(1.0, peak) <= stoutest):
        return None
    return Arrow(at_end=at_end, length=float(length), width=float(peak))


def _rule_from_mask(
    mask: np.ndarray, orientation: str, ink: np.ndarray, page_span: int
) -> Rule | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    if orientation == "horizontal":
        position = float(np.average(ys))
        start, end = float(xs.min()), float(xs.max())
        working = ink
    else:
        position = float(np.average(xs))
        start, end = float(ys.min()), float(ys.max())
        working = _transpose(ink)

    centre = int(round(position))
    lo = max(0, int(start))
    hi = min(working.shape[1], int(end) + 1)
    window = working[:, lo:hi]
    profile = _profile_run(window, centre)
    populated = profile[profile > 0]
    if populated.size == 0:
        return None
    thickness = float(np.median(populated))

    search = max(8, min(int(0.08 * page_span), 120))
    rule = Rule(
        orientation=orientation,
        position=position,
        start=start,
        end=end,
        thickness=thickness,
    )
    for at_end in (False, True):
        arrow = _find_arrow(profile, thickness, at_end=at_end, search=search)
        if arrow is not None:
            rule.arrows.append(arrow)
    return rule


def detect_rules(
    page: Page,
    *,
    ink: np.ndarray | None = None,
    minimum_length_fraction: float = 0.22,
    maximum_thickness_factor: float = 3.0,
) -> list[Rule]:
    rules: list[Rule] = []
    ink = page.ink if ink is None else ink
    thickness_limit = max(6.0, maximum_thickness_factor * page.stroke_width)

    for orientation in ("horizontal", "vertical"):
        span = page.width if orientation == "horizontal" else page.height
        run = max(24, int(minimum_length_fraction * span))
        kernel = (
            np.ones((1, run), np.uint8)
            if orientation == "horizontal"
            else np.ones((run, 1), np.uint8)
        )
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(opened, 8)
        for label in range(1, count):
            rule = _rule_from_mask(labels == label, orientation, ink, span)
            if rule is None or rule.length < run:
                continue
            # A long run through a filled shape is not a rule; a rule is thin.
            if rule.thickness > thickness_limit:
                continue
            rules.append(rule)

    rules.sort(key=lambda item: item.length, reverse=True)

    # Where another rule crosses near an end, the perpendicular thickness flares
    # exactly as an arrowhead does. The crossing is known, so the flare is not
    # evidence of anything.
    for rule in rules:
        kept = []
        for arrow in rule.arrows:
            tip = rule.end if arrow.at_end else rule.start
            reach = arrow.length + 2.0 * page.stroke_width
            crossed = any(
                other is not rule
                and other.orientation != rule.orientation
                and abs(other.position - tip) <= reach
                and min(other.start, other.end) - reach
                <= rule.position
                <= max(other.start, other.end) + reach
                for other in rules
            )
            if not crossed:
                kept.append(arrow)
        rule.arrows = kept
    return rules


def _lattice(positions: list[float], tolerance: float) -> tuple[float | None, float]:
    """Modal spacing of a set of marks, ignoring gaps where marks are missing."""
    if len(positions) < 3:
        return None, 0.0
    ordered = np.sort(np.asarray(positions, dtype=float))
    gaps = np.diff(ordered)
    gaps = gaps[gaps > tolerance]
    if gaps.size == 0:
        return None, 0.0

    candidate = float(np.median(gaps))
    if candidate <= 0:
        return None, 0.0
    offsets = np.round((ordered - ordered[0]) / candidate)
    with np.errstate(invalid="ignore"):
        slope, intercept = np.polyfit(offsets, ordered, 1)
    residual = float(np.max(np.abs(offsets * slope + intercept - ordered)))
    return float(slope), residual


def _snap(positions: list[float], spacing: float) -> tuple[list[float], float]:
    """Regular marks are a lattice; snapping removes contamination from touching ink."""
    base = positions[0]
    offsets = np.round((np.asarray(positions) - base) / spacing)
    slope, intercept = np.polyfit(offsets, np.asarray(positions), 1)
    snapped = offsets * slope + intercept
    residual = float(np.max(np.abs(snapped - np.asarray(positions))))
    return [float(v) for v in snapped], residual


def detect_ticks(
    page: Page,
    rule: Rule,
    *,
    ink: np.ndarray | None = None,
    others: list[Rule] | None = None,
    reach_factor: float = 7.0,
    snap_tolerance: float = 0.15,
    least_confidence: float = 0.5,
) -> TickSet | None:
    source = page.ink if ink is None else ink
    ink = source if rule.orientation == "horizontal" else _transpose(source)
    height, width = ink.shape

    low, high = rule.band(pad=1.0)
    reach = int(max(10.0, reach_factor * page.stroke_width))
    start, end = int(rule.start), int(rule.end)

    working = ink.copy()
    working[max(0, low) : min(height, high + 1), max(0, start) : min(width, end + 1)] = 0

    # A perpendicular rule crossing this one looks exactly like a tick on both sides.
    crossings: list[tuple[float, float]] = []
    for other in others or []:
        if other is rule or other.orientation == rule.orientation:
            continue
        crossings.append((other.position, max(2.0, other.thickness)))

    def crosses(centre: float) -> bool:
        return any(abs(centre - at) <= 1.5 * thick for at, thick in crossings)

    # An arrowhead flares on both sides of the rule, which is exactly what a
    # tick does; its extent is already known, so exclude it by position.
    arrow_zones: list[tuple[float, float]] = []
    for arrow in rule.arrows:
        pad = max(2.0, page.stroke_width)
        if arrow.at_end:
            arrow_zones.append((rule.end - arrow.length - pad, rule.end + pad))
        else:
            arrow_zones.append((rule.start - pad, rule.start + arrow.length + pad))

    def in_arrowhead(centre: float) -> bool:
        return any(low <= centre <= high for low, high in arrow_zones)

    # A tick is as narrow as the pen that drew it; that alone separates it from
    # curve ink grazing the axis, which is far wider where it enters the strip.
    narrow = max(3.0, 2.5 * page.stroke_width)
    sides: dict[str, list[tuple[float, int, int, int]]] = {"near": [], "far": []}
    windows = {
        "near": (max(0, low - reach), max(0, low)),
        "far": (min(height, high + 1), min(height, high + 1 + reach)),
    }

    for side, (top, bottom) in windows.items():
        if bottom <= top:
            continue
        strip = working[top:bottom, :]
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(strip, 8)
        for label in range(1, count):
            x, y, w, h, _area = (int(v) for v in stats[label])
            if w > narrow or h < 2:
                continue
            touching = (y == 0) if side == "far" else (y + h) >= strip.shape[0]
            if not touching:
                continue
            centre = float(centroids[label][0])
            if not (rule.start <= centre <= rule.end) or crosses(centre):
                continue
            if in_arrowhead(centre):
                continue
            sides[side].append((centre, top + y, top + y + h, w))

    candidates = sorted(sides["near"] + sides["far"])
    if len(candidates) < 2:
        return None

    spacing, _residual = _lattice([c[0] for c in candidates], tolerance=2.0)
    merge_window = max(2.0 * page.stroke_width, (spacing or 0.0) / 6.0)

    groups: list[list[tuple[float, int, int, int]]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate[0] - groups[-1][-1][0] <= merge_window:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    positions = [float(np.median([item[0] for item in group])) for group in groups]
    if len(positions) < 2:
        return None

    spacing, residual = _lattice(positions, tolerance=2.0)
    confidence = 0.75
    if spacing is not None and spacing > 0:
        relative = residual / spacing
        confidence = float(np.clip(1.0 - relative / snap_tolerance, 0.3, 1.0))
        if relative <= snap_tolerance:
            positions, residual = _snap(positions, spacing)

    # Ticks are evenly spaced by definition, and this was measuring how evenly
    # and then reporting it rather than acting on it. Where a curve crosses an
    # axis three or four times the crossings pass every other test a tick set
    # has, and were drawn as ticks. A real set fits a lattice at 0.75 or better
    # -- including one on the interference scan with ticks missing, and one on
    # waves1.png spaced three apart -- and a false one sits on the floor at 0.3.
    if confidence < least_confidence:
        return None

    # Marks are uniform, so one measured extent per side describes them all.
    near = float(np.median([item[1] for item in sides["near"]])) if sides["near"] else float(low)
    far = float(np.median([item[2] for item in sides["far"]])) if sides["far"] else float(high)
    widths = [item[3] for item in candidates]

    return TickSet(
        rule=rule,
        positions=positions,
        near=near,
        far=far,
        thickness=float(np.median(widths)),
        spacing=spacing,
        confidence=confidence,
    )
