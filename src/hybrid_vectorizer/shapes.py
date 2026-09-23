"""Telling apart the things a plot is drawn with: strokes, solids and hatching."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .components import Component, extract
from .fitting import fit_bezier, path_data


@dataclass
class Hatch:
    """Evenly spaced parallel ruling that stands for a filled area."""

    angle: float
    spacing: float
    stroke_width: float
    coverage: float


@dataclass
class Outline:
    """A closed boundary, and the simplest shape that describes it."""

    contour: np.ndarray = field(repr=False)
    holes: list[np.ndarray] = field(default_factory=list, repr=False)
    kind: str = "freeform"
    parameters: dict[str, float] = field(default_factory=dict)


def _disc(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def deep_fraction(mask: np.ndarray, stroke_width: float) -> float:
    """How much of a mark lies further from its edge than a pen could reach.

    Filling the outer contour and comparing does not work: for anything thin and
    sprawling the filled contour is the shape again, so every stroke looks solid.
    Distance from the background does not care about the shape's outline at all.
    """
    if not np.any(mask):
        return 0.0
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    ink = distance[padded > 0]
    if ink.size == 0:
        return 0.0
    return float(np.count_nonzero(ink > 1.2 * stroke_width) / ink.size)


def split_solids(ink: np.ndarray, stroke_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Separate solid areas from the strokes that touch them.

    A filled area and the axis it sits on are one component, so this has to cut
    by shape rather than by connectivity: opening with a disc wider than the pen
    erases every stroke and leaves the solid cores, which are then grown back
    inside the ink to recover the rim the opening took off.
    """
    radius = max(2, int(round(stroke_width)))
    cores = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _disc(radius))
    if not np.any(cores):
        return np.zeros_like(ink), ink
    solid = cv2.bitwise_and(cv2.dilate(cores, _disc(radius)), ink)
    return solid, cv2.bitwise_and(ink, cv2.bitwise_not(solid))


def detect_hatch(
    component: Component,
    stroke_width: float,
    *,
    minimum_lines: int = 4,
    angle_tolerance: float = 8.0,
    minimum_periods: float = 4.0,
) -> Hatch | None:
    """Find ruling: many parallel strokes at one angle and one spacing.

    Requiring several periods across the area is what separates ruling from a
    box with a line of text in it, which also has a dominant direction and a
    perfectly good autocorrelation peak at one repetition.
    """
    mask = component.mask
    if min(mask.shape) < 8 * stroke_width:
        return None

    length = max(12, int(0.35 * min(mask.shape)))
    segments = cv2.HoughLinesP(
        mask, 1, np.pi / 360.0, threshold=max(20, length // 2),
        minLineLength=length, maxLineGap=3,
    )
    if segments is None or len(segments) < minimum_lines:
        return None

    angles = []
    weights = []
    for x1, y1, x2, y2 in segments[:, 0, :]:
        angle = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))) % 180.0
        angles.append(angle)
        weights.append(float(np.hypot(x2 - x1, y2 - y1)))

    angles = np.asarray(angles)
    weights = np.asarray(weights)
    # Angles live on a circle of half-turn, so average them as unit vectors.
    doubled = np.radians(2.0 * angles)
    mean = np.degrees(np.arctan2((weights * np.sin(doubled)).sum(), (weights * np.cos(doubled)).sum())) / 2.0
    mean %= 180.0
    difference = np.abs((angles - mean + 90.0) % 180.0 - 90.0)
    aligned = difference <= angle_tolerance
    if int(aligned.sum()) < minimum_lines:
        return None
    if weights[aligned].sum() < 0.6 * weights.sum():
        return None

    # Spacing: project the ink onto the normal and look for a repeating peak.
    ys, xs = np.nonzero(mask)
    normal = np.array([-np.sin(np.radians(mean)), np.cos(np.radians(mean))])
    offsets = xs * normal[0] + ys * normal[1]
    if offsets.size < 32:
        return None
    counts, edges = np.histogram(offsets, bins=max(32, int(np.ptp(offsets))))
    centred = counts - counts.mean()
    spectrum = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    if spectrum[0] <= 0:
        return None
    spectrum = spectrum / spectrum[0]

    floor = max(2, int(1.5 * stroke_width))
    if spectrum.size <= floor + 1:
        return None
    peak = floor + int(np.argmax(spectrum[floor:]))
    if spectrum[peak] < 0.25:
        return None

    step = float(edges[1] - edges[0])
    spacing = float(peak * step)
    if spacing <= 1.5 * stroke_width or np.ptp(offsets) < minimum_periods * spacing:
        return None

    return Hatch(
        angle=float(mean),
        spacing=spacing,
        stroke_width=stroke_width,
        coverage=float(component.area) / max(1.0, float(component.width * component.height)),
    )


def classify(
    component: Component,
    stroke_width: float,
    *,
    solid_fraction: float = 0.2,
    minimum_area: float = 60.0,
) -> str:
    """Decide whether a mark is a solid, a hatched area, or drawn with a pen."""
    if component.area < minimum_area:
        return "stroke"
    if deep_fraction(component.mask, stroke_width) >= solid_fraction:
        return "solid"
    if detect_hatch(component, stroke_width) is not None:
        return "hatched"
    return "stroke"


def _identify(contour: np.ndarray, tolerance: float) -> tuple[str, dict[str, float]]:
    """Name a boundary, taking corners as the stronger evidence.

    A small square drawn with a thin pen is round enough after scanning to pass
    a circularity test, so corner count is asked first and roundness only
    settles what has no corners to speak of.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter <= 0 or area <= 0:
        return "freeform", {}

    approximation = cv2.approxPolyDP(contour, tolerance, True)
    corners = len(approximation)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)

    if corners == 3:
        return "triangle", {}
    if corners == 4:
        x, y, width, height = cv2.boundingRect(approximation)
        if abs(area - width * height) / max(1.0, width * height) < 0.25:
            return "rectangle", {
                "x": float(x), "y": float(y), "width": float(width), "height": float(height),
            }
        return "quadrilateral", {}
    if circularity > 0.80:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return "circle", {"cx": float(x), "cy": float(y), "r": float(radius)}
    if corners <= 8:
        return "polygon", {"corners": float(corners)}
    return "freeform", {}


def outline(component: Component, stroke_width: float, *, simplify: float | None = None) -> Outline | None:
    """The boundary of a mark, as the simplest shape that still describes it."""
    padded = cv2.copyMakeBorder(component.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    order = int(np.argmax([cv2.contourArea(c) for c in contours]))
    offset = np.array([component.x - 1, component.y - 1], dtype=float)
    outer = contours[order].reshape(-1, 2).astype(float) + offset

    holes: list[np.ndarray] = []
    if hierarchy is not None:
        for index, entry in enumerate(hierarchy[0]):
            if entry[3] == order and cv2.contourArea(contours[index]) > 4.0 * stroke_width**2:
                holes.append(contours[index].reshape(-1, 2).astype(float) + offset)

    tolerance = simplify if simplify is not None else max(1.5, 0.45 * stroke_width)
    kind, parameters = _identify(contours[order], tolerance)
    for key in ("cx", "x"):
        if key in parameters:
            parameters[key] += offset[0]
    for key in ("cy", "y"):
        if key in parameters:
            parameters[key] += offset[1]
    return Outline(contour=outer, holes=holes, kind=kind, parameters=parameters)


@dataclass
class Region:
    """An area of the drawing that is filled rather than drawn with a pen."""

    component: Component
    kind: str
    outline: Outline
    hatch: Hatch | None = None
    bordered: bool = False


@dataclass
class MarkerSet:
    """Congruent marks repeated across a plot: one data series."""

    shape: str
    filled: bool
    size: float
    positions: list[tuple[float, float]] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    components: list[Component] = field(default_factory=list, repr=False)


def has_border(component: Component, stroke_width: float, *, threshold: float = 0.7) -> bool:
    """Is the boundary drawn, or is it only where the ruling happens to stop?

    Testing the ink just inside the boundary does not work: eroding by the pen
    width erases a thin frame entirely. Instead the simplified boundary is drawn
    and the ink is asked whether it lies along it, which a frame does all the way
    round and bare ruling does only where each line ends.
    """
    padded = cv2.copyMakeBorder(component.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _hierarchy = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    outer = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(outer, max(2.0, 1.5 * stroke_width), True)
    if len(simplified) < 3:
        return False

    edge = np.zeros_like(padded)
    cv2.drawContours(edge, [simplified], -1, 255, 1)
    reach = max(1, int(round(stroke_width)))
    near_ink = cv2.dilate(padded, _disc(reach))
    total = int(np.count_nonzero(edge))
    if total == 0:
        return False
    return int(np.count_nonzero(cv2.bitwise_and(edge, near_ink))) / total >= threshold


def detect_regions(
    ink: np.ndarray, stroke_width: float
) -> tuple[list[Region], np.ndarray]:
    """Take the filled and ruled areas out of the ink before anything else.

    A long run through a filled shape is indistinguishable from an axis, and the
    parallel strokes of ruling are indistinguishable from a dozen curves, so both
    have to be claimed before rules and curves are looked for. Small solids are
    deliberately left behind: an arrowhead is solid too, and the rule it caps
    cannot be measured once it is gone.
    """
    solid_ink, _rest = split_solids(ink, stroke_width)
    regions: list[Region] = []
    working = ink.copy()

    area_floor = (8.0 * stroke_width) ** 2
    for component in extract(solid_ink):
        if min(component.width, component.height) <= 6.0 * stroke_width:
            continue
        if component.area <= area_floor:
            continue
        shape = outline(component, stroke_width)
        if shape is None:
            continue
        regions.append(Region(component=component, kind="solid", outline=shape))
        patch = working[component.y : component.bottom, component.x : component.right]
        patch[component.mask > 0] = 0

    for component in extract(working):
        hatch = detect_hatch(component, stroke_width)
        if hatch is None:
            continue
        shape = outline(component, stroke_width)
        if shape is None:
            continue
        regions.append(
            Region(
                component=component,
                kind="hatched",
                outline=shape,
                hatch=hatch,
                bordered=has_border(component, stroke_width),
            )
        )
        patch = working[component.y : component.bottom, component.x : component.right]
        patch[component.mask > 0] = 0

    return regions, working


def find_marker_sets(
    blocks: list,
    stroke_width: float,
    *,
    minimum: int = 3,
    threshold: float = 0.75,
) -> tuple[list[MarkerSet], set[int]]:
    """Group congruent standalone marks into series, and say which blocks they used.

    Only marks that stand alone are considered. A repeated letter is congruent
    too, but it sits in a word with its neighbours, so grouping the page into
    labels first and looking only at the single-mark ones separates the two
    without needing a distance threshold to be tuned.
    """
    from .consensus import similarity

    singles = [
        (index, block.components[0])
        for index, block in enumerate(blocks)
        if len(block.components) == 1 and block.components[0].area >= max(24.0, stroke_width**2)
    ]
    if len(singles) < minimum:
        return [], set()

    groups: list[list[tuple[int, Component]]] = []
    for entry in singles:
        component = entry[1]
        for group in groups:
            reference = group[0][1]
            ratio = max(component.width, component.height) / max(
                1.0, max(reference.width, reference.height)
            )
            if not (0.75 <= ratio <= 1.33):
                continue
            if similarity(component.mask, reference.mask) >= threshold:
                group.append(entry)
                break
        else:
            groups.append([entry])

    sets: list[MarkerSet] = []
    used: set[int] = set()
    for group in groups:
        if len(group) < minimum:
            continue
        reference = max((component for _index, component in group), key=lambda item: item.area)
        shape = outline(reference, stroke_width)
        sets.append(
            MarkerSet(
                shape=shape.kind if shape else "freeform",
                filled=deep_fraction(reference.mask, stroke_width) >= 0.2,
                size=float(max(reference.width, reference.height)),
                positions=[
                    (component.x + component.width / 2.0, component.y + component.height / 2.0)
                    for _index, component in group
                ],
                parameters=dict(shape.parameters) if shape else {},
                components=[component for _index, component in group],
            )
        )
        used.update(index for index, _component in group)
    return sets, used


def claim_similar(
    sets: list[MarkerSet],
    components: list[Component],
    *,
    threshold: float = 0.75,
) -> set[int]:
    """Find the rest of a known marker, including ones sitting beside a label.

    A legend draws its sample right next to the words it explains, so that copy
    never stands alone and the first pass cannot see it. Once the shape is known
    from the copies out in the plot, the remaining ones can be claimed by
    resemblance instead of by isolation.
    """
    from .consensus import similarity

    taken: set[int] = set()
    for series in sets:
        reference = max(series.components, key=lambda item: item.area)
        for component in components:
            if id(component) in taken or component in series.components:
                continue
            ratio = max(component.width, component.height) / max(
                1.0, max(reference.width, reference.height)
            )
            if not (0.75 <= ratio <= 1.33):
                continue
            if similarity(component.mask, reference.mask) < threshold:
                continue
            series.components.append(component)
            series.positions.append(
                (component.x + component.width / 2.0, component.y + component.height / 2.0)
            )
            taken.add(id(component))
    return taken


def contour_path(points: np.ndarray, tolerance: float) -> str:
    """A closed boundary as an SVG path, straight where it is straight."""
    if points.shape[0] < 3:
        return ""
    simplified = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2), tolerance, True)
    simplified = simplified.reshape(-1, 2)
    if simplified.shape[0] <= 12:
        commands = [f"M{simplified[0][0]:.2f} {simplified[0][1]:.2f}"]
        commands += [f"L{x:.2f} {y:.2f}" for x, y in simplified[1:]]
        return " ".join(commands) + " Z"

    closed = np.vstack([points, points[:1]])
    segments = fit_bezier(closed, tolerance)
    return path_data(segments) + " Z"


def region_path(shape: Outline, tolerance: float) -> str:
    """Outer boundary then any holes, for an even-odd fill."""
    parts = [contour_path(shape.contour, tolerance)]
    parts += [contour_path(hole, tolerance) for hole in shape.holes]
    return " ".join(part for part in parts if part)
