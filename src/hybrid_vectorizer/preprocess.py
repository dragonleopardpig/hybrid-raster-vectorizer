"""Image loading, binarisation, deskewing and global ink statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Page:
    """A binarised figure together with the statistics every later stage needs."""

    gray: np.ndarray
    ink: np.ndarray
    stroke_width: float
    skew_degrees: float
    background: str
    source_path: Path
    paper_spread: int = 0

    @property
    def height(self) -> int:
        return int(self.ink.shape[0])

    @property
    def width(self) -> int:
        return int(self.ink.shape[1])


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        flattened = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        image = flattened.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _smooth_surface(gray: np.ndarray, usable: np.ndarray, degree: int, samples: int) -> np.ndarray:
    """Least-squares surface of the given order through the usable pixels."""
    height, width = gray.shape
    ys, xs = np.nonzero(usable)
    if ys.size < 500:
        return np.full(gray.shape, 255.0)

    step = max(1, ys.size // samples)
    ys, xs = ys[::step], xs[::step]
    powers = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]

    u, v = xs / width, ys / height
    design = np.column_stack([u**i * v**j for i, j in powers])
    coefficients, *_rest = np.linalg.lstsq(design, gray[ys, xs].astype(np.float64), rcond=None)

    grid_v, grid_u = np.mgrid[0:height, 0:width]
    grid_u = grid_u / width
    grid_v = grid_v / height
    surface = sum(
        coefficient * (grid_u**i * grid_v**j)
        for coefficient, (i, j) in zip(coefficients, powers)
    )
    return np.clip(surface, 1.0, 255.0)


def estimate_paper(
    gray: np.ndarray,
    *,
    tiles: int = 48,
    degree: int = 3,
    samples: int = 40000,
    content: float = 0.92,
) -> np.ndarray:
    """Estimate the paper behind the drawing, in two passes.

    The first pass fits a surface too smooth to follow anything that was drawn,
    and whatever sits well below it is content rather than paper. The second
    pass then estimates the paper tile by tile from the pixels that survived,
    which follows blotchy staining that no smooth surface can, and carries the
    estimate across the content by inpainting.

    Both passes are needed. A tile estimate alone takes a figure's grey slabs
    for paper and divides them away, leaving them 7% darker than the page where
    they are really 24%. A smooth surface alone leaves a badly stained page
    with half again as many spurious labels.
    """
    height, width = gray.shape
    threshold, _binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    lighter_than_ink = gray > threshold

    rough = _smooth_surface(gray, lighter_than_ink, degree, samples)
    paper = lighter_than_ink & (gray.astype(np.float64) / rough > content)

    step = max(8, min(height, width) // tiles)
    rows, columns = max(1, height // step), max(1, width // step)
    coarse = np.zeros((rows, columns), np.uint8)
    missing = np.zeros((rows, columns), np.uint8)
    for row in range(rows):
        for column in range(columns):
            y0, x0 = row * step, column * step
            y1 = height if row == rows - 1 else y0 + step
            x1 = width if column == columns - 1 else x0 + step
            window = gray[y0:y1, x0:x1]
            visible = window[paper[y0:y1, x0:x1]]
            if visible.size < 0.1 * window.size:
                missing[row, column] = 255
            else:
                coarse[row, column] = int(np.median(visible))

    if np.any(missing):
        if np.all(missing):
            return np.clip(rough, 1, 255).astype(np.uint8)
        coarse = cv2.inpaint(coarse, missing, 3, cv2.INPAINT_TELEA)

    coarse = cv2.GaussianBlur(coarse, (0, 0), 1.2)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def flatten(gray: np.ndarray, *, minimum_spread: int = 25) -> tuple[np.ndarray, int]:
    """Even out the paper so one threshold can serve the whole page."""
    paper = estimate_paper(gray)
    spread = int(np.percentile(paper, 98) - np.percentile(paper, 2))
    if spread < minimum_spread:
        return gray, spread
    scaled = gray.astype(np.float32) / np.maximum(paper.astype(np.float32), 1.0)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8), spread


def _binarise(gray: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    if float(np.count_nonzero(binary)) > 0.5 * binary.size:
        # Otsu picked the wrong side: the figure is light ink on a dark ground.
        binary = 255 - binary
    return binary


def despeckle(ink: np.ndarray, minimum_area: int = 4, minimum_side: float = 0.0) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    if minimum_side > 0:
        span = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
        keep[1:] &= span >= minimum_side
    keep[0] = False
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def remove_grain(ink: np.ndarray, stroke_width: float) -> np.ndarray:
    """Drop marks smaller than the pen that drew the figure.

    A worn scan carries grain that survives a fixed speck threshold and then
    dominates every later statistic: on one of these figures the median
    component was 8px tall, so the typical glyph height was measured from dirt.
    Nothing narrower than the pen can be a mark the pen made.
    """
    if stroke_width <= 1.0:
        return ink
    return despeckle(
        ink,
        minimum_area=max(4, int(round((0.6 * stroke_width) ** 2))),
        minimum_side=max(2.0, 0.6 * stroke_width),
    )


def estimate_stroke_width(ink: np.ndarray) -> float:
    """Median pen width, measured on the medial axis so junctions do not bias it."""
    if not np.any(ink):
        return 1.0
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    medial = skeletonize(ink > 0)
    samples = distance[medial]
    if samples.size == 0:
        return 1.0
    return float(2.0 * np.median(samples))


def estimate_skew(ink: np.ndarray, maximum_degrees: float = 4.0) -> float:
    """Skew from the longest near-horizontal runs, which in a plot are the axes."""
    height, width = ink.shape
    minimum_length = max(40, int(0.2 * width))
    segments = cv2.HoughLinesP(
        ink,
        1,
        np.pi / 2880.0,
        threshold=max(40, minimum_length // 4),
        minLineLength=minimum_length,
        maxLineGap=6,
    )
    if segments is None:
        return 0.0

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in segments[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < minimum_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > maximum_degrees:
            continue
        angles.append(angle)
        weights.append(length)

    if not angles:
        return 0.0
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    midpoint = cumulative[-1] / 2.0
    return float(sorted_angles[int(np.searchsorted(cumulative, midpoint))])


def _rotate(image: np.ndarray, degrees: float, border: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def load_page(path: Path, *, deskew: bool = True, skew_tolerance: float = 0.1) -> Page:
    if not Path(path).is_file():
        raise SystemExit(f"No such image: {path}")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"Not a readable image: {path}")

    gray = _to_gray(raw)
    gray, spread = flatten(gray)
    ink = despeckle(_binarise(gray))

    skew = estimate_skew(ink) if deskew else 0.0
    if deskew and abs(skew) > skew_tolerance:
        gray = _rotate(gray, skew, border=255)
        ink = despeckle(_binarise(gray))
    else:
        skew = 0.0

    ink = remove_grain(ink, estimate_stroke_width(ink))

    background = "#ffffff"
    if np.any(ink == 0):
        level = int(np.median(gray[ink == 0]))
        background = f"#{level:02x}{level:02x}{level:02x}"

    return Page(
        gray=gray,
        ink=ink,
        stroke_width=estimate_stroke_width(ink),
        skew_degrees=skew,
        background=background,
        source_path=Path(path),
        paper_spread=spread,
    )
