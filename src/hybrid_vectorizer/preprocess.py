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


def _binarise(gray: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    if float(np.count_nonzero(binary)) > 0.5 * binary.size:
        # Otsu picked the wrong side: the figure is light ink on a dark ground.
        binary = 255 - binary
    return binary


def despeckle(ink: np.ndarray, minimum_area: int = 4) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    keep[0] = False
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


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
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"Unable to read image: {path}")

    gray = _to_gray(raw)
    ink = despeckle(_binarise(gray))

    skew = estimate_skew(ink) if deskew else 0.0
    if deskew and abs(skew) > skew_tolerance:
        gray = _rotate(gray, skew, border=255)
        ink = despeckle(_binarise(gray))
    else:
        skew = 0.0

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
    )
