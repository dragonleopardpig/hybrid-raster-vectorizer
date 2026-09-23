"""Connected components and the coarse ink/text/graphic split."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Component:
    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid: tuple[float, float]
    mask: np.ndarray = field(repr=False)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def aspect(self) -> float:
        return self.width / max(1.0, float(self.height))

    @property
    def fill_ratio(self) -> float:
        return self.area / max(1.0, float(self.width * self.height))

    def crop(self, image: np.ndarray, pad: int = 0) -> np.ndarray:
        y0 = max(0, self.y - pad)
        x0 = max(0, self.x - pad)
        return image[y0 : self.bottom + pad, x0 : self.right + pad]

    def overlaps_rows(self, top: int, bottom: int) -> bool:
        return self.y < bottom and top < self.bottom


def extract(ink: np.ndarray) -> list[Component]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    components: list[Component] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        mask = (labels[y : y + height, x : x + width] == label).astype(np.uint8) * 255
        components.append(
            Component(
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                mask=mask,
            )
        )
    return components


def split_into(
    component: Component, pieces: int, *, minimum_width: int = 4, clean_fraction: float = 0.3
) -> tuple[list[Component], bool]:
    """Cut one component into a known number of glyphs at its thinnest columns.

    Touching letters share a component, so nothing can be said about either of
    them. How many are in there is not guessed: the recogniser's reading says
    how many glyphs were laid out over this ink, and the column profile says
    where they join.

    Also reports whether every cut fell in a genuine gap. A cut that passes
    through a stroke splits one letter in two, and pieces like that are worse
    than no evidence: they look alike whatever they came from.
    """
    if pieces <= 1 or component.width < pieces * minimum_width:
        return [component], True

    profile = (component.mask > 0).sum(axis=0).astype(float)
    cuts: list[int] = []
    for _ in range(pieces - 1):
        best: tuple[float, int] | None = None
        for column in range(minimum_width, component.width - minimum_width):
            if any(abs(column - cut) < minimum_width for cut in cuts):
                continue
            value = float(profile[column])
            if best is None or value < best[0]:
                best = (value, column)
        if best is None:
            break
        cuts.append(best[1])

    ink_columns = profile[profile > 0]
    reference = float(np.median(ink_columns)) if ink_columns.size else 0.0
    clean = all(float(profile[cut]) <= clean_fraction * reference for cut in cuts)

    cuts.sort()
    result: list[Component] = []
    for left, right in zip([0] + cuts, cuts + [component.width]):
        window = component.mask[:, left:right]
        ys, xs = np.nonzero(window)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mask = window[y0:y1, x0:x1]
        result.append(
            Component(
                label=component.label,
                x=component.x + left + x0,
                y=component.y + y0,
                width=x1 - x0,
                height=y1 - y0,
                area=int(np.count_nonzero(mask)),
                centroid=(
                    float(component.x + left + x0 + (x1 - x0) / 2.0),
                    float(component.y + y0 + (y1 - y0) / 2.0),
                ),
                mask=mask.copy(),
            )
        )
    return (result or [component]), clean


def median_text_height(components: list[Component], page_height: int) -> float:
    """Typical glyph height, taken from components small enough to be glyphs."""
    candidates = [
        float(component.height)
        for component in components
        if component.height < 0.25 * page_height and component.area > 8
    ]
    if not candidates:
        return max(8.0, 0.02 * page_height)
    return float(np.median(candidates))


def mask_of(components: list[Component], shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for component in components:
        region = canvas[component.y : component.bottom, component.x : component.right]
        np.maximum(region, component.mask, out=region)
    return canvas
