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
