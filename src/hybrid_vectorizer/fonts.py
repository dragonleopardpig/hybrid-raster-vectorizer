"""Finding the installed font that best matches the ink, and measuring it."""

from __future__ import annotations

import functools
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont


@dataclass
class FontFace:
    family: str
    style: str
    path: Path

    @property
    def italic(self) -> bool:
        return "italic" in self.style.lower() or "oblique" in self.style.lower()

    @property
    def bold(self) -> bool:
        return "bold" in self.style.lower()


@dataclass
class Metrics:
    """Advance widths in em units, so a layout can be computed without rendering."""

    units_per_em: float
    advances: dict[str, float]
    cap_height: float
    x_height: float
    ascent: float
    descent: float
    fallback: float = 0.5
    face: FontFace | None = field(default=None, repr=False)

    def advance(self, character: str, size: float) -> float:
        return self.advances.get(character, self.fallback) * size

    def width(self, text: str, size: float) -> float:
        return sum(self.advance(character, size) for character in text)


def list_faces() -> list[FontFace]:
    if shutil.which("fc-list") is None:
        return []
    completed = subprocess.run(
        ["fc-list", "--format", "%{family[0]}\\t%{style[0]}\\t%{file}\\n"],
        check=False,
        capture_output=True,
        text=True,
    )
    faces: list[FontFace] = []
    seen: set[tuple[str, str]] = set()
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        family, style, path = (field.strip() for field in fields)
        if not path or Path(path).suffix.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        if (family, style) in seen:
            continue
        seen.add((family, style))
        faces.append(FontFace(family=family, style=style, path=Path(path)))
    return faces


@functools.lru_cache(maxsize=64)
def measure(path: str, index: int = 0) -> Metrics | None:
    try:
        font = TTFont(path, fontNumber=index, lazy=True)
        units = float(font["head"].unitsPerEm)
        cmap = font.getBestCmap()
        hmtx = font["hmtx"]
        advances = {}
        for code, name in cmap.items():
            try:
                advances[chr(code)] = hmtx[name][0] / units
            except KeyError:
                continue
        os2 = font["OS/2"] if "OS/2" in font else None
        hhea = font["hhea"]
        metrics = Metrics(
            units_per_em=units,
            advances=advances,
            cap_height=float(getattr(os2, "sCapHeight", 0.7 * units) or 0.7 * units) / units,
            x_height=float(getattr(os2, "sxHeight", 0.45 * units) or 0.45 * units) / units,
            ascent=float(hhea.ascent) / units,
            descent=abs(float(hhea.descent)) / units,
        )
        font.close()
        return metrics
    except Exception:
        return None


def _normalise(image: np.ndarray, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    ys, xs = np.nonzero(image)
    if xs.size == 0:
        return np.zeros(size, dtype=np.uint8)
    cropped = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Shape agreement of two ink patches, independent of size and position."""
    first = _normalise(a) > 127
    second = _normalise(b) > 127
    union = np.count_nonzero(first | second)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first & second) / union)


def render_text(face: FontFace, text: str, height: int) -> np.ndarray | None:
    """Rasterise a string with Pillow so it can be compared with the source ink."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(str(face.path), size=max(8, int(height * 1.6)))
    except Exception:
        return None
    image = Image.new("L", (max(32, len(text) * height * 3), height * 5), color=0)
    ImageDraw.Draw(image).text((height, height), text, fill=255, font=font)
    array = np.asarray(image)
    if not array.any():
        return None
    return array


def match_font(
    samples: list[tuple[np.ndarray, str]],
    *,
    faces: list[FontFace] | None = None,
    limit: int = 240,
) -> list[tuple[FontFace, float]]:
    """Rank installed faces by how closely they reproduce known ink samples."""
    faces = faces or list_faces()
    if not faces or not samples:
        return []

    scored: list[tuple[FontFace, float]] = []
    for face in faces[:limit]:
        scores = []
        for ink, text in samples:
            if not text.strip():
                continue
            rendered = render_text(face, text, height=max(8, ink.shape[0]))
            if rendered is None:
                continue
            scores.append(similarity(ink, rendered))
        if scores:
            scored.append((face, float(np.mean(scores))))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
