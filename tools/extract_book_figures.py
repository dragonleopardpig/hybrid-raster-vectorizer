"""Cut every captioned figure out of a scanned book, one PNG apiece.

The pages are images, so the figures are not stored separately and have to be
found. The OCR text layer knows where every "Fig. 4-33" caption sits, which
fixes the bottom of a figure and, where two sit side by side, which half of the
page each one owns. The rest is whitespace: walk up from the caption until the
page goes quiet for longer than a line of type, and that is the top.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

CAPTION = re.compile(r"^(?:fig|figure)\.?$", re.IGNORECASE)
NUMBER = re.compile(r"^\(?([0-9]+[-–][0-9]+)\)?[.,]?$")
XHTML = "{http://www.w3.org/1999/xhtml}"


def page_words(pdf: Path, page: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Every word the OCR layer found on one page, in PDF points."""
    raw = subprocess.run(
        ["pdftotext", "-bbox", "-f", str(page), "-l", str(page), str(pdf), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    try:
        tree = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return []

    return [
        (
            (word.text or "").strip(),
            (
                float(word.get("xMin")), float(word.get("yMin")),
                float(word.get("xMax")), float(word.get("yMax")),
            ),
        )
        for word in tree.iter(f"{XHTML}word")
    ]


def by_line(words: list) -> dict[int, list]:
    lines: dict[int, list] = {}
    for entry in words:
        lines.setdefault(round(entry[1][1] / 4.0), []).append(entry)
    return lines


def captions_in(words: list, crowd: int = 4) -> list[tuple[str, tuple[float, float, float, float]]]:
    """The figure captions among these words.

    "As shown in Fig. 1-9," reads exactly like a caption and is not one, and
    neither does counting words settle it: "(a) See Fig. 1-8." is four words, as
    two captions side by side are. What a caption line holds is captions and
    nothing else, so the line is read whole and kept only if every word on it
    belongs to one.
    """
    found = []
    for group in by_line(words).values():
        if len(group) > crowd:
            continue
        group.sort(key=lambda entry: entry[1][0])

        here: list[tuple[str, tuple[float, float, float, float]]] = []
        index = 0
        while index + 1 < len(group):
            text, box = group[index]
            following, next_box = group[index + 1]
            match = CAPTION.match(text) and NUMBER.match(following)
            if not match:
                here = []
                break
            here.append(
                (
                    match.group(1).replace("\u2013", "-"),
                    (box[0], min(box[1], next_box[1]), next_box[2], max(box[3], next_box[3])),
                )
            )
            index += 2
        if here and index == len(group):
            found.extend(here)
    return found


def quiet_rows(ink: np.ndarray, floor: float) -> np.ndarray:
    return ink.sum(axis=1) <= floor * ink.shape[1]


def prose_bottom(words: list, above: float, scale: float, crowd: int,
                 page_width: float = 612.0, span: float = 0.55) -> int:
    """The foot of the last line of running text above a point on the page.

    A figure is bounded below the prose, and the whitespace between them is
    often no wider than the gap between two paragraphs, so it cannot be found
    by silence alone.

    Many words on a line is not enough to call it prose: a graph's own scale
    reads "0 200 400 600 800 1000", six words across a third of the page, and
    taking that for text put the top of the figure below its own axis. Running
    text also runs the width of the column.
    """
    foot = 0.0
    width = page_width
    for group in by_line(words).values():
        if len(group) <= crowd:
            continue
        reach = max(box[2] for _text, box in group) - min(box[0] for _text, box in group)
        if reach < span * width:
            continue
        bottom = max(box[3] for _text, box in group)
        if bottom <= above:
            foot = max(foot, bottom)
    return int(foot * scale)


def first_ink(ink: np.ndarray, start: int, floor: float) -> int:
    """The first row at or below `start` that holds ink."""
    quiet = quiet_rows(ink, floor)
    row = max(0, start)
    while row < quiet.size and quiet[row]:
        row += 1
    return row


def top_of_figure(ink: np.ndarray, bottom: int, *, gap: int, floor: float) -> int:
    """Walk up from the caption to the silence above the figure.

    Not the first silence: that one is the space between the figure and the
    caption naming it, which is wide by design. Cross it, find the foot of the
    drawing, and only then look for the gap that ends it.
    """
    quiet = quiet_rows(ink, floor)
    row = bottom
    while row > 0 and quiet[row - 1]:
        row -= 1

    run = 0
    while row > 0:
        row -= 1
        if quiet[row]:
            run += 1
            if run >= gap:
                return row + run
        else:
            run = 0
    return 0


def own_column(piece: np.ndarray, centre: int, gutter: int) -> tuple[np.ndarray, int]:
    """Keep the block of the band that the caption sits under.

    A figure is often set beside the text rather than under it, and the band
    above the caption then runs the width of the page with a paragraph in the
    other half. The two are separated by a gutter of clean paper, which nothing
    inside a drawing is as wide as.
    """
    if piece.size == 0:
        return piece, 0
    empty = piece.sum(axis=0) == 0
    blocks: list[tuple[int, int]] = []
    start = None
    for column in range(empty.size + 1):
        if column < empty.size and not empty[column]:
            if start is None:
                start = column
        elif start is not None:
            blocks.append((start, column))
            start = None

    merged: list[list[int]] = []
    for first, last in blocks:
        if merged and first - merged[-1][1] < gutter:
            merged[-1][1] = last
        else:
            merged.append([first, last])
    for first, last in merged:
        if first <= centre < last:
            return piece[:, first:last], first
    return piece, 0


def trim(ink: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    marks = cv2.findNonZero(ink)
    if marks is None:
        return None
    x, y, width, height = cv2.boundingRect(marks)
    return (
        max(0, x - pad), max(0, y - pad),
        min(ink.shape[1], x + width + pad) - max(0, x - pad),
        min(ink.shape[0], y + height + pad) - max(0, y - pad),
    )


def figures_on_page(
    page_image: np.ndarray, boxes: list, words: list, scale: float, options,
    page_width: float = 612.0,
) -> list[tuple[str, np.ndarray]]:
    grey = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    _level, ink = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = (ink > 0).astype(np.uint8)
    height, width = ink.shape

    # Captions on one line belong to figures standing side by side.
    rows: list[list] = []
    for name, box in sorted(boxes, key=lambda item: item[1][1]):
        top = box[1] * scale
        if rows and abs(top - rows[-1][0][1][1] * scale) <= options.same_row:
            rows[-1].append((name, box))
        else:
            rows.append([(name, box)])

    cut: list[tuple[str, np.ndarray]] = []
    for row in rows:
        row.sort(key=lambda item: item[1][0])
        caption_top = int(min(box[1] for _n, box in row) * scale)
        floor_of_prose = prose_bottom(
            words, min(box[1] for _n, box in row) - 2.0, scale, options.crowd, page_width
        )
        silence = top_of_figure(
            ink[:caption_top], caption_top, gap=options.gap, floor=options.floor
        )
        band_top = first_ink(ink, max(silence, floor_of_prose + 2), options.floor)
        if caption_top - band_top < options.least_height:
            continue

        edges = [0]
        for left, right in zip(row, row[1:]):
            edges.append(int(0.5 * (left[1][2] + right[1][0]) * scale))
        edges.append(width)

        for index, (name, box) in enumerate(row):
            left, right = edges[index], edges[index + 1]
            piece = ink[band_top:caption_top, left:right]
            centre = int(0.5 * (box[0] + box[2]) * scale) - left
            piece, shift = own_column(piece, centre, options.gutter)
            left += shift
            found = trim(piece, options.pad)
            if found is None:
                continue
            x, y, w, h = found
            if w < options.least_width or h < options.least_height:
                continue
            crop = page_image[band_top + y : band_top + y + h, left + x : left + x + w]
            cut.append((name, crop))
    return cut


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--gap", type=int, default=26, help="silent rows that end a figure")
    parser.add_argument("--floor", type=float, default=0.004, help="ink share a quiet row may hold")
    parser.add_argument("--pad", type=int, default=8)
    parser.add_argument("--same-row", type=float, default=30.0)
    parser.add_argument("--least-width", type=int, default=120)
    parser.add_argument("--least-height", type=int, default=90)
    parser.add_argument("--gutter", type=int, default=46,
                        help="clean columns that separate a figure from text beside it")
    parser.add_argument("--crowd", type=int, default=4,
                        help="words a caption line may hold before it is prose")
    options = parser.parse_args()

    options.out.mkdir(parents=True, exist_ok=True)
    report = subprocess.run(
        ["pdfinfo", str(options.pdf)], capture_output=True, text=True, check=True
    ).stdout
    pages = int(report.split("Pages:")[1].split()[0])
    # Page size varies between editions, and every box the text layer gives is in
    # points on that page.
    page_width = float(report.split("Page size:")[1].split()[0])
    last = options.last or pages

    written = 0
    for page in range(options.first, last + 1):
        words = page_words(options.pdf, page)
        boxes = captions_in(words, options.crowd)
        if not boxes:
            continue
        with tempfile.TemporaryDirectory() as room:
            stem = Path(room) / "page"
            subprocess.run(
                ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(options.dpi),
                 "-gray", "-png", "-singlefile", str(options.pdf), str(stem)],
                capture_output=True, check=True,
            )
            page_image = cv2.imread(str(stem) + ".png", cv2.IMREAD_GRAYSCALE)
        if page_image is None:
            continue
        scale = page_image.shape[1] / page_width

        for name, crop in figures_on_page(
            page_image, boxes, words, scale, options, page_width
        ):
            target = options.out / f"fig-{name}.png"
            copy = 2
            while target.exists():
                target = options.out / f"fig-{name}-{copy}.png"
                copy += 1
            cv2.imwrite(str(target), crop)
            written += 1
            print(f"  page {page:3d}  {target.name:20s} {crop.shape[1]}x{crop.shape[0]}")
    print(f"{written} figure(s) -> {options.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
