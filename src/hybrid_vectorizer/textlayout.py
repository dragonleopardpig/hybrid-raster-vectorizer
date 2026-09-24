"""Grouping leftover ink into text blocks and reading their internal structure."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .components import Component


@dataclass
class Block:
    """One label: a caption, an axis name or a whole formula."""

    components: list[Component]
    bars: list[Component] = field(default_factory=list)
    kind: str = "text"
    orientation: str = "horizontal"

    @property
    def x(self) -> int:
        return min(component.x for component in self.components)

    @property
    def y(self) -> int:
        return min(component.y for component in self.components)

    @property
    def right(self) -> int:
        return max(component.right for component in self.components)

    @property
    def bottom(self) -> int:
        return max(component.bottom for component in self.components)

    @property
    def width(self) -> int:
        return self.right - self.x

    @property
    def height(self) -> int:
        return self.bottom - self.y

    @property
    def centre_x(self) -> float:
        return (self.x + self.right) / 2.0

    @property
    def centre_y(self) -> float:
        return (self.y + self.bottom) / 2.0

    def body_height(self) -> float:
        """Height of the full-size glyphs, ignoring sub- and superscripts."""
        heights = sorted(
            component.height for component in self.components if component not in self.bars
        )
        if not heights:
            return float(self.height)
        return float(heights[int(0.85 * (len(heights) - 1))])

    def baseline(self) -> float:
        """Most glyphs rest on the baseline, so their bottoms cluster there."""
        body = self.body_height()
        bottoms = [
            float(component.bottom)
            for component in self.components
            if component not in self.bars and component.height >= 0.6 * body
        ]
        if not bottoms:
            return float(self.bottom)
        return float(np.median(bottoms))


def drop_noise(components: list[Component], text_height: float) -> list[Component]:
    """Slivers left where an axis or tick was cut away are not glyphs."""
    return [
        component
        for component in components
        if component.area >= 6 or max(component.width, component.height) >= 0.3 * text_height
    ]


def _bar_shaped(component: Component, text_height: float, stroke_width: float) -> bool:
    return (
        component.width >= max(2.5 * stroke_width, 0.45 * text_height)
        and component.height <= max(3.0, 1.8 * stroke_width)
        and component.aspect >= 3.0
        and component.fill_ratio >= 0.55
    )


def find_fraction_bars(
    components: list[Component], text_height: float, stroke_width: float
) -> list[Component]:
    """A fraction bar is the only rule with its own ink both above and below it.

    That single test rejects the strokes of an equals sign and a leading minus,
    which are bar-shaped but have nothing stacked on one side.
    """
    shaped = [c for c in components if _bar_shaped(c, text_height, stroke_width)]
    others = [c for c in components if c not in shaped]
    reach = 2.2 * text_height

    bars: list[Component] = []
    for bar in shaped:
        margin = 0.25 * bar.width
        low, high = bar.x - margin, bar.right + margin
        above = below = False
        for other in others:
            centre = (other.x + other.right) / 2.0
            if not (low <= centre <= high):
                continue
            if 0 < bar.y - other.bottom <= reach:
                above = True
            elif 0 < other.y - bar.bottom <= reach:
                below = True
        if above and below:
            bars.append(bar)
    return bars


def _rlsa(
    components: list[Component], shape: tuple[int, int], gap_x: float, gap_y: float
) -> list[list[Component]]:
    canvas = np.zeros(shape, dtype=np.uint8)
    for component in components:
        canvas[component.y : component.bottom, component.x : component.right] = 255
    kernel = np.ones((max(1, int(gap_y)), max(1, int(gap_x))), np.uint8)
    _count, labels = cv2.connectedComponents(cv2.dilate(canvas, kernel), 8)

    groups: dict[int, list[Component]] = {}
    for component in components:
        centre_y = int(np.clip((component.y + component.bottom) // 2, 0, shape[0] - 1))
        centre_x = int(np.clip((component.x + component.right) // 2, 0, shape[1] - 1))
        label = int(labels[centre_y, centre_x]) or -component.label
        groups.setdefault(label, []).append(component)
    return list(groups.values())


def _overlap(a: Block, b: Block) -> float:
    top, bottom = max(a.y, b.y), min(a.bottom, b.bottom)
    if bottom <= top:
        return 0.0
    return (bottom - top) / max(1.0, float(min(a.height, b.height)))


def _merge_same_line(blocks: list[Block], text_height: float, reach: float) -> list[Block]:
    """Rejoin neighbours on one line, but never fuse two separate fractions."""
    ordered = sorted(blocks, key=lambda block: block.x)
    result: list[Block] = []
    for block in ordered:
        merged = False
        for existing in result:
            if existing.bars and block.bars:
                continue
            gap = block.x - existing.right
            if gap > reach or gap < -0.5 * text_height:
                continue
            if _overlap(existing, block) < 0.45:
                continue
            existing.components = sorted(existing.components + block.components, key=lambda c: c.x)
            existing.bars = existing.bars + block.bars
            merged = True
            break
        if not merged:
            result.append(block)
    return result


def script_components(block: Block) -> list[Component]:
    """Glyphs the ink actually draws off the baseline, at reduced size."""
    body = block.body_height()
    centre = block.baseline() - 0.5 * body
    found: list[Component] = []
    for component in block.components:
        if component in block.bars or component.height >= 0.78 * body:
            continue
        if component.height < 0.25 * body or component.width < 0.25 * body:
            continue
        if abs((component.y + component.bottom) / 2.0 - centre) > 0.22 * body:
            found.append(component)
    return found


def _has_script(block: Block) -> bool:
    body = block.body_height()
    centre = block.baseline() - 0.5 * body
    return bool(script_components(block))


def text_angle(
    block: Block, *, minimum_marks: int = 4, linearity: float = 3.0, deadband: float = 12.0
) -> float:
    """The angle a label is set at, from how its marks are strung out.

    A label following a sloping line is neither upright nor a quarter turn, so
    the angle has to be measured rather than chosen from a list. A fraction
    stacks its marks vertically without being turned at all, so blocks with a
    bar are left alone, and a short label needs four marks before a slope is
    believed: three marks of "4I" with a sunken subscript measure 22 degrees.
    """
    if block.bars or len(block.components) < minimum_marks:
        return 0.0

    centres = np.array(
        [[c.x + c.width / 2.0, c.y + c.height / 2.0] for c in block.components], dtype=float
    )
    centred = centres - centres.mean(axis=0)
    _u, spread, axes = np.linalg.svd(centred, full_matrices=False)
    # Written this way round so that marks lying exactly on a line, where the
    # spread across it is zero, read as perfectly linear rather than as a
    # division that has to be guarded away.
    if spread[0] <= 1e-9 or spread[1] * linearity > spread[0]:
        return 0.0

    angle = float(np.degrees(np.arctan2(axes[0][1], axes[0][0])))
    angle = (angle + 90.0) % 180.0 - 90.0
    return 0.0 if abs(angle) < deadband else angle


def group_vertical(
    blocks: list[Block],
    text_height: float,
    *,
    minimum: int = 3,
    column: float = 0.6,
    stacking: float = 1.2,
) -> list[Block]:
    """Join a label written up the side of the page into one block.

    Grouping runs along the line, so a label turned on its side arrives as a
    handful of unrelated pieces. They are rejoined by the one thing that makes
    them a line: narrow pieces sharing a column, stacked tightly. A column of
    tick labels also shares an x, but each of those is wider than it is tall and
    they stand much further apart.
    """
    narrow = 1.6 * text_height
    # A stack of dashes shares a column and stacks tightly too, so the pieces
    # have to be wide enough to be letters rather than marks on a broken line.
    slim = 0.35 * text_height
    candidates = [
        block
        for block in blocks
        if len(block.components) <= 3
        and slim <= block.width <= narrow
        and block.height >= slim
        and not block.bars
    ]
    if len(candidates) < minimum:
        return blocks

    used: set[int] = set()
    merged: list[Block] = []
    for seed in sorted(candidates, key=lambda block: block.y):
        if id(seed) in used:
            continue
        run = [seed]
        while True:
            last = run[-1]
            following = [
                block
                for block in candidates
                if id(block) not in used
                and all(block is not member for member in run)
                and 0 <= block.y - last.bottom <= stacking * text_height
                and abs(block.centre_x - last.centre_x) <= column * text_height
            ]
            if not following:
                break
            run.append(min(following, key=lambda block: block.y))
        if len(run) < minimum:
            continue
        for member in run:
            used.add(id(member))
        components = [c for member in run for c in member.components]
        merged.append(
            Block(
                components=sorted(components, key=lambda c: c.y),
                bars=[],
                kind="text",
                orientation="vertical",
            )
        )

    if not merged:
        return blocks
    kept = [block for block in blocks if id(block) not in used]
    return sorted(kept + merged, key=lambda block: (block.y, block.x))


def group_blocks(
    components: list[Component],
    shape: tuple[int, int],
    text_height: float,
    stroke_width: float,
    *,
    line_gap: float = 0.9,
) -> list[Block]:
    components = drop_noise(components, text_height)
    if not components:
        return []

    bars = find_fraction_bars(components, text_height, stroke_width)
    reach = 2.2 * text_height

    # Each fraction bar claims its own numerator and denominator first, so that
    # neighbouring tick labels cannot be run together by a proximity rule.
    claimed: set[int] = set()
    blocks: list[Block] = []
    for bar in bars:
        margin = 0.3 * bar.width
        low, high = bar.x - margin, bar.right + margin
        members = [bar]
        for other in components:
            if other is bar or id(other) in claimed or other in bars:
                continue
            centre = (other.x + other.right) / 2.0
            if not (low <= centre <= high):
                continue
            if 0 < bar.y - other.bottom <= reach or 0 < other.y - bar.bottom <= reach:
                members.append(other)
        for member in members:
            claimed.add(id(member))
        blocks.append(Block(components=sorted(members, key=lambda c: c.x), bars=[bar]))

    remainder = [component for component in components if id(component) not in claimed]
    for group in _rlsa(
        remainder,
        shape,
        gap_x=max(2.0, 0.55 * text_height),
        gap_y=max(2.0, 0.45 * text_height),
    ):
        blocks.append(Block(components=sorted(group, key=lambda c: c.x)))

    blocks = _merge_same_line(blocks, text_height, line_gap * text_height)
    for block in blocks:
        block.kind = "math" if (block.bars or _has_script(block)) else "text"
    blocks = group_vertical(blocks, text_height)
    blocks.sort(key=lambda block: (block.y, block.x))
    return blocks
