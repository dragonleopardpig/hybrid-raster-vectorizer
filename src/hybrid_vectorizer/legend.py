"""Reading a legend as one object: a frame, its samples, and what they are called."""

from __future__ import annotations

from dataclasses import dataclass, field

from .components import Component
from .shapes import Frame, MarkerSet


@dataclass
class Entry:
    """One row of a legend: a sample mark and the name beside it.

    Either may be missing. A sample that has run into its label cannot be told
    apart from it, and recording the row without the sample is more use than
    dropping the row.
    """

    series: int | None = None
    position: tuple[float, float] | None = None
    block: int | None = None
    text: str = ""


@dataclass
class Legend:
    frame: Frame | None = None
    entries: list[Entry] = field(default_factory=list)
    blocks: set[int] = field(default_factory=set)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def framed(self) -> bool:
        return self.frame is not None


def _overlaps(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    top, bottom = max(low_a, low_b), min(high_a, high_b)
    if bottom <= top:
        return 0.0
    return (bottom - top) / max(1.0, min(high_a - low_a, high_b - low_b))


def assemble(
    frames: list[Frame],
    marker_sets: list[MarkerSet],
    blocks: list,
    *,
    margin: float = 4.0,
) -> list[Legend]:
    """Pair the samples inside each frame with the labels beside them.

    A sample inside a legend is not a data point, so it is taken out of the
    series it names. Leaving it in would put a reading at the legend's own
    coordinates, which is nowhere the plot ever measured.
    """
    legends: list[Legend] = []
    for frame in frames:
        legend = Legend(
            frame=frame,
            bounds=(float(frame.x), float(frame.y), float(frame.width), float(frame.height)),
        )

        for index, series in enumerate(marker_sets):
            kept: list[tuple[float, float]] = []
            kept_components: list[Component] = []
            for position, component in zip(series.positions, series.components):
                if frame.contains(position[0], position[1], margin):
                    legend.entries.append(Entry(series=index, position=position))
                else:
                    kept.append(position)
                    kept_components.append(component)
            series.positions = kept
            series.components = kept_components

        for entry in legend.entries:
            best: tuple[float, int] | None = None
            for index, block in enumerate(blocks):
                if not frame.contains(block.centre_x, block.centre_y, margin):
                    continue
                if block.x + block.width / 2.0 <= entry.position[0]:
                    continue
                share = _overlaps(
                    entry.position[1] - 1.0, entry.position[1] + 1.0,
                    float(block.y), float(block.bottom),
                )
                if share <= 0.0:
                    continue
                distance = block.x - entry.position[0]
                if best is None or distance < best[0]:
                    best = (distance, index)
            if best is not None:
                entry.block = best[1]
                legend.blocks.add(best[1])

        # A row whose sample could not be separated from its label is still a
        # row; record it with the name alone.
        for index, block in enumerate(blocks):
            if index in legend.blocks:
                continue
            if not frame.contains(block.centre_x, block.centre_y, margin):
                continue
            legend.entries.append(Entry(series=None, position=None, block=index))
            legend.blocks.add(index)

        legend.entries.sort(key=lambda item: (item.position or (0.0, 0.0))[1] if item.position
                            else float(blocks[item.block].centre_y))
        if legend.entries:
            legends.append(legend)
    return legends


def find_unframed(
    marker_sets: list[MarkerSet],
    blocks: list,
    text_height: float,
    *,
    minimum_rows: int = 2,
) -> list[Legend]:
    """Find a legend drawn without a box, by the shape of its rows.

    Sitting outside the plot is not the cue: legends are as often placed inside
    the axes. What a legend always is, and a scattered annotation never is, is
    several rows lined up — the samples sharing a column, each with its name
    immediately to the right, and the names starting at a common margin.
    """
    if not marker_sets or len(blocks) < minimum_rows:
        return []

    reach = 2.2 * text_height
    pairs: list[tuple[int, tuple[float, float], object, int]] = []
    for series_index, series in enumerate(marker_sets):
        for position, component in zip(series.positions, series.components):
            best: tuple[float, int] | None = None
            for block_index, block in enumerate(blocks):
                gap = block.x - component.right
                if gap < 0 or gap > reach:
                    continue
                if not (block.y - 0.4 * text_height <= position[1] <= block.bottom + 0.4 * text_height):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, block_index)
            if best is not None:
                pairs.append((series_index, position, component, best[1]))

    if len(pairs) < minimum_rows:
        return []

    spread = 0.8 * max(
        max(c.width, c.height) for _s, _p, c, _b in pairs
    )
    pairs.sort(key=lambda item: item[1][0])
    columns: list[list] = [[pairs[0]]]
    for pair in pairs[1:]:
        if abs(pair[1][0] - columns[-1][-1][1][0]) <= spread:
            columns[-1].append(pair)
        else:
            columns.append([pair])

    legends: list[Legend] = []
    for column in columns:
        if len(column) < minimum_rows:
            continue
        if len({block for _s, _p, _c, block in column}) < minimum_rows:
            continue
        # The names of a legend start at one margin; scattered labels do not.
        margins = [blocks[block].x for _s, _p, _c, block in column]
        if max(margins) - min(margins) > text_height:
            continue

        legend = Legend(frame=None)
        for series_index, position, component, block_index in column:
            legend.entries.append(
                Entry(series=series_index, position=position, block=block_index)
            )
            legend.blocks.add(block_index)

        for series_index, position, _component, _block in column:
            series = marker_sets[series_index]
            keep = [
                (other, component)
                for other, component in zip(series.positions, series.components)
                if other != position
            ]
            series.positions = [other for other, _component in keep]
            series.components = [component for _other, component in keep]

        left = min(c.x for _s, _p, c, _b in column)
        top = min(min(c.y for _s, _p, c, _b in column),
                  min(blocks[b].y for _s, _p, _c, b in column))
        right = max(blocks[b].right for _s, _p, _c, b in column)
        bottom = max(max(c.bottom for _s, _p, c, _b in column),
                     max(blocks[b].bottom for _s, _p, _c, b in column))
        legend.bounds = (float(left), float(top), float(right - left), float(bottom - top))
        legend.entries.sort(key=lambda entry: entry.position[1] if entry.position else 0.0)
        legends.append(legend)
    return legends
