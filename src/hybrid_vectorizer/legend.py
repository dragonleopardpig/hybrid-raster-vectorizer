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
    frame: Frame
    entries: list[Entry] = field(default_factory=list)
    blocks: set[int] = field(default_factory=set)


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
        legend = Legend(frame=frame)

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
