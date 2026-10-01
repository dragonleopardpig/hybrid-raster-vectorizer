"""The automatic pipeline: a raster scientific figure in, a semantic SVG out."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ir
from . import latex as tex
from .components import Component, extract, median_text_height
from .fitting import choose_model, circular_path, fit_bezier, path_data
from .fonts import FontFace, list_faces, match_font
from .legend import Legend, assemble, find_unframed
from .ocr import (
    FormulaReader,
    Reading,
    augmentations,
    isolate,
    looks_like_prose,
    read_tesseract,
    turned,
    upright_bias,
)
from .preprocess import Page, load_page
from .primitives import Rule, TickSet, detect_rules, detect_ticks
from .preserve import mask_path
from .alphabet import apply as apply_alphabet
from .alphabet import build_clusters, solve
from .consensus import reconcile
from .refine import (
    FontSet,
    agreement,
    build_font_set,
    fit_size,
    glyph_slots,
    demote_spurious_scripts,
    rasterise,
    render_box,
    shape_iou,
    substitute_glyphs,
)
from .shapes import (
    MarkerSet,
    Region,
    claim_similar,
    detect_regions,
    find_marker_sets,
    region_path,
)
from .textlayout import Block, group_blocks, text_angle
from .tracing import (
    Trace, arrowheads_on, merge_collinear, nearest_pen, partition, pen_set,
    restore_contacts, smooth_path, straighten,
)


@dataclass
class Options:
    deskew: bool = True
    bezier_tolerance: float = 0.25
    model_tolerance: float = 0.5
    idealise: bool = True
    confidence_threshold: float = 0.55
    raster_fallback: bool = False
    typeset_uncertain: bool = False
    substitute_glyphs: bool = False
    solve_alphabet: bool = False
    background: str | None = None
    ensemble: int = 1
    largest_label: float = 3.5
    doubtful_label: float = 2.4
    doubtful_shape: float = 0.20
    font_family: str | None = None
    font_candidates: int = 400
    use_formula_ocr: bool = True
    verify: bool = True


@dataclass
class Analysis:
    page: Page
    components: list[Component]
    text_height: float
    rules: list[Rule]
    ticks: list[TickSet]
    traces: list[Trace]
    blocks: list[Block]
    regions: list = field(default_factory=list)
    marker_sets: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    legends: list = field(default_factory=list)
    dashed: list = field(default_factory=list)
    rotations: dict[int, float] = field(default_factory=dict)
    pens: list[float] = field(default_factory=list)
    rejected: list = field(default_factory=list)
    readings: dict[int, Reading] = field(default_factory=dict)
    prepared: list = field(default_factory=list)
    alternatives: dict[int, list[str]] = field(default_factory=dict)
    label_reports: list[dict] = field(default_factory=list)
    unassigned: list[Component] = field(default_factory=list)


def analyse(path: Path, options: Options) -> Analysis:
    page = load_page(path, deskew=options.deskew)
    components = extract(page.ink)
    text_height = median_text_height(components, page.height, page.stroke_width)

    regions, working = detect_regions(page.ink, page.stroke_width, page.gray)
    rules = detect_rules(page, ink=working)
    ticks = [
        tick
        for tick in (detect_ticks(page, rule, ink=working, others=rules) for rule in rules)
        if tick
    ]
    dashed, frames, traces, leftovers = partition(page, working, rules, ticks, text_height)

    # One drawing, one or two pens. Snapping here rather than at the drawing
    # keeps every later measurement -- arrowheads, dash periods -- on the width
    # the mark will actually be drawn with.
    pens = pen_set(
        [trace.stroke_width for trace in traces]
        + [line.stroke_width for line in dashed]
        + [rule.thickness for rule in rules]
        + [region.hatch.stroke_width for region in regions if region.hatch is not None],
        page.stroke_width,
    )
    for trace in traces:
        trace.stroke_width = nearest_pen(trace.stroke_width, pens)
    for line in dashed:
        line.stroke_width = nearest_pen(line.stroke_width, pens)
    for rule in rules:
        rule.thickness = nearest_pen(rule.thickness, pens)
    for region in regions:
        if region.hatch is not None:
            region.hatch.stroke_width = nearest_pen(region.hatch.stroke_width, pens)

    # A line the scan broke into pieces is still one line, and a straight one is
    # drawn straight.
    traces = merge_collinear(traces, tolerance=1.2 * page.stroke_width, reach=6.0 * page.stroke_width)
    traces = straighten(traces, tolerance=1.2 * page.stroke_width)
    restore_contacts(traces, rules)
    blocks = group_blocks(leftovers, page.ink.shape, text_height, page.stroke_width)

    marker_sets, consumed = find_marker_sets(
        blocks, page.stroke_width, page_size=(page.width, page.height)
    )
    claimed = {
        id(component)
        for index, block in enumerate(blocks)
        if index in consumed
        for component in block.components
    }

    # A legend sample sits as close to its name as a letter sits to its
    # neighbours, so isolation cannot find it. Resemblance can, but only while
    # the test is strict enough that a letter cannot pass it.
    if marker_sets:
        loose = [component for component in leftovers if id(component) not in claimed]
        claimed |= set(claim_similar(marker_sets, loose))
    leftovers = [component for component in leftovers if id(component) not in claimed]
    blocks = group_blocks(leftovers, page.ink.shape, text_height, page.stroke_width)

    legends = assemble(frames, marker_sets, blocks)
    spoken_for = {index for legend in legends for index in legend.blocks}
    legends += [
        legend
        for legend in find_unframed(marker_sets, blocks, text_height)
        if not (legend.blocks & spoken_for)
    ]

    labelled = {id(component) for block in blocks for component in block.components}
    unassigned = [component for component in leftovers if id(component) not in labelled]
    return Analysis(
        page, components, text_height, rules, ticks, traces, blocks,
        regions=regions, marker_sets=marker_sets, frames=frames, legends=legends,
        dashed=dashed, pens=pens, unassigned=unassigned,
    )


def read_blocks(analysis: Analysis, options: Options) -> None:
    """Route every block to the recogniser its structure calls for."""
    page = analysis.page
    crops: dict[int, np.ndarray] = {}
    for index, block in enumerate(analysis.blocks):
        crop = isolate(page.gray, block.components)
        slope = text_angle(block)
        if abs(slope) < 1e-6:
            crops[index] = crop
            continue
        # Which end the text starts from is not knowable from the ink. The
        # recogniser's own confidence decides it where the two differ, and where
        # they do not, which way up the type sits does.
        best = None
        for angle in (slope, slope + 180.0):
            candidate = turned(crop, angle)
            score = (read_tesseract(candidate).confidence or 0.0) + upright_bias(candidate)
            if best is None or score > best[0]:
                best = (score, angle, candidate)
        _score, angle, candidate = best
        analysis.rotations[index] = ((angle + 180.0) % 360.0) - 180.0
        crops[index] = candidate

    # A legend entry names a series, so it is prose even when it reads poorly.
    # Handing it to a formula recogniser turns a stray sample mark beside the
    # word into \square and buries the name inside it.
    legend_labels = {index for legend in analysis.legends for index in legend.blocks}

    pending: list[int] = []
    for index, block in enumerate(analysis.blocks):
        # Only a fraction bar settles the question on structure alone. Anything
        # else is offered to Tesseract first: a confident prose reading is worth
        # more than a formula recogniser's guess, and it anchors font matching.
        if block.bars:
            pending.append(index)
            continue
        reading = read_tesseract(crops[index])
        if looks_like_prose(reading) or (index in legend_labels and reading.text):
            analysis.readings[index] = reading
        else:
            pending.append(index)

    if not pending:
        return
    if not options.use_formula_ocr:
        for index in pending:
            analysis.readings[index] = read_tesseract(crops[index])
        return

    with FormulaReader() as reader:
        for index in pending:
            if options.ensemble <= 1:
                analysis.readings[index] = reader.read(crops[index])
                continue
            counts = Counter()
            variants = augmentations(crops[index], options.ensemble)
            for variant in variants:
                text = reader.read(variant).text
                if text:
                    counts[text] += 1
            if not counts:
                analysis.readings[index] = Reading("", "formulaocr", 0.0)
                continue
            text, hits = counts.most_common(1)[0]
            analysis.readings[index] = Reading(
                text, "formulaocr", None, consistency=hits / len(variants)
            )
            analysis.alternatives[index] = [
                other for other, _n in counts.most_common()[1:]
            ]


def _prose_samples(analysis: Analysis) -> list[tuple[np.ndarray, str]]:
    samples: list[tuple[np.ndarray, str]] = []
    for index, block in enumerate(analysis.blocks):
        reading = analysis.readings.get(index)
        if reading is None or reading.engine != "tesseract":
            continue
        if reading.confidence is None or reading.confidence < 0.6 or len(reading.text.replace(" ", "")) < 2:
            continue
        mask = np.zeros(analysis.page.ink.shape, dtype=np.uint8)
        for component in block.components:
            region = mask[component.y : component.bottom, component.x : component.right]
            np.maximum(region, component.mask, out=region)
        samples.append((mask[block.y : block.bottom, block.x : block.right], reading.text))
    return samples


def choose_fonts(analysis: Analysis, options: Options) -> tuple[FontSet | None, list[tuple[str, float]]]:
    faces = list_faces()
    if options.font_family:
        chosen = build_font_set(options.font_family, bold=False, faces=faces)
        return chosen, []

    ranked = match_font(_prose_samples(analysis), faces=faces, limit=options.font_candidates)
    if not ranked:
        return build_font_set("DejaVu Serif", bold=False, faces=faces), []

    summary = [(face.family + " " + face.style, score) for face, score in ranked[:8]]
    best_face, _score = ranked[0]
    return build_font_set(best_face.family, bold=best_face.bold, faces=faces), summary


def _measure_node(reading: Reading, block: Block, unknown: list[str] | None = None) -> tex.Row:
    if reading.engine == "tesseract":
        return tex.Row([tex.Run(reading.text, upright=True)])
    return tex.parse(reading.text, unknown=unknown)


def _block_ink(page: Page, block: Block, angle: float | None = None) -> np.ndarray:
    mask = np.zeros(page.ink.shape, dtype=np.uint8)
    for component in block.components:
        region = mask[component.y : component.bottom, component.x : component.right]
        np.maximum(region, component.mask, out=region)
    window = mask[block.y : block.bottom, block.x : block.right]
    if angle is None:
        return window
    # Size and score a turned label against ink turned the same way. This is a
    # mask, so its empty is black.
    from .ocr import turned

    return turned(window, angle, background=0)


def _ink_extent(image: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(image > 127)
    if xs.size == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _best_size(node: object, fonts: FontSet, ink: np.ndarray, target_width: float) -> tuple[float, float]:
    """Pick the type size whose rendering covers the same area of page as the ink.

    Shape agreement cannot choose a size, because it compares outlines after
    normalising them; the size has to be settled against the measured box.
    """
    width, height = _ink_extent(ink)
    if width == 0 or height == 0:
        return fit_size(node, fonts.metrics, target_width), 0.0

    seed = fit_size(node, fonts.metrics, target_width)
    best_size, best_error = seed, np.inf
    for factor in np.linspace(0.6, 1.5, 19):
        size = seed * float(factor)
        rendered = render_box(tex.layout(node, fonts.metrics, size), fonts)
        if rendered is None:
            continue
        drawn_width, drawn_height = _ink_extent(rendered)
        if drawn_width == 0 or drawn_height == 0:
            continue
        error = abs(np.log(drawn_width / width)) + abs(np.log(drawn_height / height))
        if error < best_error:
            best_size, best_error = size, error

    rendered = render_box(tex.layout(node, fonts.metrics, best_size), fonts)
    score = shape_iou(ink, rendered) if rendered is not None else 0.0
    return best_size, score


@dataclass
class Prepared:
    index: int
    block: Block
    reading: Reading
    node: tex.Row
    size: float
    score: float
    changes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    transform: str = ""
    unknown: list[str] = field(default_factory=list)

    @property
    def pure_fraction(self) -> bool:
        return len(self.node.items) == 1 and isinstance(self.node.items[0], tex.Frac)


def solve_alphabet(slots: list, options: Options) -> list[tuple[int, str]]:
    """Name every distinct shape in the figure at once, against installed fonts.

    Measured on the example this is worse than leaving the reading alone, so it
    is off unless asked for; see the README for the numbers.
    """
    clusters = build_clusters(slots)
    families = sorted({face.family for face in list_faces()})
    solution = solve(clusters, families)
    if solution is None:
        return []
    return apply_alphabet(solution, clusters)


def _harmonise(prepared: list[Prepared], text_height: float) -> None:
    """Repeated structures on one row are set in one size; make them agree.

    Tick labels are drawn identically, so a per-label size fit that disagrees
    across the row is fitting noise rather than the typography.
    """
    fractions = [entry for entry in prepared if entry.pure_fraction and entry.block.bars]
    rows: list[list[Prepared]] = []
    for entry in sorted(fractions, key=lambda item: item.block.bars[0].y):
        bar = entry.block.bars[0].y
        if rows and abs(rows[-1][-1].block.bars[0].y - bar) <= 0.8 * text_height:
            rows[-1].append(entry)
        else:
            rows.append([entry])

    for row in rows:
        if len(row) < 3:
            continue
        common = float(np.median([entry.size for entry in row]))
        for entry in row:
            entry.size = common


def preserve_block(
    analysis: Analysis, index: int, block: Block, options: Options, reasons: list[str]
) -> ir.Element:
    ink = _block_ink(analysis.page, block)
    common = dict(
        confidence=0.0, identifier=f"label-{index}", review_reasons=reasons,
        provenance="original ink; recognition unverified", x=float(block.x), y=float(block.y),
    )
    if options.raster_fallback:
        black = np.zeros_like(ink)
        success, buffer = cv2.imencode(".png", cv2.merge([black, black, black, ink]))
        if success:
            return ir.RasterFallback(
                kind="image", width=float(block.width), height=float(block.height),
                png=buffer.tobytes(), **common,
            )
    return ir.VectorFallback(kind="preserved", path=mask_path(ink), **common)


def label_report(
    analysis: Analysis, index: int, block: Block, element: ir.Element,
    entry: Prepared | None = None,
) -> dict:
    reading = analysis.readings.get(index)
    return {
        "id": element.identifier,
        "box": [block.x, block.y, block.width, block.height],
        "text": getattr(element, "plain", ""),
        "source_reading": reading.text if reading else "",
        "engine": reading.engine if reading else "",
        "confidence": round(element.confidence, 2),
        "reading_confidence": reading.confidence if reading else None,
        "consistency": reading.consistency if reading else None,
        "shape_score": round(entry.score, 3) if entry else None,
        "corrections": "; ".join(entry.changes) if entry else "",
        "other_readings": analysis.alternatives.get(index, []),
        "ambiguous_glyphs": entry.unresolved if entry else [],
        "unknown_commands": entry.unknown if entry else [],
        "representation": {"label": "text", "image": "raster", "preserved": "vector"}[element.kind],
        "review_reasons": list(element.review_reasons),
    }


def build_labels(analysis: Analysis, fonts: FontSet | None, options: Options) -> list[ir.Element]:
    page = analysis.page
    prepared: list[Prepared] = []
    labels: list[ir.Element] = []
    analysis.label_reports = []
    for index, block in enumerate(analysis.blocks):
        reading = analysis.readings.get(index)
        if fonts is None or reading is None or not reading.text:
            reasons = ["no usable font" if fonts is None else "empty OCR result"]
            element = preserve_block(analysis, index, block, options, reasons)
            labels.append(element)
            analysis.label_reports.append(label_report(analysis, index, block, element))
            continue

        angle = analysis.rotations.get(index)
        ink = _block_ink(page, block, angle)
        unknown: list[str] = []
        node = _measure_node(reading, block, unknown)
        along = float(_ink_extent(ink)[0] if angle is not None else block.width)
        size, score = _best_size(node, fonts, ink, along)

        # A reading that has to be set six times the size of everything else on
        # the page is not a line of text. These are stray marks -- a tick, a
        # dash, a speck -- grouped together and then read as something, and
        # drawing the answer puts large invented words across the figure.
        # Size alone cannot settle this. The largest label on the generated
        # series figure is a tick reading 2.8 times the page's type and it is
        # real; the largest on waves1.png is the same multiple and is a dozen
        # zigzag dashes read as "r=2". What separates them is whether the
        # setting looks anything like the ink: the real ones score 0.42 to 0.58
        # on shape, the invented ones 0.00 to 0.19.
        outsized = size > options.doubtful_label * analysis.text_height
        unlike = score < options.doubtful_shape
        if size > options.largest_label * analysis.text_height or (outsized and unlike):
            analysis.rejected.append(
                {
                    "box": [block.x, block.y, block.width, block.height],
                    "reading": reading.text,
                    "size_px": round(size, 1),
                    "shape_score": round(score, 2),
                    "why": "type size out of family with the page",
                }
            )
            element = preserve_block(analysis, index, block, options, ["implausible text layout"])
            labels.append(element)
            entry = Prepared(index, block, reading, node, size, score, unknown=unknown)
            analysis.label_reports.append(label_report(analysis, index, block, element, entry))
            continue

        prepared.append(
            Prepared(index=index, block=block, reading=reading, node=node, size=size, score=score,
                     unknown=unknown)
        )

    _harmonise(prepared, analysis.text_height)

    # Place everything first: consensus needs every label's glyphs at once.
    placements: dict[int, tuple[float, float]] = {}
    slots: list = []
    for entry in prepared:
        block, size = entry.block, entry.size
        box = tex.layout(entry.node, fonts.metrics, size)
        angle = analysis.rotations.get(entry.index)
        if angle is not None:
            # Laid out flat, then turned about the middle of its own ink.
            centre_x = block.x + block.width / 2.0
            centre_y = block.y + block.height / 2.0
            offset_x = box.width / 2.0
            offset_y = (box.descent - box.ascent) / 2.0
            entry.transform = (
                f"translate({centre_x:.2f} {centre_y:.2f}) rotate({angle:.0f}) "
                f"translate({-offset_x:.2f} {-offset_y:.2f})"
            )
            placements[entry.index] = (0.0, 0.0)
            continue
        if entry.pure_fraction and block.bars:
            bar = block.bars[0]
            baseline = bar.y + bar.height / 2.0 + tex.AXIS_RATIO * size
            x = block.x + (block.width - box.width) / 2.0
        else:
            baseline = block.baseline()
            x = float(block.x)
        placements[entry.index] = (x, baseline)

        if entry.transform:
            continue  # a turned label has no shared frame with the page's ink
        glyph_ink = [
            component for component in block.components if component not in block.bars
        ]
        bar_y = (block.bars[0].y + block.bars[0].height / 2.0) if block.bars else None
        demotions = demote_spurious_scripts(
            entry.node, fonts, size, x, baseline, glyph_ink, block.body_height(), bar_y
        )
        if demotions:
            entry.changes.extend(demotions)
            entry.size = size = _best_size(entry.node, fonts, _block_ink(page, block), float(block.width))[0]
            box = tex.layout(entry.node, fonts.metrics, size)
            if entry.pure_fraction and block.bars:
                bar = block.bars[0]
                baseline = bar.y + bar.height / 2.0 + tex.AXIS_RATIO * size
                x = block.x + (block.width - box.width) / 2.0
            placements[entry.index] = (x, baseline)

        found, ambiguous = glyph_slots(
            entry.node, fonts, size, x, baseline, glyph_ink, bar_y
        )
        for slot in found:
            slot.label = entry.index
        slots.extend(found)
        entry.unresolved.extend(ambiguous)

    corrections: list[tuple[int, str]] = []
    if options.solve_alphabet:
        corrections.extend(solve_alphabet(slots, options))
    if options.substitute_glyphs:
        corrections.extend(substitute_glyphs(slots, fonts))
    corrections.extend(reconcile(slots))

    by_index = {entry.index: entry for entry in prepared}
    for label_index, description in corrections:
        entry = by_index.get(label_index)
        if entry is not None:
            entry.changes.append(description)

    for entry in prepared:
        block, node, size = entry.block, entry.node, entry.size
        box = tex.layout(node, fonts.metrics, size)
        x, baseline = placements[entry.index]

        rendered = render_box(box, fonts)
        measured = _block_ink(page, block, analysis.rotations.get(entry.index))
        score = shape_iou(measured, rendered) if rendered is not None else entry.score
        confidence = entry.reading.confidence or 0.0
        reasons = []
        if entry.reading.confidence is None:
            reasons.append("recognition confidence unavailable")
        elif confidence < options.confidence_threshold:
            reasons.append("low recognition confidence")
        token = entry.reading.text.strip()
        if entry.reading.engine == "tesseract" and len(token) == 1 and token.isalpha():
            reasons.append("isolated letter may be a mathematical symbol")
        if score < options.doubtful_shape:
            reasons.append("poor visual match")
        if entry.unknown:
            reasons.append("unsupported formula commands")
        if entry.reading.engine == "formulaocr" and entry.unresolved:
            reasons.append("ambiguous formula glyphs")
        if entry.reading.consistency is not None and entry.reading.consistency < 1.0:
            reasons.append("OCR variants disagree")
        entry.score = score

        # The face in hand is not the face on the page, so a run set with its
        # own advances ends short of the printed one: on complex.png the last
        # letter of nearly every label was landing a glyph behind its ink.
        # Spacing is stretched to the width that was measured, glyph sizes are
        # not, and the stretch is bounded so a wrong reading cannot run away.
        spread = 1.0
        if rendered is not None:
            ink_width = _ink_extent(measured)[0]
            set_width = _ink_extent(rendered)[0]
            if ink_width > 0 and set_width > 0:
                spread = float(np.clip(ink_width / set_width, 0.75, 1.3))
        if entry.pure_fraction and block.bars:
            x -= 0.5 * (spread - 1.0) * box.width

        if entry.unknown or (reasons and (not options.typeset_uncertain or options.raster_fallback)):
            element = preserve_block(analysis, entry.index, block, options, reasons)
            labels.append(element)
            analysis.label_reports.append(label_report(analysis, entry.index, block, element, entry))
            continue

        labels.append(
            ir.Label(
                kind="label",
                confidence=confidence,
                provenance="; ".join(entry.changes),
                identifier=f"label-{entry.index}",
                box=box,
                x=x,
                baseline=baseline,
                source=entry.reading.text,
                plain=tex.to_text(node),
                engine=entry.reading.engine,
                transform=entry.transform,
                spread=spread,
                review_reasons=reasons,
            )
        )
        analysis.label_reports.append(label_report(analysis, entry.index, block, labels[-1], entry))
    analysis.prepared = prepared
    analysis.label_reports.sort(key=lambda record: int(record["id"].split("-")[-1]))
    return labels


def build_geometry(analysis: Analysis, options: Options) -> tuple[list[ir.Element], list[dict]]:
    page = analysis.page
    elements: list[ir.Element] = []
    notes: list[dict] = []

    # Areas go down first so that strokes and marks sit on top of them, and
    # tints go under the rest: a tint is what a solid or a ruling is drawn over.
    tolerance = max(1.0, 0.4 * page.stroke_width)
    ordered = sorted(analysis.regions, key=lambda region: region.kind != "tint")
    for index, region in enumerate(ordered):
        shape = region.outline
        elements.append(
            ir.Area(
                kind="area",
                confidence=0.9 if region.kind == "solid" else 0.8,
                provenance=region.kind,
                identifier=f"area-{index}",
                path=region_path(shape, tolerance),
                shape=shape.kind if shape.kind in {"rectangle", "circle"} else "freeform",
                parameters=dict(shape.parameters),
                hatch_angle=region.hatch.angle if region.hatch else None,
                hatch_spacing=region.hatch.spacing if region.hatch else None,
                hatch_width=region.hatch.stroke_width if region.hatch else page.stroke_width,
                bordered=region.bordered,
                border_width=page.stroke_width,
                opacity=region.opacity,
                bounds=(
                    float(region.component.x), float(region.component.y),
                    float(region.component.width), float(region.component.height),
                ),
            )
        )

    for index, trace in enumerate(analysis.traces):
        points = trace.points
        steps = np.diff(points[:, 0])
        functional = not trace.closed and (np.all(steps >= 0) or np.all(steps <= 0))
        model = choose_model(
            points[:, 0], points[:, 1], tolerance=options.model_tolerance * trace.stroke_width
        ) if functional else None
        loose = choose_model(points[:, 0], points[:, 1], tolerance=4.0 * trace.stroke_width) if functional else None
        source = smooth_path(points, int(2.0 * trace.stroke_width) | 1, closed=trace.closed)
        if options.idealise and model is not None:
            grid = np.linspace(points[0, 0], points[-1, 0], max(256, points.shape[0]))
            source = np.column_stack([grid, model.sample(grid)])

        arc = circular_path(points, options.model_tolerance * trace.stroke_width, closed=trace.closed)
        segments = []
        remaining = source
        start_tangent = None
        contact_tangent = np.array([1.0, 0.0])
        for contact in sorted(trace.contacts):
            before = remaining[remaining[:, 0] < contact[0] - 3.0 * trace.stroke_width]
            segments.extend(fit_bezier(
                np.vstack([before, contact]), options.bezier_tolerance * trace.stroke_width,
                start_tangent=start_tangent, end_tangent=contact_tangent,
            ))
            remaining = np.vstack([contact, remaining[remaining[:, 0] > contact[0] + 3.0 * trace.stroke_width]])
            start_tangent = contact_tangent
        segments.extend(fit_bezier(
            remaining, options.bezier_tolerance * trace.stroke_width, start_tangent=start_tangent,
        ))
        path = path_data(segments, closed=trace.closed)
        if len(points) == 2:
            path = f"M{points[0, 0]:.2f} {points[0, 1]:.2f} L{points[1, 0]:.2f} {points[1, 1]:.2f}"
        elif arc is not None and options.idealise and not trace.contacts and (model is None or trace.closed):
            path = arc[0]
        described = model or loose
        own_ink = np.zeros_like(page.ink)
        for component in trace.components:
            window = own_ink[component.y:component.bottom, component.x:component.right]
            np.maximum(window, component.mask, out=window)
        head_points = points
        if len(points) == 2:
            head_points = np.linspace(points[0], points[-1], max(6, int(np.linalg.norm(points[-1] - points[0])) + 1))
        start_head, end_head = (None, None) if trace.closed else arrowheads_on(head_points, own_ink, trace.stroke_width)
        elements.append(
            ir.Curve(
                kind="curve",
                confidence=0.95 if model is not None else 0.8,
                provenance=f"trace:{trace.method}",
                identifier=f"curve-{index}",
                path=path,
                stroke_width=trace.stroke_width,
                model=described.description if described else "",
                model_rms=described.rms if described else None,
                arrow_start=start_head is not None,
                arrow_end=end_head is not None,
                arrow_length=max(
                    (head.length for head in (start_head, end_head) if head), default=12.0
                ),
                arrow_width=max(
                    (head.width for head in (start_head, end_head) if head), default=10.0
                ),
                dash=trace.dash,
                gap=trace.gap,
            )
        )
        notes.append(
            {
                "curve": index,
                "points": int(points.shape[0]),
                "segments": path.count("C") + path.count("A") + path.count("L"),
                "rule_contacts": [list(contact) for contact in trace.contacts],
                "closed": trace.closed,
                "primitive": "line" if len(points) == 2 else "arc" if " A" in path else "bezier",
                "endpoints": points[[0, -1]].round(2).tolist(),
                "stroke_width": round(trace.stroke_width, 2),
                "analytic_model": described.name if described else None,
                "analytic_residual_px": round(described.rms, 2) if described else None,
                "within_tolerance": model is not None,
                "arrowheads": int(start_head is not None) + int(end_head is not None),
                "dashed": trace.dash > 0,
            }
        )

    for index, rule in enumerate(analysis.rules):
        (x1, y1), (x2, y2) = rule.endpoints()
        elements.append(
            ir.Axis(
                kind="axis",
                confidence=rule.confidence,
                provenance="morphological rule",
                identifier=f"axis-{index}",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke_width=rule.thickness,
                arrow_start=any(not arrow.at_end for arrow in rule.arrows),
                arrow_end=any(arrow.at_end for arrow in rule.arrows),
            )
        )

    for index, series in enumerate(analysis.marker_sets):
        elements.append(
            ir.MarkerField(
                kind="markers",
                confidence=0.9,
                provenance=f"{len(series.positions)} congruent marks",
                identifier=f"series-{index}",
                shape=series.shape,
                size=series.size,
                filled=series.filled,
                stroke_width=page.stroke_width,
                positions=series.positions,
                parameters={"r": series.parameters["r"]} if "r" in series.parameters else {},
            )
        )

    for index, line in enumerate(analysis.dashed):
        elements.append(
            ir.Dashed(
                kind="dashed",
                confidence=0.85,
                provenance=f"{len(line.components)} marks on one period",
                identifier=f"dashed-{index}",
                x1=line.start[0], y1=line.start[1], x2=line.end[0], y2=line.end[1],
                stroke_width=line.stroke_width,
                dash=line.dash,
                gap=line.gap,
                marks=len(line.components),
                stretches=list(line.drawn),
            )
        )

    for index, frame in enumerate(analysis.frames):
        elements.append(
            ir.Frame(
                kind="frame",
                confidence=0.9,
                provenance="closed box",
                identifier=f"frame-{index}",
                x=float(frame.x),
                y=float(frame.y),
                width=float(frame.width),
                height=float(frame.height),
                stroke_width=page.stroke_width,
            )
        )

    for index, tick_set in enumerate(analysis.ticks):
        elements.append(
            ir.Ticks(
                kind="ticks",
                confidence=tick_set.confidence,
                provenance="lattice",
                identifier=f"ticks-{index}",
                orientation=tick_set.rule.orientation,
                positions=tick_set.positions,
                near=tick_set.near,
                far=tick_set.far,
                stroke_width=tick_set.thickness,
                spacing=tick_set.spacing,
            )
        )
    for index, component in enumerate(getattr(analysis, "unassigned", [])):
        elements.append(ir.VectorFallback(
            kind="preserved", identifier=f"unclassified-{index}", confidence=0.0,
            review_reasons=["unclassified residual ink"], path=mask_path(component.mask),
            x=float(component.x), y=float(component.y),
        ))
    return elements, notes


def group_legends(
    analysis: Analysis, geometry: list[ir.Element], labels: list[ir.Element]
) -> tuple[list[ir.Element], list[ir.Element]]:
    """Gather each legend's frame, samples and names into one group.

    A legend is one object in the drawing and should be one object in the file,
    so that moving it moves the box, the samples and the words together.
    """
    if not analysis.legends:
        return geometry, labels

    by_id = {element.identifier: element for element in geometry + labels}
    spoken_for: set[str] = set()
    groups: list[ir.Group] = []

    for index, legend in enumerate(analysis.legends):
        frame_index = next(
            (i for i, frame in enumerate(analysis.frames) if frame is legend.frame), None
        )
        children: list[ir.Element] = []

        frame_element = by_id.get(f"frame-{frame_index}") if frame_index is not None else None
        if frame_element is not None:
            children.append(frame_element)
            spoken_for.add(frame_element.identifier)

        named = 0
        for position, entry in enumerate(legend.entries):
            if entry.series is not None and entry.position is not None:
                series = analysis.marker_sets[entry.series]
                children.append(
                    ir.MarkerField(
                        kind="markers",
                        confidence=0.9,
                        provenance="legend sample",
                        identifier=f"legend-{index}-sample-{position}",
                        shape=series.shape,
                        size=series.size,
                        filled=series.filled,
                        stroke_width=analysis.page.stroke_width,
                        positions=[entry.position],
                        symbol_id=f"marker-series-{entry.series}",
                    )
                )
            if entry.block is None:
                continue
            label = by_id.get(f"label-{entry.block}")
            if label is None:
                continue
            if entry.series is not None:
                label.provenance = (
                    f"names series-{entry.series}"
                    + (f"; {label.provenance}" if label.provenance else "")
                )
                named += 1
            children.append(label)
            spoken_for.add(label.identifier)

        if len(children) <= 1:
            continue
        groups.append(
            ir.Group(
                kind="legend",
                confidence=0.85,
                provenance="frame with samples and names",
                identifier=f"legend-{index}",
                label="legend",
                note=f"{len(legend.entries)} entries, {named} tied to a series",
                children=children,
            )
        )

    geometry = [element for element in geometry if element.identifier not in spoken_for]
    labels = [element for element in labels if element.identifier not in spoken_for]
    return geometry, labels + groups


def convert(path: Path, options: Options | None = None) -> ir.Document:
    options = options or Options()
    analysis = analyse(path, options)
    read_blocks(analysis, options)
    fonts, ranking = choose_fonts(analysis, options)

    labels = build_labels(analysis, fonts, options)
    geometry, curve_notes = build_geometry(analysis, options)
    geometry, labels = group_legends(analysis, geometry, labels)

    page = analysis.page
    arrow_length = max(
        (arrow.length for rule in analysis.rules for arrow in rule.arrows), default=12.0
    )
    arrow_width = max(
        (arrow.width for rule in analysis.rules for arrow in rule.arrows), default=10.0
    )

    document = ir.Document(
        width=float(page.width),
        height=float(page.height),
        background=options.background,
        title=f"Reconstruction of {path.name}",
        description=(
            f"Automatic vector reconstruction: {len(analysis.traces)} traced curve(s), "
            f"{len(analysis.rules)} axis rule(s), {len(analysis.blocks)} label(s)."
        ),
        font_family=fonts.family if fonts else "serif",
        bold=bool(fonts and fonts.bold),
        arrow_length=arrow_length,
        arrow_width=arrow_width,
        geometry=geometry,
        labels=labels,
    )

    document.report = {
        "source": str(path),
        "size": [page.width, page.height],
        "stroke_width": round(page.stroke_width, 2),
        "detected_background": page.background,
        "skew_degrees": round(page.skew_degrees, 3),
        "text_height": round(analysis.text_height, 1),
        "font": {
            "chosen": fonts.family if fonts else None,
            "bold": bool(fonts and fonts.bold),
            "ranking": [{"face": name, "score": round(score, 3)} for name, score in ranking],
        },
        "curves": curve_notes,
        "areas": [
            {
                "kind": region.kind,
                "shape": region.outline.kind,
                "box": [region.component.x, region.component.y,
                        region.component.width, region.component.height],
                "opacity": round(region.opacity, 2),
                "hatch": (
                    {"angle_degrees": round(region.hatch.angle, 1),
                     "spacing_px": round(region.hatch.spacing, 2)}
                    if region.hatch else None
                ),
            }
            for region in analysis.regions
        ],
        "dashed_lines": [
            {
                "from": [round(line.start[0], 1), round(line.start[1], 1)],
                "to": [round(line.end[0], 1), round(line.end[1], 1)],
                "dash_px": round(line.dash, 1),
                "gap_px": round(line.gap, 1),
                "marks": len(line.components),
                "stretches": len(line.drawn),
            }
            for line in analysis.dashed
        ],
        "legends": [
            {
                "box": [round(value, 1) for value in legend.bounds],
                "framed": legend.framed,
                "entries": [
                    {
                        "series": entry.series,
                        "sample": (
                            [round(entry.position[0], 1), round(entry.position[1], 1)]
                            if entry.position else None
                        ),
                        "named": entry.block is not None,
                    }
                    for entry in legend.entries
                ],
            }
            for legend in analysis.legends
        ],
        "marker_series": [
            {
                "shape": series.shape,
                "filled": series.filled,
                "size_px": round(series.size, 1),
                "count": len(series.positions),
            }
            for series in analysis.marker_sets
        ],
        "ticks": [
            {
                "orientation": tick.rule.orientation,
                "count": len(tick.positions),
                "spacing_px": round(tick.spacing, 2) if tick.spacing else None,
                "confidence": round(tick.confidence, 2),
            }
            for tick in analysis.ticks
        ],
        "labels": analysis.label_reports,
        "quality": {
            "label_blocks": len(analysis.blocks),
            "labels_accounted_for": len(analysis.label_reports),
            "typeset_labels": sum(record["representation"] == "text" for record in analysis.label_reports),
            "preserved_labels": sum(record["representation"] != "text" for record in analysis.label_reports),
            "unverified_labels": sum(bool(record["review_reasons"]) for record in analysis.label_reports),
            "unclassified_components_preserved": len(analysis.unassigned),
            "closed_curves": sum(trace.closed for trace in analysis.traces),
            "curve_primitives": dict(Counter(note["primitive"] for note in curve_notes)),
            "semantic_accuracy": None,
        },
    }

    if options.verify:
        rendered = rasterise(document.to_svg(), page.width, page.height)
        if rendered is not None:
            document.report["agreement"] = {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in agreement(page.ink, rendered).items()
            }

    low = [record["id"] for record in analysis.label_reports if record["review_reasons"]]
    low += [
        element.identifier
        for element in geometry
        if element.confidence < options.confidence_threshold or element.review_reasons
    ]
    document.report["needs_review"] = low
    document.report["not_labels"] = analysis.rejected
    return document
