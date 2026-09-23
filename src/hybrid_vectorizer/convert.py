"""The automatic pipeline: a raster scientific figure in, a semantic SVG out."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ir
from . import latex as tex
from .components import Component, extract, median_text_height
from .fitting import choose_model, fit_bezier, path_data
from .fonts import FontFace, list_faces, match_font
from .ocr import FormulaReader, Reading, isolate, looks_like_prose, read_tesseract
from .preprocess import Page, load_page
from .primitives import Rule, TickSet, detect_rules, detect_ticks
from .consensus import reconcile
from .refine import (
    FontSet,
    agreement,
    build_font_set,
    correct,
    fit_size,
    glyph_slots,
    rasterise,
    render_box,
    shape_iou,
    substitute_glyphs,
)
from .textlayout import Block, group_blocks, script_components
from .tracing import Trace, extract_curves


@dataclass
class Options:
    deskew: bool = True
    bezier_tolerance: float = 0.25
    model_tolerance: float = 0.5
    idealise: bool = False
    confidence_threshold: float = 0.55
    raster_fallback: bool = False
    substitute_glyphs: bool = False
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
    readings: dict[int, Reading] = field(default_factory=dict)
    prepared: list = field(default_factory=list)


def analyse(path: Path, options: Options) -> Analysis:
    page = load_page(path, deskew=options.deskew)
    components = extract(page.ink)
    text_height = median_text_height(components, page.height)

    rules = detect_rules(page)
    ticks = [
        tick for tick in (detect_ticks(page, rule, others=rules) for rule in rules) if tick
    ]
    traces, leftovers = extract_curves(page, rules, ticks, components, text_height)
    blocks = group_blocks(leftovers, page.ink.shape, text_height, page.stroke_width)
    return Analysis(page, components, text_height, rules, ticks, traces, blocks)


def read_blocks(analysis: Analysis, options: Options) -> None:
    """Route every block to the recogniser its structure calls for."""
    page = analysis.page
    crops = {
        index: isolate(page.gray, block.components)
        for index, block in enumerate(analysis.blocks)
    }

    pending: list[int] = []
    for index, block in enumerate(analysis.blocks):
        # Only a fraction bar settles the question on structure alone. Anything
        # else is offered to Tesseract first: a confident prose reading is worth
        # more than a formula recogniser's guess, and it anchors font matching.
        if block.bars:
            pending.append(index)
            continue
        reading = read_tesseract(crops[index])
        if looks_like_prose(reading):
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
            analysis.readings[index] = reader.read(crops[index])


def _prose_samples(analysis: Analysis) -> list[tuple[np.ndarray, str]]:
    samples: list[tuple[np.ndarray, str]] = []
    for index, block in enumerate(analysis.blocks):
        reading = analysis.readings.get(index)
        if reading is None or reading.engine != "tesseract":
            continue
        if reading.confidence < 0.6 or len(reading.text.replace(" ", "")) < 2:
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


def _measure_node(reading: Reading, block: Block) -> tex.Row:
    if reading.engine == "tesseract":
        return tex.Row([tex.Run(reading.text, upright=True)])
    return tex.parse(reading.text)


def _block_ink(page: Page, block: Block) -> np.ndarray:
    mask = np.zeros(page.ink.shape, dtype=np.uint8)
    for component in block.components:
        region = mask[component.y : component.bottom, component.x : component.right]
        np.maximum(region, component.mask, out=region)
    return mask[block.y : block.bottom, block.x : block.right]


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

    @property
    def pure_fraction(self) -> bool:
        return len(self.node.items) == 1 and isinstance(self.node.items[0], tex.Frac)


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


def build_labels(analysis: Analysis, fonts: FontSet | None, options: Options) -> list[ir.Element]:
    if fonts is None:
        return []

    page = analysis.page
    prepared: list[Prepared] = []
    for index, block in enumerate(analysis.blocks):
        reading = analysis.readings.get(index)
        if reading is None or not reading.text:
            continue

        ink = _block_ink(page, block)
        node = _measure_node(reading, block)
        outcome = correct(
            node,
            ink,
            fonts,
            float(block.width),
            script_slots=len(script_components(block)),
        )
        size, score = _best_size(outcome.node, fonts, ink, float(block.width))
        prepared.append(
            Prepared(
                index=index,
                block=block,
                reading=reading,
                node=outcome.node,
                size=size,
                score=score,
                changes=list(outcome.changes),
                unresolved=list(outcome.unresolved),
            )
        )

    _harmonise(prepared, analysis.text_height)

    # Place everything first: consensus needs every label's glyphs at once.
    placements: dict[int, tuple[float, float]] = {}
    slots: list = []
    for entry in prepared:
        block, size = entry.block, entry.size
        box = tex.layout(entry.node, fonts.metrics, size)
        if entry.pure_fraction and block.bars:
            bar = block.bars[0]
            baseline = bar.y + bar.height / 2.0 + tex.AXIS_RATIO * size
            x = block.x + (block.width - box.width) / 2.0
        else:
            baseline = block.baseline()
            x = float(block.x)
        placements[entry.index] = (x, baseline)

        found, ambiguous = glyph_slots(
            entry.node, fonts, size, x, baseline, block.components
        )
        for slot in found:
            slot.label = entry.index
        slots.extend(found)
        entry.unresolved.extend(ambiguous)

    corrections: list[tuple[int, str]] = []
    if options.substitute_glyphs:
        corrections.extend(substitute_glyphs(slots, fonts))
    corrections.extend(reconcile(slots))

    by_index = {entry.index: entry for entry in prepared}
    for label_index, description in corrections:
        entry = by_index.get(label_index)
        if entry is not None:
            entry.changes.append(description)

    labels: list[ir.Element] = []
    for entry in prepared:
        block, node, size = entry.block, entry.node, entry.size
        box = tex.layout(node, fonts.metrics, size)
        x, baseline = placements[entry.index]

        rendered = render_box(box, fonts)
        score = shape_iou(_block_ink(page, block), rendered) if rendered is not None else entry.score
        confidence = float(np.clip(0.35 + 0.9 * score, 0.0, 0.99))

        if confidence < options.confidence_threshold and options.raster_fallback:
            success, buffer = cv2.imencode(".png", 255 - _block_ink(page, block))
            if success:
                labels.append(
                    ir.RasterFallback(
                        kind="image",
                        confidence=confidence,
                        provenance=f"{entry.reading.engine}: {entry.reading.text}",
                        identifier=f"label-{entry.index}",
                        x=float(block.x),
                        y=float(block.y),
                        width=float(block.width),
                        height=float(block.height),
                        png=buffer.tobytes(),
                    )
                )
                entry.score = score
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
            )
        )
        entry.score = score
    analysis.prepared = prepared
    return labels


def build_geometry(analysis: Analysis, options: Options) -> tuple[list[ir.Element], list[dict]]:
    page = analysis.page
    elements: list[ir.Element] = []
    notes: list[dict] = []

    for index, trace in enumerate(analysis.traces):
        points = trace.points
        model = choose_model(
            points[:, 0], points[:, 1], tolerance=options.model_tolerance * trace.stroke_width
        )
        loose = choose_model(points[:, 0], points[:, 1], tolerance=4.0 * trace.stroke_width)
        source = points
        if options.idealise and model is not None:
            grid = np.linspace(points[0, 0], points[-1, 0], max(256, points.shape[0]))
            source = np.column_stack([grid, model.sample(grid)])

        segments = fit_bezier(source, options.bezier_tolerance * trace.stroke_width)
        described = model or loose
        elements.append(
            ir.Curve(
                kind="curve",
                confidence=0.95 if model is not None else 0.8,
                provenance=f"trace:{trace.method}",
                identifier=f"curve-{index}",
                path=path_data(segments),
                stroke_width=trace.stroke_width,
                model=described.description if described else "",
                model_rms=described.rms if described else None,
            )
        )
        notes.append(
            {
                "curve": index,
                "points": int(points.shape[0]),
                "segments": len(segments),
                "stroke_width": round(trace.stroke_width, 2),
                "analytic_model": described.name if described else None,
                "analytic_residual_px": round(described.rms, 2) if described else None,
                "within_tolerance": model is not None,
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
    return elements, notes


def convert(path: Path, options: Options | None = None) -> ir.Document:
    options = options or Options()
    analysis = analyse(path, options)
    read_blocks(analysis, options)
    fonts, ranking = choose_fonts(analysis, options)

    geometry, curve_notes = build_geometry(analysis, options)
    labels = build_labels(analysis, fonts, options)

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
        background=page.background,
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
        "skew_degrees": round(page.skew_degrees, 3),
        "text_height": round(analysis.text_height, 1),
        "font": {
            "chosen": fonts.family if fonts else None,
            "bold": bool(fonts and fonts.bold),
            "ranking": [{"face": name, "score": round(score, 3)} for name, score in ranking],
        },
        "curves": curve_notes,
        "ticks": [
            {
                "orientation": tick.rule.orientation,
                "count": len(tick.positions),
                "spacing_px": round(tick.spacing, 2) if tick.spacing else None,
                "confidence": round(tick.confidence, 2),
            }
            for tick in analysis.ticks
        ],
        "labels": [
            {
                "id": element.identifier,
                "text": getattr(element, "plain", ""),
                "engine": getattr(element, "engine", ""),
                "confidence": round(element.confidence, 2),
                "corrections": element.provenance,
                "ambiguous_glyphs": next(
                    (entry.unresolved for entry in analysis.prepared
                     if f"label-{entry.index}" == element.identifier),
                    [],
                ),
            }
            for element in labels
        ],
    }

    if options.verify:
        rendered = rasterise(document.to_svg(), page.width, page.height)
        if rendered is not None:
            document.report["agreement"] = {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in agreement(page.ink, rendered).items()
            }

    low = [
        element.identifier
        for element in labels + geometry
        if element.confidence < options.confidence_threshold
    ]
    document.report["needs_review"] = low
    return document
