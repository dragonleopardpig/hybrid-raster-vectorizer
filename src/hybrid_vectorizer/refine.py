"""Render what we think the figure says, compare it with the ink, and correct it."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import functools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import latex as tex
from .fonts import FontFace, Metrics, list_faces, measure

# Readings that differ by one glyph the OCR cannot separate on shape alone.
CONFUSIONS: list[set[str]] = [
    {"s", "δ", "∂"},
    {"a", "α"}, {"v", "ν"}, {"w", "ω"}, {"u", "μ"},
    {"y", "γ"}, {"x", "χ", "×"}, {"p", "ρ"},
    {"e", "ε"}, {"t", "τ"}, {"n", "η"}, {"k", "κ"},
    {"o", "θ", "0"}, {"l", "1"}, {"S", "5"}, {"g", "9"}, {"z", "2"},
    {"b", "6"}, {"B", "8"}, {"r", "γ"},
]


@dataclass
class FontSet:
    family: str
    upright: FontFace
    italic: FontFace
    metrics: Metrics
    bold: bool = False


def _pick(faces: list[FontFace], family: str, *, italic: bool, bold: bool) -> FontFace | None:
    members = [face for face in faces if face.family == family]
    if not members:
        return None
    exact = [face for face in members if face.italic == italic and face.bold == bold]
    partial = [face for face in members if face.italic == italic]
    return (exact or partial or members)[0]


def build_font_set(family: str, *, bold: bool, faces: list[FontFace] | None = None) -> FontSet | None:
    faces = faces or list_faces()
    upright = _pick(faces, family, italic=False, bold=bold)
    italic = _pick(faces, family, italic=True, bold=bold)
    if upright is None and italic is None:
        return None
    upright = upright or italic
    italic = italic or upright
    metrics = measure(str(italic.path)) or measure(str(upright.path))
    if metrics is None:
        return None
    return FontSet(family=family, upright=upright, italic=italic, metrics=metrics, bold=bold)


def shape_iou(first: np.ndarray, second: np.ndarray, height: int = 128) -> float:
    """Agreement of two ink patches after matching their height, keeping aspect."""

    def prepare(image: np.ndarray) -> np.ndarray | None:
        ys, xs = np.nonzero(image > 127)
        if xs.size == 0:
            return None
        cropped = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        scale = height / cropped.shape[0]
        width = max(1, int(round(cropped.shape[1] * scale)))
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA) > 127

    a, b = prepare(first), prepare(second)
    if a is None or b is None:
        return 0.0
    width = max(a.shape[1], b.shape[1])
    canvas_a = np.zeros((height, width), dtype=bool)
    canvas_b = np.zeros((height, width), dtype=bool)
    canvas_a[:, : a.shape[1]] = a
    canvas_b[:, : b.shape[1]] = b
    union = np.count_nonzero(canvas_a | canvas_b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(canvas_a & canvas_b) / union)


def render_box(box: tex.Box, fonts: FontSet, *, margin: int = 8) -> np.ndarray | None:
    """Draw a laid-out expression with Pillow, at the size the layout specifies."""
    if not box.glyphs and not box.bars:
        return None
    width = int(np.ceil(box.width)) + 2 * margin
    height = int(np.ceil(box.ascent + box.descent)) + 2 * margin
    if width <= 0 or height <= 0:
        return None

    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    baseline = margin + box.ascent

    for bar in box.bars:
        top = baseline + bar.y - bar.thickness / 2.0
        draw.rectangle(
            [margin + bar.x, top, margin + bar.x + bar.width, top + max(1.0, bar.thickness)],
            fill=255,
        )
    for glyph in box.glyphs:
        face = fonts.upright if glyph.upright else fonts.italic
        try:
            font = ImageFont.truetype(str(face.path), size=max(4, int(round(glyph.size))))
        except Exception:
            continue
        draw.text(
            (margin + glyph.x, baseline + glyph.y), glyph.text, fill=255, font=font, anchor="ls"
        )
    return np.asarray(image)


def fit_size(node: object, metrics: Metrics, target_width: float, probe: float = 100.0) -> float:
    """Layout scales linearly with size, so the size that fits solves in one step."""
    box = tex.layout(node, metrics, probe)
    if box.width <= 0:
        return probe
    return float(probe * target_width / box.width)


def _runs(node: object) -> list[tex.Run]:
    if isinstance(node, tex.Run):
        return [node]
    if isinstance(node, tex.Row):
        return [run for item in node.items for run in _runs(item)]
    if isinstance(node, tex.Frac):
        return _runs(node.numerator) + _runs(node.denominator)
    if isinstance(node, tex.Scripts):
        result = _runs(node.base)
        for script in (node.superscript, node.subscript):
            if script is not None:
                result.extend(_runs(script))
        return result
    return []


def _script_count(node: object) -> int:
    return sum(
        (entry.superscript is not None) + (entry.subscript is not None)
        for entry in _scripts(node)
    )


def _scripts(node: object) -> list[tex.Scripts]:
    if isinstance(node, tex.Scripts):
        found = [node] + _scripts(node.base)
        for script in (node.superscript, node.subscript):
            if script is not None:
                found.extend(_scripts(script))
        return found
    if isinstance(node, tex.Row):
        return [item for entry in node.items for item in _scripts(entry)]
    if isinstance(node, tex.Frac):
        return _scripts(node.numerator) + _scripts(node.denominator)
    return []


def _flatten_script(root: object, target: tex.Scripts) -> bool:
    """Demote a wrongly-subscripted glyph back onto the baseline, in place."""
    if isinstance(root, tex.Row):
        for index, item in enumerate(root.items):
            if item is target and target.subscript is not None and target.superscript is None:
                root.items[index] = tex.Row([target.base, target.subscript])
                return True
            if _flatten_script(item, target):
                return True
    elif isinstance(root, tex.Frac):
        return _flatten_script(root.numerator, target) or _flatten_script(root.denominator, target)
    elif isinstance(root, tex.Scripts):
        for child in (root.base, root.superscript, root.subscript):
            if child is not None and _flatten_script(child, target):
                return True
    return False


@functools.lru_cache(maxsize=4096)
def _glyph_shape(path: str, character: str, size: int = 72) -> bytes | None:
    try:
        font = ImageFont.truetype(path, size=size)
    except Exception:
        return None
    image = Image.new("L", (size * 3, size * 3), color=0)
    ImageDraw.Draw(image).text((size, size * 2), character, fill=255, font=font, anchor="ls")
    array = np.asarray(image)
    return array.tobytes() + b"|" + repr(array.shape).encode() if array.any() else None


def _shape_of(path: str, character: str, size: int = 72) -> np.ndarray | None:
    packed = _glyph_shape(path, character, size)
    if packed is None:
        return None
    body, _, shape = packed.rpartition(b"|")
    rows, columns = eval(shape.decode())  # shape tuple written above
    return np.frombuffer(body, dtype=np.uint8).reshape(rows, columns)


@functools.lru_cache(maxsize=4096)
def distinguishable(path: str, first: str, second: str, threshold: float = 0.78) -> bool:
    """Can this typeface tell these two glyphs apart at all?

    Substituting 0 for o, or 2 for z, asks the pixels a question they cannot
    answer: the shapes coincide. Offering such a swap only invites noise to
    decide it, so the reading is left as the recogniser gave it.
    """
    a = _shape_of(path, first)
    b = _shape_of(path, second)
    if a is None or b is None:
        return False
    return shape_iou(a, b) < threshold


@dataclass
class Correction:
    node: object
    score: float
    changes: list[str]
    unresolved: list[str] = field(default_factory=list)


def correct(
    node: tex.Row,
    ink: np.ndarray,
    fonts: FontSet,
    target_width: float,
    *,
    script_slots: int | None = None,
    glyph_margin: float = 0.035,
    script_margin: float = 0.02,
) -> Correction:
    """Test the readings the pixels can actually decide between, and keep the best.

    Every candidate is re-rendered in the matched typeface and measured against
    the ink, so a correction has to earn its place by explaining the drawing
    better than the recogniser's original reading did.
    """
    import copy

    def evaluate(candidate: object) -> float:
        size = fit_size(candidate, fonts.metrics, target_width)
        rendered = render_box(tex.layout(candidate, fonts.metrics, size), fonts)
        if rendered is None:
            return 0.0
        return shape_iou(ink, rendered)

    best = copy.deepcopy(node)
    best_score = evaluate(best)
    changes: list[str] = []
    unresolved: list[str] = []

    # The ink shows how many glyphs really sit off the baseline. If the reading
    # claims more scripts than that, the surplus ones are the recogniser's.
    if script_slots is not None:
        surplus = _script_count(best) - script_slots
        while surplus > 0:
            candidates = []
            for index in range(len(_scripts(best))):
                trial = copy.deepcopy(best)
                targets = _scripts(trial)
                if index >= len(targets) or not _flatten_script(trial, targets[index]):
                    continue
                candidates.append((evaluate(trial), trial))
            if not candidates:
                break
            score, trial = max(candidates, key=lambda item: item[0])
            if score <= best_score - script_margin:
                break
            best, best_score = trial, score
            changes.append("script demoted to baseline")
            surplus -= 1

    return Correction(node=best, score=best_score, changes=changes, unresolved=unresolved)


def glyph_slots(
    node: tex.Row,
    fonts: FontSet,
    size: float,
    x: float,
    baseline: float,
    components: list,
) -> tuple[list, list[str]]:
    """Pair each laid-out character with the one ink component it covers."""
    from .consensus import Slot

    box = tex.layout(node, fonts.metrics, size, merge=False)
    slots: list[Slot] = []
    ambiguous: list[str] = []

    for glyph in box.glyphs:
        if len(glyph.text) != 1 or not isinstance(glyph.origin, tex.Run):
            continue
        left = x + glyph.x
        right = left + fonts.metrics.advance(glyph.text, glyph.size)
        bottom = baseline + glyph.y
        pad = 0.35 * glyph.size
        matches = [
            component
            for component in components
            if left - pad <= (component.x + component.right) / 2.0 <= right + pad
            and abs(component.bottom - bottom) <= 0.45 * glyph.size
            # A mark far smaller or wider than the glyph is a different symbol
            # that merely sits nearby, such as a minus beside a lambda.
            and 0.3 * glyph.size <= component.height <= 1.5 * glyph.size
            and component.width <= 1.8 * max(1.0, right - left)
        ]
        confusion = next((group for group in CONFUSIONS if glyph.text in group), None)
        if len(matches) != 1:
            # Touching glyphs share one component, so this reading cannot be
            # checked against the ink at all. Say so rather than imply it was.
            if confusion is not None:
                ambiguous.append(f"{glyph.text} (unchecked: not separable from its neighbours)")
            continue

        slots.append(Slot(label=-1, run=glyph.origin, character=glyph.text, component=matches[0]))
        if confusion is not None:
            ambiguous.append(f"{glyph.text} (could be {'/'.join(sorted(confusion - {glyph.text}))})")
    return slots, ambiguous


def substitute_glyphs(
    slots: list, fonts: FontSet, *, margin: float = 0.06
) -> list[tuple[int, str]]:
    """Re-read ambiguous glyphs against rendered candidates.

    Measured on a scanned textbook figure this picked the right letter in only
    about a sixth of cases, because the scanned typeface is not installed and
    the winning margins are far below the spread between typefaces. It stays
    available behind a flag, and off by default, so that a confident wrong
    answer never replaces an honest uncertain one.
    """
    changes: list[tuple[int, str]] = []
    for slot in slots:
        confusion = next((group for group in CONFUSIONS if slot.character in group), None)
        if confusion is None:
            continue
        scores: dict[str, float] = {}
        for option in sorted(confusion):
            shape = _shape_of(str(fonts.italic.path), option)
            if shape is not None:
                scores[option] = shape_iou(slot.component.mask, shape)
        if slot.character not in scores or len(scores) < 2:
            continue
        winner = max(scores, key=lambda key: scores[key])
        if winner != slot.character and scores[winner] > scores[slot.character] + margin:
            changes.append(
                (
                    slot.label,
                    f"{slot.character!r} rendered closer to {winner!r} "
                    f"({scores[slot.character]:.2f} to {scores[winner]:.2f})",
                )
            )
            slot.run.text = winner
            slot.character = winner
    return changes


def rasterise(svg_text: str, width: int, height: int) -> np.ndarray | None:
    """Render the finished SVG so the whole page can be scored against the scan."""
    renderer = shutil.which("resvg")
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "figure.svg"
        target = Path(directory) / "figure.png"
        source.write_text(svg_text, encoding="utf-8")

        if renderer is not None:
            command = [renderer, "--width", str(width), "--height", str(height),
                       str(source), str(target)]
        else:
            inkscape = shutil.which("inkscape")
            if inkscape is None:
                return None
            command = [inkscape, str(source), f"--export-width={width}",
                       f"--export-height={height}", f"--export-filename={target}"]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not target.exists():
            return None
        image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    _threshold, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def agreement(source_ink: np.ndarray, rendered_ink: np.ndarray, tolerance: int = 3) -> dict:
    """How much of each drawing the other one covers, allowing a small offset."""
    if rendered_ink.shape != source_ink.shape:
        rendered_ink = cv2.resize(
            rendered_ink, (source_ink.shape[1], source_ink.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    source_near = cv2.dilate(source_ink, kernel)
    rendered_near = cv2.dilate(rendered_ink, kernel)

    source_total = max(1, int(np.count_nonzero(source_ink)))
    rendered_total = max(1, int(np.count_nonzero(rendered_ink)))
    covered = int(np.count_nonzero(cv2.bitwise_and(source_ink, rendered_near)))
    explained = int(np.count_nonzero(cv2.bitwise_and(rendered_ink, source_near)))

    union = int(np.count_nonzero(cv2.bitwise_or(source_ink, rendered_ink)))
    intersection = int(np.count_nonzero(cv2.bitwise_and(source_ink, rendered_ink)))
    return {
        "recall": covered / source_total,
        "precision": explained / rendered_total,
        "iou": intersection / max(1, union),
        "tolerance_px": tolerance,
    }
