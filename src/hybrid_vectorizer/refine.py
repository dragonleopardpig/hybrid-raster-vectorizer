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


def _advance(fonts: FontSet, glyph) -> float:
    return max(1.0, fonts.metrics.advance(glyph.text, glyph.size))


def _align(glyphs: list, components: list, fonts: FontSet) -> list[tuple]:
    """Match a row of characters to a row of ink, in order.

    Absolute positions drift along a line whenever the typeface in hand is not
    the one that was printed, so pairing by predicted coordinate goes wrong the
    further into an expression it gets. Reading order does not drift: the n-th
    mark is the n-th character, and the only question is which marks ran
    together. That is settled by widths.
    """
    if not glyphs or not components or len(components) > len(glyphs):
        return []

    widths = [_advance(fonts, glyph) for glyph in glyphs]
    prefix = [0.0]
    for width in widths:
        prefix.append(prefix[-1] + width)

    def cost(component, first: int, last: int) -> float:
        wanted = prefix[last] - prefix[first]
        actual = float(component.width)
        return abs(actual - wanted) / max(actual, wanted, 1.0)

    n, m = len(glyphs), len(components)
    best = np.full((n + 1, m + 1), np.inf)
    back = np.zeros((n + 1, m + 1), dtype=int)
    best[0, 0] = 0.0
    for j in range(1, m + 1):
        for i in range(j, n - (m - j) + 1):
            for k in range(j - 1, i):
                value = best[k, j - 1] + cost(components[j - 1], k, i)
                if value < best[i, j]:
                    best[i, j] = value
                    back[i, j] = k
    if not np.isfinite(best[n, m]):
        return []

    pairs: list[tuple] = []
    i = n
    for j in range(m, 0, -1):
        k = int(back[i, j])
        pairs.append((components[j - 1], glyphs[k:i]))
        i = k
    return list(reversed(pairs))


def correspond(
    node: tex.Row,
    fonts: FontSet,
    size: float,
    x: float,
    baseline: float,
    components: list,
    bar_y: float | None = None,
) -> list[tuple]:
    """Pair every laid-out character with the ink it was drawn over."""
    box = tex.layout(node, fonts.metrics, size, merge=False)
    singles = [
        glyph
        for glyph in box.glyphs
        if len(glyph.text) == 1 and isinstance(glyph.origin, tex.Run) and not glyph.text.isspace()
    ]
    if bar_y is None:
        groups = [(singles, list(components))]
    else:
        groups = [
            (
                [g for g in singles if baseline + g.y <= bar_y],
                [c for c in components if c.bottom <= bar_y + 2],
            ),
            (
                [g for g in singles if baseline + g.y > bar_y],
                [c for c in components if c.bottom > bar_y + 2],
            ),
        ]

    pairs: list[tuple] = []
    for glyphs, marks in groups:
        pairs.extend(
            _align(
                sorted(glyphs, key=lambda item: item.x),
                sorted(marks, key=lambda item: item.x),
                fonts,
            )
        )
    return pairs


def demote_spurious_scripts(
    node: tex.Row,
    fonts: FontSet,
    size: float,
    x: float,
    baseline: float,
    components: list,
    body_height: float,
    bar_y: float | None = None,
) -> list[str]:
    """Undo a subscript the ink draws at full size, on that glyph's own evidence.

    Counting how many glyphs sit off the baseline and comparing with the reading
    is too blunt: where a subscript touches its base they share one mark and the
    count silently loses it, which then demotes a genuine subscript. Asking per
    script, and declining to answer when base and script share ink, keeps the
    correction to the cases the drawing actually settles.
    """
    changes: list[str] = []
    for _ in range(4):
        pairs = correspond(node, fonts, size, x, baseline, components, bar_y)
        owner = {id(glyph): component for component, glyphs in pairs for glyph in glyphs}
        shared = {
            id(glyph)
            for component, glyphs in pairs
            if len(glyphs) > 1
            for glyph in glyphs
        }
        box = tex.layout(node, fonts.metrics, size, merge=False)
        by_run = {id(g.origin): g for g in box.glyphs if g.origin is not None}

        demoted = None
        for entry in _scripts(node):
            if entry.subscript is None or entry.superscript is not None:
                continue
            script_runs, base_runs = _runs(entry.subscript), _runs(entry.base)
            if not script_runs or not base_runs:
                continue
            script_glyph = by_run.get(id(script_runs[0]))
            if script_glyph is None or id(script_glyph) in shared:
                continue  # base and script share ink: the drawing cannot say
            component = owner.get(id(script_glyph))
            if component is None or component.height < 0.78 * body_height:
                continue  # genuinely smaller, so genuinely a script
            if _flatten_script(node, entry):
                demoted = script_glyph.text
                break
        if demoted is None:
            break
        changes.append(f"{demoted!r} is drawn at full size, so it is not a subscript")
    return changes


def glyph_slots(
    node: tex.Row,
    fonts: FontSet,
    size: float,
    x: float,
    baseline: float,
    components: list,
    bar_y: float | None = None,
) -> tuple[list, list[str]]:
    """Pair each character with its own ink, cutting marks that ran together.

    Marks that cannot be a glyph at this size, such as a fraction bar, must not
    be offered here: cutting one into pieces yields solid blocks that resemble
    each other perfectly and would poison any comparison built on them.
    """
    from .components import split_into
    from .consensus import Slot

    slots: list[Slot] = []
    ambiguous: list[str] = []

    def note(glyph, reason: str) -> None:
        confusion = next((group for group in CONFUSIONS if glyph.text in group), None)
        if confusion is not None:
            ambiguous.append(f"{glyph.text} ({reason})")

    for component, glyphs in correspond(node, fonts, size, x, baseline, components, bar_y):
        if len(glyphs) == 1:
            pieces, clean = [component], True
        else:
            pieces, clean = split_into(component, len(glyphs))
        if not clean or len(pieces) != len(glyphs):
            for glyph in glyphs:
                note(glyph, "unchecked: cutting it apart would pass through a stroke")
            continue
        for glyph, piece in zip(glyphs, pieces):
            slots.append(Slot(label=-1, run=glyph.origin, character=glyph.text, component=piece))
            confusion = next((group for group in CONFUSIONS if glyph.text in group), None)
            if confusion is not None:
                ambiguous.append(
                    f"{glyph.text} (could be {'/'.join(sorted(confusion - {glyph.text}))})"
                )
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
