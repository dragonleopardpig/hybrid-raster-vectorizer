"""A small LaTeX subset: parse it, lay it out with real font metrics, emit SVG."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from .fonts import Metrics

SYMBOLS: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓",
    "div": "÷", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "approx": "≈", "equiv": "≡", "sim": "∼",
    "propto": "∝", "infty": "∞", "partial": "∂", "nabla": "∇",
    "int": "∫", "sum": "∑", "prod": "∏", "sqrt": "√",
    "rightarrow": "→", "to": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "ldots": "…", "cdots": "⋯", "dots": "…",
    "prime": "′", "circ": "∘", "degree": "°", "angle": "∠",
    "perp": "⊥", "parallel": "∥", "in": "∈", "hbar": "ℏ",
}

UPRIGHT_WORDS = {
    "cos", "sin", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max",
    "min", "sup", "inf", "det", "arg", "dim", "deg", "gcd", "Re", "Im",
}

SPACING = {",": 0.16, ";": 0.27, ":": 0.21, "!": -0.16, " ": 0.25, "quad": 1.0, "qquad": 2.0}
IGNORED = {"displaystyle", "textstyle", "limits", "nolimits", "left", "right", "!", "bf", "rm", "it"}

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[{}^_]|\s+|[^\\{}^_\s]")


@dataclass
class Run:
    text: str
    upright: bool = False


@dataclass
class Row:
    items: list = field(default_factory=list)


@dataclass
class Frac:
    numerator: object
    denominator: object


@dataclass
class Scripts:
    base: object
    superscript: object | None = None
    subscript: object | None = None


@dataclass
class Space:
    width: float


def _tokenise(source: str) -> list[str]:
    source = source.strip()
    if source.startswith("$$") and source.endswith("$$"):
        source = source[2:-2]
    elif source.startswith("$") and source.endswith("$"):
        source = source[1:-1]
    # A bare space between symbols is how the recogniser separates them, not
    # spacing the author asked for; explicit commands like \\, still carry it.
    return [token for token in _TOKEN.findall(source) if not token.isspace()]


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def next(self) -> str | None:
        token = self.peek()
        if token is not None:
            self.index += 1
        return token

    def parse_row(self, stop: str | None = None) -> Row:
        items: list = []
        while True:
            token = self.peek()
            if token is None or token == stop:
                if token == stop and stop is not None:
                    self.next()
                break
            atom = self.parse_atom()
            if atom is None:
                continue
            atom = self.attach_scripts(atom)
            items.append(atom)
        return Row(items)

    def attach_scripts(self, base: object) -> object:
        superscript = subscript = None
        while self.peek() in {"^", "_"}:
            marker = self.next()
            argument = self.parse_atom()
            if argument is None:
                break
            if marker == "^":
                superscript = argument
            else:
                subscript = argument
        if superscript is None and subscript is None:
            return base
        return Scripts(base=base, superscript=superscript, subscript=subscript)

    def parse_atom(self) -> object | None:
        token = self.next()
        if token is None:
            return None
        if token == "{":
            return self.parse_row(stop="}")
        if token == "}":
            return None
        if token.startswith("\\"):
            return self.parse_command(token[1:])
        if token == " ":
            return Space(0.22)
        return Run(token, upright=not token.isalpha())

    def parse_command(self, name: str) -> object | None:
        if name in {"frac", "dfrac", "tfrac"}:
            numerator = self.parse_atom() or Row([])
            denominator = self.parse_atom() or Row([])
            return Frac(numerator, denominator)
        if name in {"mathrm", "mathbf", "operatorname", "text", "textrm", "mathsf"}:
            inner = self.parse_atom() or Row([])
            _set_upright(inner)
            return inner
        if name in {"mathit", "mathnormal"}:
            return self.parse_atom() or Row([])
        if name in SPACING:
            return Space(SPACING[name])
        if name in IGNORED:
            return None
        if name in UPRIGHT_WORDS:
            return Run(name, upright=True)
        if name in SYMBOLS:
            return Run(SYMBOLS[name], upright=False)
        return Run(name, upright=True)


def _set_upright(node: object) -> None:
    if isinstance(node, Run):
        node.upright = True
    elif isinstance(node, Row):
        for item in node.items:
            _set_upright(item)


def parse(source: str) -> Row:
    node = _Parser(_tokenise(source)).parse_row()
    return node


# --- layout -------------------------------------------------------------------


@dataclass
class Glyph:
    text: str
    x: float
    y: float
    size: float
    upright: bool
    origin: object | None = None


@dataclass
class Bar:
    x: float
    y: float
    width: float
    thickness: float


@dataclass
class Box:
    width: float
    ascent: float
    descent: float
    glyphs: list[Glyph] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)

    def shifted(self, dx: float, dy: float) -> Box:
        return Box(
            width=self.width,
            ascent=self.ascent - dy,
            descent=self.descent + dy,
            glyphs=[Glyph(g.text, g.x + dx, g.y + dy, g.size, g.upright, g.origin) for g in self.glyphs],
            bars=[Bar(b.x + dx, b.y + dy, b.width, b.thickness) for b in self.bars],
        )


SCRIPT_RATIO = 0.70
AXIS_RATIO = 0.26
GAP_RATIO = 0.13


def layout(node: object, metrics: Metrics, size: float, *, merge: bool = True) -> Box:
    if isinstance(node, Run):
        width = metrics.width(node.text, size)
        return Box(
            width=width,
            ascent=metrics.cap_height * size,
            descent=metrics.descent * size * 0.6,
            glyphs=[Glyph(node.text, 0.0, 0.0, size, node.upright, node)],
        )

    if isinstance(node, Space):
        return Box(width=node.width * size, ascent=0.0, descent=0.0)

    if isinstance(node, Row):
        merged = _merge_runs(node.items) if merge else list(node.items)
        box = Box(width=0.0, ascent=0.0, descent=0.0)
        cursor = 0.0
        for item in merged:
            child = layout(item, metrics, size, merge=merge).shifted(cursor, 0.0)
            cursor += child.width
            box.glyphs.extend(child.glyphs)
            box.bars.extend(child.bars)
            box.ascent = max(box.ascent, child.ascent)
            box.descent = max(box.descent, child.descent)
        box.width = cursor
        return box

    if isinstance(node, Scripts):
        base = layout(node.base, metrics, size, merge=merge)
        box = Box(width=base.width, ascent=base.ascent, descent=base.descent,
                  glyphs=list(base.glyphs), bars=list(base.bars))
        script_size = max(6.0, size * SCRIPT_RATIO)
        cursor = base.width
        advance = 0.0
        if node.superscript is not None:
            raised = layout(node.superscript, metrics, script_size, merge=merge).shifted(cursor, -0.45 * size)
            box.glyphs.extend(raised.glyphs)
            box.bars.extend(raised.bars)
            box.ascent = max(box.ascent, raised.ascent)
            advance = max(advance, raised.width)
        if node.subscript is not None:
            lowered = layout(node.subscript, metrics, script_size, merge=merge).shifted(cursor, 0.22 * size)
            box.glyphs.extend(lowered.glyphs)
            box.bars.extend(lowered.bars)
            box.descent = max(box.descent, lowered.descent)
            advance = max(advance, lowered.width)
        box.width = cursor + advance
        return box

    if isinstance(node, Frac):
        axis = AXIS_RATIO * size
        gap = GAP_RATIO * size
        thickness = max(1.0, 0.055 * size)
        numerator = layout(node.numerator, metrics, size, merge=merge)
        denominator = layout(node.denominator, metrics, size, merge=merge)
        padding = 0.18 * size
        width = max(numerator.width, denominator.width) + padding

        top = numerator.shifted((width - numerator.width) / 2.0, -axis - gap - numerator.descent)
        bottom = denominator.shifted(
            (width - denominator.width) / 2.0, -axis + gap + denominator.ascent
        )
        box = Box(
            width=width,
            ascent=max(top.ascent, axis + thickness),
            descent=max(bottom.descent, 0.0),
            glyphs=top.glyphs + bottom.glyphs,
            bars=top.bars + bottom.bars + [Bar(0.0, -axis, width, thickness)],
        )
        return box

    return Box(width=0.0, ascent=0.0, descent=0.0)


def _merge_runs(items: list) -> list:
    """Adjacent characters of the same style become one editable text run."""
    merged: list = []
    for item in items:
        if (
            isinstance(item, Run)
            and merged
            and isinstance(merged[-1], Run)
            and merged[-1].upright == item.upright
        ):
            merged[-1] = Run(merged[-1].text + item.text, item.upright)
        else:
            merged.append(item)
    return merged


def to_svg(box: Box, x: float, baseline: float, *, indent: str = "    ") -> list[str]:
    def number(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".") or "0"

    lines: list[str] = []
    for bar in box.bars:
        lines.append(
            f'{indent}<path class="fraction-line" '
            f'd="M{number(x + bar.x)} {number(baseline + bar.y)}h{number(bar.width)}" '
            f'stroke-width="{number(bar.thickness)}"/>'
        )
    for glyph in box.glyphs:
        style = "upright" if glyph.upright else "italic"
        lines.append(
            f'{indent}<text class="glyph {style}" x="{number(x + glyph.x)}" '
            f'y="{number(baseline + glyph.y)}" font-size="{number(glyph.size)}">'
            f"{html.escape(glyph.text)}</text>"
        )
    return lines


def to_text(node: object) -> str:
    """Plain reading of the expression, for the accessible label."""
    if isinstance(node, Run):
        return node.text
    if isinstance(node, Space):
        return " "
    if isinstance(node, Row):
        return "".join(to_text(item) for item in node.items)
    if isinstance(node, Frac):
        return f"({to_text(node.numerator)})/({to_text(node.denominator)})"
    if isinstance(node, Scripts):
        text = to_text(node.base)
        if node.subscript is not None:
            text += f"_{to_text(node.subscript)}"
        if node.superscript is not None:
            text += f"^{to_text(node.superscript)}"
        return text
    return ""
