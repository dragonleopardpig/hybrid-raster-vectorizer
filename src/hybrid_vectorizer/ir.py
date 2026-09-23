"""The reconstructed figure: a list of semantic elements that can write itself as SVG."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from . import latex as tex


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _attribute(value: object) -> str:
    return html.escape(str(value), quote=True)


@dataclass
class Element:
    kind: str
    confidence: float = 1.0
    provenance: str = ""
    identifier: str | None = None

    def defs(self) -> list[str]:
        """Anything this element needs declared once, such as a pattern."""
        return []

    def attributes(self) -> str:
        parts = []
        if self.identifier:
            parts.append(f'id="{_attribute(self.identifier)}"')
        parts.append(f'data-confidence="{self.confidence:.2f}"')
        if self.provenance:
            parts.append(f'data-from="{_attribute(self.provenance)}"')
        return " ".join(parts)


@dataclass
class Axis(Element):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    stroke_width: float = 1.0
    arrow_start: bool = False
    arrow_end: bool = False

    def to_svg(self, indent: str) -> list[str]:
        markers = []
        if self.arrow_end:
            markers.append('marker-end="url(#arrow)"')
        if self.arrow_start:
            markers.append('marker-start="url(#arrow-start)"')
        return [
            f'{indent}<line class="axis" {self.attributes()} '
            f'x1="{_number(self.x1)}" y1="{_number(self.y1)}" '
            f'x2="{_number(self.x2)}" y2="{_number(self.y2)}" '
            f'stroke-width="{_number(self.stroke_width)}" {" ".join(markers)}/>'
        ]


@dataclass
class Ticks(Element):
    orientation: str = "horizontal"
    positions: list[float] = field(default_factory=list)
    near: float = 0.0
    far: float = 0.0
    stroke_width: float = 1.0
    spacing: float | None = None

    def to_svg(self, indent: str) -> list[str]:
        spacing = f' data-spacing="{_number(self.spacing)}"' if self.spacing else ""
        lines = [f'{indent}<g class="ticks" {self.attributes()}{spacing}>']
        for position in self.positions:
            if self.orientation == "horizontal":
                geometry = (
                    f'x1="{_number(position)}" y1="{_number(self.near)}" '
                    f'x2="{_number(position)}" y2="{_number(self.far)}"'
                )
            else:
                geometry = (
                    f'x1="{_number(self.near)}" y1="{_number(position)}" '
                    f'x2="{_number(self.far)}" y2="{_number(position)}"'
                )
            lines.append(
                f'{indent}  <line class="tick" {geometry} stroke-width="{_number(self.stroke_width)}"/>'
            )
        lines.append(f"{indent}</g>")
        return lines


@dataclass
class Curve(Element):
    path: str = ""
    stroke_width: float = 1.0
    model: str = ""
    model_rms: float | None = None

    def to_svg(self, indent: str) -> list[str]:
        model = ""
        if self.model:
            model = f' data-model="{_attribute(self.model)}"'
            if self.model_rms is not None:
                model += f' data-model-residual="{self.model_rms:.2f}"'
        return [
            f'{indent}<path class="curve" {self.attributes()}{model} '
            f'stroke-width="{_number(self.stroke_width)}" d="{self.path}"/>'
        ]


@dataclass
class Area(Element):
    """A filled or ruled region, drawn as a shape rather than as its outline."""

    path: str = ""
    shape: str = "freeform"
    parameters: dict = field(default_factory=dict)
    hatch_angle: float | None = None
    hatch_spacing: float | None = None
    hatch_width: float = 1.0
    bordered: bool = False
    border_width: float = 1.0

    @property
    def pattern(self) -> str | None:
        return f"hatch-{self.identifier}" if self.hatch_spacing else None

    def defs(self) -> list[str]:
        if not self.pattern:
            return []
        spacing = _number(self.hatch_spacing or 1.0)
        return [
            f'    <pattern id="{_attribute(self.pattern)}" patternUnits="userSpaceOnUse"',
            f'             width="{spacing}" height="{spacing}"',
            f'             patternTransform="rotate({_number(self.hatch_angle or 0.0)})">',
            f'      <line x1="0" y1="0" x2="{spacing}" y2="0" stroke="currentColor"',
            f'            stroke-width="{_number(self.hatch_width)}"/>',
            "    </pattern>",
        ]

    def to_svg(self, indent: str) -> list[str]:
        fill = f'url(#{self.pattern})' if self.pattern else "currentColor"
        # Written as attributes, not as a class rule: a stylesheet rule would
        # beat the attribute and silently erase a frame that was really drawn.
        edge = (
            f' stroke="currentColor" stroke-width="{_number(self.border_width)}"'
            if self.bordered
            else ' stroke="none"'
        )
        common = f'class="area {self.shape}" {self.attributes()} fill="{fill}"{edge}'
        if self.shape == "rectangle" and self.parameters:
            p = self.parameters
            return [
                f'{indent}<rect {common} x="{_number(p["x"])}" y="{_number(p["y"])}" '
                f'width="{_number(p["width"])}" height="{_number(p["height"])}"/>'
            ]
        if self.shape == "circle" and self.parameters:
            p = self.parameters
            return [
                f'{indent}<circle {common} cx="{_number(p["cx"])}" cy="{_number(p["cy"])}" '
                f'r="{_number(p["r"])}"/>'
            ]
        return [f'{indent}<path {common} fill-rule="evenodd" d="{self.path}"/>']


@dataclass
class MarkerField(Element):
    """One data series, drawn as repeats of a single shape."""

    shape: str = "circle"
    size: float = 4.0
    filled: bool = True
    stroke_width: float = 1.0
    positions: list[tuple[float, float]] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    symbol_id: str | None = None

    @property
    def symbol(self) -> str:
        return self.symbol_id or f"marker-{self.identifier}"

    def _primitive(self) -> str:
        paint = (
            'fill="currentColor"'
            if self.filled
            else f'fill="none" stroke="currentColor" stroke-width="{_number(self.stroke_width)}"'
        )
        half = self.size / 2.0
        if self.shape == "rectangle":
            return (
                f'<rect x="{_number(-half)}" y="{_number(-half)}" '
                f'width="{_number(self.size)}" height="{_number(self.size)}" {paint}/>'
            )
        if self.shape == "triangle":
            return (
                f'<path d="M0 {_number(-half)} L{_number(half)} {_number(half)} '
                f'L{_number(-half)} {_number(half)} Z" {paint}/>'
            )
        radius = float(self.parameters.get("r", half))
        return f'<circle cx="0" cy="0" r="{_number(radius)}" {paint}/>'

    def defs(self) -> list[str]:
        if self.symbol_id:
            return []  # drawn with a shape another series already declared
        return [
            f'    <g id="{_attribute(self.symbol)}">',
            f"      {self._primitive()}",
            "    </g>",
        ]

    def to_svg(self, indent: str) -> list[str]:
        lines = [
            f'{indent}<g class="markers" {self.attributes()} data-count="{len(self.positions)}">'
        ]
        for x, y in self.positions:
            lines.append(
                f'{indent}  <use href="#{_attribute(self.symbol)}" '
                f'xlink:href="#{_attribute(self.symbol)}" x="{_number(x)}" y="{_number(y)}"/>'
            )
        lines.append(f"{indent}</g>")
        return lines


@dataclass
class Frame(Element):
    """A drawn box, such as the one around a legend."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    stroke_width: float = 1.0

    def to_svg(self, indent: str) -> list[str]:
        return [
            f'{indent}<rect class="frame" {self.attributes()} '
            f'x="{_number(self.x)}" y="{_number(self.y)}" '
            f'width="{_number(self.width)}" height="{_number(self.height)}" '
            f'stroke-width="{_number(self.stroke_width)}"/>'
        ]


@dataclass
class Group(Element):
    """Several elements that belong together, such as a legend and its rows."""

    label: str = "group"
    children: list[Element] = field(default_factory=list)
    note: str = ""

    def defs(self) -> list[str]:
        return [line for child in self.children for line in child.defs()]

    def to_svg(self, indent: str) -> list[str]:
        note = f' data-note="{_attribute(self.note)}"' if self.note else ""
        lines = [f'{indent}<g class="{_attribute(self.label)}" {self.attributes()}{note}>']
        for child in self.children:
            lines.extend(child.to_svg(indent + "  "))
        lines.append(f"{indent}</g>")
        return lines


@dataclass
class Label(Element):
    box: tex.Box | None = None
    x: float = 0.0
    baseline: float = 0.0
    source: str = ""
    plain: str = ""
    engine: str = ""

    def to_svg(self, indent: str) -> list[str]:
        if self.box is None:
            return []
        aria = f' aria-label="{_attribute(self.plain)}"' if self.plain else ""
        source = f' data-latex="{_attribute(self.source)}"' if self.source else ""
        engine = f' data-engine="{_attribute(self.engine)}"' if self.engine else ""
        lines = [f'{indent}<g class="label" {self.attributes()}{source}{engine}{aria}>']
        lines.extend(tex.to_svg(self.box, self.x, self.baseline, indent=indent + "  "))
        lines.append(f"{indent}</g>")
        return lines


@dataclass
class RasterFallback(Element):
    """Original pixels for a label the pipeline will not vouch for.

    Written with an alpha channel so it sits on a transparent page like every
    other mark, and inverted under a dark theme so the ink stays visible.
    """

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    png: bytes = b""

    def to_svg(self, indent: str) -> list[str]:
        payload = base64.b64encode(self.png).decode("ascii")
        return [
            f'{indent}<image class="unverified" {self.attributes()} '
            f'x="{_number(self.x)}" y="{_number(self.y)}" '
            f'width="{_number(self.width)}" height="{_number(self.height)}" '
            f'xlink:href="data:image/png;base64,{payload}" '
            f'href="data:image/png;base64,{payload}"/>'
        ]


@dataclass
class Document:
    width: float
    height: float
    background: str | None = None
    light_ink: str = "#111111"
    dark_ink: str = "#eeeeee"
    title: str = "Reconstructed figure"
    description: str = ""
    font_family: str = "serif"
    fallback_family: str = "serif"
    bold: bool = False
    arrow_length: float = 12.0
    arrow_width: float = 10.0
    geometry: list[Element] = field(default_factory=list)
    labels: list[Element] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def to_svg(self) -> str:
        weight = "700" if self.bold else "400"
        head = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
            f'     width="{_number(self.width)}" height="{_number(self.height)}"',
            f'     viewBox="0 0 {_number(self.width)} {_number(self.height)}"',
            '     role="img" aria-labelledby="title description">',
            f"  <title id=\"title\">{html.escape(self.title)}</title>",
            f"  <desc id=\"description\">{html.escape(self.description)}</desc>",
            "  <defs>",
            '    <marker id="arrow" viewBox="0 0 12 12" refX="11" refY="6"',
            f'            markerWidth="{_number(self.arrow_length)}" markerHeight="{_number(self.arrow_width)}"',
            '            markerUnits="userSpaceOnUse" orient="auto">',
            '      <path d="M0 0 L12 6 L0 12 Z" fill="currentColor"/>',
            "    </marker>",
            '    <marker id="arrow-start" viewBox="0 0 12 12" refX="1" refY="6"',
            f'            markerWidth="{_number(self.arrow_length)}" markerHeight="{_number(self.arrow_width)}"',
            '            markerUnits="userSpaceOnUse" orient="auto">',
            '      <path d="M12 0 L0 6 L12 12 Z" fill="currentColor"/>',
            "    </marker>",
        ]
        for element in self.geometry + self.labels:
            head.extend(element.defs())
        head += [
            "  </defs>",
            "  <style>",
            "    /* Every mark paints with currentColor, so an inline SVG simply takes",
            "       the colour of the text around it. The rules below only set a",
            "       sensible default for the file viewed on its own. */",
            f"    svg {{ color: {_attribute(self.light_ink)}; }}",
            "    @media (prefers-color-scheme: dark) {",
            f"      svg {{ color: {_attribute(self.dark_ink)}; }}",
            "      .unverified { filter: invert(1); }",
            "    }",
            "    .curve { fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; }",
            "    .axis { stroke: currentColor; stroke-linecap: butt; }",
            "    .tick { stroke: currentColor; stroke-linecap: butt; }",
            "    .fraction-line { stroke: currentColor; fill: none; stroke-linecap: butt; }",
            f'    .glyph {{ fill: currentColor; font-family: "{_attribute(self.font_family)}", {self.fallback_family};'
            f" font-weight: {weight}; }}",
            "    .italic { font-style: italic; }",
            "    .upright { font-style: normal; }",
            "    .frame { fill: none; stroke: currentColor; }",
            "    .markers { color: inherit; }",
            "  </style>",
        ]
        if self.background:
            head.append(
                f'  <rect width="{_number(self.width)}" height="{_number(self.height)}" '
                f'fill="{_attribute(self.background)}"/>'
            )

        body = ['  <g id="geometry">']
        for element in self.geometry:
            body.extend(element.to_svg("    "))
        body.append("  </g>")
        body.append('  <g id="labels">')
        for element in self.labels:
            body.extend(element.to_svg("    "))
        body.append("  </g>")

        metadata = [
            "  <metadata>",
            f"    {html.escape(json.dumps(self.report, ensure_ascii=False, default=str))}",
            "  </metadata>",
        ]
        return "\n".join(head + body + metadata + ["</svg>", ""])
