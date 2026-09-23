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
    background: str = "#ffffff"
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
            "  </defs>",
            "  <style>",
            "    .curve { fill: none; stroke: #000; stroke-linecap: round; stroke-linejoin: round; }",
            "    .axis { stroke: #000; color: #000; stroke-linecap: butt; }",
            "    .tick { stroke: #000; stroke-linecap: butt; }",
            "    .fraction-line { stroke: #000; fill: none; stroke-linecap: butt; }",
            f'    .glyph {{ fill: #000; font-family: "{_attribute(self.font_family)}", {self.fallback_family};'
            f" font-weight: {weight}; }}",
            "    .italic { font-style: italic; }",
            "    .upright { font-style: normal; }",
            "  </style>",
            f'  <rect width="{_number(self.width)}" height="{_number(self.height)}" '
            f'fill="{_attribute(self.background)}"/>',
        ]

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
