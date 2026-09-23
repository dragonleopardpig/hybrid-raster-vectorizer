from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float | int) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _attribute(value: object) -> str:
    return html.escape(str(value), quote=True)


def _curve_path(curve: dict[str, Any]) -> str:
    start_x = float(curve["start_minimum_x"])
    half_period = float(curve["half_period"])
    baseline_y = float(curve["baseline_y"])
    peak_y = float(curve["peak_y"])
    segments = int(curve["segments"])

    commands = [f"M{_number(start_x)} {_number(baseline_y)}"]
    for index in range(segments):
        x0 = start_x + (index * half_period)
        x1 = x0 + half_period
        y0 = baseline_y if index % 2 == 0 else peak_y
        y1 = peak_y if index % 2 == 0 else baseline_y
        control_x1 = x0 + (half_period / 3.0)
        control_x2 = x0 + ((2.0 * half_period) / 3.0)
        commands.append(
            "C"
            f"{_number(control_x1)} {_number(y0)} "
            f"{_number(control_x2)} {_number(y1)} "
            f"{_number(x1)} {_number(y1)}"
        )
    return " ".join(commands)


def _text_element(item: dict[str, Any]) -> str:
    attributes = {
        "id": item.get("id"),
        "class": item.get("class", "math"),
        "x": item["x"],
        "y": item["y"],
        "font-size": item["font_size"],
        "text-anchor": item.get("anchor"),
        "aria-label": item.get("aria_label"),
    }
    rendered = " ".join(
        f'{name}="{_attribute(value)}"'
        for name, value in attributes.items()
        if value is not None
    )
    return f"    <text {rendered}>{html.escape(item['text'])}</text>"


def _fraction_element(item: dict[str, Any]) -> str:
    line_half_width = float(item["line_width"]) / 2.0
    font_size = item["font_size"]
    x = item["x"]
    y = item["y"]
    numerator_y = item.get("numerator_y", 0)
    line_y = item.get("line_y", 9)
    denominator_y = item.get("denominator_y", 38)
    aria_label = item.get("aria_label")
    aria = f' aria-label="{_attribute(aria_label)}"' if aria_label else ""
    return "\n".join(
        [
            f'    <g class="fraction" transform="translate({_number(x)} {_number(y)})" text-anchor="middle"{aria}>',
            f'      <text class="math" x="0" y="{_number(numerator_y)}" font-size="{_attribute(font_size)}">{html.escape(item["numerator"])}</text>',
            f'      <path class="fraction-line" d="M-{_number(line_half_width)} {_number(line_y)}H{_number(line_half_width)}"/>',
            f'      <text class="math" x="0" y="{_number(denominator_y)}" font-size="{_attribute(font_size)}">{html.escape(item["denominator"])}</text>',
            "    </g>",
        ]
    )


def render_svg(spec: dict[str, Any]) -> str:
    canvas = spec["canvas"]
    geometry = spec["geometry"]
    curve = geometry["curve"]
    horizontal_axis = geometry["horizontal_axis"]
    vertical_axis = geometry["vertical_axis"]
    clip = curve["clip"]
    font_family = spec["typography"]["font_family"]
    fallback_family = spec["typography"].get("fallback_family", "serif")

    labels = []
    for item in spec["labels"]:
        if item["type"] == "text":
            labels.append(_text_element(item))
        elif item["type"] == "fraction":
            labels.append(_fraction_element(item))
        else:
            raise ValueError(f"Unsupported label type: {item['type']}")

    tick_lines = "\n".join(
        f'      <line class="tick" x1="{_number(x)}" y1="{_number(geometry["tick_top"])}" '
        f'x2="{_number(x)}" y2="{_number(geometry["tick_bottom"])}"/>'
        for x in geometry["ticks"]
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{_number(canvas['width'])}" height="{_number(canvas['height'])}"
     viewBox="0 0 {_number(canvas['width'])} {_number(canvas['height'])}"
     role="img" aria-labelledby="title description">
  <title id="title">{html.escape(spec['title'])}</title>
  <desc id="description">{html.escape(spec['description'])}</desc>
  <metadata>{html.escape(spec['metadata'])}</metadata>
  <defs>
    <marker id="axis-arrow" viewBox="0 0 12 12" refX="11" refY="6"
            markerWidth="{_number(geometry['arrow_size'])}"
            markerHeight="{_number(geometry['arrow_size'])}"
            markerUnits="userSpaceOnUse" orient="auto">
      <path d="M0 0 L12 6 L0 12 Z" fill="#000"/>
    </marker>
    <clipPath id="curve-clip">
      <rect x="{_number(clip['x'])}" y="{_number(clip['y'])}"
            width="{_number(clip['width'])}" height="{_number(clip['height'])}"/>
    </clipPath>
  </defs>
  <rect width="{_number(canvas['width'])}" height="{_number(canvas['height'])}" fill="{_attribute(canvas['background'])}"/>
  <style>
    .curve {{ fill: none; stroke: #000; stroke-width: {_number(curve['stroke_width'])}; stroke-linecap: round; stroke-linejoin: round; }}
    .axis {{ fill: none; stroke: #000; stroke-width: {_number(geometry['axis_width'])}; stroke-linecap: square; marker-end: url(#axis-arrow); }}
    .tick {{ stroke: #000; stroke-width: {_number(geometry['tick_width'])}; stroke-linecap: square; }}
    .math {{ fill: #000; font-family: "{_attribute(font_family)}", {fallback_family}; font-style: italic; font-weight: 600; }}
    .caption {{ fill: #000; font-family: "{_attribute(font_family)}", {fallback_family}; font-weight: 700; }}
    .fraction-line {{ fill: none; stroke: #000; stroke-width: 2.6; stroke-linecap: square; }}
  </style>
  <g id="plot-geometry">
    <path class="curve" clip-path="url(#curve-clip)" d="{_curve_path(curve)}"/>
    <line class="axis" x1="{_number(horizontal_axis['x1'])}" y1="{_number(horizontal_axis['y'])}"
          x2="{_number(horizontal_axis['x2'])}" y2="{_number(horizontal_axis['y'])}"/>
    <line class="axis" x1="{_number(vertical_axis['x'])}" y1="{_number(vertical_axis['y1'])}"
          x2="{_number(vertical_axis['x'])}" y2="{_number(vertical_axis['y2'])}"/>
    <g id="ticks">
{tick_lines}
    </g>
  </g>
  <g id="labels">
{chr(10).join(labels)}
  </g>
</svg>
'''


def render_file(spec_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(load_spec(spec_path)), encoding="utf-8")
