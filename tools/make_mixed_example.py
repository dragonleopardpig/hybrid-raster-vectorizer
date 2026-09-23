"""Draw a figure containing the features a plot has beyond strokes and text.

The interference example is all strokes, so it cannot show whether a filled
region, hatching, data markers or a legend are handled, nor whether detecting
them misfires. This draws all four with known ground truth, then roughens the
result so it resembles something scanned rather than something rendered.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1200, 720
LEFT, RIGHT, BASE, TOP = 120, 1080, 600, 120

TRUTH = {
    "filled_regions": 1,
    "hatched_regions": 1,
    "marker_sets": 2,
    "markers_per_set": 9,
    "legend_entries": 2,
    "hatch_angle_degrees": 45.0,
    # 14px apart along x, which at 45 degrees is 14/sqrt(2) measured across
    # the ruling -- the perpendicular spacing is what a detector recovers.
    "hatch_spacing_px": 14.0 / (2 ** 0.5),
}


def curve(x, amplitude, phase):
    return BASE - amplitude * np.sin((x - LEFT) / (RIGHT - LEFT) * np.pi * 1.5 + phase) ** 2


def main() -> None:
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)

    # a filled region under the first curve
    xs = np.linspace(LEFT + 40, LEFT + 380, 200)
    ys = curve(xs, 300, 0.0)
    polygon = np.vstack([
        np.column_stack([xs, ys]),
        [[xs[-1], BASE], [xs[0], BASE]],
    ]).astype(np.int32)
    cv2.fillPoly(page, [polygon], 0, lineType=cv2.LINE_AA)

    # a hatched rectangle, 45 degrees at 14px spacing
    x0, y0, x1, y1 = 640, 200, 900, 400
    hatch = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for offset in range(-HEIGHT, WIDTH, 14):
        cv2.line(hatch, (offset, 0), (offset + HEIGHT, HEIGHT), 0, 2, cv2.LINE_AA)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    page[mask > 0] = np.minimum(page[mask > 0], hatch[mask > 0])
    cv2.rectangle(page, (x0, y0), (x1, y1), 0, 2, cv2.LINE_AA)

    # axes with arrowheads
    cv2.line(page, (LEFT - 40, BASE), (RIGHT + 60, BASE), 0, 3, cv2.LINE_AA)
    cv2.fillPoly(page, [np.array([[RIGHT + 60, BASE - 10], [RIGHT + 90, BASE], [RIGHT + 60, BASE + 10]])], 0)
    cv2.line(page, (LEFT - 40, BASE), (LEFT - 40, TOP - 40), 0, 3, cv2.LINE_AA)
    cv2.fillPoly(page, [np.array([[LEFT - 50, TOP - 40], [LEFT - 40, TOP - 70], [LEFT - 30, TOP - 40]])], 0)

    # two data series, drawn as markers only
    sample_x = np.linspace(LEFT + 60, RIGHT - 60, TRUTH["markers_per_set"])
    for series, (amplitude, phase) in enumerate([(240, 0.6), (150, 2.2)]):
        for x in sample_x:
            y = int(round(curve(np.array([x]), amplitude, phase)[0]))
            if series == 0:
                cv2.circle(page, (int(x), y), 9, 0, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(page, (int(x) - 8, y - 8), (int(x) + 8, y + 8), 0, 2, cv2.LINE_AA)

    # a legend box with two samples and their labels
    lx, ly = 760, 470
    cv2.rectangle(page, (lx, ly), (lx + 250, ly + 90), 0, 2, cv2.LINE_AA)
    cv2.circle(page, (lx + 30, ly + 28), 9, 0, -1, cv2.LINE_AA)
    cv2.rectangle(page, (lx + 22, ly + 54), (lx + 38, ly + 70), 0, 2, cv2.LINE_AA)
    cv2.putText(page, "signal", (lx + 60, ly + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    cv2.putText(page, "noise", (lx + 60, ly + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)

    cv2.putText(page, "Fig. 9-1", (520, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)

    # roughen it: a scan is never this clean
    page = cv2.GaussianBlur(page, (3, 3), 0.6)
    noise = np.random.default_rng(7).normal(0, 4, page.shape)
    page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    target = Path(__file__).resolve().parents[1] / "examples" / "mixed" / "figure.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), page)
    (target.parent / "truth.json").write_text(__import__("json").dumps(TRUTH, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
