"""Draw a scatter plot whose legend has no box around it.

Most legends are not framed, so the framed example cannot show whether one is
found by its structure alone: samples sharing a column, each with its name
immediately to the right.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1000, 640
LEFT, RIGHT, BASE, TOP = 110, 930, 540, 90

TRUTH = {
    "legends": 1,
    "framed": False,
    "legend_entries": 2,
    "marker_sets": 2,
    "points_per_set": 8,
    "names": ["alpha", "beta"],
}


def main() -> None:
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    rng = np.random.default_rng(11)

    cv2.line(page, (LEFT, BASE), (RIGHT + 40, BASE), 0, 3, cv2.LINE_AA)
    cv2.fillPoly(page, [np.array([[RIGHT + 40, BASE - 9], [RIGHT + 68, BASE], [RIGHT + 40, BASE + 9]])], 0)
    cv2.line(page, (LEFT, BASE), (LEFT, TOP - 30), 0, 3, cv2.LINE_AA)
    cv2.fillPoly(page, [np.array([[LEFT - 9, TOP - 30], [LEFT, TOP - 58], [LEFT + 9, TOP - 30]])], 0)

    def triangle(centre, size, thickness):
        x, y = centre
        half = size // 2
        points = np.array([[x, y - half], [x + half, y + half], [x - half, y + half]])
        cv2.drawContours(page, [points], 0, 0, thickness, cv2.LINE_AA)

    xs = np.linspace(LEFT + 70, RIGHT - 70, TRUTH["points_per_set"])
    for x, jitter in zip(xs, rng.normal(0, 22, len(xs))):
        triangle((int(x), int(BASE - 90 - jitter - 0.25 * (x - LEFT))), 20, -1)
    for x, jitter in zip(xs, rng.normal(0, 22, len(xs))):
        cv2.circle(page, (int(x), int(BASE - 250 - jitter + 0.18 * (x - LEFT))), 10, 0, 2, cv2.LINE_AA)

    # the legend: two rows, no box, samples sharing a column
    sx, sy = 640, 150
    triangle((sx, sy), 20, -1)
    cv2.putText(page, "alpha", (sx + 26, sy + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2, cv2.LINE_AA)
    cv2.circle(page, (sx, sy + 46), 10, 0, 2, cv2.LINE_AA)
    cv2.putText(page, "beta", (sx + 26, sy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2, cv2.LINE_AA)

    cv2.putText(page, "Fig. 4-2", (430, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2, cv2.LINE_AA)

    page = cv2.GaussianBlur(page, (3, 3), 0.6)
    page = np.clip(page.astype(np.float32) + rng.normal(0, 4, page.shape), 0, 255).astype(np.uint8)

    target = Path(__file__).resolve().parents[1] / "examples" / "series" / "figure.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), page)
    (target.parent / "truth.json").write_text(__import__("json").dumps(TRUTH, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
