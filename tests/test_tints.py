"""Tints are found in the scan as it was, by enclosure and flatness; shading is never heavier than the pen."""
import unittest

import cv2
import numpy as np

from hybrid_vectorizer.components import Component
from hybrid_vectorizer.preprocess import estimate_stroke_width, flatten
from hybrid_vectorizer.shapes import detect_hatch, detect_regions, detect_tints, has_border


def _component(mask):
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    width, height = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    return Component(label=1, x=x, y=y, width=width, height=height, area=int(np.count_nonzero(mask)),
                     centroid=(x + width / 2.0, y + height / 2.0), mask=mask[y:y + height, x:x + width].copy())


def _sheet(paper=255):
    """A framed drawing with two rules across it, as a scan would hold it."""
    sheet = np.full((600, 900), float(paper), np.float32)
    cv2.rectangle(sheet, (30, 30), (870, 570), 15, 6)
    cv2.line(sheet, (30, 300), (870, 300), 15, 6)
    cv2.line(sheet, (450, 30), (450, 570), 15, 6)
    return sheet


def _finish(sheet, seed=3):
    rng = np.random.default_rng(seed)
    gray = np.clip(sheet + rng.normal(0, 3, sheet.shape), 0, 255).astype(np.uint8)
    ink = np.zeros(gray.shape, np.uint8)
    ink[gray < 100] = 255
    return gray, ink


class RawTintTest(unittest.TestCase):
    def test_a_mottled_slab_is_found_although_flattening_lifted_it(self):
        """fig-3-7's slab: 218-240 on a 255 paper, lifted to 253 by the paper estimate."""
        sheet = _sheet()
        rng = np.random.default_rng(1)
        slab = np.clip(230 + rng.normal(0, 8, (268, 838)), 200, 255)
        sheet[302:570, 31:869] = slab
        gray, ink = _finish(sheet)
        flat, _spread = flatten(gray)
        self.assertGreater(float(np.median(flat[400:500, 100:400])), 245.0, "flattening did not lift the slab")
        tints = detect_tints(flat, ink, 6.0, raw=gray)
        self.assertEqual(len(tints), 1)
        self.assertAlmostEqual(tints[0].opacity, 0.1, delta=0.04)
        self.assertGreater(tints[0].component.area, 150000)

    def test_a_slab_crossed_by_lines_is_one_region(self):
        sheet = _sheet()
        sheet[302:570, 31:869] = 225
        for x in range(150, 850, 90):
            cv2.line(sheet, (x, 302), (x + 60, 569), 15, 5)
        gray, ink = _finish(sheet)
        tints = detect_tints(gray, ink, 6.0)
        self.assertEqual(len(tints), 1)
        self.assertGreater(tints[0].component.width, 800)

    def test_a_screened_band_without_an_outline_is_a_tint(self):
        """fig-2-2: a halftone rectangle, invisible to the threshold, bounded by its own edge."""
        sheet = np.full((600, 900), 255.0, np.float32)
        cv2.line(sheet, (40, 560), (860, 560), 15, 6)
        rng = np.random.default_rng(2)
        band = np.where(rng.random((500, 300)) < 0.35, 90.0, 255.0)
        sheet[40:540, 300:600] = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.2)
        gray, ink = _finish(sheet)
        tints = detect_tints(gray, ink, 6.0)
        self.assertEqual(len(tints), 1)
        x, y, width, height = (tints[0].component.x, tints[0].component.y,
                               tints[0].component.width, tints[0].component.height)
        self.assertLess(abs(x - 300), 20); self.assertLess(abs(width - 300), 40)

    def test_a_stain_gradient_inside_a_frame_is_not_a_tint(self):
        """fig-1-9: the scan darkens toward the top edge; its boundary runs along the frame."""
        sheet = _sheet()
        rows = np.arange(600, dtype=np.float32)[:, None]
        sheet = sheet - np.clip(60.0 - rows * 60.0 / 350.0, 0, 60)   # 195 at the top, paper by row 350
        sheet[sheet < 20] = 15
        gray, ink = _finish(sheet)
        self.assertEqual(detect_tints(gray, ink, 6.0), [])

    def test_a_soft_unoutlined_blob_is_not_a_tint(self):
        sheet = _sheet()
        blob = np.zeros(sheet.shape, np.float32)
        cv2.circle(blob, (650, 420), 120, 1.0, -1)
        sheet = sheet - 50 * cv2.GaussianBlur(blob, (0, 0), 60)
        gray, ink = _finish(sheet)
        self.assertEqual(detect_tints(gray, ink, 6.0), [])

    def test_the_region_stage_reads_the_scan_not_the_flattened_page(self):
        sheet = _sheet()
        sheet[302:570, 31:869] = 228
        gray, ink = _finish(sheet)
        flat, _spread = flatten(gray)
        regions, _working = detect_regions(ink, 6.0, flat, raw=gray)
        self.assertEqual([region.kind for region in regions], ["tint"])


class RulingHeavinessTest(unittest.TestCase):
    def test_parallel_lines_heavier_than_the_pen_are_not_shading(self):
        """fig-4-18: five wavefront lines, 2.7 pens wide at nine pens apart."""
        mask = np.zeros((300, 240), np.uint8)
        for x in range(20, 240, 44):
            cv2.line(mask, (x, 10), (x, 290), 255, 12)
        self.assertIsNone(detect_hatch(_component(mask), 4.4))

    def test_ruling_at_the_pen_or_finer_is_still_shading(self):
        mask = np.zeros((300, 240), np.uint8)
        for x in range(8, 240, 12):
            cv2.line(mask, (x, 10), (x, 290), 255, 3)
        self.assertIsNotNone(detect_hatch(_component(mask), estimate_stroke_width(mask)))

    def test_a_drawing_with_some_parallel_lines_is_not_a_ruled_area(self):
        """fig-4-18 whole: wavefront lines carried the Hough vote but a third of the ink."""
        mask = np.zeros((446, 878), np.uint8)
        cv2.rectangle(mask, (3, 3), (874, 442), 255, 5)
        cv2.ellipse(mask, (520, 220), (120, 170), 0, 0, 360, 255, 5)
        cv2.line(mask, (10, 220), (868, 220), 255, 4)
        for x in range(640, 860, 44):
            cv2.line(mask, (x, 40), (x, 400), 255, 12)
        for x in range(440, 600, 30):
            cv2.line(mask, (x, 60), (x, 380), 255, 4)
        for radius in range(60, 300, 45):
            cv2.ellipse(mask, (200, 220), (radius, radius), 0, -40, 40, 255, 4)
        self.assertIsNone(detect_hatch(_component(mask), estimate_stroke_width(mask)))

    def test_bare_ruling_is_not_bordered_by_its_own_edges(self):
        mask = np.zeros((300, 240), np.uint8)
        for x in range(20, 240, 44):
            cv2.line(mask, (x, 10), (x, 290), 255, 12)
        self.assertFalse(has_border(_component(mask), 4.4))
        framed = mask.copy()
        cv2.rectangle(framed, (4, 4), (235, 295), 255, 4)
        self.assertTrue(has_border(_component(framed), 4.4))


if __name__ == "__main__":
    unittest.main()
