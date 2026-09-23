"""Tests for the automatic pipeline, on synthetic figures and the real example."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from hybrid_vectorizer import latex as tex
from hybrid_vectorizer.components import extract, median_text_height
from hybrid_vectorizer.consensus import Slot, cluster, reconcile
from hybrid_vectorizer.convert import Options, analyse
from hybrid_vectorizer.fitting import choose_model, fit_bezier, fit_sinusoid, path_data, _bezier
from hybrid_vectorizer.preprocess import Page, estimate_stroke_width
from hybrid_vectorizer.primitives import detect_rules, detect_ticks
from hybrid_vectorizer.textlayout import find_fraction_bars, group_blocks

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "interference" / "raster.png"


def blank(width=800, height=400):
    return np.zeros((height, width), dtype=np.uint8)


def as_page(ink):
    return Page(
        gray=255 - ink,
        ink=ink,
        stroke_width=estimate_stroke_width(ink),
        skew_degrees=0.0,
        background="#ffffff",
        source_path=Path("synthetic"),
    )


class FittingTest(unittest.TestCase):
    def test_sinusoid_period_and_amplitude_are_recovered(self):
        x = np.linspace(0, 1000, 900)
        y = 491 - 263 * np.sin(np.pi * (x - 84) / 194.0) ** 2
        model = fit_sinusoid(x, y)
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model.parameters["period"], 194.0, delta=0.5)
        self.assertAlmostEqual(model.parameters["amplitude"], 131.5, delta=1.0)
        self.assertLess(model.rms, 0.05)

    def test_a_straight_line_prefers_the_line_model(self):
        x = np.linspace(0, 500, 200)
        model = choose_model(x, 3.0 * x + 17.0, tolerance=1.0)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "line")

    def test_bezier_fit_stays_within_its_tolerance(self):
        x = np.linspace(0, 600, 600)
        points = np.column_stack([x, 200 + 150 * np.sin(x / 80.0)])
        for tolerance in (0.5, 2.0):
            segments = fit_bezier(points, tolerance)
            # Sample finely: a coarse sample overstates the distance by half its
            # own spacing, which at this tolerance is the whole measurement.
            dense = np.vstack([_bezier(s, np.linspace(0, 1, 1500)) for s in segments])
            worst = max(
                float(np.min(np.linalg.norm(dense - point, axis=1))) for point in points
            )
            self.assertLessEqual(worst, tolerance, f"tolerance {tolerance}")

    def test_path_data_starts_with_a_move(self):
        segments = fit_bezier(np.column_stack([np.arange(50.0), np.arange(50.0)]), 1.0)
        self.assertTrue(path_data(segments).startswith("M"))


class PrimitiveTest(unittest.TestCase):
    def _figure(self):
        ink = blank()
        cv2.line(ink, (40, 300), (740, 300), 255, 5)
        cv2.fillPoly(ink, [np.array([[740, 288], [770, 300], [740, 312]])], 255)
        for x in range(120, 700, 80):
            cv2.line(ink, (x, 285), (x, 315), 255, 3)
        return ink

    def test_axis_extent_and_arrowhead(self):
        rules = detect_rules(as_page(self._figure()))
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertEqual(rule.orientation, "horizontal")
        self.assertAlmostEqual(rule.position, 300.0, delta=1.5)
        self.assertGreater(rule.end, 760)
        self.assertTrue(any(arrow.at_end for arrow in rule.arrows))
        self.assertFalse(any(not arrow.at_end for arrow in rule.arrows))

    def test_ticks_are_found_on_a_regular_lattice(self):
        page = as_page(self._figure())
        rules = detect_rules(page)
        ticks = detect_ticks(page, rules[0], others=rules)
        self.assertIsNotNone(ticks)
        self.assertEqual(len(ticks.positions), 8)
        self.assertAlmostEqual(ticks.spacing, 80.0, delta=0.5)
        self.assertGreater(ticks.confidence, 0.8)
        for expected, found in zip(range(120, 700, 80), ticks.positions):
            self.assertAlmostEqual(found, expected, delta=2.0)

    def test_a_crossing_rule_is_not_mistaken_for_a_tick(self):
        ink = self._figure()
        cv2.line(ink, (400, 80), (400, 380), 255, 5)
        page = as_page(ink)
        rules = detect_rules(page)
        horizontal = next(rule for rule in rules if rule.orientation == "horizontal")
        ticks = detect_ticks(page, horizontal, others=rules)
        self.assertFalse(
            any(abs(position - 400) < 6 for position in ticks.positions),
            f"the vertical rule was read as a tick: {ticks.positions}",
        )


class TextLayoutTest(unittest.TestCase):
    def _bar(self, ink, x, y, width, thickness=4):
        cv2.rectangle(ink, (x, y), (x + width, y + thickness), 255, -1)

    def _glyph(self, ink, x, y, width=16, height=22):
        cv2.rectangle(ink, (x, y), (x + width, y + height), 255, -1)

    def test_equals_sign_is_not_a_fraction_bar(self):
        ink = blank(300, 200)
        self._bar(ink, 100, 96, 40)
        self._bar(ink, 100, 110, 40)
        components = extract(ink)
        bars = find_fraction_bars(components, text_height=22.0, stroke_width=4.0)
        self.assertEqual(bars, [], "an equals sign has no ink above and below")

    def test_leading_minus_is_not_a_fraction_bar(self):
        ink = blank(300, 240)
        self._bar(ink, 60, 60, 30)
        self._glyph(ink, 100, 55)
        self._bar(ink, 55, 110, 90)
        self._glyph(ink, 80, 130)
        components = extract(ink)
        bars = find_fraction_bars(components, text_height=22.0, stroke_width=4.0)
        self.assertEqual(len(bars), 1)
        self.assertAlmostEqual(bars[0].y, 110, delta=2)

    def test_adjacent_fractions_stay_separate(self):
        ink = blank(600, 260)
        for offset in (60, 260, 460):
            self._glyph(ink, offset + 10, 60)
            self._bar(ink, offset, 100, 80)
            self._glyph(ink, offset + 10, 120)
        components = extract(ink)
        blocks = group_blocks(components, ink.shape, 22.0, 4.0)
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertEqual(len(block.bars), 1)
            self.assertEqual(block.kind, "math")


class LatexTest(unittest.TestCase):
    def test_parses_fractions_scripts_and_symbols(self):
        node = tex.parse(r"I=4I_{0}\cos^{2}\frac{ya\pi}{\delta\lambda_{0}}")
        plain = tex.to_text(node)
        self.assertIn("π", plain)
        self.assertIn("δ", plain)
        self.assertIn("/", plain)
        self.assertIn("_0", plain)

    def test_layout_width_scales_with_size(self):
        from hybrid_vectorizer.fonts import Metrics

        metrics = Metrics(
            units_per_em=1000, advances={c: 0.5 for c in "abc"},
            cap_height=0.7, x_height=0.45, ascent=0.9, descent=0.2,
        )
        node = tex.parse("abc")
        first = tex.layout(node, metrics, 20.0).width
        second = tex.layout(node, metrics, 40.0).width
        self.assertAlmostEqual(second, 2.0 * first, places=6)

    def test_upright_and_italic_runs_are_separated(self):
        from hybrid_vectorizer.fonts import Metrics

        metrics = Metrics(
            units_per_em=1000, advances={}, cap_height=0.7, x_height=0.45,
            ascent=0.9, descent=0.2,
        )
        box = tex.layout(tex.parse(r"2x"), metrics, 20.0)
        styles = {glyph.upright for glyph in box.glyphs}
        self.assertEqual(styles, {True, False})


class ConsensusTest(unittest.TestCase):
    def _slot(self, character, pattern, label=0):
        from hybrid_vectorizer.components import Component

        mask = np.zeros((40, 30), dtype=np.uint8)
        cv2.rectangle(mask, (4, 4), (26, 36), 255, pattern)

        class Run:
            def __init__(self):
                self.text = character

        return Slot(label=label, run=Run(), character=character, component=Component(
            label=1, x=0, y=0, width=30, height=40, area=int(np.count_nonzero(mask)),
            centroid=(15.0, 20.0), mask=mask,
        ))

    def test_identical_shapes_cluster_together(self):
        slots = [self._slot("s", 3) for _ in range(5)]
        groups = cluster(slots)
        self.assertEqual(len(groups), 1)

    def test_majority_rewrites_the_odd_one_out(self):
        slots = [self._slot("s", 3, label=i) for i in range(5)]
        slots[2].character = "B"
        slots[2].run.text = "B"
        changes = reconcile(slots)
        self.assertEqual(len(changes), 1)
        self.assertEqual(slots[2].run.text, "s")

    def test_a_bare_majority_of_two_is_not_enough(self):
        slots = [self._slot("s", 3, label=i) for i in range(3)]
        slots[2].character = "B"
        slots[2].run.text = "B"
        self.assertEqual(reconcile(slots), [])
        self.assertEqual(slots[2].run.text, "B")


@unittest.skipUnless(EXAMPLE.exists(), "example figure is not present")
class ExampleGeometryTest(unittest.TestCase):
    """The detection stages run without OCR, so this stays fast and offline."""

    @classmethod
    def setUpClass(cls):
        cls.analysis = analyse(EXAMPLE, Options())

    def test_both_axes_are_found_with_arrowheads(self):
        rules = self.analysis.rules
        self.assertEqual(len(rules), 2)
        horizontal = next(r for r in rules if r.orientation == "horizontal")
        vertical = next(r for r in rules if r.orientation == "vertical")
        self.assertAlmostEqual(horizontal.position, 491.5, delta=2.0)
        self.assertAlmostEqual(vertical.position, 763.0, delta=2.0)
        self.assertTrue(any(a.at_end for a in horizontal.arrows))
        self.assertTrue(any(not a.at_end for a in vertical.arrows))

    def test_ten_ticks_on_a_regular_lattice(self):
        ticks = self.analysis.ticks
        self.assertEqual(len(ticks), 1)
        self.assertEqual(len(ticks[0].positions), 10)
        self.assertAlmostEqual(ticks[0].spacing, 96.9, delta=1.0)

    def test_the_curve_is_traced_as_one_stroke(self):
        self.assertEqual(len(self.analysis.traces), 1)
        points = self.analysis.traces[0].points
        self.assertGreater(points.shape[0], 1000)
        self.assertLess(points[:, 0].min(), 120)
        self.assertGreater(points[:, 0].max(), 1400)

    def test_labels_are_segmented_into_eleven_blocks(self):
        blocks = self.analysis.blocks
        self.assertEqual(len(blocks), 11)
        self.assertEqual(sum(1 for b in blocks if len(b.bars) == 1), 7)

    def test_the_curve_is_recognisably_a_squared_cosine(self):
        points = self.analysis.traces[0].points
        model = fit_sinusoid(points[:, 0], points[:, 1])
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model.parameters["period"], 194.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()
