"""Tests for the automatic pipeline, on synthetic figures and the real example."""

import unittest
import xml.etree.ElementTree as ET
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
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


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


class SplittingTest(unittest.TestCase):
    def _component(self, mask):
        from hybrid_vectorizer.components import Component

        return Component(
            label=1, x=0, y=0, width=mask.shape[1], height=mask.shape[0],
            area=int(np.count_nonzero(mask)), centroid=(0.0, 0.0), mask=mask,
        )

    def test_a_gap_between_letters_is_a_clean_cut(self):
        from hybrid_vectorizer.components import split_into

        mask = np.zeros((30, 44), dtype=np.uint8)
        cv2.rectangle(mask, (2, 4), (18, 26), 255, 3)
        cv2.rectangle(mask, (26, 4), (42, 26), 255, 3)
        pieces, clean = split_into(self._component(mask), 2)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(clean)

    def test_cutting_through_a_stroke_is_not_clean(self):
        from hybrid_vectorizer.components import split_into

        mask = np.zeros((30, 44), dtype=np.uint8)
        cv2.rectangle(mask, (2, 4), (42, 26), 255, -1)
        _pieces, clean = split_into(self._component(mask), 2)
        self.assertFalse(clean, "a solid block has no gap to cut at")

    def test_one_piece_is_returned_unchanged(self):
        from hybrid_vectorizer.components import split_into

        mask = np.zeros((20, 12), dtype=np.uint8)
        mask[4:16, 2:10] = 255
        pieces, clean = split_into(self._component(mask), 1)
        self.assertEqual(len(pieces), 1)
        self.assertTrue(clean)


class AlignmentTest(unittest.TestCase):
    """Order-based pairing must survive a typeface whose widths are wrong."""

    def setUp(self):
        from hybrid_vectorizer.refine import build_font_set

        self.fonts = build_font_set("DejaVu Serif", bold=False)
        if self.fonts is None:
            self.skipTest("no usable font installed")

    def _component(self, x, width, height=20):
        from hybrid_vectorizer.components import Component

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[2 : height - 2, 1 : width - 1] = 255
        return Component(
            label=1, x=x, y=0, width=width, height=height,
            area=int(np.count_nonzero(mask)), centroid=(x + width / 2.0, height / 2.0),
            mask=mask,
        )

    def test_each_letter_gets_its_own_mark(self):
        from hybrid_vectorizer.refine import correspond

        node = tex.parse("abc")
        size = 20.0
        widths = [self.fonts.metrics.advance(c, size) for c in "abc"]
        cursor, marks = 0.0, []
        for width in widths:
            marks.append(self._component(int(cursor), max(2, int(width))))
            cursor += width
        pairs = correspond(node, self.fonts, size, 0.0, 20.0, marks)
        self.assertEqual([len(glyphs) for _c, glyphs in pairs], [1, 1, 1])
        self.assertEqual("".join(g[0].text for _c, g in pairs), "abc")

    def test_letters_that_ran_together_share_one_mark(self):
        from hybrid_vectorizer.refine import correspond

        node = tex.parse("abc")
        size = 20.0
        widths = [self.fonts.metrics.advance(c, size) for c in "abc"]
        joined = self._component(0, max(2, int(widths[0] + widths[1])))
        last = self._component(int(widths[0] + widths[1]), max(2, int(widths[2])))
        pairs = correspond(node, self.fonts, size, 0.0, 20.0, [joined, last])
        self.assertEqual([len(glyphs) for _c, glyphs in pairs], [2, 1])
        self.assertEqual("".join(g.text for g in pairs[0][1]), "ab")

    def test_more_marks_than_letters_is_refused(self):
        from hybrid_vectorizer.refine import correspond

        marks = [self._component(i * 10, 8) for i in range(5)]
        pairs = correspond(tex.parse("ab"), self.fonts, 20.0, 0.0, 20.0, marks)
        self.assertEqual(pairs, [])


class ShapeTest(unittest.TestCase):
    def _component(self, mask):
        from hybrid_vectorizer.components import Component

        return Component(
            label=1, x=0, y=0, width=mask.shape[1], height=mask.shape[0],
            area=int(np.count_nonzero(mask)), centroid=(0.0, 0.0), mask=mask,
        )

    def test_a_stroke_has_no_depth_but_a_blob_does(self):
        from hybrid_vectorizer.shapes import deep_fraction

        stroke = np.zeros((80, 80), np.uint8)
        cv2.line(stroke, (5, 40), (75, 40), 255, 4)
        blob = np.zeros((80, 80), np.uint8)
        cv2.circle(blob, (40, 40), 30, 255, -1)
        self.assertLess(deep_fraction(stroke, 4.0), 0.05)
        self.assertGreater(deep_fraction(blob, 4.0), 0.5)

    def test_a_filled_area_is_separated_from_the_stroke_it_touches(self):
        from hybrid_vectorizer.shapes import split_solids

        ink = np.zeros((160, 200), np.uint8)
        cv2.circle(ink, (80, 70), 40, 255, -1)
        cv2.line(ink, (0, 140), (199, 140), 255, 4)
        cv2.line(ink, (80, 70), (80, 140), 255, 4)
        solid, rest = split_solids(ink, 4.0)
        self.assertGreater(np.count_nonzero(solid), 3000)
        self.assertGreater(np.count_nonzero(rest[138:143, :]), 500, "the rule must survive")
        self.assertLess(np.count_nonzero(solid[138:143, :]), 60)

    def test_ruling_reports_its_angle_and_spacing(self):
        from hybrid_vectorizer.shapes import detect_hatch

        mask = np.zeros((220, 260), np.uint8)
        for offset in range(-260, 260, 16):
            cv2.line(mask, (offset, 0), (offset + 220, 220), 255, 2)
        cv2.rectangle(mask, (2, 2), (257, 217), 255, 2)
        hatch = detect_hatch(self._component(mask), 2.0)
        self.assertIsNotNone(hatch)
        self.assertAlmostEqual(hatch.angle, 45.0, delta=4.0)
        self.assertAlmostEqual(hatch.spacing, 16.0 / (2 ** 0.5), delta=2.0)

    def test_a_box_with_one_line_in_it_is_not_ruling(self):
        from hybrid_vectorizer.shapes import detect_hatch

        mask = np.zeros((120, 260), np.uint8)
        cv2.rectangle(mask, (2, 2), (257, 117), 255, 2)
        cv2.line(mask, (20, 60), (240, 60), 255, 3)
        self.assertIsNone(detect_hatch(self._component(mask), 2.0))

    def test_boundaries_are_named(self):
        from hybrid_vectorizer.shapes import outline

        circle = np.zeros((60, 60), np.uint8)
        cv2.circle(circle, (30, 30), 22, 255, -1)
        square = np.zeros((60, 60), np.uint8)
        cv2.rectangle(square, (8, 8), (50, 50), 255, 2)
        self.assertEqual(outline(self._component(circle), 3.0).kind, "circle")
        self.assertEqual(outline(self._component(square), 2.0).kind, "rectangle")

    def test_a_drawn_frame_is_told_from_where_ruling_stops(self):
        from hybrid_vectorizer.shapes import has_border

        framed = np.zeros((160, 200), np.uint8)
        cv2.rectangle(framed, (4, 4), (195, 155), 255, 2)
        bare = np.zeros((160, 200), np.uint8)
        for offset in range(-160, 200, 16):
            cv2.line(bare, (offset, 0), (offset + 160, 159), 255, 2)
        self.assertTrue(has_border(self._component(framed), 2.0))
        self.assertFalse(has_border(self._component(bare), 2.0))


class FrameTest(unittest.TestCase):
    def _component(self, mask):
        from hybrid_vectorizer.components import Component

        return Component(
            label=1, x=0, y=0, width=mask.shape[1], height=mask.shape[0],
            area=int(np.count_nonzero(mask)), centroid=(0.0, 0.0), mask=mask,
        )

    def test_a_hollow_box_is_a_frame(self):
        from hybrid_vectorizer.shapes import is_frame

        mask = np.zeros((120, 240), np.uint8)
        cv2.rectangle(mask, (3, 3), (236, 116), 255, 3)
        self.assertTrue(is_frame(self._component(mask), 3.0))

    def test_a_filled_box_is_not_a_frame(self):
        from hybrid_vectorizer.shapes import is_frame

        mask = np.zeros((120, 240), np.uint8)
        cv2.rectangle(mask, (3, 3), (236, 116), 255, -1)
        self.assertFalse(is_frame(self._component(mask), 3.0))

    def test_a_curve_of_the_same_extent_is_not_a_frame(self):
        from hybrid_vectorizer.shapes import is_frame

        mask = np.zeros((120, 240), np.uint8)
        xs = np.arange(5, 235)
        ys = (60 + 45 * np.sin(xs / 25.0)).astype(int)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (int(x), int(y)), 2, 255, -1)
        self.assertFalse(is_frame(self._component(mask), 3.0))


class LegendAssemblyTest(unittest.TestCase):
    def _block(self, x, y, width, height):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.textlayout import Block

        mask = np.full((height, width), 255, np.uint8)
        return Block(components=[Component(
            label=1, x=x, y=y, width=width, height=height, area=width * height,
            centroid=(x + width / 2.0, y + height / 2.0), mask=mask,
        )])

    def _series(self, positions):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.shapes import MarkerSet

        components = [
            Component(label=1, x=int(x) - 5, y=int(y) - 5, width=10, height=10, area=100,
                      centroid=(x, y), mask=np.full((10, 10), 255, np.uint8))
            for x, y in positions
        ]
        return MarkerSet(shape="circle", filled=True, size=10.0,
                         positions=list(positions), components=components)

    def _frame(self):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.shapes import Frame

        component = Component(label=1, x=100, y=100, width=200, height=80, area=1,
                              centroid=(200.0, 140.0), mask=np.zeros((80, 200), np.uint8))
        return Frame(component=component, x=100, y=100, width=200, height=80)

    def test_a_sample_inside_the_frame_is_not_a_data_point(self):
        from hybrid_vectorizer.legend import assemble

        series = self._series([(20.0, 20.0), (50.0, 30.0), (130.0, 130.0)])
        legends = assemble([self._frame()], [series], [self._block(160, 118, 60, 24)])
        self.assertEqual(len(legends), 1)
        self.assertEqual(len(series.positions), 2, "the sample must leave the series")
        self.assertNotIn((130.0, 130.0), series.positions)

    def test_the_sample_is_paired_with_the_name_to_its_right(self):
        from hybrid_vectorizer.legend import assemble

        series = self._series([(20.0, 20.0), (130.0, 130.0)])
        blocks = [self._block(160, 118, 60, 24)]
        legend = assemble([self._frame()], [series], blocks)[0]
        tied = [entry for entry in legend.entries if entry.series is not None]
        self.assertEqual(len(tied), 1)
        self.assertEqual(tied[0].block, 0)

    def test_a_row_whose_sample_cannot_be_separated_is_still_recorded(self):
        from hybrid_vectorizer.legend import assemble

        series = self._series([(20.0, 20.0), (130.0, 130.0)])
        blocks = [self._block(160, 118, 60, 24), self._block(120, 150, 90, 24)]
        legend = assemble([self._frame()], [series], blocks)[0]
        self.assertEqual(len(legend.entries), 2)
        self.assertEqual(sum(1 for e in legend.entries if e.series is None), 1)

    def test_nothing_is_assembled_without_a_frame(self):
        from hybrid_vectorizer.legend import assemble

        series = self._series([(20.0, 20.0)])
        self.assertEqual(assemble([], [series], [self._block(160, 118, 60, 24)]), [])


class UnframedLegendTest(unittest.TestCase):
    def _block(self, x, y, width, height):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.textlayout import Block

        return Block(components=[Component(
            label=1, x=x, y=y, width=width, height=height, area=width * height,
            centroid=(x + width / 2.0, y + height / 2.0),
            mask=np.full((height, width), 255, np.uint8),
        )])

    def _series(self, positions):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.shapes import MarkerSet

        components = [
            Component(label=1, x=int(x) - 8, y=int(y) - 8, width=16, height=16, area=256,
                      centroid=(x, y), mask=np.full((16, 16), 255, np.uint8))
            for x, y in positions
        ]
        return MarkerSet(shape="circle", filled=True, size=16.0,
                         positions=list(positions), components=components)

    def test_rows_sharing_a_column_are_a_legend(self):
        from hybrid_vectorizer.legend import find_unframed

        first = self._series([(600.0, 100.0), (120.0, 400.0)])
        second = self._series([(600.0, 150.0), (300.0, 420.0)])
        blocks = [self._block(620, 90, 70, 22), self._block(620, 140, 60, 22)]
        legends = find_unframed([first, second], blocks, 20.0)
        self.assertEqual(len(legends), 1)
        self.assertFalse(legends[0].framed)
        self.assertEqual(len(legends[0].entries), 2)
        self.assertEqual(first.positions, [(120.0, 400.0)])
        self.assertEqual(second.positions, [(300.0, 420.0)])

    def test_one_labelled_point_is_not_a_legend(self):
        from hybrid_vectorizer.legend import find_unframed

        series = self._series([(600.0, 100.0), (120.0, 400.0)])
        legends = find_unframed([series], [self._block(620, 90, 70, 22)], 20.0)
        self.assertEqual(legends, [])
        self.assertEqual(len(series.positions), 2, "a lone annotation is left alone")

    def test_names_must_start_at_a_common_margin(self):
        from hybrid_vectorizer.legend import find_unframed

        first = self._series([(600.0, 100.0)])
        second = self._series([(600.0, 150.0)])
        # the second name is indented far past the first, so these are not rows
        blocks = [self._block(620, 90, 70, 22), self._block(700, 140, 60, 22)]
        self.assertEqual(find_unframed([first, second], blocks, 20.0), [])

    def test_a_label_on_the_wrong_side_is_not_paired(self):
        from hybrid_vectorizer.legend import find_unframed

        first = self._series([(600.0, 100.0)])
        second = self._series([(600.0, 150.0)])
        blocks = [self._block(480, 90, 70, 22), self._block(480, 140, 60, 22)]
        self.assertEqual(find_unframed([first, second], blocks, 20.0), [])


class ArrowTest(unittest.TestCase):
    def test_a_long_thin_spike_is_not_an_arrowhead(self):
        """A cut-away filled area leaves a taper far longer than it is wide."""
        from hybrid_vectorizer.primitives import _find_arrow

        spike = np.full(160, 4, dtype=np.int32)
        spike[-96:] = np.linspace(20, 4, 96).astype(np.int32)
        self.assertIsNone(_find_arrow(spike, 4.0, at_end=True, search=140))

    def test_a_short_very_wide_flare_is_not_an_arrowhead(self):
        """A crossing rule flares far wider than it is long."""
        from hybrid_vectorizer.primitives import _find_arrow

        flare = np.full(160, 4, dtype=np.int32)
        flare[-5:] = 77
        self.assertIsNone(_find_arrow(flare, 4.0, at_end=True, search=140))

    def test_a_proper_head_is_accepted(self):
        from hybrid_vectorizer.primitives import _find_arrow

        profile = np.full(160, 4, dtype=np.int32)
        profile[-24:] = np.linspace(26, 4, 24).astype(np.int32)
        arrow = _find_arrow(profile, 4.0, at_end=True, search=140)
        self.assertIsNotNone(arrow)
        self.assertGreater(arrow.width, 8)


@unittest.skipUnless((ROOT / "examples" / "mixed" / "figure.png").exists(), "mixed example missing")
class MixedFigureTest(unittest.TestCase):
    """Areas, ruling and marker series, on a figure whose contents are known."""

    @classmethod
    def setUpClass(cls):
        from hybrid_vectorizer.convert import Options, analyse

        cls.analysis = analyse(ROOT / "examples" / "mixed" / "figure.png", Options())

    def test_one_filled_area_and_one_ruled_area(self):
        kinds = sorted(region.kind for region in self.analysis.regions)
        self.assertEqual(kinds, ["hatched", "solid"])

    def test_the_ruling_is_measured(self):
        hatched = next(r for r in self.analysis.regions if r.kind == "hatched")
        self.assertAlmostEqual(hatched.hatch.angle, 45.0, delta=5.0)
        self.assertAlmostEqual(hatched.hatch.spacing, 14.0 / (2 ** 0.5), delta=2.0)
        self.assertEqual(hatched.outline.kind, "rectangle")
        self.assertTrue(hatched.bordered)

    def test_the_legend_is_one_object_tied_to_a_series(self):
        self.assertEqual(len(self.analysis.frames), 1)
        self.assertEqual(len(self.analysis.legends), 1)
        legend = self.analysis.legends[0]
        self.assertEqual(len(legend.entries), 2)
        self.assertEqual(sum(1 for e in legend.entries if e.series is not None), 1)

    def test_no_legend_sample_is_counted_as_data(self):
        legend = self.analysis.legends[0]
        for series in self.analysis.marker_sets:
            for x, y in series.positions:
                self.assertFalse(
                    legend.frame.contains(x, y, 4.0),
                    f"a {series.shape} at ({x:.0f},{y:.0f}) is the legend's own sample",
                )

    def test_two_marker_series_with_the_right_shapes(self):
        shapes = sorted(series.shape for series in self.analysis.marker_sets)
        self.assertEqual(shapes, ["circle", "rectangle"])
        for series in self.analysis.marker_sets:
            self.assertGreaterEqual(len(series.positions), 4)
        filled = {series.shape: series.filled for series in self.analysis.marker_sets}
        self.assertTrue(filled["circle"])
        self.assertFalse(filled["rectangle"])

    def test_each_axis_keeps_exactly_one_arrowhead(self):
        self.assertEqual(len(self.analysis.rules), 2)
        for rule in self.analysis.rules:
            self.assertEqual(len(rule.arrows), 1, f"{rule.orientation} axis")

    def test_a_filled_area_is_not_read_as_a_thick_rule(self):
        for rule in self.analysis.rules:
            self.assertLess(rule.thickness, 20.0)


@unittest.skipUnless((ROOT / "examples" / "series" / "figure.png").exists(), "series example missing")
class SeriesFigureTest(unittest.TestCase):
    """A legend with no box around it, found by the shape of its rows."""

    @classmethod
    def setUpClass(cls):
        from hybrid_vectorizer.convert import Options, analyse

        cls.analysis = analyse(ROOT / "examples" / "series" / "figure.png", Options())

    def test_an_unframed_legend_is_found(self):
        self.assertEqual(len(self.analysis.frames), 0)
        self.assertEqual(len(self.analysis.legends), 1)
        legend = self.analysis.legends[0]
        self.assertFalse(legend.framed)
        self.assertEqual(len(legend.entries), 2)

    def test_every_row_is_tied_to_a_series(self):
        legend = self.analysis.legends[0]
        self.assertTrue(all(entry.series is not None for entry in legend.entries))
        self.assertEqual(len({entry.series for entry in legend.entries}), 2)

    def test_the_samples_are_not_counted_as_data(self):
        legend = self.analysis.legends[0]
        x, y, width, height = legend.bounds
        for series in self.analysis.marker_sets:
            for px, py in series.positions:
                inside = x <= px <= x + width and y <= py <= y + height
                self.assertFalse(inside, f"{series.shape} at ({px:.0f},{py:.0f}) is a sample")


class NoRegressionTest(unittest.TestCase):
    """The stroke-only example must gain no areas and no marker series."""

    @unittest.skipUnless(EXAMPLE.exists(), "example figure is not present")
    def test_the_interference_figure_stays_all_strokes(self):
        from hybrid_vectorizer.convert import Options, analyse

        analysis = analyse(EXAMPLE, Options())
        self.assertEqual(analysis.regions, [])
        self.assertEqual(analysis.marker_sets, [])
        self.assertEqual(analysis.frames, [])
        self.assertEqual(analysis.legends, [])
        self.assertEqual(len(analysis.traces), 1)
        self.assertEqual(len(analysis.blocks), 11)


class OutputTest(unittest.TestCase):
    def _document(self, **kwargs):
        from hybrid_vectorizer import ir

        document = ir.Document(width=100, height=50, **kwargs)
        document.geometry.append(
            ir.Axis(kind="axis", x1=0, y1=25, x2=100, y2=25, stroke_width=2, arrow_end=True)
        )
        return document

    def test_the_page_is_transparent_by_default(self):
        content = self._document().to_svg()
        root = ET.fromstring(content)
        self.assertEqual(root.findall(f"{SVG_NAMESPACE}rect"), [])

    def test_a_background_can_be_asked_for(self):
        content = self._document(background="#ffffff").to_svg()
        root = ET.fromstring(content)
        rects = root.findall(f"{SVG_NAMESPACE}rect")
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0].attrib["fill"], "#ffffff")

    def test_every_mark_paints_with_currentcolor(self):
        content = self._document().to_svg()
        self.assertNotRegex(content, r'(fill|stroke)(=")#|(fill|stroke): #')
        self.assertIn("currentColor", content)

    def test_both_themes_are_described(self):
        content = self._document().to_svg()
        self.assertIn("prefers-color-scheme: dark", content)
        self.assertIn("svg { color:", content)

    def test_a_group_nests_its_children_and_their_definitions(self):
        from hybrid_vectorizer import ir

        document = ir.Document(width=200, height=100)
        document.geometry.append(
            ir.MarkerField(kind="markers", identifier="series-0", shape="circle",
                           size=8, positions=[(20, 20)])
        )
        document.labels.append(
            ir.Group(kind="legend", identifier="legend-0", label="legend", children=[
                ir.Frame(kind="frame", identifier="frame-0", x=100, y=10,
                         width=80, height=50, stroke_width=2),
                ir.MarkerField(kind="markers", identifier="legend-0-sample-0", shape="circle",
                               size=8, positions=[(112, 25)], symbol_id="marker-series-0"),
            ])
        )
        content = document.to_svg()
        root = ET.fromstring(content)
        group = root.find(f".//{SVG_NAMESPACE}g[@id='legend-0']")
        self.assertIsNotNone(group)
        self.assertEqual(len(group.findall(f"{SVG_NAMESPACE}rect")), 1)
        self.assertEqual(content.count('id="marker-series-0"'), 1, "one definition only")

    def test_the_document_still_parses_as_svg(self):
        root = ET.fromstring(self._document().to_svg())
        self.assertEqual(root.tag, f"{SVG_NAMESPACE}svg")


class EnsembleTest(unittest.TestCase):
    def test_augmentations_are_distinct_and_keep_the_original_first(self):
        from hybrid_vectorizer.ocr import augmentations

        image = np.full((40, 90), 255, dtype=np.uint8)
        cv2.putText(image, "ab", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        variants = augmentations(image, 5)
        self.assertEqual(len(variants), 5)
        self.assertTrue(np.array_equal(variants[0], image))
        shapes = {(v.shape, int(v.sum())) for v in variants}
        self.assertGreaterEqual(len(shapes), 4, "augmentations must actually differ")

    def test_asking_for_one_returns_only_the_original(self):
        from hybrid_vectorizer.ocr import augmentations

        image = np.full((20, 20), 255, dtype=np.uint8)
        self.assertEqual(len(augmentations(image, 1)), 1)


class PaperTest(unittest.TestCase):
    def test_a_filled_area_survives_flattening(self):
        """The paper estimate must carry across ink, not treat it as shading."""
        from hybrid_vectorizer.preprocess import estimate_paper

        gray = np.full((400, 500), 245, np.uint8)
        cv2.rectangle(gray, (120, 120), (360, 300), 20, -1)
        paper = estimate_paper(gray)
        inside = paper[180:240, 180:300]
        self.assertGreater(int(inside.min()), 180, "the fill was taken for background")

    def test_uneven_shading_is_measured_and_removed(self):
        """Shading that is lighter than the ink, which is the case on a scan."""
        from hybrid_vectorizer.preprocess import flatten

        gradient = np.tile(np.linspace(150, 250, 500).astype(np.uint8), (400, 1))
        for y in (120, 200, 280):
            cv2.line(gradient, (40, y), (460, y), 0, 4)
        flat, spread = flatten(gradient)
        self.assertGreater(spread, 25)
        corners = [flat[10, 10], flat[10, -10], flat[-10, 10], flat[-10, -10]]
        self.assertLess(
            int(max(corners)) - int(min(corners)), 30, "the page is still uneven"
        )
        self.assertLess(int(flat[120, 250]), 120, "the ink must survive")

    def test_an_even_page_is_left_alone(self):
        from hybrid_vectorizer.preprocess import flatten

        page = np.full((300, 400), 250, np.uint8)
        cv2.circle(page, (200, 150), 60, 0, -1)
        flat, spread = flatten(page)
        self.assertLess(spread, 25)
        self.assertTrue(np.array_equal(flat, page), "a clean page must not be touched")

    def test_grain_smaller_than_the_pen_is_dropped(self):
        from hybrid_vectorizer.preprocess import remove_grain

        ink = np.zeros((200, 300), np.uint8)
        cv2.line(ink, (20, 100), (280, 100), 255, 6)
        rng = np.random.default_rng(3)
        for x, y in rng.integers([0, 0], [300, 200], size=(120, 2)):
            ink[y:y + 2, x:x + 2] = 255
        cleaned = remove_grain(ink, 6.0)
        self.assertGreater(np.count_nonzero(cleaned[97:104, 20:280]), 1200)
        self.assertLess(np.count_nonzero(cleaned) - np.count_nonzero(cleaned[95:106, :]), 200)


class LoadingTest(unittest.TestCase):
    def test_a_missing_file_is_reported_plainly(self):
        from hybrid_vectorizer.preprocess import load_page

        with self.assertRaises(SystemExit) as raised:
            load_page(Path("/nonexistent/figure.png"))
        self.assertIn("No such image", str(raised.exception))

    @unittest.skipUnless(EXAMPLE.exists(), "example figure is not present")
    def test_a_real_image_loads_and_is_measured(self):
        from hybrid_vectorizer.preprocess import load_page

        page = load_page(EXAMPLE)
        self.assertEqual((page.width, page.height), (1593, 717))
        self.assertGreater(page.stroke_width, 1.0)
        self.assertEqual(page.background, "#ffffff")


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
