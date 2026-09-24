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
from hybrid_vectorizer.fitting import (
    _bezier,
    choose_model,
    fit_bezier,
    fit_polynomial,
    fit_sinusoid,
    path_data,
)
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

    def test_a_high_degree_fit_over_real_coordinates_stays_conditioned(self):
        """Raising pixel coordinates to the fifteenth power is not conditioned."""
        import warnings

        x = np.linspace(200.0, 1800.0, 900)
        y = 40.0 * np.sin(x / 300.0) + 300.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = fit_polynomial(x, y, 5)
        self.assertIsNotNone(model)
        self.assertLess(float(np.max(np.abs(model.sample(x) - y))), 1.0)

    def test_a_fitted_polynomial_samples_where_it_was_fitted(self):
        x = np.linspace(500.0, 2500.0, 400)
        model = fit_polynomial(x, 3.0 * x + 17.0, 1)
        self.assertLess(float(np.max(np.abs(model.sample(x) - (3.0 * x + 17.0)))), 1e-6)

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


class RotatedTextTest(unittest.TestCase):
    def _block(self, x, y, width, height):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.textlayout import Block

        return Block(components=[Component(
            label=1, x=x, y=y, width=width, height=height, area=width * height,
            centroid=(x + width / 2.0, y + height / 2.0),
            mask=np.full((height, width), 255, np.uint8),
        )])

    def test_a_column_of_letters_becomes_one_label(self):
        from hybrid_vectorizer.textlayout import group_vertical

        blocks = [self._block(40, 100 + row * 34, 22, 24) for row in range(5)]
        merged = group_vertical(blocks, 22.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].orientation, "vertical")
        self.assertEqual(len(merged[0].components), 5)

    def test_a_stack_of_dashes_is_not_text(self):
        from hybrid_vectorizer.textlayout import group_vertical

        blocks = [self._block(40, 100 + row * 30, 4, 16) for row in range(5)]
        merged = group_vertical(blocks, 22.0)
        self.assertTrue(all(block.orientation == "horizontal" for block in merged))

    def test_tick_labels_down_an_axis_are_left_alone(self):
        from hybrid_vectorizer.textlayout import group_vertical

        blocks = [self._block(40, 100 + row * 90, 46, 24) for row in range(4)]
        merged = group_vertical(blocks, 22.0)
        self.assertEqual(len(merged), 4)
        self.assertTrue(all(block.orientation == "horizontal" for block in merged))

    def test_a_slope_is_measured_not_chosen_from_a_list(self):
        from hybrid_vectorizer.textlayout import text_angle

        blocks = [self._block(100 + step * 30, 300 - step * 30, 20, 20) for step in range(6)]
        block = blocks[0]
        block.components = [b.components[0] for b in blocks]
        self.assertAlmostEqual(text_angle(block), -45.0, delta=4.0)

    def test_upright_text_measures_no_slope(self):
        from hybrid_vectorizer.textlayout import text_angle

        blocks = [self._block(100 + step * 30, 300, 20, 20) for step in range(6)]
        block = blocks[0]
        block.components = [b.components[0] for b in blocks]
        self.assertEqual(text_angle(block), 0.0)

    def test_three_marks_are_too_few_to_show_a_slope(self):
        """'4I' with a sunken subscript measures 22 degrees and is not turned."""
        from hybrid_vectorizer.textlayout import text_angle

        block = self._block(100, 300, 18, 22)
        block.components += [
            self._block(120, 300, 12, 22).components[0],
            self._block(134, 312, 14, 14).components[0],
        ]
        self.assertEqual(text_angle(block), 0.0)

    def test_marks_leaning_with_their_run_are_a_line_not_a_label(self):
        """Four diagonal dashed lines were being read as slanted labels."""
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.textlayout import Block, text_angle

        marks = []
        for step in range(6):
            patch = np.zeros((46, 46), np.uint8)
            cv2.line(patch, (6, 40), (40, 6), 255, 4)
            marks.append(Component(
                label=1, x=100 + step * 52, y=400 - step * 52, width=46, height=46,
                area=int(np.count_nonzero(patch)),
                centroid=(123 + step * 52, 423 - step * 52), mask=patch,
            ))
        self.assertEqual(text_angle(Block(components=marks)), 0.0)

    def test_upright_glyphs_on_a_slope_are_still_a_label(self):
        from hybrid_vectorizer.components import Component
        from hybrid_vectorizer.textlayout import Block, text_angle

        marks = []
        for step in range(6):
            patch = np.zeros((30, 22), np.uint8)
            cv2.rectangle(patch, (4, 3), (18, 27), 255, 3)
            marks.append(Component(
                label=1, x=100 + step * 40, y=400 - step * 40, width=22, height=30,
                area=int(np.count_nonzero(patch)),
                centroid=(111 + step * 40, 415 - step * 40), mask=patch,
            ))
        self.assertAlmostEqual(text_angle(Block(components=marks)), -45.0, delta=6.0)

    def test_a_fraction_is_stacked_but_not_turned(self):
        from hybrid_vectorizer.textlayout import text_angle

        block = self._block(100, 300, 40, 20)
        block.components += [
            self._block(100, 326, 60, 6).components[0],
            self._block(105, 340, 34, 20).components[0],
            self._block(108, 366, 30, 20).components[0],
        ]
        block.bars = [block.components[1]]
        self.assertEqual(text_angle(block), 0.0)

    def test_turning_by_any_angle_keeps_the_corners(self):
        from hybrid_vectorizer.ocr import turned

        image = np.full((40, 120), 255, np.uint8)
        image[8:32, 10:110] = 0
        grown = turned(image, -45.0)
        self.assertGreater(grown.shape[0], 40)
        self.assertGreater(grown.shape[1], 40)
        self.assertGreater(
            np.count_nonzero(grown < 128),
            0.85 * np.count_nonzero(image < 128),
            "the ink was clipped by the box it arrived in",
        )

    def test_turning_a_crop_is_reversible(self):
        from hybrid_vectorizer.ocr import turned

        image = np.zeros((40, 90), np.uint8)
        image[5:12, 10:60] = 255
        self.assertEqual(turned(image, -90.0).shape, (90, 40))
        self.assertEqual(turned(image, 90.0).shape, (90, 40))
        self.assertFalse(np.array_equal(turned(image, -90.0), turned(image, 90.0)))

    def test_type_sitting_the_right_way_up_scores_higher(self):
        from hybrid_vectorizer.ocr import upright_bias

        line = np.full((90, 260), 255, np.uint8)
        cv2.putText(line, "Alphabet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 4)
        self.assertGreater(upright_bias(line), upright_bias(cv2.rotate(line, cv2.ROTATE_180)))

    def test_the_tie_break_cannot_outweigh_confidence(self):
        """It is a weak signal, so it must never overrule the recogniser."""
        from hybrid_vectorizer.ocr import upright_bias

        line = np.full((90, 260), 255, np.uint8)
        cv2.putText(line, "Alphabet", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 4)
        self.assertLessEqual(abs(upright_bias(line)), 0.1)

    def test_a_turned_label_is_placed_by_transform(self):
        from hybrid_vectorizer import ir
        from hybrid_vectorizer.fonts import Metrics

        metrics = Metrics(units_per_em=1000, advances={}, cap_height=0.7,
                          x_height=0.45, ascent=0.9, descent=0.2)
        box = tex.layout(tex.parse("abc"), metrics, 20.0)
        document = ir.Document(width=200, height=200)
        document.labels.append(
            ir.Label(kind="label", identifier="label-0", box=box,
                     transform="translate(50 50) rotate(-90) translate(-15 5)")
        )
        content = document.to_svg()
        root = ET.fromstring(content)
        group = root.find(f".//{SVG_NAMESPACE}g[@id='label-0']")
        self.assertIn("rotate(-90)", group.attrib["transform"])
        self.assertEqual(group.find(f"{SVG_NAMESPACE}text").attrib["y"], "0")


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

    def test_style_commands_keep_what_they_wrap(self):
        self.assertEqual(tex.to_text(tex.parse(r"\boldsymbol{x}=\boldsymbol{A}")), "x=A")
        self.assertEqual(tex.to_text(tex.parse(r"\mathbb{R}^{2}")), "R^2")

    def test_an_unknown_wrapper_keeps_its_argument(self):
        """Rendering the command's own name put 'boldsymbol' into a figure."""
        self.assertEqual(tex.to_text(tex.parse(r"\unknowncmd{q}+x")), "q+x")

    def test_a_bare_unknown_command_draws_nothing_and_says_so(self):
        """Setting its name as a word put 'twoheadrightarrow' across a figure."""
        seen = []
        self.assertEqual(tex.to_text(tex.parse(r"\weird", unknown=seen)), "")
        self.assertEqual(seen, ["weird"])

    def test_the_symbols_a_recogniser_offers_for_arrows_are_drawn(self):
        for source, drawn in (
            (r"\twoheadrightarrow", "\u21a0"),
            (r"\nwarrow", "\u2196"),
            (r"\longrightarrow", "\u27f6"),
            (r"\therefore", "\u2234"),
            (r"\forall", "\u2200"),
        ):
            seen = []
            self.assertEqual(tex.to_text(tex.parse(source, unknown=seen)), drawn)
            self.assertEqual(seen, [], source)

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


class WordVersusLineTest(unittest.TestCase):
    """In heavy type a whole word arrives as one component, wide enough to pass
    for a curve, and is then traced as a squiggle and never read."""

    def _component(self, mask):
        from hybrid_vectorizer.components import Component

        ys, xs = np.nonzero(mask)
        return Component(
            label=1, x=0, y=0, width=mask.shape[1], height=mask.shape[0],
            area=int(np.count_nonzero(mask)), centroid=(0.0, 0.0), mask=mask,
        )

    def test_a_drawn_line_has_two_ends_and_no_branches(self):
        from hybrid_vectorizer.tracing import skeleton_nodes

        mask = np.zeros((200, 400), np.uint8)
        xs = np.arange(20, 380)
        for x in xs:
            cv2.circle(mask, (int(x), int(100 + 60 * np.sin(x / 60.0))), 3, 255, -1)
        ends, junctions = skeleton_nodes(mask)
        self.assertEqual(ends, 2)
        self.assertEqual(junctions, 0)

    def test_a_word_has_many_ends_and_branches(self):
        from hybrid_vectorizer.tracing import skeleton_nodes

        mask = np.zeros((60, 320), np.uint8)
        cv2.putText(mask, "Imaginary", (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 5)
        ends, junctions = skeleton_nodes(mask)
        self.assertGreaterEqual(ends + junctions, 10)

    def test_a_touching_word_is_not_taken_for_a_curve(self):
        from hybrid_vectorizer.tracing import looks_like_text

        mask = np.zeros((44, 300), np.uint8)
        cv2.putText(mask, "Imaginary", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 6)
        self.assertTrue(looks_like_text(self._component(mask), 26.0))

    def test_an_arrow_branches_only_at_its_heads(self):
        """A dimension arrow was being read as a word and drawn as one."""
        from hybrid_vectorizer.tracing import looks_like_text, skeleton_nodes

        mask = np.zeros((40, 200), np.uint8)
        cv2.line(mask, (20, 20), (180, 20), 255, 4)
        cv2.fillPoly(mask, [np.array([[20, 20], [44, 8], [44, 32]])], 255)
        cv2.fillPoly(mask, [np.array([[180, 20], [156, 8], [156, 32]])], 255)
        whole = sum(skeleton_nodes(mask))
        middle = sum(skeleton_nodes(mask, middle=0.6))
        self.assertLess(middle, whole, "the heads are at the ends")
        self.assertFalse(looks_like_text(self._component(mask), 16.0))

    def test_a_word_branches_all_the_way_along(self):
        from hybrid_vectorizer.tracing import looks_like_text

        mask = np.zeros((44, 300), np.uint8)
        cv2.putText(mask, "Imaginary", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 6)
        self.assertTrue(looks_like_text(self._component(mask), 26.0))

    def test_a_tall_curve_is_still_a_curve(self):
        from hybrid_vectorizer.tracing import looks_like_text

        mask = np.zeros((300, 400), np.uint8)
        for x in range(20, 380):
            cv2.circle(mask, (x, int(150 + 120 * np.sin(x / 50.0))), 3, 255, -1)
        self.assertFalse(looks_like_text(self._component(mask), 26.0))


class DashedLineTest(unittest.TestCase):
    def _marks(self, image):
        from hybrid_vectorizer.components import extract

        return extract(image)

    def _canvas(self):
        return np.zeros((300, 600), np.uint8)

    def test_a_broken_line_is_one_line(self):
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x in range(40, 520, 44):
            cv2.line(image, (x, 150), (x + 30, 150), 255, 4)
        lines, used = find_dashed_lines(self._marks(image), 4.0, (600, 300))
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertGreater(line.length, 440)
        self.assertAlmostEqual(line.dash, 31.0, delta=4.0)
        self.assertAlmostEqual(line.gap, 14.0, delta=5.0)
        self.assertEqual(len(used), len(line.components))

    def test_a_diagonal_broken_line_is_found(self):
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for step in range(8):
            x, y = 40 + step * 40, 40 + step * 28
            cv2.line(image, (x, y), (x + 26, y + 18), 255, 4)
        lines, _used = find_dashed_lines(self._marks(image), 4.0, (600, 300))
        self.assertEqual(len(lines), 1)
        self.assertGreater(lines[0].length, 280)

    def test_marks_at_an_irregular_period_are_not_a_line(self):
        """Tick labels offer a fraction bar apiece; only the period tells them apart."""
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x, width in ((40, 40), (120, 30), (230, 34), (300, 26), (420, 38)):
            cv2.line(image, (x, 150), (x + width, 150), 255, 4)
        lines, used = find_dashed_lines(self._marks(image), 4.0, (600, 300))
        self.assertEqual(lines, [])
        self.assertEqual(used, set())

    def test_a_missing_dash_does_not_break_the_period(self):
        """A line passing behind something loses a dash and leaves a double gap."""
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x in (40, 84, 128, 216, 260, 304, 348):   # 172 is missing
            cv2.line(image, (x, 150), (x + 30, 150), 255, 4)
        lines, _used = find_dashed_lines(self._marks(image), 4.0, (600, 300))
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines[0].components), 7)
        self.assertAlmostEqual(lines[0].gap, 14.0, delta=6.0)

    def test_a_mark_with_ink_above_and_below_is_not_a_dash(self):
        """Fraction bars are collinear and evenly spaced; what they have is
        a numerator over them and a denominator under."""
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x in range(40, 520, 100):
            cv2.line(image, (x, 150), (x + 60, 150), 255, 4)
            cv2.rectangle(image, (x + 15, 110), (x + 45, 138), 255, -1)
            cv2.rectangle(image, (x + 20, 162), (x + 40, 190), 255, -1)
        lines, used = find_dashed_lines(self._marks(image), 4.0, (600, 300), ink=image)
        self.assertEqual(lines, [])
        self.assertEqual(used, set())

    def test_the_result_does_not_depend_on_which_mark_comes_first(self):
        """Walking outward from a seed made this swing with the reach."""
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x in range(40, 520, 44):
            cv2.line(image, (x, 100), (x + 30, 100), 255, 4)
        for y in range(180, 290, 30):
            cv2.line(image, (500, y), (500, y + 18), 255, 4)
        marks = self._marks(image)
        first = find_dashed_lines(marks, 4.0, (600, 300))[0]
        second = find_dashed_lines(list(reversed(marks)), 4.0, (600, 300))[0]
        self.assertEqual(len(first), len(second))
        self.assertEqual(
            sorted(len(line.components) for line in first),
            sorted(len(line.components) for line in second),
        )

    def test_two_marks_are_not_a_line(self):
        from hybrid_vectorizer.dashes import find_dashed_lines

        image = self._canvas()
        for x in (40, 120):
            cv2.line(image, (x, 150), (x + 30, 150), 255, 4)
        self.assertEqual(find_dashed_lines(self._marks(image), 4.0, (600, 300))[0], [])

    def test_a_long_stroke_is_not_a_dash(self):
        from hybrid_vectorizer.dashes import is_dash

        image = self._canvas()
        cv2.line(image, (20, 150), (580, 150), 255, 4)
        self.assertFalse(is_dash(self._marks(image)[0], 4.0, (600, 300)))

    def test_a_fat_blob_is_not_a_dash(self):
        from hybrid_vectorizer.dashes import is_dash

        image = self._canvas()
        cv2.circle(image, (300, 150), 22, 255, -1)
        self.assertFalse(is_dash(self._marks(image)[0], 4.0, (600, 300)))


class TintTest(unittest.TestCase):
    """A grey fill is not ink: one threshold cannot hold a dark stroke and a
    light tint, so it is looked for in the greyscale instead."""

    def _sheet(self):
        """A page with ink on it, and grain, as any scan has."""
        sheet = np.full((500, 700), 250, np.float32)
        for y in (60, 100, 430, 470):
            cv2.line(sheet, (40, y), (660, y), 15, 5)
        for x in (60, 200, 340, 480, 620):
            cv2.line(sheet, (x, 40), (x, 480), 15, 5)
        return sheet

    def _finish(self, sheet, seed=7):
        rng = np.random.default_rng(seed)
        gray = np.clip(sheet + rng.normal(0, 4, sheet.shape), 0, 255).astype(np.uint8)
        ink = np.zeros(gray.shape, np.uint8)
        ink[gray < 100] = 255
        return gray, ink

    def _outlined_tint(self, sheet):
        cv2.rectangle(sheet, (110, 150), (560, 380), 200, -1)
        cv2.rectangle(sheet, (110, 150), (560, 380), 15, 4)
        return sheet

    def test_a_drawn_tint_is_found_at_the_density_it_was_printed(self):
        from hybrid_vectorizer.shapes import detect_tints

        tints = detect_tints(*self._finish(self._outlined_tint(self._sheet())), 4.0)
        self.assertEqual(len(tints), 1)
        self.assertAlmostEqual(tints[0].opacity, 0.2, delta=0.06)
        self.assertEqual(tints[0].kind, "tint")

    def test_a_soft_stain_is_not_a_tint(self):
        """A stain spreads across the page instead of being outlined."""
        from hybrid_vectorizer.shapes import detect_tints

        sheet = self._sheet()
        blob = np.zeros(sheet.shape, np.float32)
        cv2.circle(blob, (330, 260), 160, 1.0, -1)
        sheet = sheet - 55 * cv2.GaussianBlur(blob, (0, 0), 70)
        self.assertEqual(detect_tints(*self._finish(sheet), 4.0), [])

    def test_a_screened_tint_survives_its_own_holes(self):
        """A printed tint is mostly band with a scatter of holes in it."""
        from hybrid_vectorizer.shapes import detect_tints

        sheet = self._outlined_tint(self._sheet())
        rng = np.random.default_rng(5)
        holes = (rng.random(sheet.shape) < 0.3)
        region = np.zeros(sheet.shape, bool)
        region[155:375, 115:555] = True
        sheet[region & holes] = 250
        self.assertEqual(len(detect_tints(*self._finish(sheet), 4.0)), 1)

    def test_an_area_already_claimed_is_not_tinted_as_well(self):
        from hybrid_vectorizer.shapes import detect_tints

        gray, ink = self._finish(self._outlined_tint(self._sheet()))
        claimed = np.zeros(gray.shape, np.uint8)
        claimed[140:390, 100:570] = 255
        self.assertEqual(detect_tints(gray, ink, 4.0, claimed=claimed), [])

    def test_a_tint_survives_the_paper_estimate(self):
        """A tile estimate took these for paper and divided them away."""
        from hybrid_vectorizer.preprocess import estimate_paper

        gray = np.full((500, 700), 250, np.uint8)
        cv2.rectangle(gray, (120, 140), (560, 380), 190, -1)
        paper = estimate_paper(gray)
        inside = paper[200:320, 200:480]
        self.assertGreater(int(inside.min()), 225, "the tint was taken for paper")


class DashedCurveTest(unittest.TestCase):
    """A straight broken line is found by the line its marks share; a bent one
    has no such line and has to be followed."""

    def _pieces(self, image):
        from hybrid_vectorizer.components import extract

        return extract(image)

    def test_a_dashed_wave_is_followed_into_one_curve(self):
        """The case these figures actually draw: a sine, shown dashed."""
        from hybrid_vectorizer.tracing import follow_dashed_curves

        image = np.zeros((400, 900), np.uint8)
        xs = np.arange(40, 860)
        ys = (200 + 120 * np.sin(xs / 90.0)).astype(int)
        for start in range(0, xs.size - 26, 38):
            for x, y in zip(xs[start : start + 26], ys[start : start + 26]):
                cv2.circle(image, (int(x), int(y)), 3, 255, -1)
        curves = follow_dashed_curves(self._pieces(image), 6.0, 30.0)
        self.assertEqual(len(curves), 1)
        self.assertGreaterEqual(len(curves[0].components), 15)
        self.assertGreater(curves[0].dash, 0.0)
        self.assertGreater(curves[0].gap, 0.0)

    def test_marks_at_no_regular_spacing_are_not_a_curve(self):
        from hybrid_vectorizer.tracing import follow_dashed_curves

        image = np.zeros((400, 700), np.uint8)
        for x, width in ((40, 30), (110, 18), (230, 34), (300, 22), (470, 28)):
            cv2.line(image, (x, 200), (x + width, 205), 255, 5)
        self.assertEqual(follow_dashed_curves(self._pieces(image), 6.0, 30.0), [])

    def test_a_short_huddle_of_marks_is_not_a_curve(self):
        """A legend's sample and its letters sit within a few dash lengths."""
        from hybrid_vectorizer.tracing import follow_dashed_curves

        image = np.zeros((400, 700), np.uint8)
        for step in range(5):
            cv2.line(image, (100 + step * 14, 200), (100 + step * 14 + 9, 205), 255, 5)
        self.assertEqual(follow_dashed_curves(self._pieces(image), 6.0, 30.0), [])

    def test_too_few_marks_are_not_a_curve(self):
        from hybrid_vectorizer.tracing import follow_dashed_curves

        image = np.zeros((400, 700), np.uint8)
        for x in (40, 140, 240):
            cv2.line(image, (x, 200), (x + 50, 200), 255, 5)
        self.assertEqual(follow_dashed_curves(self._pieces(image), 6.0, 30.0), [])


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

    def test_the_ruled_area_is_not_also_reported_as_a_tint(self):
        self.assertEqual([r for r in self.analysis.regions if r.kind == "tint"], [])

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


@unittest.skipUnless((ROOT / "examples" / "thicklens_cascade.png").exists(), "scan missing")
class LensCascadeTest(unittest.TestCase):
    """Its broken lines run behind the lenses, losing a dash where they pass."""

    @classmethod
    def setUpClass(cls):
        from hybrid_vectorizer.convert import Options, analyse

        cls.analysis = analyse(ROOT / "examples" / "thicklens_cascade.png", Options())

    def test_the_long_horizontal_broken_lines_are_found(self):
        long_ones = [line for line in self.analysis.dashed if line.length > 700]
        self.assertGreaterEqual(len(long_ones), 2)
        for line in long_ones:
            self.assertGreaterEqual(len(line.components), 8)

    def test_its_dimension_arrows_are_drawn_not_read(self):
        """Each was a block holding an arrow and its label, read as a symbol."""
        row = [
            trace for trace in self.analysis.traces
            if 540 < trace.points[:, 1].mean() < 600 and 500 < trace.points[:, 0].mean() < 1200
        ]
        self.assertGreaterEqual(len(row), 4)

    def test_the_shaded_lens_elements_are_tints(self):
        tints = [region for region in self.analysis.regions if region.kind == "tint"]
        self.assertEqual(len(tints), 4)


class NoRegressionTest(unittest.TestCase):
    """The stroke-only example must gain no areas and no marker series."""

    @unittest.skipUnless(EXAMPLE.exists(), "example figure is not present")
    def test_the_interference_figure_stays_all_strokes(self):
        from hybrid_vectorizer.convert import Options, analyse

        analysis = analyse(EXAMPLE, Options())
        self.assertEqual(analysis.regions, [], "a stroke figure has no areas or tints")
        self.assertEqual(analysis.marker_sets, [])
        self.assertEqual(analysis.frames, [])
        self.assertEqual(analysis.legends, [])
        self.assertEqual(analysis.dashed, [], "tick-label bars are not a broken line")
        self.assertEqual([t for t in analysis.traces if t.dash > 0], [])
        self.assertTrue(all(b.orientation == "horizontal" for b in analysis.blocks))
        self.assertEqual(len(analysis.traces), 1, "the curve must stay a curve")
        self.assertEqual(len(analysis.blocks), 11)


@unittest.skipUnless((ROOT / "examples" / "complex.png").exists(), "scan missing")
class ScannedFigureTest(unittest.TestCase):
    """A real scan, where the broken lines were the largest gap."""

    @classmethod
    def setUpClass(cls):
        from hybrid_vectorizer.convert import Options, analyse

        cls.analysis = analyse(ROOT / "examples" / "complex.png", Options())

    def test_both_broken_lines_are_found(self):
        self.assertEqual(len(self.analysis.dashed), 2)
        lengths = sorted(line.length for line in self.analysis.dashed)
        self.assertGreater(lengths[0], 250)

    def test_words_in_heavy_type_are_read_not_traced(self):
        boxes = {(b.x, b.y, b.width, b.height) for b in self.analysis.blocks}
        self.assertIn((66, 34, 218, 36), boxes, "'Imaginary' must reach the text stage")
        self.assertIn((644, 500, 78, 29), boxes, "'Real' must reach the text stage")

    def test_only_the_drawn_vector_is_traced(self):
        self.assertEqual(len(self.analysis.traces), 1)

    def test_the_label_up_the_side_is_read_as_one_line(self):
        vertical = [b for b in self.analysis.blocks if b.orientation == "vertical"]
        self.assertEqual(len(vertical), 1)
        self.assertGreater(vertical[0].height, 2 * vertical[0].width)

    def test_the_label_set_along_the_vector_is_found_at_its_own_angle(self):
        from hybrid_vectorizer.textlayout import text_angle

        block = next(b for b in self.analysis.blocks if (b.x, b.y) == (174, 310))
        self.assertAlmostEqual(text_angle(block), -44.8, delta=5.0)

    def test_they_run_at_right_angles_to_each_other(self):
        import numpy as np

        angles = []
        for line in self.analysis.dashed:
            dx = line.end[0] - line.start[0]
            dy = line.end[1] - line.start[1]
            angles.append(abs(np.degrees(np.arctan2(dy, dx))) % 180.0)
        self.assertAlmostEqual(abs(angles[0] - angles[1]) % 180.0, 90.0, delta=8.0)


@unittest.skipUnless((ROOT / "examples" / "waves1.png").exists(), "scan missing")
class OutOfFamilyLabelTest(unittest.TestCase):
    """Stray marks grouped together get read as something, and drawing the
    answer put large invented words across this figure."""

    @classmethod
    def setUpClass(cls):
        from hybrid_vectorizer.convert import Options, analyse

        cls.page = ROOT / "examples" / "waves1.png"
        cls.options = Options()
        cls.blocks = len(analyse(cls.page, cls.options).blocks)

    def _labelled(self, options):
        """Read every block as a short word, so only the fit decides."""
        from hybrid_vectorizer.convert import analyse, build_labels, choose_fonts
        from hybrid_vectorizer.ocr import Reading

        analysis = analyse(self.page, options)
        for index in range(len(analysis.blocks)):
            analysis.readings[index] = Reading("ab", "tesseract", 0.9)
        fonts, _ranking = choose_fonts(analysis, options)
        return analysis, build_labels(analysis, fonts, options)

    def test_a_reading_needing_outsized_type_is_not_drawn(self):
        analysis, labels = self._labelled(self.options)
        self.assertTrue(analysis.rejected, "sparse scatters should be refused")
        limit = self.options.largest_label * analysis.text_height
        for entry in analysis.rejected:
            self.assertGreater(entry["size_px"], limit)
        self.assertTrue(labels, "real labels must survive")
        self.assertLess(len(analysis.rejected), self.blocks, "not everything is refused")

    def test_raising_the_limit_lets_them_through(self):
        from hybrid_vectorizer.convert import Options

        analysis, labels = self._labelled(Options(largest_label=1000.0))
        self.assertEqual(analysis.rejected, [])
        self.assertEqual(len(labels), self.blocks)

    def test_what_is_refused_is_reported(self):
        analysis, _labels = self._labelled(self.options)
        entry = analysis.rejected[0]
        self.assertEqual(sorted(entry), ["box", "reading", "size_px", "why"])
        self.assertEqual(len(entry["box"]), 4)


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

    def test_a_tint_carries_its_density(self):
        from hybrid_vectorizer import ir

        document = ir.Document(width=100, height=50)
        document.geometry.append(
            ir.Area(kind="area", identifier="area-0", shape="rectangle",
                    parameters={"x": 5, "y": 5, "width": 40, "height": 20}, opacity=0.18)
        )
        root = ET.fromstring(document.to_svg())
        rect = root.find(f".//{SVG_NAMESPACE}rect[@id='area-0']")
        self.assertEqual(rect.attrib["fill-opacity"], "0.18")

    def test_a_solid_area_carries_no_density(self):
        from hybrid_vectorizer import ir

        document = ir.Document(width=100, height=50)
        document.geometry.append(
            ir.Area(kind="area", identifier="area-0", path="M0 0 L9 0 L9 9 Z")
        )
        self.assertNotIn("fill-opacity", document.to_svg())

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
