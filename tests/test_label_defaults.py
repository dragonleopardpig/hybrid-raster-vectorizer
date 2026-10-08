"""The default label contract: typeset when verified, keep the ink on disagreement.

Under the earlier rule any review reason preserved the ink, and because the
formula worker reports no probability every formula label was kept as pixel
contours -- a traced bitmap. These tests pin the new default and the two flags
that move it, plus two curve-report defects found in the same review.
"""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from hybrid_vectorizer import ir
from hybrid_vectorizer.components import extract
from hybrid_vectorizer.convert import Options, build_geometry, build_labels, foreign_script
from hybrid_vectorizer.ocr import Reading
from hybrid_vectorizer.refine import build_font_set
from hybrid_vectorizer.textlayout import Block
from hybrid_vectorizer.tracing import Trace, trace_strokes


class ForeignScriptTest(unittest.TestCase):
    def test_latin_greek_and_latex_are_clean(self):
        for text in ['λ_0', 'x+iy', r'\frac{a}{b}', 'Imaginary', r'\alpha^2 + \beta', 'ℏω', 'é']:
            self.assertEqual(foreign_script(text), [], text)

    def test_cjk_cyrillic_and_arabic_letters_are_reported(self):
        self.assertEqual(foreign_script('x 中文'), ['中', '文'])
        self.assertEqual(foreign_script('ひらがな'), list('ひらがな'))
        self.assertEqual(foreign_script('한'), ['한'])
        self.assertEqual(foreign_script('Жx'), ['Ж'])
        self.assertEqual(foreign_script('ب'), ['ب'])
        self.assertEqual(foreign_script('א'), ['א'])
        self.assertEqual(foreign_script('ก'), ['ก'])
        self.assertEqual(foreign_script('क'), ['क'])


class LabelDefaultTest(unittest.TestCase):
    def setUp(self):
        self.fonts = build_font_set('DejaVu Serif', bold=False)
        if self.fonts is None:
            self.skipTest('test font unavailable')

    def analysis(self, reading):
        ink = np.zeros((100, 100), np.uint8)
        cv2.putText(ink, 'x', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        return SimpleNamespace(
            page=SimpleNamespace(ink=ink), blocks=[Block(components=extract(ink))],
            readings={0: reading}, prepared=[], rotations={}, alternatives={},
            text_height=20.0, rejected=[], traces=[], pens=[],
        )

    def test_default_typesets_a_reading_without_confidence_and_records_the_flag(self):
        analysis = self.analysis(Reading('x', 'formulaocr', None))
        labels = build_labels(analysis, self.fonts, Options())
        self.assertEqual(len(labels), 1)
        self.assertIsInstance(labels[0], ir.Label)
        self.assertEqual(labels[0].plain, 'x')
        self.assertIn('recognition confidence unavailable', labels[0].review_reasons)
        report = analysis.label_reports[0]
        self.assertEqual(report['representation'], 'text')
        self.assertIn('recognition confidence unavailable', report['review_reasons'])
        self.assertIsNone(report['reading_confidence'])
        # The element's confidence is the shape-based estimate, not 0.00.
        self.assertAlmostEqual(labels[0].confidence, min(0.99, 0.35 + 0.9 * report['shape_score']), places=2)
        self.assertGreater(labels[0].confidence, 0.35)

    def test_preserve_uncertain_keeps_the_ink_of_a_flagged_label(self):
        analysis = self.analysis(Reading('x', 'formulaocr', None))
        labels = build_labels(analysis, self.fonts, Options(preserve_uncertain=True))
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertIn('recognition confidence unavailable', labels[0].review_reasons)
        self.assertEqual(analysis.label_reports[0]['representation'], 'vector')

    def test_low_confidence_is_a_flag_not_a_preservation_trigger(self):
        analysis = self.analysis(Reading('x', 'formulaocr', 0.3))
        labels = build_labels(analysis, self.fonts, Options())
        self.assertIsInstance(labels[0], ir.Label)
        self.assertIn('low recognition confidence', labels[0].review_reasons)
        self.assertAlmostEqual(labels[0].confidence, 0.3)

    def test_foreign_script_reading_is_preserved(self):
        analysis = self.analysis(Reading('中文', 'formulaocr', None))
        labels = build_labels(analysis, self.fonts, Options())
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertEqual(labels[0].review_reasons, ['foreign script in reading'])
        report = analysis.label_reports[0]
        self.assertEqual(report['source_reading'], '中文')
        self.assertEqual(report['representation'], 'vector')
        # Not even the permissive flag typesets a script the page cannot carry.
        labels = build_labels(analysis, self.fonts, Options(typeset_uncertain=True))
        self.assertIsInstance(labels[0], ir.VectorFallback)

    def test_poor_shape_match_preserves_unless_typesetting_is_forced(self):
        # 'x' ink read as 'O': the ring set at the mark's size scores 0.13.
        analysis = self.analysis(Reading('O', 'formulaocr', None))
        labels = build_labels(analysis, self.fonts, Options())
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertIn('poor visual match', labels[0].review_reasons)
        self.assertLess(analysis.label_reports[0]['shape_score'], 0.20)
        labels = build_labels(analysis, self.fonts, Options(typeset_uncertain=True))
        self.assertIsInstance(labels[0], ir.Label)
        self.assertIn('poor visual match', labels[0].review_reasons)

    def test_unknown_commands_are_preserved_whatever_the_flags(self):
        analysis = self.analysis(Reading(r'x\unknownsymbol', 'formulaocr', None))
        for options in (Options(), Options(typeset_uncertain=True)):
            labels = build_labels(analysis, self.fonts, options)
            self.assertIsInstance(labels[0], ir.VectorFallback)
            self.assertIn('unsupported formula commands', labels[0].review_reasons)

    def test_raster_fallback_changes_the_representation_not_the_decision(self):
        analysis = self.analysis(Reading('x', 'formulaocr', None))
        labels = build_labels(analysis, self.fonts, Options(raster_fallback=True))
        self.assertIsInstance(labels[0], ir.Label)
        labels = build_labels(analysis, self.fonts, Options(raster_fallback=True, preserve_uncertain=True))
        self.assertIsInstance(labels[0], ir.RasterFallback)


class CurveReportTest(unittest.TestCase):
    @staticmethod
    def geometry(traces, ink):
        analysis = SimpleNamespace(page=SimpleNamespace(ink=ink, stroke_width=2.0),
                                   traces=traces, regions=[], rules=[], marker_sets=[],
                                   dashed=[], frames=[], ticks=[])
        return build_geometry(analysis, Options())

    def test_a_trace_ending_just_past_a_contact_keeps_its_tail(self):
        xs = np.concatenate([np.arange(20, 48, dtype=float), np.array([53.0, 54.0])])
        points = np.column_stack([xs, 100.0 - (xs - 50.0) ** 2 / 20.0])
        trace = Trace(points=points, stroke_width=2.0, contacts=[(50.0, 100.0)])
        geometry, notes = self.geometry([trace], np.zeros((200, 200), np.uint8))
        path = geometry[0].path
        self.assertIn('50 100', path)
        last = path.split('C')[-1].split()
        self.assertAlmostEqual(float(last[-2]), 54.0, places=1)
        self.assertEqual(notes[0]['rule_contacts'], [[50.0, 100.0]])

    def test_a_drawn_arc_is_reported_as_a_circle(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.circle(image, (200, 200), 120, 255, 5)
        traces = trace_strokes(max(extract(image), key=lambda component: component.area), shortest=15.0)
        geometry, notes = self.geometry(traces, image)
        self.assertIn(' A', geometry[0].path)
        self.assertEqual(notes[0]['primitive'], 'arc')
        self.assertEqual(notes[0]['analytic_model'], 'circle')
        self.assertTrue(notes[0]['within_tolerance'])
        self.assertIsNotNone(notes[0]['analytic_residual_px'])
        self.assertLess(notes[0]['analytic_residual_px'], 2.0)
        self.assertTrue(geometry[0].model.startswith('circle r='))
        self.assertAlmostEqual(float(geometry[0].model.split('r=')[1]), 120.0, delta=3.0)
        self.assertAlmostEqual(geometry[0].model_rms, notes[0]['analytic_residual_px'], places=2)


if __name__ == '__main__':
    unittest.main()
