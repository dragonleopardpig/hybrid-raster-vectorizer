import io
import json
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hybrid_vectorizer import ir
from hybrid_vectorizer.components import extract
from hybrid_vectorizer.convert import Options, analyse, build_geometry, build_labels, convert
from hybrid_vectorizer.ocr import FormulaReader, Reading
from hybrid_vectorizer.preserve import mask_path
from hybrid_vectorizer.refine import agreement, build_font_set, rasterise
from hybrid_vectorizer.textlayout import Block
from hybrid_vectorizer.tracing import nearest_pen, straighten, trace_strokes


ROOT = Path(__file__).resolve().parents[1]


class ConnectivityTest(unittest.TestCase):
    def test_a_thin_axis_is_not_forced_to_use_the_curve_pen(self):
        self.assertEqual(nearest_pen(5.0, [7.9]), 5.0)
        self.assertEqual(nearest_pen(7.5, [7.9]), 7.9)

    def trace(self, image):
        return trace_strokes(max(extract(image), key=lambda component: component.area), shortest=15.0)

    def assert_straight(self, trace, tolerance=2.0):
        direction = trace.end - trace.start
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        self.assertLessEqual(float(np.max(np.abs((trace.points - trace.start) @ normal))), tolerance)

    def test_plus_pairs_opposite_arms(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.line(image, (40, 200), (360, 200), 255, 5)
        cv2.line(image, (200, 40), (200, 360), 255, 5)
        traces = self.trace(image)
        self.assertEqual(len(traces), 2)
        for trace in traces:
            self.assert_straight(trace)
            self.assertGreater(np.linalg.norm(trace.end - trace.start), 310)

    def test_diagonal_cross_pairs_opposite_arms(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.line(image, (50, 50), (350, 350), 255, 5)
        cv2.line(image, (50, 350), (350, 50), 255, 5)
        traces = self.trace(image)
        self.assertEqual(len(traces), 2)
        for trace in traces:
            self.assert_straight(trace)
            self.assertGreater(np.linalg.norm(trace.end - trace.start), 415)

    def test_tee_keeps_the_stem_at_the_crossbar(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.line(image, (40, 200), (360, 200), 255, 5)
        cv2.line(image, (200, 40), (200, 200), 255, 5)
        traces = self.trace(image)
        self.assertEqual(len(traces), 2)
        for trace in traces:
            self.assert_straight(trace, tolerance=3.0)
        stem = min(traces, key=lambda trace: np.linalg.norm(trace.end - trace.start))
        self.assertLess(min(np.linalg.norm(stem.start - [200, 200]),
                            np.linalg.norm(stem.end - [200, 200])), 3.0)

    def test_circle_is_one_closed_stroke_and_svg_arc(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.circle(image, (200, 200), 120, 255, 5)
        traces = self.trace(image)
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0].closed)
        np.testing.assert_array_equal(traces[0].start, traces[0].end)
        analysis = SimpleNamespace(page=SimpleNamespace(ink=image, stroke_width=5.0),
                                   traces=traces, regions=[], rules=[], marker_sets=[],
                                   dashed=[], frames=[], ticks=[])
        geometry, notes = build_geometry(analysis, Options())
        self.assertTrue(geometry[0].path.endswith('Z'))
        self.assertIn(' A', geometry[0].path)
        self.assertEqual(notes[0]['primitive'], 'arc')
        self.assertFalse(geometry[0].arrow_start or geometry[0].arrow_end)

    def test_an_ellipse_remains_a_single_closed_path(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.ellipse(image, (200, 200), (140, 70), 25, 0, 360, 255, 5)
        traces = self.trace(image)
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0].closed)

    def test_straightening_keeps_a_genuine_arrowhead(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.line(image, (50, 200), (350, 200), 255, 5)
        cv2.fillPoly(image, [np.array([[350, 200], [325, 186], [325, 214]])], 255)
        traces = straighten(self.trace(image), tolerance=6.0)
        self.assertEqual(len(traces[0].points), 2)
        analysis = SimpleNamespace(page=SimpleNamespace(ink=image, stroke_width=5.0),
                                   traces=traces, regions=[], rules=[], marker_sets=[],
                                   dashed=[], frames=[], ticks=[])
        geometry, _notes = build_geometry(analysis, Options())
        self.assertEqual(sum(element.arrow_start + element.arrow_end for element in geometry), 1)

    def test_erased_axis_contacts_are_retained_as_curve_constraints(self):
        analysis = analyse(ROOT / 'examples/interference/raster.png', Options())
        trace = analysis.traces[0]
        self.assertEqual(len(trace.contacts), 6)
        geometry, notes = build_geometry(analysis, Options())
        curve = next(element for element in geometry if isinstance(element, ir.Curve))
        self.assertFalse(curve.arrow_start or curve.arrow_end)
        for column, row in trace.contacts:
            self.assertIn(f'{column:.2f}'.rstrip('0').rstrip('.'), curve.path)
            self.assertIn(f'{row:.2f}'.rstrip('0').rstrip('.'), curve.path)
        controls = [np.array([float(value) for value in command.split()]).reshape(3, 2)
                    for command in re.findall(r'C([^CMLZ]+)', curve.path)]
        for contact in trace.contacts:
            incoming = next(index for index, control in enumerate(controls)
                            if np.linalg.norm(control[-1] - contact) < 0.02)
            self.assertAlmostEqual(controls[incoming][-2, 1], contact[1], places=2)
            self.assertAlmostEqual(controls[incoming + 1][0, 1], contact[1], places=2)
            self.assertLess(controls[incoming][-2, 0], contact[0])
            self.assertGreater(controls[incoming + 1][0, 0], contact[0])
        self.assertEqual(len(notes[0]['rule_contacts']), 6)


class PreservationTest(unittest.TestCase):
    def test_vector_contours_keep_holes_dots_and_diagonal_contacts(self):
        ink = np.zeros((40, 40), np.uint8)
        cv2.rectangle(ink, (4, 4), (22, 22), 255, 3)
        ink[29, 29] = ink[30, 30] = ink[35, 8] = 255
        document = ir.Document(width=40, height=40, labels=[
            ir.VectorFallback(kind='preserved', path=mask_path(ink))
        ])
        rendered = rasterise(document.to_svg(), 40, 40)
        if rendered is None:
            self.skipTest('SVG rasterizer unavailable')
        np.testing.assert_array_equal(rendered, ink)

    def analysis(self, reading):
        ink = np.zeros((100, 100), np.uint8)
        cv2.putText(ink, 'x', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        return SimpleNamespace(
            page=SimpleNamespace(ink=ink), blocks=[Block(components=extract(ink))],
            readings={0: reading}, prepared=[], rotations={}, alternatives={},
            text_height=20.0, rejected=[], traces=[], pens=[],
        )

    def test_empty_reading_is_preserved_without_fonts(self):
        analysis = self.analysis(Reading('', 'formulaocr', 0.0))
        labels = build_labels(analysis, None, Options())
        self.assertEqual(len(labels), 1)
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertTrue(labels[0].review_reasons)
        self.assertEqual(analysis.label_reports[0]['representation'], 'vector')

    def test_empty_reading_reaches_requested_raster_fallback(self):
        analysis = self.analysis(Reading('', 'formulaocr', 0.0))
        labels = build_labels(analysis, object(), Options(raster_fallback=True))
        self.assertEqual(len(labels), 1)
        self.assertIsInstance(labels[0], ir.RasterFallback)

    def test_unknown_confidence_is_not_upgraded_by_shape_match(self):
        fonts = build_font_set('DejaVu Serif', bold=False)
        if fonts is None:
            self.skipTest('test font unavailable')
        analysis = self.analysis(Reading('x', 'formulaocr', None))
        labels = build_labels(analysis, fonts, Options())
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertIsNone(analysis.label_reports[0]['reading_confidence'])
        self.assertIn('recognition confidence unavailable', labels[0].review_reasons)
        labels = build_labels(analysis, fonts, Options(typeset_uncertain=True))
        self.assertIsInstance(labels[0], ir.Label)
        self.assertIn('recognition confidence unavailable', labels[0].review_reasons)

    def test_confident_isolated_letters_are_not_proof_of_symbol_identity(self):
        fonts = build_font_set('DejaVu Serif', bold=False)
        if fonts is None:
            self.skipTest('test font unavailable')
        analysis = self.analysis(Reading('A', 'tesseract', 0.99))
        labels = build_labels(analysis, fonts, Options())
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertIn('isolated letter may be a mathematical symbol', labels[0].review_reasons)

    def test_unknown_commands_cannot_silently_disappear(self):
        fonts = build_font_set('DejaVu Serif', bold=False)
        if fonts is None:
            self.skipTest('test font unavailable')
        analysis = self.analysis(Reading(r'x\unknownsymbol', 'formulaocr', 0.99))
        labels = build_labels(analysis, fonts, Options(typeset_uncertain=True))
        self.assertIsInstance(labels[0], ir.VectorFallback)
        self.assertEqual(analysis.label_reports[0]['unknown_commands'], ['unknownsymbol'])


class RecognitionTest(unittest.TestCase):
    def test_formula_worker_does_not_invent_a_probability(self):
        reader = FormulaReader(command='unused')
        reader._directory = SimpleNamespace(name='/tmp')
        reader._process = SimpleNamespace(
            stdin=io.StringIO(), stdout=io.StringIO(json.dumps({'ok': True, 'formula': 'x'}) + '\n')
        )
        with patch('hybrid_vectorizer.ocr.cv2.imwrite', return_value=True):
            reading = reader.read(np.zeros((5, 5), np.uint8))
        self.assertIsNone(reading.confidence)

    def test_unsupported_marker_requires_its_actual_shape(self):
        marker = ir.MarkerField(kind='markers', shape='freeform')
        with self.assertRaises(ValueError):
            marker.defs()
        marker.path = 'M0 0 L10 0 L5 5 Z'
        self.assertIn('<path', ''.join(marker.defs()))
        self.assertNotIn('<circle', ''.join(marker.defs()))

    def test_wave_labels_are_not_marker_series(self):
        analysis = analyse(ROOT / 'examples/waves1.png', Options())
        self.assertEqual(analysis.marker_sets, [])
        expected = [(1741, 278), (1740.5, 768), (1741, 1240),
                    (510, 64.5), (518, 626), (516.5, 1083.5)]
        for column, row in expected:
            self.assertTrue(any(block.x <= column < block.right and block.y <= row < block.bottom
                                for block in analysis.blocks), (column, row))


class VerificationTest(unittest.TestCase):
    def test_conversion_keeps_unread_labels_and_reports_them(self):
        image = np.full((200, 400), 255, np.uint8)
        cv2.putText(image, 'unknown', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'text.png'
            cv2.imwrite(str(source), image)
            with patch('hybrid_vectorizer.convert.read_blocks'):
                document = convert(source, Options(font_family='DejaVu Serif'))
        report = document.report
        self.assertGreater(report['quality']['label_blocks'], 0)
        self.assertEqual(report['quality']['label_blocks'], report['quality']['labels_accounted_for'])
        self.assertEqual(report['quality']['typeset_labels'], 0)
        self.assertTrue(all(label['id'] in report['needs_review'] for label in report['labels']))
        root = ET.fromstring(document.to_svg())
        self.assertEqual(root.findall('.//{http://www.w3.org/2000/svg}image'), [])
        self.assertTrue(root.findall('.//{http://www.w3.org/2000/svg}path[@class="preserved-ink"]'))
        if 'agreement' in report:
            self.assertGreater(report['agreement']['exact_recall'], 0.99)

    def test_semantic_scoring_checks_readings_even_when_the_ink_is_preserved(self):
        from hybrid_vectorizer.evaluation import evaluate

        reference = {'counts': {'marker_sets': 0}, 'labels': [
            {'box': [10, 10, 20, 20], 'latex': r'\delta'}
        ]}
        report = {'marker_series': [], 'labels': [
            {'box': [10, 10, 20, 20], 'source_reading': 's', 'engine': 'formulaocr',
             'representation': 'vector', 'id': 'label-0'}
        ]}
        scores = evaluate(report, reference)
        self.assertEqual(scores['exact_ocr_labels'], 0)
        self.assertEqual(scores['character_error_rate'], 1.0)
        self.assertTrue(scores['object_counts']['marker_sets']['correct'])

    def test_missing_reference_labels_are_errors(self):
        from hybrid_vectorizer.evaluation import evaluate

        scores = evaluate({'labels': []}, {'labels': [{'box': [0, 0, 10, 10], 'text': 'x'}]})
        self.assertEqual(scores['matched_labels'], 0)
        self.assertEqual(scores['exact_ocr_labels'], 0)
        self.assertEqual(scores['character_error_rate'], 1.0)

    def test_tolerant_overlap_does_not_conceal_exact_displacement(self):
        source = np.zeros((100, 100), np.uint8)
        rendered = np.zeros_like(source)
        source[40, 10:91] = 255
        rendered[43, 10:91] = 255
        scores = agreement(source, rendered)
        self.assertEqual(scores['recall'], 1.0)
        self.assertEqual(scores['exact_recall'], 0.0)
        self.assertEqual(scores['exact_precision'], 0.0)
        self.assertAlmostEqual(scores['source_distance_mean_px'], 3.0)

    def test_missing_objects_are_reported_by_location_and_size(self):
        source = np.zeros((100, 100), np.uint8)
        source[10:20, 10:20] = 255
        source[60:80, 60:80] = 255
        rendered = source.copy()
        rendered[60:80, 60:80] = 0
        scores = agreement(source, rendered)
        self.assertEqual(scores['missing_ink_pixels'], 400)
        self.assertEqual(scores['missing_components'], 1)
        self.assertEqual(scores['largest_missing_components'][0]['box'], [60, 60, 20, 20])


if __name__ == '__main__':
    unittest.main()
