import unittest
from pathlib import Path

from hybrid_vectorizer.evaluation import (
    LABEL_SCHEMA,
    SCORABLE_COUNTS,
    evaluate,
    reading_spellings,
    score_reading,
    truth_spellings,
    unique_output_names,
    validate_reference_labels,
)


def _report(*labels: dict) -> dict:
    return {'marker_series': [{'count': 9}, {'count': 9}], 'curves': [{'closed': True, 'arrowheads': 1}],
            'labels': list(labels)}


class ObjectCountScoringTest(unittest.TestCase):
    def test_every_scorable_count_is_compared_to_the_report(self):
        reference = {'counts': {'marker_sets': 2, 'markers': 18, 'curves': 1, 'closed_curves': 1,
                                'curve_arrowheads': 2}}
        counts = evaluate(_report(), reference)['object_counts']
        self.assertEqual(set(counts), set(SCORABLE_COUNTS))
        self.assertTrue(all(counts[name]['correct'] for name in SCORABLE_COUNTS if name != 'curve_arrowheads'))
        self.assertEqual(counts['curve_arrowheads'], {'expected': 2, 'actual': 1, 'correct': False})

    def test_a_count_name_the_report_cannot_answer_is_unscorable_not_a_crash(self):
        # examples/mixed/truth.json uses markers_per_set and examples/series/truth.json
        # uses points_per_set; a bare dict index on either used to raise KeyError.
        reference = {'counts': {'markers_per_set': 9, 'marker_sets': 2}}
        counts = evaluate(_report(), reference)['object_counts']
        self.assertEqual(counts['markers_per_set'],
                         {'expected': 9, 'actual': None, 'correct': None, 'unscorable': True})
        self.assertEqual(counts['marker_sets'], {'expected': 2, 'actual': 2, 'correct': True})


class ReadingScoringTest(unittest.TestCase):
    def test_a_tesseract_fraction_scores_exact_against_a_latex_truth(self):
        # Truth '(λ)/(4)' versus a correct raw 'λ/4' used to cost four character errors.
        label = {'box': [0, 0, 10, 10], 'latex': r'\frac{\lambda}{4}'}
        candidate = {'box': [0, 0, 10, 10], 'source_reading': 'λ/4', 'engine': 'tesseract',
                     'representation': 'vector', 'id': 'label-0'}
        scores = evaluate({'labels': [candidate]}, {'labels': [label]})
        self.assertEqual(scores['exact_ocr_labels'], 1)
        self.assertEqual(scores['character_error_rate'], 0.0)
        self.assertEqual(scores['labels'][0]['expected'], '(λ)/(4)')
        self.assertEqual(scores['labels'][0]['reading'], 'λ/4')
        self.assertTrue(scores['labels'][0]['correct'])

    def test_a_tesseract_scripted_label_is_also_read_as_latex(self):
        label = {'box': [0, 0, 10, 10], 'latex': r'I_0'}
        self.assertEqual(truth_spellings(label), ['I_0'])
        self.assertIn('I_0', reading_spellings({'source_reading': 'I_{0}', 'engine': 'tesseract'}))
        detail = score_reading(label, {'source_reading': 'I_{0}', 'engine': 'tesseract'})
        self.assertTrue(detail['correct'])
        self.assertEqual(detail['character_errors'], 0)

    def test_only_single_token_groups_are_dropped_from_the_truth(self):
        self.assertEqual(truth_spellings({'latex': r'\frac{\lambda}{4}'}), ['(λ)/(4)', 'λ/4'])
        self.assertEqual(truth_spellings({'latex': r'\frac{3\lambda}{4}'}), ['(3λ)/(4)', '(3λ)/4'])
        self.assertEqual(truth_spellings({'text': 'Fig. 6-7'}), ['Fig.6-7'])
        # Literal parentheses are part of the label: a reading that drops them is wrong.
        self.assertEqual(truth_spellings({'text': '(a)'}), ['(a)'])
        self.assertEqual(truth_spellings({'latex': '(a)'}), ['(a)'])

    def test_a_formula_engine_reading_is_only_read_as_latex(self):
        self.assertEqual(reading_spellings({'source_reading': r'\frac{\lambda}{4}', 'engine': 'formulaocr'}),
                         ['(λ)/(4)'])

    def test_a_reading_with_an_unknown_command_is_not_offered_as_latex(self):
        spellings = reading_spellings({'source_reading': r'\nosuch{x}', 'engine': 'tesseract'})
        self.assertEqual(spellings, [r'\nosuch{x}'])

    def test_a_wrong_reading_still_counts_its_errors(self):
        detail = score_reading({'box': [0, 0, 1, 1], 'latex': r'\delta'},
                               {'source_reading': 's', 'engine': 'formulaocr'})
        self.assertFalse(detail['correct'])
        self.assertEqual(detail['character_errors'], 1)

    def test_a_missing_prediction_costs_the_whole_truth(self):
        detail = score_reading({'box': [0, 0, 1, 1], 'text': 'abc'}, {})
        self.assertEqual(detail['character_errors'], 3)
        self.assertEqual(detail['reading'], '')
        self.assertIsNone(detail['id'])

    def test_a_candidate_without_a_source_reading_scores_as_empty(self):
        # label_report always emits source_reading; the old candidate['text'] fallback
        # was dead code and a stray 'text' key must not be mistaken for a reading.
        detail = score_reading({'box': [0, 0, 1, 1], 'text': 'x'}, {'text': 'x', 'engine': 'tesseract'})
        self.assertFalse(detail['correct'])
        self.assertEqual(detail['character_errors'], 1)


class ReferenceValidationTest(unittest.TestCase):
    def test_a_well_formed_reference_passes(self):
        validate_reference_labels([{'box': [0, 0, 10, 10.5], 'text': 'x'},
                                   {'box': (1, 2, 3, 4), 'latex': r'\lambda'}], 'figure')

    def test_a_label_without_a_box_names_the_figure_index_and_schema(self):
        reference = {'labels': [{'box': [0, 0, 10, 10], 'text': 'x'}, {'text': 'y'}]}
        with self.assertRaises(ValueError) as caught:
            evaluate({'labels': []}, reference, figure='examples/waves1.png')
        message = str(caught.exception)
        self.assertIn('examples/waves1.png', message)
        self.assertIn('label 1', message)
        self.assertIn(LABEL_SCHEMA, message)

    def test_a_box_must_hold_exactly_four_numbers(self):
        for box in ([0, 0, 10], [0, 0, '10', 10], [0, 0, 10, True], None, 'box'):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_reference_labels([{'box': box, 'text': 'x'}])

    def test_a_label_needs_exactly_one_reading(self):
        with self.assertRaises(ValueError):
            validate_reference_labels([{'box': [0, 0, 1, 1]}])
        with self.assertRaises(ValueError):
            validate_reference_labels([{'box': [0, 0, 1, 1], 'text': 'x', 'latex': 'x'}])
        with self.assertRaises(ValueError):
            validate_reference_labels([{'box': [0, 0, 1, 1], 'text': 3}])

    def test_validation_runs_before_any_label_is_indexed(self):
        # Even with no predictions at all, the malformed truth is reported first.
        with self.assertRaises(ValueError):
            evaluate({}, {'labels': [{'latex': 'x'}]})


class OutputNameTest(unittest.TestCase):
    def test_generic_stems_take_their_directory_name(self):
        names = unique_output_names([Path('examples/mixed/figure.png'), Path('examples/series/raster.png'),
                                     Path('examples/waves1.png')])
        self.assertEqual(names, ['mixed', 'series', 'waves1'])
        # A lone generic stem takes its directory without needing a collision to grow.
        self.assertEqual(unique_output_names([Path('examples/mixed/figure.png')]), ['mixed'])
        self.assertEqual(unique_output_names([Path('examples/mixed/figure.png'), Path('examples/waves1.png')]),
                         ['mixed', 'waves1'])

    def test_colliding_stems_gain_their_parent_directory_instead_of_refusing(self):
        names = unique_output_names([Path('/data/mixed/figure.png'), Path('/data/other/mixed/figure.png'),
                                     Path('/a/x.png'), Path('/b/x.png')])
        self.assertEqual(names, ['data-mixed', 'other-mixed', 'a-x', 'b-x'])
        self.assertEqual(len(set(names)), len(names))

    def test_the_same_path_twice_is_still_told_apart(self):
        names = unique_output_names([Path('/a/x.png'), Path('/a/x.png')])
        self.assertEqual(len(set(names)), 2)


if __name__ == '__main__':
    unittest.main()
