import unittest

import numpy as np

from hybrid_vectorizer.primitives import Rule
from hybrid_vectorizer.tracing import Trace, nearest_pen, restore_contacts, smooth_path


def parabola_trace(curvature, *, rule_height=200.0, band=3.0, stroke_width=4.0, vertex_x=300.0):
    """A column trace of y = rule_height - curvature (x - vertex_x)^2 with the rule's band erased.

    The band is what erasing a horizontal rule of thickness ~2*band removes, so the
    trace arrives with a gap around the vertex exactly as the converter hands it on.
    """
    x = np.arange(200.0, 401.0)
    y = rule_height - curvature * (x - vertex_x) ** 2
    keep = np.abs(y - rule_height) > band
    return Trace(points=np.column_stack([x[keep], y[keep]]), stroke_width=stroke_width, method="column")


def horizontal_rule(position, thickness=3.0):
    return Rule(orientation="horizontal", position=position, start=0.0, end=600.0, thickness=thickness)


class RestoreContactsTest(unittest.TestCase):
    def assert_single_contact_near(self, trace, expected, tolerance=1.0):
        self.assertEqual(len(trace.contacts), 1, trace.contacts)
        contact = trace.contacts[0]
        self.assertLessEqual(abs(contact[0] - expected[0]), tolerance, contact)
        self.assertLessEqual(abs(contact[1] - expected[1]), tolerance, contact)

    def test_a_gentle_contact_on_the_axis_is_recovered(self):
        for curvature in (0.01, 0.005):
            with self.subTest(curvature=curvature):
                trace = parabola_trace(curvature)
                restore_contacts([trace], [horizontal_rule(200.0)])
                self.assert_single_contact_near(trace, (300.0, 200.0))

    def test_a_sharper_contact_is_still_recovered(self):
        trace = parabola_trace(0.05)
        restore_contacts([trace], [horizontal_rule(200.0)])
        self.assert_single_contact_near(trace, (300.0, 200.0))

    def test_a_vertex_outside_the_gap_is_rejected(self):
        # The curve crosses the rule on a slope: its parabola vertex is far to the
        # left of the gap, so the gap is a crossing, not a contact.
        x = np.arange(200.0, 401.0)
        y = 200.0 + (x - 300.0) + 0.01 * (x - 300.0) ** 2
        keep = np.abs(y - 200.0) > 3.0
        trace = Trace(points=np.column_stack([x[keep], y[keep]]), stroke_width=4.0, method="column")
        restore_contacts([trace], [horizontal_rule(200.0)])
        self.assertEqual(trace.contacts, [])

    def test_one_gap_yields_one_contact_from_the_nearest_rule(self):
        trace = parabola_trace(0.01)
        restore_contacts([trace], [horizontal_rule(201.0), horizontal_rule(200.0)])
        self.assertEqual(len(trace.contacts), 1, trace.contacts)
        self.assertEqual(trace.contacts[0][1], 200.0)
        self.assertLessEqual(abs(trace.contacts[0][0] - 300.0), 1.0)

    def test_a_rule_out_of_reach_contributes_nothing(self):
        trace = parabola_trace(0.01)
        restore_contacts([trace], [horizontal_rule(260.0)])
        self.assertEqual(trace.contacts, [])


class SmoothPathTest(unittest.TestCase):
    def closed_loop(self, count=41):
        angle = np.linspace(0.0, 2.0 * np.pi, count)
        points = np.column_stack([100.0 + 40.0 * np.cos(angle), 100.0 + 40.0 * np.sin(angle)])
        points[-1] = points[0]
        return points

    def test_a_closed_loop_keeps_its_point_count_for_an_even_window(self):
        points = self.closed_loop()
        smoothed = smooth_path(points, 6, closed=True)
        self.assertEqual(smoothed.shape, points.shape)
        np.testing.assert_array_equal(smoothed[0], smoothed[-1])

    def test_a_closed_loop_keeps_its_point_count_for_an_odd_window(self):
        points = self.closed_loop()
        smoothed = smooth_path(points, 7, closed=True)
        self.assertEqual(smoothed.shape, points.shape)
        np.testing.assert_array_equal(smoothed[0], smoothed[-1])

    def test_an_open_path_keeps_its_ends(self):
        x = np.arange(0.0, 50.0)
        points = np.column_stack([x, 10.0 + np.where(x.astype(int) % 2 == 0, 1.0, -1.0)])
        smoothed = smooth_path(points, 6)
        np.testing.assert_array_equal(smoothed[0], points[0])
        np.testing.assert_array_equal(smoothed[-1], points[-1])


class NearestPenTest(unittest.TestCase):
    def test_a_width_far_from_every_pen_keeps_its_own(self):
        self.assertEqual(nearest_pen(5.0, [7.9]), 5.0)
        # pen_set never splits a lighter group off the median: a minority of thin
        # axes beside heavy curves has only the curve pen and must keep its width.
        self.assertEqual(nearest_pen(3.0, [8.0]), 3.0)
        self.assertEqual(nearest_pen(3.9, [8.0]), 3.9)

    def test_a_width_within_the_group_spread_snaps(self):
        self.assertEqual(nearest_pen(7.5, [7.9]), 7.9)
        self.assertEqual(nearest_pen(6.0, [7.9]), 7.9)

    def test_a_degenerate_width_takes_the_lightest_pen(self):
        self.assertEqual(nearest_pen(1.0, [4.0, 8.0]), 4.0)
        self.assertEqual(nearest_pen(0.0, [4.0, 8.0]), 4.0)

    def test_a_tie_falls_to_the_lighter_pen(self):
        self.assertEqual(nearest_pen(5.5, [4.0, 7.0]), 4.0)
        self.assertEqual(nearest_pen(5.5, [7.0, 4.0]), 4.0)


if __name__ == "__main__":
    unittest.main()
