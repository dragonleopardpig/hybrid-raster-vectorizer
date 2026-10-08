"""Skeleton-graph tracing: crossings that thinning splits, outlines that a
third branch would open, and the length filters.

Every figure here failed on the raw graph: an acute crossing came back as 3
strokes at pens 3, 5 and 8, the +/-30 grid as 255 strokes, the tangent
circle as an open arc and the rectangle with a tail as one open stroke.
"""

import math
import unittest

import cv2
import numpy as np

from hybrid_vectorizer.strokegraph import strokes


def chord_deviation(points: np.ndarray) -> float:
    """Largest distance of any point from the straight line through the ends."""
    chord = points[-1] - points[0]
    normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    return float(np.max(np.abs((points - points[0]) @ normal)))


def is_closed(points: np.ndarray) -> bool:
    return bool(np.array_equal(points[0], points[-1]))


def horizontal_line(pen: int) -> np.ndarray:
    """A 360 px horizontal line through (200, 200)."""
    image = np.zeros((400, 400), np.uint8)
    cv2.line(image, (20, 200), (380, 200), 255, pen)
    return image


def oblique_line(angle_degrees: float, pen: int) -> np.ndarray:
    """A 340 px line through (200, 200) at the angle to the horizontal."""
    image = np.zeros((400, 400), np.uint8)
    along = math.cos(math.radians(angle_degrees)), math.sin(math.radians(angle_degrees))
    cv2.line(image, (int(200 - 170 * along[0]), int(200 + 170 * along[1])),
             (int(200 + 170 * along[0]), int(200 - 170 * along[1])), 255, pen)
    return image


def crossing(angle_degrees: float, pen: int) -> np.ndarray:
    return horizontal_line(pen) | oblique_line(angle_degrees, pen)


def orthogonal_grid() -> np.ndarray:
    """20 horizontal and 20 vertical 3 px lines, 40 px apart, overhanging the
    outer crossings so no line ends in a rounded corner."""
    image = np.zeros((900, 900), np.uint8)
    for index in range(20):
        coordinate = 70 + index * 40
        cv2.line(image, (40, coordinate), (860, coordinate), 255, 3)
        cv2.line(image, (coordinate, 40), (coordinate, 860), 255, 3)
    return image


def oblique_grid(angle_degrees: float, count: int = 15, spacing: int = 40) -> np.ndarray:
    """``count`` 3 px lines at +angle and ``count`` at -angle, every crossing
    inside the image and every line overhanging its outermost crossing."""
    size = 1400
    image = np.zeros((size, size), np.uint8)
    centre = np.array([size / 2, size / 2])
    for sign in (1, -1):
        angle = math.radians(sign * angle_degrees)
        direction = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        for index in range(count):
            base = centre + (index - (count - 1) / 2) * spacing * normal
            first, second = base - 520 * direction, base + 520 * direction
            cv2.line(image, tuple(int(round(value)) for value in first),
                     tuple(int(round(value)) for value in second), 255, 3)
    return image


class SplitJunctionTest(unittest.TestCase):
    def test_acute_crossings_return_two_straight_full_length_strokes(self):
        for pen in (3, 5, 8):
            for angle in (60, 45, 30, 20):
                with self.subTest(pen=pen, angle=angle):
                    result = strokes(crossing(angle, pen), (0, 0), 3.0 * pen)
                    self.assertEqual(len(result), 2)
                    spans = sorted(float(np.linalg.norm(points[-1] - points[0])) for points in result)
                    self.assertGreater(spans[0], 330.0)
                    self.assertGreater(spans[1], 355.0)
                    for points in result:
                        self.assertLessEqual(chord_deviation(points), 2.0)

    def test_merged_crossing_is_centred_on_the_ink_overlap(self):
        """cv2 rounds the oblique line's endpoints, so the drawn lines cross
        about 2 px left of the nominal (200, 200); the merged centre must
        follow the ink, not the nominal point."""
        overlap = np.argwhere((horizontal_line(5) > 0) & (oblique_line(20, 5) > 0))
        true_crossing = overlap.mean(axis=0)[::-1]
        result = strokes(crossing(20, 5), (0, 0), 15.0)
        self.assertEqual(len(result), 2)
        for points in result:
            self.assertLess(float(np.min(np.linalg.norm(points - true_crossing, axis=1))), 1.0)

    def test_oblique_grid_returns_one_stroke_per_line(self):
        result = strokes(oblique_grid(30), (0, 0), 9.0)
        self.assertEqual(len(result), 30)
        for points in result:
            self.assertGreater(float(np.linalg.norm(points[-1] - points[0])), 1000.0)

    def test_orthogonal_grid_still_returns_one_stroke_per_line(self):
        result = strokes(orthogonal_grid(), (0, 0), 9.0)
        self.assertEqual(len(result), 40)
        for points in result:
            self.assertGreater(float(np.linalg.norm(points[-1] - points[0])), 800.0)


class LoopFirstPairingTest(unittest.TestCase):
    def test_tangent_line_does_not_open_the_circle(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.circle(image, (200, 200), 80, 255, 3)
        cv2.line(image, (50, 120), (350, 120), 255, 3)
        result = strokes(image, (0, 0), 9.0)
        self.assertEqual(len(result), 2)
        closed = [points for points in result if is_closed(points)]
        self.assertEqual(len(closed), 1)
        radii = np.linalg.norm(closed[0] - [200.0, 200.0], axis=1)
        self.assertLess(float(np.max(np.abs(radii - 80.0))), 4.0)
        away_from_shared_run = closed[0][:, 1] > 130.0
        self.assertLess(float(np.max(np.abs(radii[away_from_shared_run] - 80.0))), 1.5)
        line = next(points for points in result if not is_closed(points))
        self.assertLessEqual(chord_deviation(line), 2.0)
        self.assertGreater(float(np.linalg.norm(line[-1] - line[0])), 290.0)

    def test_corner_tail_does_not_open_the_rectangle(self):
        image = np.zeros((400, 400), np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), 255, 3)
        cv2.line(image, (300, 100), (380, 20), 255, 3)
        result = strokes(image, (0, 0), 9.0)
        self.assertEqual(len(result), 2)
        closed = [points for points in result if is_closed(points)]
        self.assertEqual(len(closed), 1)
        corners = cv2.approxPolyDP(closed[0][:-1].astype(np.float32), 6.0, True)
        self.assertEqual(len(corners), 4)
        tail = next(points for points in result if not is_closed(points))
        self.assertLessEqual(chord_deviation(tail), 2.0)
        self.assertGreater(float(np.linalg.norm(tail[-1] - tail[0])), 105.0)

    def test_tee_plus_and_cross_are_unchanged(self):
        tee = np.zeros((400, 400), np.uint8)
        cv2.line(tee, (40, 200), (360, 200), 255, 5)
        cv2.line(tee, (200, 40), (200, 200), 255, 5)
        plus = np.zeros((400, 400), np.uint8)
        cv2.line(plus, (40, 200), (360, 200), 255, 5)
        cv2.line(plus, (200, 40), (200, 360), 255, 5)
        cross = np.zeros((400, 400), np.uint8)
        cv2.line(cross, (50, 50), (350, 350), 255, 5)
        cv2.line(cross, (50, 350), (350, 50), 255, 5)
        for name, image in (('tee', tee), ('plus', plus), ('cross', cross)):
            with self.subTest(figure=name):
                result = strokes(image, (0, 0), 15.0)
                self.assertEqual(len(result), 2)
                for points in result:
                    self.assertFalse(is_closed(points))
                    self.assertLessEqual(chord_deviation(points), 2.0)


class GeometricLengthTest(unittest.TestCase):
    def diamond_ring(self) -> np.ndarray:
        """24 skeleton pixels, all diagonal steps: 33.9 px of path."""
        image = np.zeros((40, 40), np.uint8)
        for row in range(40):
            for column in range(40):
                if abs(row - 20) + abs(column - 20) == 6:
                    image[row, column] = 255
        return image

    def test_bare_cycle_is_filtered_by_path_length_not_pixel_count(self):
        kept = strokes(self.diamond_ring(), (0, 0), 30.0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(is_closed(kept[0]))
        self.assertEqual(len(kept[0]), 25)
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(kept[0], axis=0), axis=1).sum()), 24 * math.sqrt(2), places=6)
        self.assertEqual(strokes(self.diamond_ring(), (0, 0), 35.0), [])


if __name__ == '__main__':
    unittest.main()
