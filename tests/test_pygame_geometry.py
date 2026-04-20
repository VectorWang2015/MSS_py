import math
import unittest

import numpy as np

from demo.pygame_demo import DemoRuntimeConfig, PygameOSVDemo


class TestPygameGeometry(unittest.TestCase):
    def setUp(self):
        self.demo = PygameOSVDemo(
            DemoRuntimeConfig(width=1000, height=800, map_scale_px_per_m=1.0)
        )

    def test_heading_psi_zero_points_north(self):
        nose, _, _ = self.demo._compute_left_ship_triangle(psi=0.0)
        _, cy = self.demo._left_center()
        self.assertLess(nose[1], cy)

    def test_heading_psi_90deg_points_east(self):
        nose, _, _ = self.demo._compute_left_ship_triangle(psi=math.pi / 2.0)
        cx, _ = self.demo._left_center()
        self.assertGreater(nose[0], cx)

    def test_ship_drawn_large_enough_to_see_shape(self):
        nose, _, _ = self.demo._compute_left_ship_triangle(psi=0.0)
        _, cy = self.demo._left_center()
        ship_length_px = abs(cy - nose[1])
        self.assertGreater(ship_length_px, 40)

    def test_thruster_vector_geometry_available(self):
        vectors = self.demo._compute_thruster_vectors_body(control=np.zeros(6))
        self.assertEqual(len(vectors), 4)

    def test_world_to_screen_keeps_ship_centered(self):
        s1 = self.demo._map_world_to_screen(10.0, -5.0, ref_n=10.0, ref_e=-5.0)
        s2 = self.demo._left_center()
        self.assertAlmostEqual(s1[0], s2[0])
        self.assertAlmostEqual(s1[1], s2[1])


if __name__ == "__main__":
    unittest.main()
