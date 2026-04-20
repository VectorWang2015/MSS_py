import unittest

from MSS_simulator_py.osv.params import load_osv_custom_params
from demo.pygame_demo import PygameOSVDemo


class TestCustomParams(unittest.TestCase):
    def test_custom_params_use_symmetric_stern_limits(self):
        p = load_osv_custom_params()
        self.assertEqual(p.n_max[2], p.n_max[3])
        self.assertEqual(p.k_max[2], p.k_max[3])

    def test_demo_uses_custom_params_by_default(self):
        demo = PygameOSVDemo()
        self.assertEqual(demo.model.params.n_max[2], demo.model.params.n_max[3])


if __name__ == "__main__":
    unittest.main()
