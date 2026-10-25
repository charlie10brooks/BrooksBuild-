import unittest

import numpy as np

from app.core.dtm import GridSpec
from app.core.planning import bilinear_sample


class BilinearSampleTests(unittest.TestCase):
    def test_mixed_inside_and_outside_points(self):
        g = GridSpec(0.0, 10.0, 1.0, 10, 10)                       # x 0..10, y 0..10
        arr = np.fromfunction(lambda r, c: c * 1.0, (10, 10))      # value = column index
        v = bilinear_sample(arr, g, np.array([5.0, 50.0, 3.5, -4.0]), np.array([5.0, 5.0, 5.0, 5.0]))
        self.assertAlmostEqual(v[0], 4.5, places=6)                # cell-centre convention
        self.assertTrue(np.isnan(v[1]) and np.isnan(v[3]))        # outside: NaN, not invented
        self.assertAlmostEqual(v[2], 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
