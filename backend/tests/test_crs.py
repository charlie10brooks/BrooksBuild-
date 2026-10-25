import unittest

import numpy as np

from app.core import crs


class CrsTests(unittest.TestCase):
    def test_known_points(self):
        # central meridian of zone 16 on the equator
        x, y = crs.lonlat_to_utm(-87.0, 0.0, 16, True)
        self.assertAlmostEqual(float(x), 500000.0, places=3)
        self.assertAlmostEqual(float(y), 0.0, places=3)
        # on the central meridian, northing = k0 * meridian arc length (independent numerical integration)
        import math
        a, e2 = crs.A, crs._E ** 2
        phis = np.linspace(0, math.radians(48.8584), 200001)
        arc = np.trapezoid(a * (1 - e2) / (1 - e2 * np.sin(phis) ** 2) ** 1.5, phis)
        x, y = crs.lonlat_to_utm(3.0, 48.8584, 31, True)
        self.assertAlmostEqual(float(x), 500000.0, places=3)
        self.assertAlmostEqual(float(y), 0.9996 * float(arc), places=3)
        # equator, 1 degree east of the central meridian: easting ~ 500000 + k0 * a * dlon (+ tiny series term)
        x, _ = crs.lonlat_to_utm(4.0, 0.0, 31, True)
        self.assertAlmostEqual(float(x), 611280.65, delta=0.05)
        # southern hemisphere false northing
        x, y = crs.lonlat_to_utm(18.4241, -33.9249, 34, False)       # Cape Town
        self.assertAlmostEqual(float(x), 261878, delta=30)
        self.assertAlmostEqual(float(y), 6243185, delta=30)

    def test_round_trip_is_millimetre_accurate(self):
        rng = np.random.default_rng(1)
        lon = rng.uniform(-90, -84, 500)
        lat = rng.uniform(-80, 80, 500)
        for north in (True, False):
            sel = (lat >= 0) == north
            x, y = crs.lonlat_to_utm(lon[sel], lat[sel], 16, north)
            lo, la = crs.utm_to_lonlat(x, y, 16, north)
            self.assertLess(float(np.abs(lo - lon[sel]).max()), 1e-8)
            self.assertLess(float(np.abs(la - lat[sel]).max()), 1e-8)

    def test_transformer_and_refusals(self):
        t = crs.transformer(4326, 32616)
        x, y = t(-86.1581, 39.7684)
        back = crs.transformer(32616, 4326)(x, y)
        self.assertAlmostEqual(float(back[0]), -86.1581, places=7)
        self.assertAlmostEqual(float(back[1]), 39.7684, places=7)
        self.assertEqual(float(crs.transformer(32616, 32616)(5.0, 6.0)[0]), 5.0)
        wm = crs.transformer(4326, 3857)(0.0, 0.0)
        self.assertAlmostEqual(float(wm[0]), 0.0, places=6)
        for bad in (26916, 2965):                       # NAD83 UTM 16N, Indiana State Plane
            with self.assertRaises(crs.UnsupportedCrs):
                crs.transformer(4326, bad)

    def test_geometry(self):
        g = {"type": "Polygon", "coordinates": [[[-86.0, 39.0], [-85.99, 39.0], [-85.99, 39.01], [-86.0, 39.0]]]}
        out = crs.transform_geometry(g, 4326, 32616)
        self.assertGreater(out["coordinates"][0][0][0], 100000)
        self.assertEqual(len(out["coordinates"][0]), 4)


if __name__ == "__main__":
    unittest.main()
