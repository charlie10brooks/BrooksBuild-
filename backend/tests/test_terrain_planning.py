import unittest

import numpy as np

from app.core import dtm as D
from app.core import planning as P
from app.core import terrain_analysis as T
from app.core.vectorize import mask_to_polygons

G = D.GridSpec.from_bounds(0, 0, 40, 40, 1.0)


def xy_grid(g=G):
    r, c = np.mgrid[0:g.height, 0:g.width]
    return g.cell_centers_xy(r, c)


class SlopeTests(unittest.TestCase):
    def test_slope_and_aspect_on_planes(self):
        x, y = xy_grid()
        s, a = T.slope_aspect((100 + 0.1 * x).astype(np.float32), G.res)       # rises to the east
        self.assertAlmostEqual(float(np.nanmedian(s)), np.degrees(np.arctan(0.1)), places=2)
        self.assertAlmostEqual(float(np.nanmedian(a)), 270.0, delta=0.5)         # downslope faces west
        s2, a2 = T.slope_aspect((100 + 0.2 * y).astype(np.float32), G.res)       # rises to the north
        self.assertAlmostEqual(float(np.nanmedian(a2)), 180.0, delta=0.5)        # downslope faces south
        s3, a3 = T.slope_aspect(np.full(G.shape, 5.0, np.float32), G.res)
        self.assertEqual(float(np.nanmax(s3)), 0.0)
        self.assertTrue(np.isnan(a3).all())

    def test_nodata_is_not_invented(self):
        x, y = xy_grid()
        z = (100 + 0.1 * x).astype(np.float32)
        z[20, 20] = np.nan
        s, _ = T.slope_aspect(z, G.res)
        self.assertTrue(np.isnan(s[19:22, 19:22]).all())
        self.assertFalse(np.isnan(s[5, 5]))


class HydrologyTests(unittest.TestCase):
    def test_pit_is_filled_and_reported(self):
        x, y = xy_grid()
        z = (100 + 0.2 * y).astype(np.float32)           # drains south
        z[20, 20] -= 3.0                                  # a 3 m pit
        filled = T.fill_depressions(z)
        # a pit spills over its lowest rim: the south (downslope) neighbour, plus epsilon
        self.assertAlmostEqual(float(filled[20, 20]), float(z[21, 20]), delta=1e-3)
        self.assertTrue((filled >= z - 1e-6).all())       # fill never lowers anything
        rep = T.depression_report(z, filled, G.res)
        self.assertEqual(rep["count"], 1)
        self.assertGreater(rep["max_depth_m"], 2.5)
        d = T.flow_direction_d8(filled)
        interior = np.zeros(G.shape, bool); interior[1:-1, 1:-1] = True
        self.assertTrue((d[interior] >= 0).all(), "no sinks may remain after filling")

    def test_accumulation_on_south_sloping_plane(self):
        x, y = xy_grid()
        z = (100 + 0.1 * y).astype(np.float32)           # y increases north, so drains south (down rows)
        filled = T.fill_depressions(z)
        d = T.flow_direction_d8(filled)
        acc = T.flow_accumulation(filled, d)
        col = 17
        self.assertEqual(float(acc[0, col]), 1.0)                      # top row: nothing upstream but itself
        self.assertEqual(float(acc[G.height - 1, col]), float(G.height))
        # conservation: every cell drains somewhere, outlets hold all of the area
        outlets = (T._downstream_index(d) < 0).reshape(G.shape)
        self.assertEqual(float(acc[outlets].sum()), float(G.width * G.height))

    def test_valley_channel_is_detected(self):
        x, y = xy_grid()
        z = (100 + 0.05 * y + 0.3 * np.abs(x - 20)).astype(np.float32)    # V valley along x = 20
        res = T.analyse_drainage(z, G, min_channel_area_m2=300)
        segs = res["segments"]
        self.assertGreater(len(segs), 5)
        xs = np.array([s["coords"][0][0] for s in segs])
        self.assertLess(float(np.abs(xs - 20.5).mean()), 2.5)               # channel sits in the valley


class VectorizeTests(unittest.TestCase):
    def test_polygon_with_hole_and_area(self):
        m = np.zeros(G.shape, bool)
        m[10:30, 10:30] = True
        m[18:22, 18:22] = False
        polys = mask_to_polygons(m, G)
        self.assertEqual(len(polys), 1)
        self.assertEqual(len(polys[0]["coordinates"]), 2, "hole must be kept")
        outer = P.polygon_area(polys[0]["coordinates"][0])
        hole = P.polygon_area(polys[0]["coordinates"][1])
        self.assertAlmostEqual(outer - hole, m.sum(), delta=0.12 * m.sum())   # within ~half-cell outline shrink

    def test_small_specks_dropped(self):
        m = np.zeros(G.shape, bool); m[5, 5] = True
        self.assertEqual(mask_to_polygons(m, G, min_cells=4), [])


class PlanningTests(unittest.TestCase):
    def setUp(self):
        x, y = xy_grid()
        self.z = (100 + 0.1 * x).astype(np.float32)          # 10 % grade rising east
        self.conf = np.full(G.shape, D.CONF_HIGH, np.uint8)
        self.sq = [(10, 10), (30, 10), (30, 30), (10, 30)]

    def test_area_and_bilinear_exactness(self):
        self.assertAlmostEqual(P.polygon_area(self.sq), 400.0)
        v = P.bilinear_sample(self.z, G, np.array([20.3]), np.array([17.7]))
        self.assertAlmostEqual(float(v[0]), 100 + 0.1 * 20.3, places=4)
        zz = self.z.copy(); zz[22, 20] = np.nan
        self.assertTrue(np.isnan(P.bilinear_sample(zz, G, np.array([20.5]), np.array([17.5]))[0]))
        self.assertTrue(np.isnan(P.bilinear_sample(self.z, G, np.array([-5.0]), np.array([5.0]))[0]))

    def test_profile(self):
        pr = P.elevation_profile(self.z, G, [(5, 20), (35, 20)], spacing=1.0, conf=self.conf)
        self.assertAlmostEqual(pr["total_length"], 30.0)
        self.assertAlmostEqual(pr["max_z"] - pr["min_z"], 3.0, places=3)
        self.assertAlmostEqual(pr["max_grade_percent"], 10.0, places=2)
        self.assertTrue((pr["confidence"] == D.CONF_HIGH).all())
        with self.assertRaises(ValueError):
            P.elevation_profile(self.z, G, [(5, 5)])

    def test_profile_does_not_bridge_nodata(self):
        zz = self.z.copy(); zz[:, 18:22] = np.nan
        pr = P.elevation_profile(zz, G, [(5, 20), (35, 20)], spacing=1.0)
        self.assertTrue(np.isnan(pr["ground_z"]).any())
        self.assertLess(pr["valid_fraction"], 1.0)

    def test_footprint_report_on_known_plane(self):
        r = P.footprint_report(self.z, G, self.sq, conf=self.conf)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["mean_slope_percent"], 10.0, places=1)
        self.assertAlmostEqual(r["relief"], 2.0, delta=0.15)
        self.assertAlmostEqual(r["downslope_azimuth_deg"], 270.0, delta=1.0)
        self.assertLess(r["plane_residual_rms"], 1e-3)
        self.assertEqual(r["warnings"], [])
        self.assertIn("Preliminary", r["disclaimer"])

    def test_footprint_warns_on_weak_support(self):
        weak = self.conf.copy(); weak[10:30, 10:30] = D.CONF_UNKNOWN
        r = P.footprint_report(self.z, G, self.sq, conf=weak)
        self.assertTrue(any("survey" in w.lower() for w in r["warnings"]))

    def test_cut_fill_matches_analytic_volume(self):
        out = P.cut_fill(self.z, G, self.sq, P.ProposedSurface.flat(102.0), conf=self.conf)   # pad at mean elevation
        self.assertEqual(out["status"], "ok")
        self.assertAlmostEqual(out["cut"], 0.1 * 20 ** 3 / 8, delta=1.0)
        self.assertAlmostEqual(out["fill"], 0.1 * 20 ** 3 / 8, delta=1.0)
        self.assertAlmostEqual(out["net"], 0.0, delta=1.0)
        self.assertIn("Preliminary", out["disclaimer"])

    def test_cut_fill_refuses_without_defensible_ground(self):
        self.assertEqual(P.cut_fill(self.z, G, self.sq, P.ProposedSurface.flat(102.0), conf=None)["status"], "refused")
        weak = self.conf.copy(); weak[10:30, 10:30] = D.CONF_LOW
        r = P.cut_fill(self.z, G, self.sq, P.ProposedSurface.flat(102.0), conf=weak)
        self.assertEqual(r["status"], "refused")
        self.assertIn("survey", r["reason"].lower())
        holes = self.z.copy(); holes[10:30, 10:20] = np.nan
        r2 = P.cut_fill(holes, G, self.sq, P.ProposedSurface.flat(102.0), conf=self.conf)
        self.assertEqual(r2["status"], "refused")
        self.assertIn("estimate", r2["reason"])

    def test_uncertainty_reported_when_rmse_given(self):
        out = P.cut_fill(self.z, G, self.sq, P.ProposedSurface.flat(102.0), conf=self.conf, error_rmse_m=0.25)
        self.assertAlmostEqual(out["volume_uncertainty_worst_case"], 400 * 0.25, delta=1.0)

    def test_plane_from_three_points(self):
        s = P.ProposedSurface.from_three_points((0, 0, 10), (10, 0, 11), (0, 10, 9))
        self.assertAlmostEqual(float(s.z_at(10, 10)), 10.0)
        with self.assertRaises(ValueError):
            P.ProposedSurface.from_three_points((0, 0, 1), (1, 1, 2), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
