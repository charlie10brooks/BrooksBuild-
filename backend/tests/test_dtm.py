import unittest

import numpy as np

from app.core import dtm as D


def truth(x, y):
    """Known synthetic terrain: tilt + hill + shallow swale. Test fixture only."""
    return (100 + 0.05 * x + 0.02 * y
            + 8 * np.exp(-((x - 60) ** 2 + (y - 50) ** 2) / (2 * 18 ** 2))
            - 1.5 * np.exp(-((y - 20) ** 2) / (2 * 3 ** 2)))


def plane(x, y):
    return 50 + 0.10 * x - 0.06 * y


def make_points(fn, grid, gap=None, density=2.0, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = grid.bounds()
    n = int(density * (maxx - minx) * (maxy - miny))
    x = rng.uniform(minx, maxx, n)
    y = rng.uniform(miny, maxy, n)
    if gap is not None:
        cx, cy, r = gap
        keep = (x - cx) ** 2 + (y - cy) ** 2 > r ** 2
        x, y = x[keep], y[keep]
    z = fn(x, y) + rng.normal(0, noise, x.size)
    return x, y, z


GRID = D.GridSpec.from_bounds(0, 0, 120, 100, 1.0)


class GriddingTests(unittest.TestCase):
    def test_bin_stats(self):
        g = D.GridSpec.from_bounds(0, 0, 4, 4, 1.0)
        x = np.array([0.5, 0.6, 0.7, 2.5]); y = np.array([3.5, 3.4, 3.6, 0.5]); z = np.array([10., 12., 20., 5.])
        for stat, expect in [("min", 10), ("max", 20), ("median", 12), ("mean", 14)]:
            zg, cnt = D.bin_points(x, y, z, g, stat)
            self.assertAlmostEqual(float(zg[0, 0]), expect, msg=stat)
            self.assertEqual(int(cnt[0, 0]), 3)
        self.assertAlmostEqual(float(zg[3, 2]), 5.0)
        self.assertTrue(np.isnan(zg[1, 1]))

    def test_points_outside_grid_ignored(self):
        zg, cnt = D.bin_points(np.array([-5., 500.]), np.array([1., 1.]), np.array([1., 2.]), GRID)
        self.assertEqual(int(cnt.sum()), 0)

    def test_grid_roundtrip(self):
        r, c = GRID.xy_to_rc(10.25, 99.0)
        self.assertAlmostEqual(float(r), 1.0)
        self.assertAlmostEqual(float(c), 10.25)
        x, y = GRID.cell_centers_xy(0, 0)
        self.assertAlmostEqual(float(x), 0.5)
        self.assertAlmostEqual(float(y), 99.5)


class BuildDTMTests(unittest.TestCase):
    def setUp(self):
        self.gap = (60, 50, 14)         # canopy-like hole in ground coverage
        self.gx, self.gy, self.gz = make_points(truth, GRID, gap=self.gap, density=2.0)

    def _truth_grid(self):
        r, c = np.mgrid[0:GRID.height, 0:GRID.width]
        x, y = GRID.cell_centers_xy(r, c)
        return truth(x, y), x, y

    def test_measured_cells_are_preserved_exactly(self):
        res = D.build_dtm(self.gx, self.gy, self.gz, GRID)
        zg, _ = D.bin_points(self.gx, self.gy, self.gz, GRID, "median")
        m = ~np.isnan(zg)
        np.testing.assert_array_equal(res.dtm[m], zg[m])
        self.assertTrue((res.flag[m] == D.FLAG_MEASURED).all())
        self.assertTrue((res.flag[~m & ~np.isnan(res.dtm)] == D.FLAG_INTERPOLATED).all())

    def test_gap_over_gentle_terrain_is_accurate(self):
        def gentle(x, y):
            return 100 + 0.05 * x + 0.02 * y + 8 * np.exp(-((x - 20) ** 2 + (y - 80) ** 2) / (2 * 18 ** 2))
        gx, gy, gz = make_points(gentle, GRID, gap=(60, 50, 14), density=2.0)
        res = D.build_dtm(gx, gy, gz, GRID, D.InterpParams(method="local_plane", max_radius_m=40))
        r, c = np.mgrid[0:GRID.height, 0:GRID.width]
        x, y = GRID.cell_centers_xy(r, c)
        in_gap = (x - 60) ** 2 + (y - 50) ** 2 < 13 ** 2
        e = res.dtm[in_gap] - gentle(x, y)[in_gap]
        self.assertFalse(np.isnan(e).any())
        self.assertLess(float(np.sqrt(np.mean(e ** 2))), 0.25)

    def test_peak_hidden_inside_gap_is_underestimated_but_relief_is_not_flattened(self):
        """Documented physical limit: a hilltop with no ground data under it cannot be recovered.

        Every interpolator is biased low there. What we require is that the estimate (a) is not
        invented beyond the evidence radius, (b) keeps real relief rather than flattening, and
        (c) preserves more relief than IDW, which flattens by construction.
        """
        tz, x, y = self._truth_grid()
        in_gap = (x - self.gap[0]) ** 2 + (y - self.gap[1]) ** 2 < (self.gap[2] - 1) ** 2
        lp = D.build_dtm(self.gx, self.gy, self.gz, GRID, D.InterpParams(method="local_plane", max_radius_m=40))
        idw = D.build_dtm(self.gx, self.gy, self.gz, GRID, D.InterpParams(method="idw", max_radius_m=40))
        est = lp.dtm[in_gap]
        self.assertFalse(np.isnan(est).any())
        self.assertLess(float(np.mean(est - tz[in_gap])), 0.0, "peak under a gap should be underestimated")
        self.assertGreater(float(np.ptp(est)), 0.6 * float(np.ptp(tz[in_gap])))
        self.assertGreater(float(np.ptp(est)), float(np.ptp(idw.dtm[in_gap])))
        # and the confidence layer must not call that region well-supported
        conf = D.terrain_confidence(lp.dtm, lp.measured, GRID.res)
        self.assertLess(int((conf[in_gap] == D.CONF_HIGH).sum()), 0.2 * int(in_gap.sum()))

    def test_local_plane_beats_idw_on_tilted_ground(self):
        gx, gy, gz = make_points(plane, GRID, gap=(60, 50, 16), density=1.5)
        errs = {}
        r, c = np.mgrid[0:GRID.height, 0:GRID.width]
        x, y = GRID.cell_centers_xy(r, c)
        in_gap = (x - 60) ** 2 + (y - 50) ** 2 < 15 ** 2
        for m in ("local_plane", "idw"):
            res = D.build_dtm(gx, gy, gz, GRID, D.InterpParams(method=m, max_radius_m=40))
            e = res.dtm[in_gap] - plane(x, y)[in_gap]
            errs[m] = float(np.sqrt(np.nanmean(e ** 2)))
        self.assertLess(errs["local_plane"], 0.15)
        self.assertLess(errs["local_plane"], errs["idw"])

    def test_all_methods_produce_finite_estimates_in_gap(self):
        r, c = np.mgrid[0:GRID.height, 0:GRID.width]
        x, y = GRID.cell_centers_xy(r, c)
        in_gap = (x - 60) ** 2 + (y - 50) ** 2 < 10 ** 2
        for m in ("local_plane", "idw", "tin", "tps"):
            res = D.build_dtm(self.gx, self.gy, self.gz, GRID, D.InterpParams(method=m, max_radius_m=40))
            self.assertTrue(np.isfinite(res.dtm[in_gap]).all(), m)

    def test_nearest_respects_radius(self):
        res = D.build_dtm(self.gx, self.gy, self.gz, GRID, D.InterpParams(method="nearest", max_radius_m=3))
        self.assertTrue(np.isnan(res.dtm[res.dist_m > 3.0]).all())

    def test_no_evidence_beyond_radius_stays_unknown(self):
        gx, gy, gz = make_points(truth, GRID, gap=(60, 50, 30), density=2.0)      # 60 m wide hole
        res = D.build_dtm(gx, gy, gz, GRID, D.InterpParams(method="local_plane", max_radius_m=10))
        centre = res.dtm[GRID.height - 50, 60]
        self.assertTrue(np.isnan(centre), "centre of a hole larger than max_radius must not be invented")
        self.assertEqual(int(res.flag[GRID.height - 50, 60]), D.FLAG_NONE)
        conf = D.terrain_confidence(res.dtm, res.measured, GRID.res)
        self.assertEqual(int(conf[GRID.height - 50, 60]), D.CONF_UNKNOWN)

    def test_insufficient_ground_raises_with_guidance(self):
        with self.assertRaises(D.InsufficientGroundError) as cm:
            D.build_dtm(np.array([1., 2.]), np.array([1., 2.]), np.array([1., 2.]), GRID)
        self.assertIn("LiDAR", str(cm.exception))


class ConfidenceTests(unittest.TestCase):
    def test_confidence_degrades_with_distance(self):
        gx, gy, gz = make_points(plane, GRID, gap=(60, 50, 22), density=2.0)
        res = D.build_dtm(gx, gy, gz, GRID, D.InterpParams(max_radius_m=40))
        conf = D.terrain_confidence(res.dtm, res.measured, GRID.res, D.ConfidenceParams(slope_penalty=0.0))
        order = {}
        for k in (D.CONF_HIGH, D.CONF_MODERATE, D.CONF_LOW):
            sel = conf == k
            order[k] = float(res.dist_m[sel].mean()) if sel.any() else None
        self.assertLess(order[D.CONF_HIGH], order[D.CONF_MODERATE])
        self.assertLess(order[D.CONF_MODERATE], order[D.CONF_LOW])
        self.assertTrue((conf[res.dist_m > 15.0] == D.CONF_UNKNOWN).all())
        self.assertTrue((conf[np.isnan(res.dtm)] == D.CONF_UNKNOWN).all())

    def test_steep_slopes_reduce_confidence(self):
        steep = lambda x, y: 100 + 0.8 * x           # ~39 degrees
        gx, gy, gz = make_points(steep, GRID, gap=(60, 50, 12), density=2.0)
        res = D.build_dtm(gx, gy, gz, GRID, D.InterpParams(max_radius_m=40))
        c0 = D.terrain_confidence(res.dtm, res.measured, GRID.res, D.ConfidenceParams(slope_penalty=0.0))
        c1 = D.terrain_confidence(res.dtm, res.measured, GRID.res, D.ConfidenceParams(slope_penalty=3.0))
        self.assertLessEqual(int(c1.astype(int).sum()), int(c0.astype(int).sum()))
        self.assertLess(int((c1 >= D.CONF_MODERATE).sum()), int((c0 >= D.CONF_MODERATE).sum()))

    def test_summary_recommends_survey_when_weak(self):
        conf = np.full((10, 10), D.CONF_UNKNOWN, np.uint8)
        s = D.confidence_summary(conf, 1.0)
        self.assertIn("survey", s["advice"].lower())
        conf[:] = D.CONF_HIGH
        self.assertEqual(D.confidence_summary(conf, 1.0)["advice"], "")


class SourceTests(unittest.TestCase):
    def test_priority_and_provenance(self):
        a = np.full((4, 4), np.nan, np.float32); a[:, :2] = 10
        b = np.full((4, 4), np.nan, np.float32); b[:, 1:] = 20
        lo = D.ElevationSource(1, "photo", "photogrammetry_ground", 1, a)
        hi = D.ElevationSource(2, "lidar", "lidar", 5, b, z_offset=1.0)
        m = D.merge_sources([lo, hi])
        self.assertEqual(float(m.dtm[0, 0]), 10.0)           # only photo covers col 0
        self.assertEqual(float(m.dtm[0, 1]), 21.0)           # lidar wins, with its vertical offset
        self.assertEqual(int(m.source_id[0, 0]), 1)
        self.assertEqual(int(m.source_id[0, 3]), 2)
        swapped = D.merge_sources([D.ElevationSource(1, "photo", "x", 9, a), hi])
        self.assertEqual(int(swapped.source_id[0, 1]), 1)    # user can re-prioritise

    def test_vertical_offset_estimation_is_robust_to_outliers(self):
        rng = np.random.default_rng(3)
        base = rng.uniform(100, 130, (60, 60)).astype(np.float32)
        other = base - 12.345 + rng.normal(0, 0.05, base.shape).astype(np.float32)
        other[:3, :3] += 50                                   # a changed area (outliers)
        out = D.estimate_vertical_offset(base, other)
        self.assertAlmostEqual(out["offset_m"], 12.345, delta=0.03)
        self.assertEqual(out["warning"], "")
        with self.assertRaises(ValueError):
            D.estimate_vertical_offset(base[:3, :3], other[:3, :3])


class CrossValidationTests(unittest.TestCase):
    def test_error_grows_with_distance_from_observations(self):
        gx, gy, gz = make_points(truth, GRID, density=1.5, noise=0.02)
        zg, _ = D.bin_points(gx, gy, gz, GRID, "median")
        rep = D.cross_validate(zg, GRID, D.InterpParams(max_radius_m=60), holdout="block", fraction=0.2, block_m=24)
        rows = [r for r in rep["by_distance"] if r["n"] >= 10]
        self.assertGreaterEqual(len(rows), 3)
        self.assertLess(rows[0]["rmse_m"], rows[-1]["rmse_m"])
        self.assertLess(rep["overall_rmse_m"], 2.0)
        self.assertIn("never observed", rep["note"])

    def test_random_holdout_is_more_optimistic_than_block(self):
        gx, gy, gz = make_points(truth, GRID, density=1.5, noise=0.02)
        zg, _ = D.bin_points(gx, gy, gz, GRID, "median")
        p = D.InterpParams(max_radius_m=60)
        rnd = D.cross_validate(zg, GRID, p, holdout="random", fraction=0.2)
        blk = D.cross_validate(zg, GRID, p, holdout="block", fraction=0.2, block_m=24)
        self.assertLess(rnd["overall_rmse_m"], blk["overall_rmse_m"])


if __name__ == "__main__":
    unittest.main()
