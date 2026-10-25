import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core import classes as C
from app.core import dtm as D
from app.core.dtm import GridSpec
from app.services import geotiff as gt
from app.services import groundfilter as G
from app.services import las as L
from app.services import synthetic
from app.services import terrain_pipeline as T

BUILTIN = G.GroundFilterParams(engine="builtin_pmf")


def _params(**kw):
    return T.TerrainParams(resolution_m=1.0, ground=BUILTIN, **kw)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        cls.site = synthetic.make_site(100, 80, 0.5, seed=5, tree_cover=0.5)
        cls.pts = cls.site.points(density=3.0, canopy_gap=0.05, seed=2)
        cls.cloud = cls.d / "cloud.las"
        L.write_las(cls.cloud, cls.pts["x"], cls.pts["y"], cls.pts["z"], epsg=32616)
        cls.res = T.run_terrain(cls.cloud, cls.d / "run1", 32616, _params(),
                                veg_mask_fn=lambda g: np.isin(_to(cls.site, g), (C.TREES, C.BRUSH)))
        cls.g = cls.res.grid
        cls.dtm = gt.read_full(Path(cls.res.paths["dtm"]))
        cls.conf = gt.read_full(Path(cls.res.paths["confidence"]))
        cls.ev = gt.read_full(Path(cls.res.paths["evidence"]))
        cls.truth = _truth_on(cls.site, cls.g)

    def test_outputs_are_georeferenced_with_units_and_sidecars(self):
        for k in ("dsm", "dtm", "ndsm", "confidence", "evidence", "source"):
            p = Path(self.res.paths[k])
            info = gt.read_georef(p)
            self.assertTrue(info.georeferenced, k)
            self.assertEqual(info.epsg, 32616)
            self.assertEqual(info.vertical_units, "metre")
            self.assertTrue(Path(str(p) + ".meta.json").exists())
        self.assertEqual(L.read_header(Path(self.res.paths["classified_cloud"])).epsg, 32616)

    def test_dtm_tracks_true_terrain_and_is_not_flat(self):
        ok = np.isfinite(self.dtm)
        err = (self.dtm - self.truth)[ok]
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.6)
        self.assertGreater(float(np.nanstd(self.dtm)), 0.5 * float(np.std(self.truth)))     # relief preserved
        meas = self.ev == T.EV_MEASURED
        self.assertLess(float(np.abs((self.dtm - self.truth)[meas]).mean()), 0.2)

    def test_confidence_orders_error(self):
        err = np.abs(self.dtm - self.truth)
        hi = err[(self.conf == D.CONF_HIGH) & np.isfinite(err)]
        lo = err[(self.conf == D.CONF_LOW) & np.isfinite(err)]
        self.assertGreater(hi.size, 100)
        if lo.size > 50:
            self.assertLess(float(np.median(hi)), float(np.median(lo)) + 1e-6)

    def test_dsm_is_canopy_and_ndsm_positive_under_trees(self):
        dsm = gt.read_full(Path(self.res.paths["dsm"]))
        ndsm = gt.read_full(Path(self.res.paths["ndsm"]))
        tall = _truth_on(self.site, self.g, self.site.canopy_h) > 10
        self.assertGreater(float(np.nanmedian(ndsm[tall])), 6.0)
        self.assertTrue(np.nanmedian(dsm[tall] - self.dtm[tall]) > 6.0)

    def test_evidence_and_unknown_are_distinguished(self):
        s = self.res.summary
        cov = s["coverage"]
        self.assertAlmostEqual(sum(cov.values()), 1.0, places=6)
        self.assertGreater(cov["measured"], 0.05)
        self.assertEqual((self.conf[self.ev == T.EV_NONE] == D.CONF_UNKNOWN).all(), True)     # never filled silently
        self.assertTrue(np.isnan(self.dtm[self.ev == T.EV_NONE]).all())
        self.assertIsNotNone(s["cross_validation"])
        self.assertIn("PRELIMINARY", Path(str(self.res.paths["dtm"]) + ".meta.json").read_text())

    def test_point_labels_include_ground_and_high_vegetation(self):
        lab = self.res.summary["point_labels"]
        self.assertIn("2", lab)
        self.assertIn("5", lab)

    def test_supplemental_sources_priority_provenance_and_trust(self):
        # survey points override photogrammetry where they exist; external DEM fills with its trust level
        g = self.g
        rs = np.random.RandomState(0)
        sx = g.x0 + rs.rand(40) * g.width * g.res
        sy = g.y0 - rs.rand(40) * g.height * g.res
        sz = np.full(40, 777.0)
        csv = self.d / "survey.csv"
        csv.write_text("x,y,z\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(sx, sy, sz)))
        dem = self.d / "ext.tif"
        gt.write_geotiff(dem, self.truth.astype("float32"), g, 32616, nodata=float("nan"))
        specs = [T.SourceSpec(2, "Field survey", "survey", csv, priority=50),
                 T.SourceSpec(3, "County DEM", "external_dem", dem, priority=1, trust="moderate")]
        r = T.run_terrain(self.cloud, self.d / "run2", 32616, _params(cross_validate=False), specs)
        dtm = gt.read_full(Path(r.paths["dtm"]))
        src = gt.read_full(Path(r.paths["source"]))
        ev = gt.read_full(Path(r.paths["evidence"]))
        conf = gt.read_full(Path(r.paths["confidence"]))
        self.assertTrue((dtm[src == 2] == 777.0).all() and (src == 2).sum() > 0)
        self.assertTrue(((src == 3) & (ev == T.EV_EXTERNAL)).sum() > 0)
        ext_cells = (ev == T.EV_EXTERNAL)
        self.assertTrue((conf[ext_cells] >= D.CONF_MODERATE).all())
        self.assertEqual({s["id"] for s in r.summary["sources"]}, {1, 2, 3})

    def test_crs_mismatch_refused_not_reprojected_silently(self):
        dem = self.d / "wrong.tif"
        gt.write_geotiff(dem, np.zeros(self.g.shape, "float32"), self.g, 32617)
        with self.assertRaises(T.TerrainError) as cm:
            T.run_terrain(self.cloud, self.d / "run3", 32616, _params(cross_validate=False),
                          [T.SourceSpec(2, "wrong crs", "external_dem", dem)])
        self.assertIn("Reproject", str(cm.exception))


def _to(site, g):
    """Nearest label of the synthetic site on grid g (cell centres)."""
    return _sample(site, g, site.labels)


def _truth_on(site, g, arr=None):
    return _sample(site, g, site.terrain if arr is None else arr).astype(np.float32)


def _sample(site, g, arr):
    rows, cols = np.mgrid[0:g.height, 0:g.width]
    x, y = g.cell_centers_xy(rows, cols)
    r, c = site.grid.xy_to_rc(x, y)
    r, c = np.clip(r.astype(int), 0, site.grid.height - 1), np.clip(c.astype(int), 0, site.grid.width - 1)
    return arr[r, c]


class FailureModes(unittest.TestCase):
    def test_solid_canopy_without_ground_reports_not_fabricates(self):
        d = Path(tempfile.mkdtemp())
        site = synthetic.make_site(60, 60, 0.5, seed=1, tree_cover=0.5)
        pts = site.points(density=3, canopy_gap=0.0, seed=1)
        keep = ~pts["is_ground_truth"] | (site.canopy_h[
            np.clip(site.grid.xy_to_rc(pts["x"], pts["y"])[0].astype(int), 0, site.grid.height - 1),
            np.clip(site.grid.xy_to_rc(pts["x"], pts["y"])[1].astype(int), 0, site.grid.width - 1)] < 0.5)
        # drop every ground return, i.e. a flight that saw only canopy tops and nothing else
        canopy_only = ~pts["is_ground_truth"]
        L.write_las(d / "c.las", pts["x"][canopy_only], pts["y"][canopy_only], pts["z"][canopy_only], epsg=32616)
        try:
            r = T.run_terrain(d / "c.las", d / "w", 32616, _params(cross_validate=False))
        except (T.TerrainError, D.InsufficientGroundError) as e:
            self.assertTrue("survey" in str(e).lower() or "ground" in str(e).lower())
            return
        # if the filter did find 'ground' it was only canopy: the run must say it could not verify that
        self.assertTrue(any("land-cover" in w for w in r.summary["warnings"]))

    def test_missing_crs_is_documented_not_hidden(self):
        d = Path(tempfile.mkdtemp())
        site = synthetic.make_site(50, 40, 0.5, seed=2, tree_cover=0.3)
        p = site.points(density=3, seed=3)
        L.write_las(d / "n.las", p["x"], p["y"], p["z"])
        r = T.run_terrain(d / "n.las", d / "w", None, _params(cross_validate=False))
        self.assertFalse(gt.read_georef(Path(r.paths["dtm"])).georeferenced)
        self.assertTrue(any("WITHOUT georeferencing" in w for w in r.summary["warnings"]))
        self.assertIn('"georeferenced": false', Path(r.paths["dtm"] + ".meta.json").read_text())

    def test_demotes_canopy_misclassified_as_ground(self):
        """A crown wider than the filter window gets called ground. The land-cover check must catch it."""
        d = Path(tempfile.mkdtemp())
        site = synthetic.make_site(80, 60, 0.5, seed=4, tree_cover=0.0)
        yy, xx = np.mgrid[0:site.grid.height, 0:site.grid.width]
        blob = ((xx - 80) ** 2 + (yy - 60) ** 2) < 50 ** 2              # ~25 m radius stand, solid canopy
        site.canopy_h[blob] = 14.0
        site.labels[blob] = C.TREES
        pts = site.points(density=3, canopy_gap=0.0, seed=1)
        # open ground elsewhere, nothing reaches the ground under the stand
        L.write_las(d / "b.las", pts["x"], pts["y"], pts["z"], epsg=32616)
        veg = lambda g: np.isin(_to(site, g), (C.TREES, C.BRUSH))
        plain = T.run_terrain(d / "b.las", d / "w1", 32616, _params(cross_validate=False))
        checked = T.run_terrain(d / "b.las", d / "w2", 32616, _params(cross_validate=False), veg_mask_fn=veg)
        truth = _truth_on(site, plain.grid)
        inside = _to(site, plain.grid) == C.TREES
        e_plain = np.nanmean(np.abs(gt.read_full(Path(plain.paths["dtm"]))[inside] - truth[inside]))
        e_checked = np.nanmean(np.abs(gt.read_full(Path(checked.paths["dtm"]))[inside] - truth[inside]))
        self.assertGreater(checked.summary["demoted_canopy_cells"], 0)
        self.assertLess(e_checked, e_plain)
        conf = gt.read_full(Path(checked.paths["confidence"]))
        from scipy import ndimage
        deep = ndimage.binary_erosion(inside, iterations=8)                 # >8 m from the stand edge
        self.assertGreater(int(deep.sum()), 100)
        self.assertFalse((conf[deep] == D.CONF_HIGH).any())


class SurveyCsv(unittest.TestCase):
    def test_parsing(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.csv").write_text("# comment\nname,x,y,z\nP1,1,2,3\nP2,4,5,6\n")
        x, y, z = T.load_survey_csv(d / "a.csv")
        self.assertEqual((x.tolist(), y.tolist(), z.tolist()), ([1.0, 4.0], [2.0, 5.0], [3.0, 6.0]))
        (d / "b.csv").write_text("1,2,3\n4,abc,6\n")
        with self.assertRaises(T.TerrainError):
            T.load_survey_csv(d / "b.csv")


if __name__ == "__main__":
    unittest.main()
