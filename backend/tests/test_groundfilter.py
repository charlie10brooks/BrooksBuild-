import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.core.dtm import GridSpec, bin_points
from app.services import groundfilter as G
from app.services import las as L
from app.services import synthetic


class PdalPipelineTests(unittest.TestCase):
    def test_smrf_pipeline_structure(self):
        pl = G.build_pdal_pipeline("in.laz", "out.las", G.GroundFilterParams())
        types = [s["type"] for s in pl]
        self.assertEqual(types, ["readers.las", "filters.assign", "filters.outlier", "filters.smrf", "writers.las"])
        smrf = pl[3]
        self.assertEqual(smrf["ignore"], "Classification[7:7]")
        self.assertEqual((smrf["cell"], smrf["slope"], smrf["window"], smrf["threshold"]), (1.0, 0.15, 18.0, 0.5))
        self.assertEqual(pl[-1]["compression"], "false")
        json.dumps(pl)

    def test_pmf_pipeline_and_no_outlier(self):
        pl = G.build_pdal_pipeline("a", "b", G.GroundFilterParams(engine="pdal_pmf", outlier_removal=False))
        self.assertEqual([s["type"] for s in pl], ["readers.las", "filters.assign", "filters.pmf", "writers.las"])
        self.assertNotIn("ignore", pl[2])
        self.assertEqual(pl[2]["max_window_size"], 33.0)

    def test_builtin_is_not_a_pdal_engine_and_bad_engine_rejected(self):
        with self.assertRaises(ValueError):
            G.build_pdal_pipeline("a", "b", G.GroundFilterParams(engine="builtin_pmf"))
        with self.assertRaises(ValueError):
            G.GroundFilterParams(engine="magic").validate()

    def test_missing_pdal_fails_loudly_with_instructions(self):
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(L.MissingDependency) as cm:
            G.run_pdal_pipeline([{"type": "readers.las", "filename": "x"}], d, pdal_bin="no-such-pdal-binary")
        self.assertIn("conda install", str(cm.exception))
        # classify_ground with a PDAL engine must also fail rather than fall back
        with self.assertRaises(L.MissingDependency):
            G.classify_ground(d / "a.las", d / "b.las", G.GroundFilterParams(engine="pdal_smrf"), d, pdal_bin="no-such-pdal-binary")

    def test_fake_pdal_binary_success_and_failure_paths(self):
        d = Path(tempfile.mkdtemp())
        ok = d / "pdal_ok"
        ok.write_text("#!/bin/sh\necho running\nexit 0\n"); ok.chmod(0o755)
        bad = d / "pdal_bad"
        bad.write_text("#!/bin/sh\necho 'PDAL: Unable to read file' 1>&2\nexit 3\n"); bad.chmod(0o755)
        lines = []
        G.run_pdal_pipeline([{"type": "readers.las", "filename": "x"}], d / "w1", str(ok), log=lines.append)
        self.assertIn("running", lines)
        self.assertTrue((d / "w1" / "pdal_pipeline.json").exists())
        with self.assertRaises(G.GroundFilterError) as cm:
            G.run_pdal_pipeline([{"type": "readers.las", "filename": "x"}], d / "w2", str(bad))
        self.assertIn("code 3", str(cm.exception))
        self.assertIn("Unable to read", str(cm.exception))
        with self.assertRaises(G.Cancelled):
            G.run_pdal_pipeline([{"type": "readers.las", "filename": "x"}], d / "w3", str(ok), cancelled=lambda: True)


class BuiltinFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        cls.site = synthetic.make_site(width_m=100, height_m=80, res=0.5, seed=3, tree_cover=0.45)
        pts = cls.site.points(density=3.0, canopy_gap=0.05, seed=4)
        cls.pts = pts
        cls.src = cls.d / "cloud.las"
        L.write_las(cls.src, pts["x"], pts["y"], pts["z"], epsg=32616)
        cls.dst = cls.d / "classified.las"
        cls.counts = G.classify_ground(cls.src, cls.dst, G.GroundFilterParams(engine="builtin_pmf"), cls.d)

    def test_counts_and_output(self):
        c = self.counts
        self.assertEqual(c["total"], len(self.pts["x"]))
        self.assertEqual(c["ground"] + c["non_ground"] + c["noise"], c["total"])
        self.assertGreater(c["ground"], 1000)
        self.assertEqual(L.read_header(self.dst).epsg, 32616)       # CRS preserved through classification

    def test_ground_points_follow_true_terrain_not_canopy(self):
        g = self.site.grid
        xs, ys, zs = [], [], []
        for x, y, z in G.iter_ground_points(self.dst):
            xs.append(x); ys.append(y); zs.append(z)
        x, y, z = map(np.concatenate, (xs, ys, zs))
        r, c = g.xy_to_rc(x, y)
        r, c = np.clip(r.astype(int), 0, g.height - 1), np.clip(c.astype(int), 0, g.width - 1)
        err = z - self.site.terrain[r, c]
        self.assertLess(abs(float(np.median(err))), 0.1)
        # almost no canopy-top points may be called ground (canopy >= 5 m above terrain)
        high_canopy = self.site.canopy_h[r, c] > 5
        self.assertLess(float((err > 3.0).mean()), 0.03)
        self.assertGreater(len(x), 0.5 * float(self.pts["is_ground_truth"].sum()))

    def test_height_above_ground_labels(self):
        g = self.site.grid
        out = self.d / "veg.las"
        counts = G.label_by_height_above_ground(self.dst, out, self.site.terrain, g, 32616)
        self.assertIn("5", counts)                      # tall canopy found
        self.assertIn("2", counts)
        self.assertEqual(sum(counts.values()), self.counts["total"])
        # no DTM estimate -> label is left alone (no guessing)
        nan_dtm = np.full(g.shape, np.nan, dtype=np.float32)
        c2 = G.label_by_height_above_ground(self.dst, self.d / "veg2.las", nan_dtm, g, 32616)
        self.assertNotIn("5", c2)

    def test_surface_does_not_flatten_terrain(self):
        yy, xx = np.mgrid[0:100, 0:100]
        hill = 6 * np.exp(-(((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 15 ** 2))).astype("float32")
        s = G.pmf_surface(hill, 1.0, G.GroundFilterParams())
        self.assertGreater(float(s.max() - s.min()), 5.9)         # relief preserved
        self.assertLess(float(np.abs(s - hill).max()), 0.05)

    def test_known_limit_wide_tall_features_are_shaved_with_default_window(self):
        """Documented PMF limit: a steep mound taller than the cap, with a too-low slope setting, is treated as an object."""
        yy, xx = np.mgrid[0:100, 0:100]
        mound = 12 * np.exp(-(((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 8 ** 2))).astype("float32")
        s = G.pmf_surface(mound, 1.0, G.GroundFilterParams())
        self.assertGreater(float(np.abs(s - mound).max()), 1.0)
        steep_ok = G.pmf_surface(mound, 1.0, G.GroundFilterParams(builtin_slope=1.5, builtin_max_distance=15))
        self.assertLess(float(np.abs(steep_ok - mound).max()), 0.05)   # raising slope + cap keeps it; the trade-off is the user's


if __name__ == "__main__":
    unittest.main()
