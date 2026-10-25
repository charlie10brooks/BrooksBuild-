import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from app.core.dtm import GridSpec
from app.core.tiling import Window
from app.services import geotiff as gt


class GeoTiffTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.grid = GridSpec(571000.0, 4402000.0, 0.25, 700, 500)

    def test_roundtrip_float_dem_with_crs_nodata_units(self):
        z = np.random.RandomState(0).rand(500, 700).astype("float32") * 10 + 250
        z[10:20, 10:20] = np.nan
        p = self.d / "dtm.tif"
        rep = gt.write_geotiff(p, z, self.grid, 32616, nodata=float("nan"), vertical_epsg=5703, vertical_units="metre")
        self.assertTrue(rep["georeferenced"])
        info = gt.read_georef(p)
        self.assertEqual(info.epsg, 32616)
        self.assertEqual(info.vertical_epsg, 5703)
        self.assertEqual(info.horizontal_units, "metre")
        self.assertEqual(info.vertical_units, "metre")
        self.assertTrue(np.isnan(info.nodata))
        self.assertEqual(info.grid, self.grid)
        self.assertEqual(info.tile, (256, 256))
        np.testing.assert_array_equal(gt.read_full(p), z)

    def test_us_survey_foot_units(self):
        p = self.d / "f.tif"
        gt.write_geotiff(p, np.zeros((500, 700), "float32"), self.grid, 2965, horizontal_units="us_survey_foot",
                         vertical_units="us_survey_foot")
        i = gt.read_georef(p)
        self.assertEqual((i.horizontal_units, i.vertical_units), ("us_survey_foot", "us_survey_foot"))

    def test_no_crs_is_not_claimed_georeferenced(self):
        p = self.d / "n.tif"
        rep = gt.write_geotiff(p, np.zeros((50, 60), "uint8"), None, None)
        self.assertFalse(rep["georeferenced"])
        i = gt.read_georef(p)
        self.assertFalse(i.georeferenced)
        self.assertIsNone(i.grid)
        with self.assertRaises(gt.GeoTiffError):       # grid/array mismatch is an error, not silent
            gt.write_geotiff(p, np.zeros((50, 60), "uint8"), self.grid, 32616)

    def test_windowed_read_matches_full_tiled_and_striped(self):
        rgb = (np.random.RandomState(1).rand(500, 700, 3) * 255).astype("uint8")
        pt = self.d / "tiled.tif"
        gt.write_geotiff(pt, rgb, self.grid, 32616)
        ps = self.d / "striped.tif"
        tifffile.imwrite(str(ps), rgb, photometric="rgb", rowsperstrip=37, compression="deflate")
        for p in (pt, ps):
            with gt.TiffWindowReader(p) as r:
                for w in (Window(0, 0, 700, 500), Window(250, 250, 300, 260), Window(255, 255, 258, 258),
                          Window(600, 400, 700, 500), Window(5, 490, 690, 500)):
                    np.testing.assert_array_equal(r.read(w), rgb[w.slices()], err_msg=f"{p.name} {w}")
                self.assertEqual(r.read_rgb(Window(0, 0, 10, 10)).shape, (10, 10, 3))
                with self.assertRaises(gt.GeoTiffError):
                    r.read(Window(600, 400, 800, 600))

    def test_streamed_tile_write_matches_source(self):
        src = (np.random.RandomState(2).rand(500, 700) * 7).astype("uint8")
        p = self.d / "mask.tif"
        calls = []

        def tf(w):
            calls.append(w)
            return src[w.slices()]
        gt.write_geotiff_tiles(p, 700, 500, "uint8", tf, self.grid, 32616, nodata=0)
        np.testing.assert_array_equal(gt.read_full(p), src)
        self.assertEqual(len(calls), 3 * 2)
        self.assertEqual(gt.read_georef(p).nodata, 0.0)

    def test_sidecar(self):
        p = self.d / "x.tif"
        gt.write_geotiff(p, np.zeros((10, 10), "uint8"), None, None)
        s = gt.write_sidecar(p, {"crs": "unknown", "note": "local coordinates"})
        self.assertTrue(s.exists() and s.name == "x.tif.meta.json")


if __name__ == "__main__":
    unittest.main()
