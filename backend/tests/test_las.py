import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services import las


class LasTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        rs = np.random.RandomState(0)
        self.n = 25_000
        self.x = 571000 + rs.rand(self.n) * 400
        self.y = 4402000 + rs.rand(self.n) * 300
        self.z = 250 + rs.rand(self.n) * 30
        self.c = rs.randint(0, 8, self.n)
        self.rgb = rs.randint(0, 65535, (self.n, 3))

    def test_roundtrip_with_crs_class_rgb(self):
        p = self.d / "a.las"
        las.write_las(p, self.x, self.y, self.z, self.c, self.rgb, epsg=32616)
        h = las.read_header(p)
        self.assertEqual((h.count, h.point_format, h.epsg, h.version), (self.n, 3, 32616, (1, 2)))
        np.testing.assert_allclose(h.mins, [self.x.min(), self.y.min(), self.z.min()], atol=2e-3)
        np.testing.assert_allclose(h.maxs, [self.x.max(), self.y.max(), self.z.max()], atol=2e-3)
        d = las.read_all(p)
        np.testing.assert_allclose(d["x"], self.x, atol=1e-3)
        np.testing.assert_allclose(d["z"], self.z, atol=1e-3)
        np.testing.assert_array_equal(d["classification"], self.c)
        np.testing.assert_array_equal(d["rgb"], self.rgb)

    def test_chunked_write_and_read(self):
        p = self.d / "b.las"
        with las.LasWriter(p, epsg=4326, scale=1e-7, offset=(-90.0, 39.0, 250.0)) as w:
            for i in range(0, self.n, 7000):
                s = slice(i, i + 7000)
                w.write(self.x[s] * 0 - 86.1 + (self.x[s] % 1) * 1e-3, self.y[s] * 0 + 39.7, self.z[s], self.c[s])
        chunks = list(las.iter_points(p, chunk=9000))
        self.assertEqual(sum(len(c["x"]) for c in chunks), self.n)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(las.read_header(p).epsg, 4326)

    def test_no_crs_written_when_unknown(self):
        p = self.d / "c.las"
        las.write_las(p, self.x[:10], self.y[:10], self.z[:10])
        self.assertIsNone(las.read_header(p).epsg)

    def test_errors_are_explicit(self):
        bad = self.d / "bad.las"
        bad.write_bytes(b"NOPE" * 100)
        with self.assertRaises(las.LasError):
            las.read_header(bad)
        p = self.d / "t.las"
        las.write_las(p, self.x, self.y, self.z)
        data = p.read_bytes()
        (self.d / "trunc.las").write_bytes(data[:-3400])           # drop 100 points
        with self.assertRaises(las.LasError) as cm:
            las.read_all(self.d / "trunc.las")
        self.assertIn("truncated", str(cm.exception))
        with self.assertRaises(las.LasError):
            las.write_las(self.d / "nan.las", [1.0, float("nan")], [1.0, 2.0], [1.0, 2.0])
        with self.assertRaises(las.LasError):
            las.write_las(self.d / "cls.las", [1.0], [1.0], [1.0], classification=[40])

    def test_laz_requires_pdal_with_install_hint(self):
        z = self.d / "x.laz"
        z.write_bytes(b"x")
        with self.assertRaises(las.MissingDependency) as cm:
            las.ensure_las(z, self.d, pdal_bin="definitely-not-installed-pdal")
        self.assertIn("conda install", str(cm.exception))
        self.assertEqual(las.ensure_las(self.d / "k.las", self.d), self.d / "k.las")

    def test_compressed_flag_rejected_on_read(self):
        p = self.d / "z.las"
        las.write_las(p, self.x[:5], self.y[:5], self.z[:5])
        b = bytearray(p.read_bytes())
        b[104] |= 0x80
        p.write_bytes(bytes(b))
        with self.assertRaises(las.LasError):
            list(las.iter_points(p))


if __name__ == "__main__":
    unittest.main()
