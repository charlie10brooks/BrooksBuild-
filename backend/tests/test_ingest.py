import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core import exif, flight
from app.core.uploadstore import MIN_CHUNK_BYTES, UploadError, UploadStore, sanitize_filename


def make_jpeg(path: Path, lat=None, lon=None, alt=None, size=(64, 48), dt="2026:05:01 10:00:00",
              xmp: str = "", model="FC6310"):
    im = Image.fromarray((np.random.RandomState(1).rand(size[1], size[0], 3) * 255).astype("uint8"))
    ex = Image.Exif()
    ex[271], ex[272], ex[306] = "DJI", model, dt
    sub = ex.get_ifd(0x8769)
    sub[36867] = dt
    sub[41989] = 24
    if lat is not None:
        g = ex.get_ifd(0x8825)
        g[1], g[3] = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        def dms(v):
            v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
            return (d, m, s)
        g[2], g[4] = dms(lat), dms(lon)
        if alt is not None:
            g[5], g[6] = (1 if alt < 0 else 0), abs(alt)
    im.save(path, "JPEG", exif=ex.tobytes())
    if xmp:  # append an XMP-like blob after the image data, which the scanner reads from the file head
        data = path.read_bytes()
        path.write_bytes(data[:2] + b"\xff\xe1" + (len(xmp) + 2 + 29).to_bytes(2, "big") +
                         b"http://ns.adobe.com/xap/1.0/\x00" + xmp.encode() + data[2:])


class ExifTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_gps_and_camera(self):
        p = self.d / "a.jpg"
        make_jpeg(p, lat=39.7684, lon=-86.1581, alt=250.5)
        m = exif.read_image_metadata(p)
        self.assertAlmostEqual(m["lat"], 39.7684, places=3)
        self.assertAlmostEqual(m["lon"], -86.1581, places=3)
        self.assertAlmostEqual(m["alt_msl"], 250.5, places=1)
        self.assertEqual((m["width"], m["height"]), (64, 48))
        self.assertEqual(m["make"], "DJI")
        self.assertEqual(m["captured_at"], "2026-05-01T10:00:00")
        a = exif.georef_assessment(m)
        self.assertEqual(a["status"], "ok")                       # horizontal + MSL altitude present
        self.assertTrue(any("relative flight altitude" in i for i in a["issues"]))   # but flagged for QA
        p2 = self.d / "noalt.jpg"
        make_jpeg(p2, lat=39.0, lon=-86.0)
        self.assertEqual(exif.georef_assessment(exif.read_image_metadata(p2))["status"], "partial")

    def test_southern_western_hemisphere_signs(self):
        p = self.d / "s.jpg"
        make_jpeg(p, lat=-33.9, lon=18.4, alt=10)
        m = exif.read_image_metadata(p)
        self.assertLess(m["lat"], 0)
        self.assertGreater(m["lon"], 0)

    def test_missing_gps_is_reported_not_guessed(self):
        p = self.d / "n.jpg"
        make_jpeg(p)
        m = exif.read_image_metadata(p)
        self.assertIsNone(m["lat"])
        a = exif.georef_assessment(m)
        self.assertEqual(a["status"], "missing")
        self.assertFalse(a["has_gps"])

    def test_zero_zero_fix_rejected(self):
        a = exif.georef_assessment({"lat": 0.0, "lon": 0.0})
        self.assertEqual(a["status"], "invalid")

    def test_xmp_relative_altitude(self):
        p = self.d / "x.jpg"
        make_jpeg(p, lat=39.0, lon=-86.0, alt=300, xmp='drone-dji:RelativeAltitude="+80.12" drone-dji:GimbalPitchDegree="-90.0"')
        m = exif.read_image_metadata(p)
        self.assertAlmostEqual(m["rel_alt"], 80.12)
        self.assertAlmostEqual(m["gimbal_pitch"], -90.0)
        self.assertEqual(exif.georef_assessment(m)["status"], "ok")

    def test_unreadable_and_pixel_limit(self):
        bad = self.d / "bad.jpg"
        bad.write_bytes(b"\xff\xd8\xff" + b"garbage" * 10)
        with self.assertRaises(ValueError):
            exif.read_image_metadata(bad)
        p = self.d / "big.jpg"
        make_jpeg(p, size=(100, 100))
        with self.assertRaises(ValueError) as cm:
            exif.read_image_metadata(p, max_pixels=5000)
        self.assertIn("above the configured limit", str(cm.exception))

    def test_sniff_ignores_extension(self):
        self.assertEqual(exif.sniff_format(b"\xff\xd8\xff\xe0abc"), "jpeg")
        self.assertEqual(exif.sniff_format(b"II*\x00abcd"), "tiff")
        self.assertIsNone(exif.sniff_format(b"MZ\x90\x00"))     # an .exe renamed to .jpg is not an image


class UploadStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.s = UploadStore(self.root / "u", min_free_bytes=0)
        self.data = np.random.RandomState(0).bytes(MIN_CHUNK_BYTES * 3 + 123)
        self.sha = hashlib.sha256(self.data).hexdigest()

    def _chunks(self, st):
        cs = st.chunk_size
        return [self.data[i * cs:(i + 1) * cs] for i in range(st.total_chunks)]

    def test_roundtrip_out_of_order_with_retry(self):
        st = self.s.create("p1", "IMG_0001.JPG", len(self.data), MIN_CHUNK_BYTES, self.sha)
        self.assertEqual(st.total_chunks, 4)
        ch = self._chunks(st)
        for i in (3, 1, 1, 0):                       # out of order, chunk 1 sent twice
            self.s.put_chunk(st.id, i, [ch[i]], hashlib.sha256(ch[i]).hexdigest())
        self.assertEqual(self.s.status(st.id)["missing_chunks"], [2])
        with self.assertRaises(UploadError) as cm:
            self.s.assemble(st.id, self.root / "out.jpg")
        self.assertEqual(cm.exception.code, "incomplete")
        self.assertFalse((self.root / "out.jpg").exists())
        self.s.put_chunk(st.id, 2, iter([ch[2][:100], ch[2][100:]]))     # streamed in blocks
        r = self.s.assemble(st.id, self.root / "out.jpg")
        self.assertEqual(r["sha256"], self.sha)
        self.assertEqual((self.root / "out.jpg").read_bytes(), self.data)

    def test_resume_after_restart(self):
        st = self.s.create("p1", "a.tif", len(self.data), MIN_CHUNK_BYTES)
        ch = self._chunks(st)
        self.s.put_chunk(st.id, 0, [ch[0]])
        s2 = UploadStore(self.root / "u", min_free_bytes=0)       # new process, same disk
        self.assertEqual(s2.status(st.id)["received_chunks"], [0])

    def test_corruption_detected(self):
        st = self.s.create("p1", "a.jpg", len(self.data), MIN_CHUNK_BYTES, self.sha)
        ch = self._chunks(st)
        with self.assertRaises(UploadError) as cm:
            self.s.put_chunk(st.id, 0, [ch[0]], "0" * 64)
        self.assertEqual(cm.exception.code, "checksum_mismatch")
        self.assertEqual(self.s.received(st.id), [])             # bad chunk not kept
        with self.assertRaises(UploadError):
            self.s.put_chunk(st.id, 0, [ch[0][:-1]])              # wrong length
        # flipped byte with right length passes chunk check but fails whole-file declared checksum
        bad = bytearray(self.data); bad[5] ^= 0xFF
        cs = st.chunk_size
        for i in range(st.total_chunks):
            self.s.put_chunk(st.id, i, [bytes(bad[i * cs:(i + 1) * cs])])
        with self.assertRaises(UploadError) as cm:
            self.s.assemble(st.id, self.root / "o.jpg")
        self.assertEqual(cm.exception.code, "checksum_mismatch")
        self.assertFalse((self.root / "o.jpg").exists())

    def test_validation(self):
        for bad in ("evil.exe", "noext", "x.jpg.php"):
            with self.assertRaises(UploadError):
                self.s.create("p", bad, 10_000, MIN_CHUNK_BYTES)
        with self.assertRaises(UploadError):
            self.s.create("p", "a.jpg", 0, MIN_CHUNK_BYTES)
        with self.assertRaises(UploadError) as cm:
            self.s.create("p", "a.jpg", 10 ** 9, MIN_CHUNK_BYTES, max_file_bytes=10 ** 6)
        self.assertEqual(cm.exception.code, "too_large")
        with self.assertRaises(UploadError):
            self.s.create("p", "a.jpg", 100, 10)                  # chunk too small
        tight = UploadStore(self.root / "t", min_free_bytes=10 ** 18)
        with self.assertRaises(UploadError) as cm:
            tight.create("p", "a.jpg", 100, MIN_CHUNK_BYTES)
        self.assertEqual(cm.exception.code, "insufficient_storage")

    def test_no_file_count_limit(self):
        for i in range(300):                                      # nothing caps the number of uploads
            self.s.create("p", f"IMG_{i}.jpg", 1000, MIN_CHUNK_BYTES)
        self.assertEqual(len(list((self.root / "u").iterdir())), 300)

    def test_path_traversal(self):
        self.assertEqual(sanitize_filename("../../etc/passwd.jpg"), "passwd.jpg")
        self.assertEqual(sanitize_filename("C:\\x\\y\\z.JPG"), "z.JPG")
        with self.assertRaises(UploadError):
            self.s.status("../../etc")
        with self.assertRaises(UploadError):
            self.s.state("A" * 32)


class FlightTests(unittest.TestCase):
    def _grid(self, spacing_m, n=5, rel_alt=80.0):
        lon0, lat0 = -86.0, 39.0
        out = []
        for r in range(n):
            for c in range(n):
                x, y = c * spacing_m, r * spacing_m
                lon, lat = flight.local_m_to_lonlat(x, y, lon0, lat0)
                out.append(flight.ImageGeo(f"{r}_{c}", float(lon), float(lat), f"2026-05-01T10:{r:02d}:{c:02d}",
                                           rel_alt, 24.0, 0.0, 4000, 3000, -90.0))
        return out

    def test_geometry(self):
        self.assertEqual(flight.suggest_utm_epsg(-86.15, 39.77), 32616)
        self.assertEqual(flight.suggest_utm_epsg(18.4, -33.9), 32734)
        x, y = flight.lonlat_to_local_m(*flight.local_m_to_lonlat(120.0, -45.0, -86.0, 39.0), -86.0, 39.0)
        self.assertAlmostEqual(float(x), 120.0, places=4)
        self.assertAlmostEqual(float(y), -45.0, places=4)
        # DJI-class 4:3 camera, 24 mm eq, 80 m AGL -> ~2.2 cm GSD-ish (sanity band)
        self.assertTrue(1.0 < flight.ground_sample_distance_cm(80, 24, 4000, 4 / 3) < 4.0)

    def test_overlap_fraction(self):
        sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        self.assertAlmostEqual(flight.overlap_fraction(sq, sq), 1.0)
        self.assertAlmostEqual(flight.overlap_fraction(sq, sq + [5, 0]), 0.5)
        self.assertAlmostEqual(flight.overlap_fraction(sq, sq + [50, 0]), 0.0)

    def test_dense_grid_is_fine_and_sparse_grid_is_poor(self):
        # 80 m AGL, 24 mm eq, 4:3 -> footprint ~ 80*36/24*... ~120 x 90 m
        dense, sd = flight.analyse_overlap(self._grid(30))
        self.assertEqual(sd["poor_overlap"], 0)
        sparse, ss = flight.analyse_overlap(self._grid(200))
        self.assertEqual(ss["poor_overlap"], 25)
        self.assertTrue(all(r.poor and "overlap" in r.reason for r in sparse.values()))

    def test_unknown_stays_unknown(self):
        ims = [flight.ImageGeo("a", None, None), flight.ImageGeo("b", -86.0, 39.0, rel_alt=None)]
        res, summ = flight.analyse_overlap(ims)
        self.assertFalse(res["a"].footprint_available)
        self.assertIn("GPS", res["a"].reason)
        self.assertFalse(res["b"].footprint_available)
        self.assertFalse(res["a"].poor)         # not flagged poor: we simply do not know
        self.assertIsNone(flight.footprint_geojson(ims[1], -86.0, 39.0))

    def test_flight_grouping(self):
        ims = [flight.ImageGeo("a", 0, 0, "2026-05-01T10:00:00"), flight.ImageGeo("b", 0, 0, "2026-05-01T10:05:00"),
               flight.ImageGeo("c", 0, 0, "2026-05-01T14:00:00"), flight.ImageGeo("d", 0, 0, None)]
        g = flight.group_flights(ims)
        self.assertEqual((g["a"], g["b"], g["c"], g["d"]), (1, 1, 2, 0))

    def test_footprint_geojson_closed_ring(self):
        im = self._grid(30, n=1)[0]
        poly = flight.footprint_geojson(im, im.lon, im.lat)
        ring = poly["coordinates"][0]
        self.assertEqual(ring[0], ring[-1])
        self.assertEqual(len(ring), 5)


if __name__ == "__main__":
    unittest.main()
