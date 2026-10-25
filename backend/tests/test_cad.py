import math
import unittest

import numpy as np

from app.services import cad
from tests.test_exports import ExportTests as _Base


def parse_dxf(text: str):
    """Group-code/value pairs -> list; fails if the stream is not strictly code/value pairs."""
    lines = text.replace("\r\n", "\n").split("\n")
    assert lines[-1] == "", "DXF must end with a newline"
    lines = lines[:-1]
    assert len(lines) % 2 == 0, "odd number of lines"
    return [(int(lines[i].strip()), lines[i + 1]) for i in range(0, len(lines), 2)]


class ContourTests(unittest.TestCase):
    def test_plane_gives_straight_line_at_exact_position(self):
        y, x = np.mgrid[0:20, 0:30]
        z = (x * 0.5).astype(float)
        lines = cad.contour_lines(z, 5.25)
        self.assertEqual(len(lines), 1)
        cols = {round(c, 6) for _, c in lines[0]}
        self.assertEqual(cols, {10.5})
        self.assertEqual(sorted(r for r, _ in lines[0])[0], 0.0)
        self.assertEqual(sorted(r for r, _ in lines[0])[-1], 19.0)

    def test_cone_gives_one_closed_ring_with_right_radius(self):
        y, x = np.mgrid[0:41, 0:41]
        z = np.hypot(x - 20, y - 20).astype(float)
        lines = cad.contour_lines(z, 8.0)
        self.assertEqual(len(lines), 1)
        ring = lines[0]
        self.assertEqual(ring[0], ring[-1])
        rad = [math.hypot(r - 20, c - 20) for r, c in ring]
        self.assertLess(max(abs(v - 8.0) for v in rad), 0.05)

    def test_nan_cells_are_skipped_not_bridged(self):
        y, x = np.mgrid[0:10, 0:10]
        z = (x * 1.0).astype(float)
        z[:, 4:6] = np.nan
        lines = cad.contour_lines(z, 7.5)
        self.assertEqual(len(lines), 1)
        self.assertEqual(cad.contour_lines(z, 4.5), [])           # crosses only through the NaN gap

    def test_nice_interval(self):
        self.assertEqual(cad.nice_interval(10), 0.5)
        self.assertEqual(cad.nice_interval(100), 5.0)
        self.assertGreater(cad.nice_interval(0), 0)

    def test_terrain_contours_split_by_confidence_and_carry_z(self):
        y, x = np.mgrid[0:60, 0:60]
        dem = (x * 0.2).astype(float)                              # 0..11.8
        conf = np.full(dem.shape, 3, np.uint8)
        conf[:, 30:] = 1                                           # right half low confidence
        r = cad.terrain_contours(dem, 1000.0, 2000.0, 1.0, 2.0, conf, smooth_cells=0)
        self.assertTrue(r["lines"])
        weak = [l for l in r["lines"] if l[2]]
        strong = [l for l in r["lines"] if not l[2]]
        self.assertTrue(weak and strong)
        for level, major, w, pts in r["lines"]:
            self.assertTrue(all(abs(p[2] - level) < 1e-9 for p in pts))
            xs = [p[0] for p in pts]
            self.assertTrue(all(1000.0 <= v <= 1060.0 for v in xs))
            # contours here are vertical lines: a weak run must sit in the low-confidence half, a strong run in the other
            if w:
                self.assertGreaterEqual(min(xs), 1029.4)
            else:
                self.assertLessEqual(max(xs), 1030.6)
        self.assertTrue(any(l[1] for l in r["lines"]))             # some major lines (every 5th interval = 10)


class DxfWriterTests(unittest.TestCase):
    def test_structure(self):
        d = cad.Dxf("metre")
        d.layer("A", 1)
        d.polyline("A", [(0, 0), (10, 0), (10, 5)], closed=True)
        d.polyline("A", [(0, 0, 1.5), (3, 3, 1.5)], elevated=True)
        d.point("A", 1, 2, 3)
        d.text("A", 0, 0, 1, "héllo\nworld")
        pairs = parse_dxf(d.dumps())
        self.assertEqual(pairs[0], (0, "SECTION"))
        self.assertEqual(pairs[-1], (0, "EOF"))
        ents = [v for c, v in pairs if c == 0]
        self.assertEqual(ents.count("POLYLINE"), ents.count("SEQEND"), "every POLYLINE needs a SEQEND")
        self.assertEqual(ents.count("VERTEX"), 5)
        self.assertEqual(ents.count("SECTION"), ents.count("ENDSEC"))
        txt = [v for c, v in pairs if c == 1 and v != "AC1009"]
        self.assertEqual(txt, ["h?llo world"])                      # non-ASCII is replaced rather than corrupting the file


class CadExportTests(_Base):
    def test_cad_export_end_to_end(self):
        self.prepare()
        self.c.post(f"{self.u}/control-points", json={"csv": "label,x,y,z\nGCP1,571050,4402050,252.5\n", "kind": "gcp", "replace": True})
        eid, j = self.export(items=["cad_dxf"], contour_interval=1.0)
        self.assertEqual(j["status"], "succeeded", j)
        m = self.c.get(f"{self.u}/exports/{eid}").json()
        f = next(x for x in m["files"] if x["item"] == "cad_dxf")
        self.assertTrue(f["name"].endswith("_PRELIMINARY.dxf") and "epsg32616" in f["name"])
        self.assertEqual(f["cad"]["contour_interval"], 1.0)
        raw = self.c.get(f"{self.u}/exports/{eid}/files/{f['name']}").content.decode("ascii")
        pairs = parse_dxf(raw)
        layers = {v for c, v in pairs if c == 8}
        for need in ("TS_BOUNDARY", "TS_FOOTPRINTS", "TS_PROFILES", "TS_CONTROL_GCP", "TS_NOTES", "TS_CONTOUR_EST_MINOR", "TS_CONTOUR_LABELS"):
            self.assertIn(need, layers, sorted(layers))
        ents = [v for c, v in pairs if c == 0]
        self.assertEqual(ents.count("POLYLINE"), ents.count("SEQEND"))
        # contour vertices are 3D at integer metres inside the DTM's elevation range
        zs, poly_layer, kind = [], None, None
        for c, v in pairs:
            if c == 0:
                kind = v
                if v == "POLYLINE":
                    poly_layer = None
            elif c == 8 and kind == "POLYLINE" and poly_layer is None:
                poly_layer = v
            elif c == 30 and kind == "VERTEX" and poly_layer in ("TS_CONTOUR_EST_MINOR", "TS_CONTOUR_EST_MAJOR"):
                zs.append(float(v))
        self.assertTrue(zs)
        self.assertTrue(all(abs(z - round(z)) < 1e-6 for z in zs))
        lo, hi = self.c.get(f"{self.u}/terrain/stats/dtm").json()["stats"]["min"], self.c.get(f"{self.u}/terrain/stats/dtm").json()["stats"]["max"]
        self.assertTrue(all(lo - 1 <= z <= hi + 1 for z in zs))
        # the preliminary-estimate note is IN the drawing
        self.assertTrue(any("PRELIMINARY PLANNING ESTIMATE" in v for c, v in pairs if c == 1))
        # the catalog offers it
        cat = {i["item"]: i for i in self.c.get(f"{self.u}/exports/catalog").json()["items"]}
        self.assertTrue(cat["cad_dxf"]["available"])

    def test_cad_with_nothing_to_draw_is_refused_honestly(self):
        pid = self.c.post("/api/projects", json={"name": "empty"}).json()["id"]
        r = self.c.post(f"/api/projects/{pid}/exports", json={"items": ["cad_dxf"]})
        self.assertEqual(r.status_code, 202)
        from tests.helpers import drain
        drain(self.ctx)
        j = self.c.get(f"/api/jobs/{r.json()['job_id']}").json()
        self.assertIn(j["status"], ("failed",))
        self.assertIn("nothing", (j["error"] or "").lower())


if __name__ == "__main__":
    unittest.main()
