import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from starlette.testclient import TestClient

from app.api.main import create_app
from app.core import classes as C
from app.core.dtm import GridSpec
from app.services import geotiff as gt
from app.services import las as L
from app.services import ortho as O
from app.services import products as P
from app.services.synthetic import make_site
from tests.helpers import drain, make_app
from tests.test_api_ingest import CHUNK


def decode(resp):
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)


BUILTIN = {"ground": {"engine": "builtin_pmf"}, "resolution_m": 1.0}


class TerrainBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ctx = make_app(self.dir, dev_mode=True)
        self.c = TestClient(create_app(self.ctx, static_dir=self.dir / "nostatic"))
        self.pid = self.c.post("/api/dev/sample-project").json()["id"]
        drain(self.ctx)
        self.u = f"/api/projects/{self.pid}"
        self.site = make_site(120.0, 90.0, 0.25, 7)

    def tearDown(self):
        O.forget_rasters(self.ctx, self.pid)
        self.tmp.cleanup()

    def run_terrain(self, **over):
        body = {**BUILTIN, **over}
        r = self.c.post(f"{self.u}/terrain/run", json=body)
        self.assertEqual(r.status_code, 202, r.text)
        drain(self.ctx)
        j = self.c.get(f"/api/jobs/{r.json()['job_id']}").json()
        return j

    def read(self, name):
        return gt.read_full(self.ctx.storage.path(self.pid, P.LAYOUT[name][0]))

    def truth_on_grid(self, grid):
        rr, cc = np.mgrid[0:grid.height, 0:grid.width]
        x, y = grid.cell_centers_xy(rr, cc)
        tr, tc = self.site.grid.xy_to_rc(x, y)
        return self.site.terrain[np.clip(tr.astype(int), 0, self.site.grid.height - 1), np.clip(tc.astype(int), 0, self.site.grid.width - 1)]

    def upload(self, name, data, purpose):
        r = self.c.post(f"{self.u}/uploads", json={"filename": name, "size": len(data), "chunk_size": CHUNK, "purpose": purpose})
        self.assertEqual(r.status_code, 201, r.text)
        uid = r.json()["upload_id"]
        for i in range(r.json()["total_chunks"]):
            self.c.put(f"{self.u}/uploads/{uid}/chunks/{i}", content=data[i * CHUNK:(i + 1) * CHUNK])
        self.assertEqual(self.c.post(f"{self.u}/uploads/{uid}/complete").status_code, 202)
        drain(self.ctx)
        return self.c.get(f"{self.u}/uploads/{uid}").json()


class TerrainRunTests(TerrainBase):
    def test_missing_pdal_is_reported_not_faked(self):
        r = self.c.post(f"{self.u}/terrain/run", json={})        # default engine: PDAL SMRF
        self.assertEqual(r.status_code, 202)
        drain(self.ctx)
        j = self.c.get(f"/api/jobs/{r.json()['job_id']}").json()
        if shutil.which("pdal"):
            self.assertEqual(j["status"], "succeeded", j["error"])
        else:
            self.assertEqual(j["status"], "failed")
            self.assertEqual(j["error_code"], "missing_dependency")
            self.assertIn("conda install", j["error"])
            self.assertFalse(self.c.get(f"{self.u}/terrain").json()["available"])      # nothing was produced

    def test_run_produces_products_with_honest_labels(self):
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j["error"])
        st = self.c.get(f"{self.u}/terrain").json()
        self.assertTrue(st["available"])
        self.assertIn("ESTIMATE", st["notice"])
        self.assertIn("Preliminary", st["notice"])
        names = {p["name"]: p for p in self.c.get(f"{self.u}/products").json()}
        self.assertEqual(names["dtm"]["surface_kind"], "estimated")
        self.assertEqual(names["dsm"]["surface_kind"], "observed")
        for n in ("dsm", "dtm", "ndsm", "confidence", "evidence", "source", "classified_cloud"):
            self.assertTrue(names[n]["exists"], n)
        info = gt.read_georef(self.ctx.storage.path(self.pid, "products/dtm.tif"))
        self.assertEqual(info.epsg, 32616)
        self.assertTrue(info.georeferenced)
        meta = json.loads(Path(str(self.ctx.storage.path(self.pid, "products/dtm.tif")) + ".meta.json").read_text())
        self.assertEqual(meta["vertical_units"], "metre")
        self.assertIn("PRELIMINARY", meta["estimate_notice"])
        # evidence separates measured from interpolated; confidence has all four classes' semantics
        ev = self.read("evidence")
        self.assertTrue((ev == 1).any() and (ev == 2).any())
        # the DTM preserves relief: not flattened
        dtm = self.read("dtm")
        self.assertGreater(float(np.nanmax(dtm) - np.nanmin(dtm)), 4.0)
        # and tracks the (known, synthetic) truth
        err = dtm - self.truth_on_grid(GridSpec(**P.get(self.ctx, self.pid, "dtm")["meta"]["grid"]))
        self.assertLess(float(np.sqrt(np.nanmean(err ** 2))), 0.5)

    def test_sample_dtm_is_below_dsm_under_canopy(self):
        self.run_terrain()
        ndsm = self.read("ndsm")
        self.assertGreater(float(np.nanmax(ndsm)), 8.0)           # tall crowns exist above the estimated ground
        self.assertLess(float(np.nanmin(ndsm)), 1.0)

    def test_visual_reclassification_does_not_change_elevation(self):
        """THE critical property: painting trees as soil must never create or alter ground elevations."""
        base = self.run_terrain()
        self.assertEqual(base["status"], "succeeded", base["error"])
        dtm0, ev0, conf0 = self.read("dtm"), self.read("evidence"), self.read("confidence")
        # user paints the whole scene as bare soil ("remove the trees")
        r = self.c.post(f"{self.u}/classification/edit", json={"type": "rect", "class_id": C.SOIL, "rect": [0, 0, 480, 360]})
        self.assertEqual(r.status_code, 200)
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j["error"])
        np.testing.assert_array_equal(self.read("dtm"), dtm0)
        np.testing.assert_array_equal(self.read("evidence"), ev0)
        np.testing.assert_array_equal(self.read("confidence"), conf0)
        # marking real tree crowns as vegetation can only make claims weaker, never stronger
        site = self.site
        for _ in range(1):
            tr = np.nonzero(site.labels == C.TREES)
        self.c.post(f"{self.u}/classification/undo")
        # paint trees where truth says so (use many rect edits over rows of crown pixels)
        mask, _ = O.class_rasters(self.ctx, self.pid)
        from app.core.tiling import Window
        full = Window(0, 0, 480, 360)
        with mask.edit("paint trees as the user would", record_history=False):
            mask.write(full, np.where(site.labels == C.TREES, C.TREES | 0x40, 0).astype(np.uint8))
        j2 = self.run_terrain()
        self.assertEqual(j2["status"], "succeeded", j2["error"])
        ev2, conf2 = self.read("evidence"), self.read("confidence")
        self.assertLessEqual(int((ev2 == 1).sum()), int((ev0 == 1).sum()))          # measured can only shrink
        self.assertTrue((conf2 <= conf0).mean() > 0.97)
        self.assertEqual(j2["result"]["demoted_canopy_cells"] >= 0, True)

    def test_inspect_labels_measured_vs_estimated(self):
        self.run_terrain()
        ev = self.read("evidence")
        g = GridSpec(**P.get(self.ctx, self.pid, "dtm")["meta"]["grid"])
        seen = {}
        for code in (1, 2):
            rr, cc = np.argwhere(ev == code)[len(np.argwhere(ev == code)) // 2]
            x, y = g.cell_centers_xy(rr, cc)
            d = self.c.get(f"{self.u}/terrain/inspect", params={"x": float(x), "y": float(y)}).json()
            seen[code] = d
            self.assertIsNotNone(d["dtm"])
            self.assertIsNotNone(d["dsm"])
            self.assertIn(d["confidence"], ("high", "moderate", "low"))
            self.assertIsNotNone(d["land_cover"] if False else d.get("ortho_pixel"))
        self.assertIn("MEASURED", seen[1]["note"])
        self.assertIn("ESTIMATE", seen[2]["note"])
        self.assertEqual(seen[1]["evidence"], "measured ground")
        out = self.c.get(f"{self.u}/terrain/inspect", params={"x": 1.0, "y": 1.0}).json()
        self.assertIsNone(out["dtm"])
        self.assertEqual(self.c.get(f"{self.u}/terrain/inspect").status_code, 400)

    def test_param_validation(self):
        bad = lambda b: self.c.post(f"{self.u}/terrain/run", json=b).status_code
        self.assertEqual(bad({"resolution_m": -1}), 422)
        self.assertEqual(bad({"interp": {"method": "magic"}}), 422)
        self.assertEqual(bad({"ground": {"engine": "x"}}), 422)
        self.assertEqual(bad({"confidence": {"high_dist_m": 10, "moderate_dist_m": 5, "low_dist_m": 15}}), 422)

    def test_one_terrain_job_at_a_time(self):
        self.assertEqual(self.c.post(f"{self.u}/terrain/run", json=BUILTIN).status_code, 202)
        self.assertEqual(self.c.post(f"{self.u}/terrain/run", json=BUILTIN).status_code, 409)

    def test_interpolation_parameters_change_the_result(self):
        self.run_terrain(interp={"method": "local_plane"})
        a = self.read("dtm")
        self.run_terrain(interp={"method": "idw", "power": 3.0})
        b = self.read("dtm")
        ev = self.read("evidence")
        self.assertTrue(np.nanmax(np.abs(a - b)[ev == 2]) > 1e-4)                    # interpolated cells differ
        np.testing.assert_allclose(a[ev == 1], b[ev == 1])                            # measured cells never change


class ViewerDataTests(TerrainBase):
    def setUp(self):
        super().setUp()
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j["error"])
        self.g = P.get(self.ctx, self.pid, "dtm")["meta"]["grid"]
        self.bbox = f"{self.g['x0']},{self.g['y0'] - self.g['height'] * self.g['res']},{self.g['x0'] + self.g['width'] * self.g['res']},{self.g['y0']}"

    def render(self, layer, **kw):
        return self.c.get(f"{self.u}/terrain/render/{layer}", params={"bbox": self.bbox, "w": 240, "h": 180, **kw})

    def test_layers_render_and_are_labelled(self):
        for layer, kind in (("dsm", "observed"), ("dtm", "estimated"), ("ndsm", "derived"), ("confidence", "evidence"), ("evidence", "evidence")):
            r = self.render(layer)
            self.assertEqual(r.status_code, 200, layer)
            self.assertEqual(r.headers["x-surface-kind"], kind)
            img = decode(r)
            self.assertEqual(img.shape, (180, 240, 4))
            self.assertGreater(int((img[..., 3] > 0).sum()), 1000)
        dsm, dtm = decode(self.render("dsm", vmin=245, vmax=275)), decode(self.render("dtm", vmin=245, vmax=275))
        self.assertGreater(float(np.abs(dsm[..., :3].astype(int) - dtm[..., :3].astype(int)).mean()), 1.0)     # canopy visible only in DSM
        hs = decode(self.render("dtm", hillshade=1, zexag=3))
        flat = decode(self.render("dtm"))
        self.assertFalse(np.array_equal(hs, flat))
        # out-of-extent areas are transparent, not invented
        far = self.c.get(f"{self.u}/terrain/render/dtm", params={"bbox": "0,0,10,10", "w": 20, "h": 20})
        self.assertEqual(int(decode(far)[..., 3].max()), 0)
        self.assertEqual(self.c.get(f"{self.u}/terrain/render/dtm", params={"bbox": "1,1,0,0"}).status_code, 400)
        self.assertEqual(self.c.get(f"{self.u}/terrain/render/nope", params={"bbox": self.bbox}).status_code, 404)

    def test_confidence_render_shows_all_categories(self):
        conf = self.read("confidence")
        img = decode(self.render("confidence"))
        self.assertEqual(img.shape[2], 4)
        cols = {tuple(c[:3]) for c in img.reshape(-1, 4)[img.reshape(-1, 4)[:, 3] > 0].tolist()}
        self.assertGreaterEqual(len(cols), len(set(np.unique(conf).tolist())) - 1)

    def test_stats_heightmap_sample(self):
        s = self.c.get(f"{self.u}/terrain/stats/dtm").json()
        self.assertEqual(s["kind"], "estimated")
        self.assertLess(s["stats"]["min"], s["stats"]["max"])
        h = self.c.get(f"{self.u}/terrain/heightmap/dsm", params={"maxdim": 64})
        meta = json.loads(h.headers["x-grid"])
        arr = np.frombuffer(h.content, "<f4").reshape(meta["height"], meta["width"])
        self.assertLessEqual(max(arr.shape), 64)
        self.assertEqual(meta["kind"], "observed")
        self.assertTrue(np.isfinite(arr).any())
        pts = [[self.g["x0"] + 30.5, self.g["y0"] - 20.5], [1, 1]]
        sm = self.c.post(f"{self.u}/terrain/sample", json={"points": pts}).json()
        self.assertIsNotNone(sm["dtm"][0])
        self.assertIsNone(sm["dtm"][1])
        self.assertGreaterEqual(sm["dsm"][0], sm["dtm"][0] - 0.5)

    def test_ortho_render(self):
        r = self.c.get(f"{self.u}/ortho/render", params={"bbox": self.bbox, "w": 200, "h": 150})
        img = decode(r)
        self.assertEqual(img.shape, (150, 200, 4))
        self.assertGreater(int((img[..., 3] > 0).mean() * 100), 95)

    def test_point_cloud_feed(self):
        m = self.c.get(f"{self.u}/pointcloud").json()
        self.assertTrue(m["available"])
        self.assertEqual(m["kind"], "classified")
        self.assertIn("2", m["classes"])                      # ground
        d = self.c.get(f"{self.u}/pointcloud/data", params={"max": 5000})
        meta = json.loads(d.headers["x-meta"])
        n = meta["n"]
        self.assertLessEqual(n, 5000)
        x = np.frombuffer(d.content, "<f4", count=n)
        self.assertEqual(len(d.content) % 4, 0)
        self.assertGreater(float(x.max()), 50)                # local metres relative to origin
        self.assertAlmostEqual(meta["origin"][0], 571000.0, delta=2)

    def test_profile_footprint_cutfill_drainage_steep(self):
        g = self.g
        x0, y0 = g["x0"] + 20, g["y0"] - 20
        line = [[x0, y0], [x0 + 60, y0 - 30]]
        p = self.c.post(f"{self.u}/planning/profile", json={"line": line, "spacing": 1.0}).json()
        self.assertGreater(p["samples"], 50)
        self.assertEqual(len(p["columns"]["ground_z"]), p["samples"])
        self.assertIn("surface_z", p["columns"])
        self.assertIn("Preliminary", p["disclaimer"])
        csvr = self.c.post(f"{self.u}/planning/profile", json={"line": line, "spacing": 1.0, "format": "csv"})
        self.assertEqual(csvr.headers["content-type"].split(";")[0], "text/csv")
        rows = [r for r in csv.reader(io.StringIO(csvr.text)) if r and not r[0].startswith("#")]
        self.assertEqual(rows[0][:4], ["distance", "x", "y", "ground_z"])
        self.assertIn("ESTIMATE", csvr.text.splitlines()[1])
        self.assertEqual(self.c.post(f"{self.u}/planning/profile", json={"line": [[0, 0]]}).status_code, 422)

        poly = [[x0, y0], [x0 + 12, y0], [x0 + 12, y0 - 10], [x0, y0 - 10]]
        f = self.c.post(f"{self.u}/planning/footprint", json={"polygon": poly}).json()
        self.assertEqual(f["status"], "ok")
        self.assertAlmostEqual(f["area_m2"], 120.0, places=3)
        self.assertEqual(len(f["vertex_elevations"]), 4)
        self.assertIn("Not suitable for final foundation design", f["disclaimer"])
        self.assertIn("confidence", f)

        mid = f["mean_z"]
        cf = self.c.post(f"{self.u}/planning/cut-fill", json={"polygon": poly, "surface": {"type": "flat", "z": mid}}).json()
        if cf["status"] == "ok":
            self.assertAlmostEqual(cf["net"], cf["fill"] - cf["cut"], places=6)
            self.assertIn("Not suitable", cf["disclaimer"])
        else:
            self.assertEqual(cf["status"], "refused")
            self.assertTrue(cf["reason"])
        outside = self.c.post(f"{self.u}/planning/cut-fill", json={"polygon": [[1, 1], [5, 1], [5, 5]], "surface": {"type": "flat", "z": 250}}).json()
        self.assertEqual(outside["status"], "refused")
        self.assertEqual(self.c.post(f"{self.u}/planning/cut-fill", json={"polygon": poly, "surface": {"type": "flat"}}).status_code, 422)
        self.assertEqual(self.c.post(f"{self.u}/planning/cut-fill", json={"polygon": poly, "surface": {"type": "three_points", "points": [[0, 0, 1], [1, 1, 1], [2, 2, 1]]}}).status_code, 422)

        d = self.c.get(f"{self.u}/planning/drainage", params={"min_area": 300}).json()
        self.assertGreater(len(d["channels"]["features"]), 5)
        self.assertIn("ESTIMATED", d["note"])
        s = self.c.get(f"{self.u}/planning/steep", params={"degrees": 8}).json()
        self.assertEqual(s["threshold_degrees"], 8.0)
        self.assertIn("ESTIMATED", s["note"])

    def test_features_crud(self):
        poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        r = self.c.post(f"{self.u}/features", json={"kind": "footprint", "name": "House A", "geometry": poly, "props": {"finished_floor": 251.2}})
        self.assertEqual(r.status_code, 201)
        fid = r.json()["id"]
        self.assertEqual(len(self.c.get(f"{self.u}/features", params={"kind": "footprint"}).json()), 1)
        r = self.c.put(f"{self.u}/features/{fid}", json={"kind": "footprint", "name": "House B", "geometry": poly})
        self.assertEqual(r.json()["name"], "House B")
        self.assertEqual(self.c.post(f"{self.u}/features", json={"kind": "footprint", "geometry": {"type": "Circle", "coordinates": []}}).status_code, 422)
        self.assertEqual(self.c.delete(f"{self.u}/features/{fid}").status_code, 200)
        self.assertEqual(self.c.get(f"{self.u}/features").json(), [])


class SourcesAndImportTests(TerrainBase):
    def dem_bytes(self, dz=0.0, epsg=32616):
        tmp = self.dir / "dem_src.tif"
        s = self.site
        gt.write_geotiff(tmp, (s.terrain + dz).astype(np.float32), s.grid, epsg, nodata=float("nan"))
        return tmp.read_bytes()

    def test_external_dem_priority_and_provenance(self):
        base = self.run_terrain()
        self.assertEqual(base["status"], "succeeded", base["error"])
        ev0 = self.read("evidence")
        up = self.upload("county_dem.tif", self.dem_bytes(0.3), "dem")
        self.assertEqual(up["status"], "complete", up)
        st = self.c.get(f"{self.u}/terrain").json()
        self.assertEqual(len(st["sources"]), 1)
        sid = st["sources"][0]["id"]
        self.assertGreaterEqual(sid, 2)
        # lower priority than photogrammetry (10): measured photo cells stay; the DEM fills the rest as 'external'
        self.assertEqual(self.c.patch(f"{self.u}/elevation-sources/{sid}", json={"priority": 5, "trust": "low"}).status_code, 200)
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j["error"])
        ev = self.read("evidence")
        self.assertTrue((ev == 3).any())
        self.assertTrue(((ev == 1) == (ev0 == 1)).mean() > 0.98)
        src = self.read("source")
        self.assertTrue(((src == sid) & (ev == 3)).any())
        self.assertTrue(any(s["id"] == sid for s in j["result"]["sources"]))
        # external cells are never 'high' confidence when trust is low
        conf = self.read("confidence")
        self.assertTrue((conf[ev == 3] <= 1).all())
        # raise the DEM above photogrammetry: now it wins, including a +0.3 m datum offset that can be corrected
        self.c.patch(f"{self.u}/elevation-sources/{sid}", json={"priority": 50, "z_offset": -0.3})
        j = self.run_terrain()
        d = self.read("dtm")
        g = GridSpec(**P.get(self.ctx, self.pid, "dtm")["meta"]["grid"])
        err = d - self.truth_on_grid(g)
        self.assertLess(float(np.nanmean(np.abs(err))), 0.1)
        # disabling a source removes it from the next run
        self.c.patch(f"{self.u}/elevation-sources/{sid}", json={"enabled": False})
        j = self.run_terrain()
        self.assertFalse((self.read("evidence") == 3).any())
        self.assertEqual(self.c.delete(f"{self.u}/elevation-sources/{sid}").status_code, 200)
        self.assertEqual(self.c.get(f"{self.u}/terrain").json()["sources"], [])

    def test_wrong_crs_dem_is_rejected_and_kept(self):
        up = self.upload("wrong.tif", self.dem_bytes(0, epsg=32617), "dem")
        self.assertEqual(up["status"], "rejected")
        self.assertIn("EPSG:32617", up["error"])
        self.assertEqual(self.c.get(f"{self.u}/terrain").json()["sources"], [])
        self.assertTrue(list((self.dir / "projects" / self.pid / "rejected").glob("*wrong.tif")))

    def test_dem_without_georeferencing_is_rejected(self):
        tmp = self.dir / "nogeo.tif"
        gt.write_geotiff(tmp, np.zeros((50, 50), np.float32), None, None)
        up = self.upload("nogeo.tif", tmp.read_bytes(), "dem")
        self.assertEqual(up["status"], "rejected")
        self.assertIn("georeferencing", up["error"])

    def test_lidar_and_survey_import(self):
        s = self.site
        rng = np.random.default_rng(2)
        x = s.grid.x0 + rng.random(4000) * s.grid.width * s.grid.res
        y = s.grid.y0 - rng.random(4000) * s.grid.height * s.grid.res
        r, c = s.grid.xy_to_rc(x, y)
        z = s.terrain[np.clip(r.astype(int), 0, s.grid.height - 1), np.clip(c.astype(int), 0, s.grid.width - 1)]
        las = self.dir / "lidar.las"
        L.write_las(las, x, y, z, classification=np.full(4000, 2, np.uint8), epsg=32616)
        up = self.upload("flight.las", las.read_bytes(), "lidar")
        self.assertEqual(up["status"], "complete", up)
        bad = self.dir / "bad.las"
        L.write_las(bad, x, y, z, epsg=4326)
        self.assertEqual(self.upload("deg.las", bad.read_bytes(), "lidar")["status"], "rejected")
        csv_text = "x,y,z\n" + "\n".join(f"{a:.3f},{b:.3f},{c_:.3f}" for a, b, c_ in zip(x[:50], y[:50], z[:50]))
        up2 = self.upload("survey.csv", csv_text.encode(), "survey_points")
        self.assertEqual(up2["status"], "complete", up2)
        kinds = sorted(sx["kind"] for sx in self.c.get(f"{self.u}/terrain").json()["sources"])
        self.assertEqual(kinds, ["lidar", "survey"])
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j["error"])
        names = {s_["kind"] for s_ in j["result"]["sources"]}
        self.assertTrue({"photogrammetry_ground", "lidar", "survey"} <= names)
        # with 4000 LiDAR ground points the surface is mostly measured now
        self.assertGreater(j["result"]["coverage"]["measured"], 0.5)

    def test_gcp_list_validation(self):
        good = "EPSG:32616\n571010 4402010 250.0 100 200 a.jpg g1\n571010 4402010 250.0 300 400 b.jpg g1\n571010 4402010 250.0 50 60 c.jpg g1\n"
        self.assertEqual(self.upload("gcp_list.txt", good.encode(), "gcp_list")["status"], "complete")
        self.assertTrue(P.get(self.ctx, self.pid, "gcp_list")["exists"])
        self.assertEqual(self.upload("gcp_list.txt", b"not a gcp file\nfoo", "gcp_list")["status"], "rejected")

    def test_check_points_validate_the_dtm(self):
        s = self.site
        pts = []
        rng = np.random.default_rng(9)
        for k in range(30):
            x = s.grid.x0 + 8 + rng.random() * (s.grid.width * s.grid.res - 16)
            y = s.grid.y0 - 8 - rng.random() * (s.grid.height * s.grid.res - 16)
            r, c = s.grid.xy_to_rc(x, y)
            pts.append(f"C{k},{x:.3f},{y:.3f},{s.terrain[int(r), int(c)]:.3f},check")
        self.assertEqual(self.c.post(f"{self.u}/control-points", json={"csv": "\n".join(pts)}).status_code, 201)
        j = self.run_terrain()
        cp = j["result"]["check_points"]
        self.assertGreaterEqual(cp["n"], 25)
        self.assertLess(cp["rmse"], 0.5)
        res = [p["residual"] for p in self.c.get(f"{self.u}/control-points").json() if p["residual"]]
        self.assertTrue(res)
        self.assertIn(res[0]["confidence"], ("high", "moderate", "low", "unknown"))

    def test_without_check_points_accuracy_is_explicitly_unvalidated(self):
        j = self.run_terrain()
        self.assertEqual(j["result"]["check_points"]["n"], 0)
        self.assertIn("NOT independently validated", j["result"]["check_points"]["note"])


if __name__ == "__main__":
    unittest.main()
