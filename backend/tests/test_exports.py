import csv
import hashlib
import io
import json
import unittest
import zipfile

import numpy as np

from app.core import classes as C
from app.services import geotiff as gt
from tests.helpers import drain
from tests.test_terrain_api import TerrainBase


class ExportTests(TerrainBase):
    def prepare(self):
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j)
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "heuristic"})
        self.assertEqual(r.status_code, 202, r.text)
        drain(self.ctx)
        info = self.c.get(f"{self.u}/terrain").json()
        g = self.c.get(f"{self.u}/products").json()
        prod = {p["name"]: p for p in g} if isinstance(g, list) else {p["name"]: p for p in g.get("products", [])}
        grid = prod["dtm"]["meta"]["grid"]
        cx, cy = grid["x0"] + grid["width"] * grid["res"] / 2, grid["y0"] - grid["height"] * grid["res"] / 2
        for body in ({"kind": "footprint", "name": "House A", "geometry": {"type": "Polygon", "coordinates": [[[cx - 8, cy - 6], [cx + 8, cy - 6], [cx + 8, cy + 6], [cx - 8, cy + 6], [cx - 8, cy - 6]]]}},
                     {"kind": "profile", "name": "North/South cut", "geometry": {"type": "LineString", "coordinates": [[cx, cy - 30], [cx, cy + 30]]}}):
            self.assertEqual(self.c.post(f"{self.u}/features", json=body).status_code, 201)

    def export(self, **body):
        r = self.c.post(f"{self.u}/exports", json=body)
        self.assertEqual(r.status_code, 202, r.text)
        drain(self.ctx)
        j = self.c.get(f"/api/jobs/{r.json()['job_id']}").json()
        return r.json()["export_id"], j

    def test_full_export_is_complete_honest_and_intact(self):
        self.prepare()
        eid, j = self.export(items="all")
        self.assertEqual(j["status"], "succeeded", j)
        m = self.c.get(f"{self.u}/exports/{eid}").json()
        names = {f["name"]: f for f in m["files"]}
        for n in ("orthophoto.tif", "classification.tif", "classification_provenance.tif", "classification_classes.csv", "class_statistics.csv",
                  "land_cover_polygons.geojson", "dsm_OBSERVED_surface.tif", "dtm_ESTIMATED_bare_earth.tif", "terrain_confidence.tif",
                  "terrain_evidence.tif", "pointcloud.las", "pointcloud_ground_classified.las", "footprint_reports.json", "site_report.pdf",
                  "metadata.json", "README.txt", "features_epsg32616.geojson", "features_wgs84.geojson"):
            self.assertIn(n, names, sorted(names))
        self.assertTrue(any(n.startswith("profile_01_") and n.endswith(".csv") for n in names))
        # originals: the sample has no images -> skipped with a reason, not silently dropped
        self.assertTrue(any(s["item"] == "originals" for s in m["skipped"]))
        # every listed file downloads and matches its recorded checksum
        for n, f in names.items():
            r = self.c.get(f"{self.u}/exports/{eid}/files/{n}")
            self.assertEqual(r.status_code, 200, n)
            self.assertEqual(hashlib.sha256(r.content).hexdigest(), f["sha256"], n)
        # rasters keep their CRS
        d = self.ctx.storage.path(self.pid, f"exports/{eid}")
        for n in ("orthophoto.tif", "classification.tif", "dsm_OBSERVED_surface.tif", "dtm_ESTIMATED_bare_earth.tif", "terrain_confidence.tif"):
            gi = gt.read_georef(d / n)
            self.assertTrue(gi.georeferenced, n)
            self.assertEqual(gi.epsg, 32616, n)
        cls = gt.read_full(d / "classification.tif")
        self.assertLessEqual(int(cls.max()), 9)
        self.assertGreater(int((cls > 0).sum()), 0)
        # DSM (observed) and DTM (estimated) are different surfaces
        dsm, dtm = gt.read_full(d / "dsm_OBSERVED_surface.tif"), gt.read_full(d / "dtm_ESTIMATED_bare_earth.tif")
        self.assertGreater(float(np.nanmean(dsm - dtm)), 0.3)
        # class statistics add up to the raster
        rows = [r for r in csv.reader(l for l in (d / "class_statistics.csv").read_text().splitlines() if not l.startswith("#"))]
        self.assertEqual(sum(int(r[2]) for r in rows[1:]), cls.size)
        # GeoJSON: CRS declared, WGS84 copy is plausible lon/lat
        gj = json.loads((d / "features_epsg32616.geojson").read_text())
        self.assertIn("32616", gj["crs"]["properties"]["name"])
        w = json.loads((d / "features_wgs84.geojson").read_text())
        x, y = w["features"][0]["geometry"]["coordinates"][0][0][:2] if w["features"][0]["geometry"]["type"] == "Polygon" else w["features"][0]["geometry"]["coordinates"][0][:2]
        self.assertTrue(-180 < x < 180 and -90 < y < 90)
        # profile CSV labels estimated vs observed and carries the disclaimer
        prof = next(n for n in names if n.startswith("profile_01_"))
        txt = (d / prof).read_text()
        self.assertIn("ground_z_ESTIMATED", txt)
        self.assertIn("Preliminary planning estimate", txt)
        # PDF
        pdf = (d / "site_report.pdf").read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 20000)
        # metadata
        meta = json.loads((d / "metadata.json").read_text())
        self.assertEqual(meta["project"]["horizontal_epsg"], 32616)
        self.assertTrue(meta["project"]["synthetic_sample"])
        self.assertIn("estimated_bare_earth", meta["surface_definitions"])
        # bundle
        bundle = next(n for n in names if n.endswith(".zip"))
        with zipfile.ZipFile(d / bundle) as z:
            self.assertIn("dtm_ESTIMATED_bare_earth.tif", z.namelist())
            self.assertIn("metadata.json", z.namelist())
        # listing & delete
        self.assertEqual(self.c.get(f"{self.u}/exports").json()[0]["id"], eid)
        self.assertEqual(self.c.delete(f"{self.u}/exports/{eid}").status_code, 200)
        self.assertEqual(self.c.get(f"{self.u}/exports/{eid}").status_code, 404)

    def test_traversal_and_validation(self):
        self.prepare()
        eid, _ = self.export(items=["class_stats"])
        for bad in ("../../../project.json", "..%2F..%2Fx", "metadata.json/../../../db"):
            self.assertEqual(self.c.get(f"{self.u}/exports/{eid}/files/{bad}").status_code, 404)
        self.assertEqual(self.c.get(f"{self.u}/exports/..%2F..").status_code, 404)
        self.assertEqual(self.c.post(f"{self.u}/exports", json={"items": ["nope"]}).status_code, 422)
        self.assertEqual(self.c.post(f"{self.u}/exports", json={"items": []}).status_code, 422)

    def test_nothing_available_fails_honestly(self):
        # sample project before any terrain: terrain items are skipped; if *nothing* can be produced the job fails
        eid, j = self.export(items=["dtm", "confidence", "classified_cloud", "profiles"])
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error_code"], "nothing_to_export")
        self.assertIn("dtm", j["error"])

    def test_originals_zip(self):
        from tests.helpers import jpeg_bytes
        from tests.test_api_ingest import upload
        data = jpeg_bytes(3, gps=(39.0, -86.0, 250.0))
        for name in ("A.JPG", "B.JPG"):
            upload(self.c, self.pid, name, jpeg_bytes(len(name) + ord(name[0]), gps=(39.0, -86.0, 250.0)))
        drain(self.ctx)
        eid, j = self.export(items=["originals"])
        self.assertEqual(j["status"], "succeeded", j)
        d = self.ctx.storage.path(self.pid, f"exports/{eid}")
        with zipfile.ZipFile(d / "original_images.zip") as z:
            names = z.namelist()
            self.assertIn("images/A.JPG", names)
            self.assertIn("images_manifest.csv", names)
            self.assertEqual(z.read("images/A.JPG"), jpeg_bytes(ord("A") + 5, gps=(39.0, -86.0, 250.0)))      # byte-identical

    def test_lonlat_boundary_is_converted_to_project_crs(self):
        site_x, site_y = 571060.0, 4402045.0
        from app.core import crs as CRS
        lon, lat = (float(v) for v in CRS.utm_to_lonlat(site_x, site_y, 16, True))
        ring = [[lon, lat], [lon + 0.0004, lat], [lon + 0.0004, lat + 0.0003], [lon, lat]]
        r = self.c.patch(self.u, json={"boundary": {"type": "Polygon", "coordinates": [ring]}})
        self.assertEqual(r.status_code, 200, r.text)
        eid, j = self.export(items=["features"])
        self.assertEqual(j["status"], "succeeded", j)
        d = self.ctx.storage.path(self.pid, f"exports/{eid}")
        gj = json.loads((d / "features_epsg32616.geojson").read_text())
        x, y = gj["features"][0]["geometry"]["coordinates"][0][0]
        self.assertAlmostEqual(x, site_x, delta=0.01)
        self.assertAlmostEqual(y, site_y, delta=0.01)

    def test_photogrammetry_api(self):
        r = self.c.get(f"{self.u}/photogrammetry").json()
        self.assertIn("docker", r)
        self.assertTrue(r["blocking"])                                     # sample project has no images
        self.assertEqual(self.c.post(f"{self.u}/photogrammetry", json={"options": {"rm-rf": True}}).status_code, 422)
        self.assertEqual(self.c.post(f"{self.u}/photogrammetry", json={"preset": "nope"}).status_code, 422)
        ok = self.c.post(f"{self.u}/photogrammetry", json={"preset": "draft"})
        self.assertEqual(ok.status_code, 202, ok.text)
        self.assertEqual(self.c.post(f"{self.u}/photogrammetry", json={}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
