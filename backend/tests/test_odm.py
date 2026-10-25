"""ODM orchestration tests using a STUB ``docker`` executable.

REAL OpenDroneMap IS NOT RUN HERE. The stub imitates only the command-line contract this platform relies on (``info``,
``run --rm IMAGE --help``, ``run ... code``, ``stop``) and writes tiny synthetic outputs. These tests prove the platform's
own behaviour (option verification, input linking, progress, output validation, failure reporting, product registration);
they say nothing about ODM's photogrammetry. See README "Unverified".
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from app.api.main import create_app
from app.services import products as P
from tests.helpers import drain, jpeg_bytes, make_app
from tests.test_api_ingest import upload

BACKEND = str(Path(__file__).resolve().parents[1])

STUB = '''#!{py}
import sys, os, json
sys.path.insert(0, {backend!r})
mode = open(os.path.join(os.path.dirname(__file__), "mode")).read().strip()
a = sys.argv[1:]
if a[:1] == ["info"]:
    sys.exit(0 if mode != "nodaemon" else 1)
if a[:1] == ["stop"]:
    sys.exit(0)
HELP = "--project-path --dsm --dtm --pc-las --pc-quality --feature-quality --orthophoto-resolution --dem-resolution --gcp --fast-orthophoto"
if mode == "badopt":
    HELP = HELP.replace("--dtm", "")
if "--help" in a:
    print("usage: run.py " + HELP); sys.exit(0)
work = a[a.index("-v") + 1].split(":")[0]
log = os.path.join(os.path.dirname(__file__), "calls.json")
json.dump(a, open(log, "w"))
for st in ("dataset", "opensfm", "odm_dem", "odm_orthophoto"):
    print("[INFO]    Running %s stage" % st, flush=True)
if mode == "crash":
    print("[ERROR] out of memory"); sys.exit(137)
import numpy as np
from app.core.dtm import GridSpec
from app.services import geotiff as gt, las
g = GridSpec(571000.0, 4402030.0, 1.0, 40, 30)
epsg = 32617 if mode == "wrongcrs" else 32616
def put(rel, arr, e=epsg):
    p = os.path.join(work, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    gt.write_geotiff(__import__("pathlib").Path(p), arr, g, e)
rgb = np.full((30, 40, 3), 120, np.uint8)
put("odm_orthophoto/odm_orthophoto.tif", rgb)
if mode != "nodsm":
    put("odm_dem/dsm.tif", np.full((30, 40), 250.0, np.float32))
put("odm_dem/dtm.tif", np.full((30, 40), 248.0, np.float32))
os.makedirs(os.path.join(work, "odm_georeferencing"), exist_ok=True)
r = np.random.default_rng(0)
las.write_las(__import__("pathlib").Path(work, "odm_georeferencing/odm_georeferenced_model.las"),
              571000 + r.random(500) * 40, 4402000 + r.random(500) * 30, 250 + r.random(500), epsg=epsg)
if mode != "nomesh":
    td = os.path.join(work, "odm_texturing"); os.makedirs(td, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(os.path.join(td, "tex0.png"))
    open(os.path.join(td, "odm_textured_model_geo.mtl"), "w").write("newmtl m0\\nKd 1 1 1\\nmap_Kd tex0.png\\n")
    ox, oy = (571000.0, 4402000.0) if mode == "offsetmesh" else (0.0, 0.0)
    if mode == "offsetmesh":
        os.makedirs(os.path.join(work, "odm_georeferencing"), exist_ok=True)
        open(os.path.join(work, "odm_georeferencing/coords.txt"), "w").write("WGS84 UTM 16N\\n571000 4402000\\n")
    vv = [(571000.0 + 5 - ox, 4402000.0 + 5 - oy, 250.0), (571000.0 + 25 - ox, 4402000.0 + 5 - oy, 250.0),
          (571000.0 + 25 - ox, 4402000.0 + 20 - oy, 255.0), (571000.0 + 5 - ox, 4402000.0 + 20 - oy, 255.0)]
    with open(os.path.join(td, "odm_textured_model_geo.obj"), "w") as fo:
        fo.write("mtllib odm_textured_model_geo.mtl\\n")
        for a in vv: fo.write("v %.3f %.3f %.3f\\n" % a)
        fo.write("vt 0 0\\nvt 1 0\\nvt 1 1\\nvt 0 1\\nusemtl m0\\nf 1/1 2/2 3/3\\nf 1/1 3/3 4/4\\n")
os.makedirs(os.path.join(work, "odm_report"), exist_ok=True)
json.dump({{"reconstruction_statistics": {{"reconstructed_shots_count": 8, "initial_shots_count": 9,
          "reprojection_error_pixels": 0.7}}}}, open(os.path.join(work, "odm_report/stats.json"), "w"))
'''


class OdmStubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bin = self.dir / "bin"
        self.bin.mkdir()
        stub = self.bin / "docker"
        stub.write_text(STUB.format(py=sys.executable, backend=BACKEND))
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.set_mode("ok")
        self.ctx = make_app(self.dir / "data", docker_bin=str(stub))
        self.c = TestClient(create_app(self.ctx, static_dir=self.dir / "nostatic"))
        self.pid = self.c.post("/api/projects", json={"name": "ODM stub", "epsg": 32616}).json()["id"]
        self.add_images(9)

    def tearDown(self):
        self.ctx.db.close()
        self.tmp.cleanup()

    def set_mode(self, m):
        (self.bin / "mode").write_text(m)

    def add_images(self, n):
        for k in range(n):
            lat, lon = 39.0 + (k // 3) * 0.00018, -86.0 + (k % 3) * 0.00023
            r = upload(self.c, self.pid, f"IMG_{k}.JPG", jpeg_bytes(k, gps=(lat, lon, 250.0), rel_alt=40.0))
            assert r["complete"].status_code == 202
        drain(self.ctx)

    def run_job(self, **p):
        jid = self.ctx.queue.enqueue("odm", self.pid, {"project_id": self.pid, **p})
        drain(self.ctx)
        return self.ctx.db.one("SELECT * FROM jobs WHERE id=?", (jid,))

    def test_success_registers_products_and_quality(self):
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)
        for kind in ("pointcloud", "odm_dsm", "odm_dtm", "odm_quality"):
            self.assertTrue(P.get(self.ctx, self.pid, kind)["exists"], kind)
        import json
        res = json.loads(j["result_json"])
        names = {c["name"]: c["status"] for c in res["checks"]}
        self.assertEqual(names["orthomosaic produced"], "pass")
        self.assertEqual(names["georeferenced outputs"], "pass")
        self.assertEqual(res["stats"]["images_reconstructed"], 8)
        # inputs were hard-linked (or copied) into the ODM working directory, one per image
        imgs = list((self.ctx.storage.path(self.pid, "odm") / "images").iterdir())
        self.assertEqual(len(imgs), 9)
        # the command line the stub saw carries the mandatory options and the image/volume contract
        call = json.loads((self.bin / "calls.json").read_text())
        for o in ("--dsm", "--dtm", "--pc-las", "--project-path"):
            self.assertIn(o, call)
        self.assertEqual(call[-1], "code")
        # a follow-up job prepares the orthomosaic for the editor
        self.assertTrue(self.ctx.db.one("SELECT 1 FROM jobs WHERE kind='prepare_ortho'"))

    def test_build_terrain_chain_queues_terrain_after_success_only(self):
        import json
        j = self.run_job(terrain_chain={"ground": {"engine": "builtin_pmf"}, "resolution_m": 0.5})
        self.assertEqual(j["status"], "succeeded", j)
        row = self.ctx.db.one("SELECT params_json FROM jobs WHERE kind='terrain'")
        self.assertIsNotNone(row, "terrain job should be queued after a successful ODM run")
        self.assertEqual(json.loads(row["params_json"])["params"]["ground"]["engine"], "builtin_pmf")

    def test_build_terrain_api_picks_engine_and_refuses_missing_pdal(self):
        self.add_images(3)
        r = self.c.post(f"/api/projects/{self.pid}/photogrammetry", json={"preset": "draft", "build_terrain": True, "ground_engine": "pdal_smrf"})
        self.assertEqual(r.status_code, 409, r.text)                       # no PDAL in the test environment: refused, not silently swapped
        self.assertEqual(r.json()["error"]["code"], "no_pdal")
        r = self.c.post(f"/api/projects/{self.pid}/photogrammetry", json={"preset": "draft", "build_terrain": True})
        self.assertEqual(r.status_code, 202, r.text)
        self.assertEqual(r.json()["ground_engine"], "builtin_pmf")
        self.assertIn("built-in ground filter", r.json()["note"])

    def test_textured_model_prepared_and_served(self):
        import json
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)
        m = self.c.get(f"/api/projects/{self.pid}/mesh").json()
        self.assertTrue(m["available"], m)
        self.assertTrue(m["aligned"], m)
        self.assertEqual((m["vertices"], m["triangles"], m["textures"]), (4, 2, 1))
        self.assertEqual(m["groups"][0]["texture"], "tex_0.jpg")
        b = self.c.get(f"/api/projects/{self.pid}/mesh/data")
        self.assertEqual(len(b.content), 4 * 12 + 4 * 8 + 6 * 4)          # positions + uvs + indices
        t = self.c.get(f"/api/projects/{self.pid}/mesh/texture/tex_0.jpg")
        self.assertEqual((t.status_code, t.headers["content-type"]), (200, "image/jpeg"))
        self.assertEqual(self.c.get(f"/api/projects/{self.pid}/mesh/texture/..%2fx.jpg").status_code, 404)
        self.assertEqual(self.c.get(f"/api/projects/{self.pid}/mesh/texture/tex_9.jpg").status_code, 404)

    def test_textured_model_offset_coordinates_use_coords_txt(self):
        self.set_mode("offsetmesh")
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)
        m = self.c.get(f"/api/projects/{self.pid}/mesh").json()
        self.assertTrue(m["aligned"], m)
        self.assertIn("coords.txt", m["alignment_note"])
        self.assertAlmostEqual(m["origin"][0], 571015.0, delta=0.01)

    def test_no_textured_model_is_reported_not_faked(self):
        self.set_mode("nomesh")
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)                      # the rest of the run is still valid
        m = self.c.get(f"/api/projects/{self.pid}/mesh").json()
        self.assertFalse(m["available"])
        self.assertIn("draft", m["reason"])
        r = self.c.post(f"/api/projects/{self.pid}/mesh/build")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (409, "no_textured_model"))
        self.assertEqual(self.c.get(f"/api/projects/{self.pid}/mesh/data").status_code, 404)

    def test_mesh_labels_seed_edit_undo_redo(self):
        import numpy as np
        from app.core.tiling import Window
        from app.services import ortho as O
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)
        mask, _ = O.class_rasters(self.ctx, self.pid)
        mask.write(Window(0, 0, 15, 30), np.full((30, 15), 3, np.uint8))          # west part of the map labelled class 3
        u = f"/api/projects/{self.pid}/mesh/labels"
        st = self.c.get(u).json()
        self.assertTrue(st["available"] and st["can_seed"] and not st["has_labels"], st)
        self.assertEqual(self.c.get(u + "/data").content, bytes(4))                # no labels yet: all unclassified
        sd = self.c.post(u + "/seed").json()
        self.assertEqual(sd["counts"], {"0": 2, "3": 2}, sd)                         # vertices at x=5 are inside the labelled part, x=25 are not
        self.assertEqual(sd["seeded_outside"], 0)
        self.assertEqual(list(self.c.get(u + "/data").content), [3, 0, 0, 3])
        ed = self.c.post(u + "/edit?cls=2", content=np.array([1, 2], "<u4").tobytes()).json()
        self.assertEqual(ed["counts"], {"2": 2, "3": 2}, ed)
        self.assertTrue(ed["can_undo"] and not ed["can_redo"])
        self.assertEqual(self.c.post(u + "/undo").json()["counts"], {"0": 2, "3": 2})
        self.assertEqual(self.c.post(u + "/redo").json()["counts"], {"2": 2, "3": 2})
        self.assertEqual(self.c.post(u + "/clear").json()["counts"], {"0": 4})
        self.assertEqual(self.c.post(u + "/undo").json()["counts"], {"2": 2, "3": 2})
        self.assertEqual(self.c.post(u + "/edit?cls=99", content=np.array([0], "<u4").tobytes()).json()["error"]["code"], "bad_class")
        self.assertEqual(self.c.post(u + "/edit?cls=2", content=np.array([77], "<u4").tobytes()).json()["error"]["code"], "bad_selection")
        self.assertEqual(self.c.post(u + "/edit?cls=2", content=b"abc").status_code, 400)

    def test_mesh_labels_discarded_when_the_model_is_rebuilt(self):
        import numpy as np
        j = self.run_job()
        self.assertEqual(j["status"], "succeeded", j)
        u = f"/api/projects/{self.pid}/mesh/labels"
        self.c.post(u + "/edit?cls=2", content=np.array([0], "<u4").tobytes())
        self.assertTrue(self.c.get(u).json()["has_labels"])
        self.run_job()                                                           # a new ODM run rebuilds the model
        self.assertFalse(self.c.get(u).json()["has_labels"])

    def test_crash_is_failed_not_success(self):
        self.set_mode("crash")
        j = self.run_job()
        self.assertEqual(j["status"], "failed")
        self.assertIn("137", j["error"])
        self.assertFalse(P.get(self.ctx, self.pid, "pointcloud") and P.get(self.ctx, self.pid, "pointcloud")["exists"])

    def test_missing_output_is_failed(self):
        self.set_mode("nodsm")
        j = self.run_job()
        self.assertEqual(j["status"], "failed")
        self.assertIn("DSM missing", j["error"])

    def test_option_mismatch_detected_before_running(self):
        self.set_mode("badopt")
        j = self.run_job()
        self.assertEqual(j["status"], "failed")
        self.assertIn("--dtm", j["error"])
        self.assertFalse((self.bin / "calls.json").exists())

    def test_crs_mismatch_refused(self):
        self.set_mode("wrongcrs")
        j = self.run_job()
        self.assertEqual(j["status"], "failed")
        self.assertIn("32617", j["error"])

    def test_docker_unavailable_is_missing_dependency(self):
        self.set_mode("nodaemon")
        j = self.run_job()
        self.assertEqual(j["status"], "failed")
        self.assertIn("daemon", j["error"].lower())

    def test_guards(self):
        j = self.run_job(options={"rm-rf": True})
        self.assertEqual(j["status"], "failed")
        self.assertIn("not allowed", j["error"])
        p2 = self.c.post("/api/projects", json={"name": "empty", "epsg": 32616}).json()["id"]
        jid = self.ctx.queue.enqueue("odm", p2, {"project_id": p2})
        drain(self.ctx)
        self.assertIn("at least 3", self.ctx.db.one("SELECT error FROM jobs WHERE id=?", (jid,))["error"])


if __name__ == "__main__":
    unittest.main()
