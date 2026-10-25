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
