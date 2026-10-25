import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from starlette.testclient import TestClient

from app.api.main import create_app
from app.core import classes as C
from app.core.tiling import Window
from app.services import classify as K
from app.services import ortho as O
from app.services.synthetic import make_site
from tests import onnx_fixture
from tests.helpers import drain, make_app


def png(resp):
    a = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)
    return a[..., 2], a[..., 1], a[..., 0]       # class, provenance, confidence


class EditorBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ctx = make_app(self.dir, dev_mode=True)
        self.app = create_app(self.ctx, static_dir=self.dir / "nostatic")
        self.c = TestClient(self.app)
        r = self.c.post("/api/dev/sample-project")
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        drain(self.ctx)
        self.u = f"/api/projects/{self.pid}"
        self.site = make_site(120.0, 90.0, 0.25, 7)

    def tearDown(self):
        O.forget_rasters(self.ctx, self.pid)
        self.tmp.cleanup()

    def edit(self, **kw):
        r = self.c.post(f"{self.u}/classification/edit", json=kw)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def stats(self):
        return {c["class_id"]: c for c in self.c.get(f"{self.u}/classification/stats").json()["classes"]}

    def tile(self, x=0, y=0, z=0):
        return png(self.c.get(f"{self.u}/classification/tiles/{z}/{x}/{y}"))

    def job(self, jid):
        return self.c.get(f"/api/jobs/{jid}").json()

    def uniform_rects(self, label, n=3, size=24):
        """n disjoint size x size blocks that are entirely one true label (used to simulate a user's careful corrections)."""
        t = self.site.labels
        out = []
        for y in range(0, t.shape[0] - size, size):
            for x in range(0, t.shape[1] - size, size):
                if (t[y:y + size, x:x + size] == label).all():
                    out.append([x, y, x + size, y + size])
                    if len(out) == n:
                        return out
        return out


class EditorTests(EditorBase):
    def test_tiles_and_info(self):
        info = self.c.get(f"{self.u}/classification").json()
        self.assertEqual((info["width"], info["height"]), (480, 360))
        self.assertEqual(len(info["classes"]), 10)
        self.assertIn("NOT elevation evidence", info["notice"])
        t = self.c.get(f"{self.u}/ortho/tiles/0/0/0")
        img = cv2.imdecode(np.frombuffer(t.content, np.uint8), 1)
        self.assertEqual(img.shape, (256, 256, 3))
        top = cv2.imdecode(np.frombuffer(self.c.get(f"{self.u}/ortho/tiles/1/0/0").content, np.uint8), 1)
        self.assertEqual(top.shape[:2], (256, 256))
        edge = png(self.c.get(f"{self.u}/classification/tiles/0/1/1"))[0]
        self.assertEqual(edge.shape, (104, 224))                      # partial edge tile has its true size
        self.assertEqual(self.c.get(f"{self.u}/classification/tiles/0/9/9").status_code, 404)
        self.assertEqual(self.c.get(f"{self.u}/classification/tiles/3/0/0").status_code, 400)
        cls, prov, conf = self.tile()
        self.assertEqual(int(cls.max()), 0)

    def test_brush_undo_redo_and_stats(self):
        r = self.edit(type="brush", class_id=C.TREES, points=[[60, 60], [100, 60]], radius=8)
        self.assertGreater(r["changed_pixels"], 500)
        self.assertTrue(r["can_undo"])
        cls, prov, _ = self.tile()
        self.assertEqual(int(cls[60, 80]), C.TREES)
        self.assertEqual(int(prov[60, 80]), 1)                         # manual provenance
        self.assertEqual(self.stats()[C.TREES]["pixels"], r["changed_pixels"])
        self.assertEqual(self.stats()[C.TREES]["manual_pixels"], r["changed_pixels"])
        u = self.c.post(f"{self.u}/classification/undo").json()
        self.assertFalse(u["can_undo"])
        self.assertTrue(u["can_redo"])
        self.assertEqual(self.stats()[C.TREES]["pixels"], 0)
        self.assertEqual(int(self.tile()[0][60, 80]), 0)
        self.c.post(f"{self.u}/classification/redo")
        self.assertEqual(self.stats()[C.TREES]["pixels"], r["changed_pixels"])
        self.assertEqual(self.c.post(f"{self.u}/classification/redo").json()["window"], None)   # nothing to redo

    def test_selection_tools(self):
        r = self.edit(type="rect", class_id=C.SOIL, rect=[10, 10, 30, 20])
        self.assertEqual(r["changed_pixels"], 200)
        r = self.edit(type="polygon", class_id=C.ROCK, points=[[100, 100], [140, 100], [100, 140]])
        self.assertGreater(r["changed_pixels"], 700)
        self.assertLess(r["changed_pixels"], 900)
        r = self.edit(type="lasso", class_id=C.WATER, points=[[200, 200], [230, 205], [225, 235], [205, 230]])
        self.assertGreater(r["changed_pixels"], 300)
        # erase to unknown: pixels become class 0 but stay marked as a manual decision
        r = self.edit(type="rect", class_id=0, rect=[10, 10, 20, 20])
        self.assertEqual(r["changed_pixels"], 100)
        cls, prov, _ = self.tile()
        self.assertEqual(int(cls[12, 12]), 0)
        self.assertEqual(int(prov[12, 12]), 1)
        # protect_classes keeps rock safe from a rectangle painted over it
        r = self.edit(type="rect", class_id=C.OTHER, rect=[90, 90, 150, 150], protect_classes=[C.ROCK])
        self.assertEqual(int(self.tile()[0][105, 105]), C.ROCK)
        # only_classes restricts what may change
        self.edit(type="rect", class_id=C.ROAD, rect=[90, 90, 150, 150], only_classes=[C.OTHER])
        self.assertEqual(int(self.tile()[0][105, 105]), C.ROCK)

    def test_fill_and_select_by_color(self):
        # find a tree-crown pixel (flat colour region in the synthetic ortho) and flood-fill it
        ys, xs = np.nonzero(self.site.labels == C.TREES)
        i = len(ys) // 2
        x, y = int(xs[i]), int(ys[i])
        sel = self.c.post(f"{self.u}/classification/select-preview", json={"type": "fill", "class_id": 1, "points": [[x + .5, y + .5]], "tolerance": 40})
        self.assertEqual(sel.status_code, 200)
        n = int(sel.headers["x-selected-pixels"])
        self.assertGreater(n, 50)
        r = self.edit(type="fill", class_id=C.TREES, points=[[x + .5, y + .5]], tolerance=40)
        self.assertEqual(r["changed_pixels"], n)                      # preview == what gets applied
        r2 = self.edit(type="similar", class_id=C.BRUSH, points=[[x + .5, y + .5]], tolerance=25)
        self.assertGreaterEqual(r2["changed_pixels"], 0)
        s = self.c.get(f"{self.u}/classification/sample", params={"x": x, "y": y, "segment": 1, "tolerance": 40}).json()
        self.assertIn("rgb", s)
        self.assertGreater(s["segment"]["pixels"], 50)
        self.assertEqual(s["map"]["epsg"], 32616)
        self.assertEqual(s["provenance"], "manual")

    def test_validation_errors(self):
        post = lambda **kw: self.c.post(f"{self.u}/classification/edit", json=kw)
        self.assertEqual(post(type="polygon", class_id=1, points=[[1, 1], [2, 2]]).status_code, 422)
        self.assertEqual(post(type="brush", class_id=1, points=[[1, 1]], radius=99999).status_code, 422)
        self.assertEqual(post(type="brush", class_id=99, points=[[1, 1]]).status_code, 422)
        self.assertEqual(self.c.post(f"{self.u}/classification/edit", content=b'{"type":"brush","class_id":1,"points":[[Infinity,1]]}').status_code, 422)
        self.assertEqual(post(type="spray", class_id=1, points=[[1, 1]]).status_code, 422)
        self.assertEqual(post(type="fill", class_id=1, points=[[9999, 1]]).status_code, 422)
        self.assertEqual(self.c.post(f"{self.u}/classification/edit", content=b"not json").status_code, 400)
        # a class that the project does not define
        self.c.patch(self.u, json={"classes": [c for c in self.c.get(self.u).json()["classes"] if c["id"] != C.WATER]})
        self.assertEqual(post(type="rect", class_id=C.WATER, rect=[0, 0, 5, 5]).status_code, 422)

    def test_class_configuration_validated_and_persisted(self):
        cl = self.c.get(self.u).json()["classes"]
        cl[1]["color"] = "#00ff00"
        self.assertEqual(self.c.patch(self.u, json={"classes": cl}).json()["classes"][1]["color"], "#00ff00")
        cl[1]["color"] = "green"
        self.assertEqual(self.c.patch(self.u, json={"classes": cl}).status_code, 422)
        dup = [dict(cl[0]), dict(cl[0])]
        self.assertEqual(self.c.patch(self.u, json={"classes": dup}).status_code, 422)

    def test_classification_never_touches_imagery_or_terrain(self):
        ortho = self.dir / "projects" / self.pid / "products" / "orthophoto.tif"
        before = hashlib.sha256(ortho.read_bytes()).hexdigest()
        self.edit(type="rect", class_id=C.SOIL, rect=[0, 0, 200, 200])         # "remove the trees"
        self.assertEqual(hashlib.sha256(ortho.read_bytes()).hexdigest(), before)
        names = set(self.c.get(self.u).json()["products"])
        self.assertFalse(names & {"dtm", "dsm", "confidence", "evidence"})     # no elevation product appears from painting

    def test_persistence_across_restart(self):
        r = self.edit(type="rect", class_id=C.ROAD, rect=[5, 5, 55, 25])
        O.forget_rasters(self.ctx, self.pid)
        ctx2 = make_app(self.dir, dev_mode=True)
        c2 = TestClient(create_app(ctx2, static_dir=self.dir / "nostatic"))
        st = c2.get(f"{self.u}/classification/stats").json()
        road = [c for c in st["classes"] if c["class_id"] == C.ROAD][0]
        self.assertEqual(road["pixels"], r["changed_pixels"])
        self.assertEqual(int(png(c2.get(f"{self.u}/classification/tiles/0/0/0"))[0][10, 10]), C.ROAD)
        self.assertTrue(c2.get(f"{self.u}/classification").json()["can_undo"])        # undo history survives too
        O.forget_rasters(ctx2, self.pid)

    def test_overview_matches_tiles(self):
        self.edit(type="rect", class_id=C.WATER, rect=[100, 100, 300, 250])
        ov = png(self.c.get(f"{self.u}/classification/overview", params={"size": 256}))[0]
        self.assertEqual(ov.shape, (180, 240))            # scale 2
        self.assertEqual(int(ov[90, 100]), C.WATER)
        self.assertEqual(int(ov[10, 10]), 0)
        self.assertEqual(self.c.get(f"{self.u}/classification/overview", params={"size": 256},
                                    headers={"if-none-match": self.c.get(f"{self.u}/classification/overview", params={"size": 256}).headers["etag"]}).status_code, 304)


class AutoClassifyTests(EditorBase):
    def test_tiled_inference_equals_full_image_inference(self):
        """Overlapping tiles must give the same answer as one big pass: no seams."""
        m = K.HeuristicModel()
        full, fconf = m.predict(self.site.rgb)
        res = K.run_classification(self.ctx, self.pid, m, None, "all", tile=64, overlap=16)
        self.assertGreater(res["tiles"], 20)
        mask, conf = O.class_rasters(self.ctx, self.pid)
        got = mask.read(Window(0, 0, mask.width, mask.height))
        self.assertTrue(((got & K.TAG_AUTO) != 0).all())
        np.testing.assert_array_equal(got & K.CLASS_MASK, full)
        np.testing.assert_array_equal(conf.read(Window(0, 0, mask.width, mask.height)), np.clip(fconf * 100, 0, 100).astype(np.uint8))

    def test_auto_job_respects_manual_edits_and_can_be_rolled_back(self):
        self.edit(type="rect", class_id=C.OTHER, rect=[0, 0, 60, 60])                # a deliberate manual decision
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "heuristic"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(self.c.post(f"{self.u}/classification/auto", json={"model": "heuristic"}).status_code, 409)   # one at a time
        drain(self.ctx)
        j = self.job(r.json()["job_id"])
        self.assertEqual(j["status"], "succeeded", j.get("error"))
        self.assertGreater(j["result"]["pixels_written"], 100000)
        cls, prov, conf = self.tile()
        self.assertTrue((cls[:60, :60] == C.OTHER).all())                            # manual region untouched
        self.assertTrue((prov[:60, :60] == 1).all())
        auto = prov == 2
        self.assertGreater(auto.sum(), 20000)
        self.assertGreater(int(conf[auto].max()), 0)
        self.assertLessEqual(int(conf.max()), 80)                                    # heuristic never claims > 80
        st = self.stats()
        self.assertGreater(st[C.TREES]["automatic_pixels"], 0)
        self.assertEqual(st[C.OTHER]["manual_pixels"], 3600)
        # veg vs non-veg on the synthetic scene
        truth_veg = np.isin(self.site.labels[:256, :256], [1, 2, 3])
        pred_veg = np.isin(cls, [1, 2, 3])
        self.assertGreater((truth_veg == pred_veg)[100:, 100:].mean(), 0.98)
        # roll back using the snapshot taken automatically before the run
        snap = j["result"]["snapshot"]
        rr = self.c.post(f"{self.u}/classification/snapshots/{snap}/restore")
        self.assertEqual(rr.status_code, 200)
        st = self.stats()
        self.assertEqual(st[C.TREES]["pixels"], 0)
        self.assertEqual(st[C.OTHER]["pixels"], 3600)

    def test_region_and_unclassified_only(self):
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "heuristic", "region": {"type": "rect", "rect": [200, 100, 320, 220]}})
        drain(self.ctx)
        self.assertEqual(self.job(r.json()["job_id"])["status"], "succeeded")
        cls, prov, _ = self.tile()
        self.assertTrue((prov[:100, :] == 0).all())
        self.assertTrue((prov[100:220, 200:320] == 2).any())
        self.assertTrue((prov[:, :200] == 0).all())
        # bad region is rejected
        self.assertEqual(self.c.post(f"{self.u}/classification/auto", json={"region": {"type": "polygon", "points": [[1, 1]]}}).status_code, 422)

    def test_learn_from_corrections(self):
        for label, cid, size in ((C.TREES, C.TREES, 16), (C.GRASS, C.GRASS, 16), (C.BUILDING, C.BUILDING, 8)):
            rects = self.uniform_rects(label, n=6, size=size)
            self.assertGreaterEqual(len(rects), 3, f"synthetic scene lacks uniform {label} blocks")
            for rc in rects:
                self.edit(type="rect", class_id=cid, rect=rc)
        r = self.c.post(f"{self.u}/classification/train")
        drain(self.ctx)
        j = self.job(r.json()["job_id"])
        self.assertEqual(j["status"], "succeeded", j.get("error"))
        rep = j["result"]
        self.assertEqual(set(rep["classes"]), {str(C.TREES), str(C.GRASS), str(C.BUILDING)})
        self.assertIn("accuracy", rep["holdout"] or {"accuracy": None})
        models = {m["id"]: m for m in self.c.get(f"{self.u}/classification").json()["models"]}
        self.assertTrue(models["learned"]["available"])
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "learned", "overwrite": "auto"})
        drain(self.ctx)
        self.assertEqual(self.job(r.json()["job_id"])["status"], "succeeded")
        cls, prov, conf = self.tile()
        t = self.site.labels[:256, :256]
        auto = (prov == 2) & np.isin(t, [C.TREES, C.GRASS, C.BUILDING])
        acc = (cls[auto] == t[auto]).mean()
        self.assertGreater(acc, 0.75)

    def test_training_without_corrections_fails_with_guidance(self):
        r = self.c.post(f"{self.u}/classification/train")
        drain(self.ctx)
        j = self.job(r.json()["job_id"])
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error_code"], "no_training_data")
        self.assertIn("Paint", j["error"])

    def test_missing_model_falls_back_with_clear_error(self):
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "onnx"})
        drain(self.ctx)
        j = self.job(r.json()["job_id"])
        self.assertEqual(j["status"], "failed")
        self.assertEqual(j["error_code"], "no_model")
        self.assertIn("manual classification", j["error"])
        models = {m["id"]: m for m in self.c.get(f"{self.u}/classification").json()["models"]}
        self.assertFalse(models["onnx"]["available"])
        self.assertTrue(models["heuristic"]["available"])

    def test_onnx_model_runs_through_the_same_tiled_pipeline(self):
        mp = onnx_fixture.write_model(self.dir)
        self.ctx.settings.model_path = str(mp)
        m = K.OnnxSegModel(mp)
        rgb = np.zeros((40, 50, 3), np.uint8)
        rgb[:, :20] = (200, 10, 10)        # red-dominant  -> class 4
        rgb[:, 20:35] = (10, 200, 10)      # green-dominant -> class 3
        rgb[:, 35:] = (10, 10, 200)        # blue-dominant -> class 8
        cls, conf = m.predict(rgb)
        self.assertEqual(set(np.unique(cls[:, :20])), {4})
        self.assertEqual(set(np.unique(cls[:, 20:35])), {3})
        self.assertEqual(set(np.unique(cls[:, 35:])), {8})
        self.assertTrue(((conf > 0) & (conf <= 1)).all())
        r = self.c.post(f"{self.u}/classification/auto", json={"model": "onnx", "overwrite": "all"})
        drain(self.ctx)
        j = self.job(r.json()["job_id"])
        self.assertEqual(j["status"], "succeeded", j.get("error"))
        self.assertEqual(j["result"]["model"], "dominant-colour-test")
        self.assertTrue(set(j["result"]["by_class"]) <= {"4", "3", "8"})

    def test_unlisted_onnx_classes_become_unclassified(self):
        mp = onnx_fixture.write_model(self.dir, class_map={"1": 3})
        m = K.OnnxSegModel(mp)
        rgb = np.zeros((8, 8, 3), np.uint8)
        rgb[...] = (200, 10, 10)
        self.assertEqual(int(m.predict(rgb)[0].max()), 0)


if __name__ == "__main__":
    unittest.main()
