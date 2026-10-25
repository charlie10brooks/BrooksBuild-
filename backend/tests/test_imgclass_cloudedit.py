import io
import json
import unittest

import cv2
import numpy as np
from PIL import Image

from app.services import cloudedit as CE
from tests.helpers import drain, jpeg_bytes
from tests.test_api_ingest import upload
from tests.test_terrain_api import TerrainBase


def aerial_jpeg(seed: int, size=(240, 180)) -> bytes:
    """Left third green (vegetation), middle brown (soil), right grey (road), plus mild noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    a = np.zeros((h, w, 3), np.uint8)
    a[:, : w // 3] = (40, 110, 35)
    a[:, w // 3: 2 * w // 3] = (140, 105, 80)
    a[:, 2 * w // 3:] = (95, 98, 102)
    a = np.clip(a.astype(int) + rng.integers(-6, 7, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    return buf.getvalue()


class ImageClassificationTests(TerrainBase):
    def upload_photos(self, n=3):
        batch = self.c.post(f"{self.u}/batches", json={"label": "B"}).json()["id"]
        for k in range(n):
            upload(self.c, self.pid, f"p{k}.jpg", aerial_jpeg(k), batch)
        drain(self.ctx)
        return batch

    def test_classify_uploaded_photos(self):
        batch = self.upload_photos(3)
        st = self.c.get(f"{self.u}/image-classification").json()
        self.assertEqual((st["images"], st["classified"]), (3, 0))
        self.assertTrue(any(m["id"] == "heuristic" and m["available"] for m in st["models"]))
        self.assertIn("Not georeferenced", st["disclaimer"])
        r = self.c.post(f"{self.u}/image-classification", json={"scope": "batch", "batch_id": batch})
        self.assertEqual(r.status_code, 202, r.text)
        self.assertEqual(r.json()["photos"], 3)
        drain(self.ctx)
        j = self.c.get(f"/api/jobs/{r.json()['job_id']}").json()
        self.assertEqual(j["status"], "succeeded", j)
        st = self.c.get(f"{self.u}/image-classification").json()
        self.assertEqual(st["classified"], 3)
        self.assertEqual(st["by_model"], {"heuristic": 3})
        its = self.c.get(f"{self.u}/image-classification/items?state=classified").json()
        self.assertEqual(its["total"], 3)
        fr = its["items"][0]["fractions"]
        # vegetation (trees/brush/grass) and soil must be found; the heuristic must NOT invent buildings
        veg = sum(fr.get(k, 0) for k in ("1", "2", "3"))
        self.assertGreater(veg, 0.15, fr)
        self.assertGreater(fr.get("4", 0), 0.15, fr)
        self.assertEqual(fr.get("6", 0), 0)
        iid = its["items"][0]["id"]
        d = self.c.get(f"{self.u}/images/{iid}/classification").json()
        self.assertEqual(sum(d["counts"].values()), d["width"] * d["height"])
        ov = self.c.get(f"{self.u}/images/{iid}/classification/overlay.png?opacity=0.5")
        self.assertEqual(ov.status_code, 200)
        a = cv2.imdecode(np.frombuffer(ov.content, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(a.shape, (d["height"], d["width"], 4))
        self.assertGreater(int((a[..., 3] > 0).sum()), 0)
        # hidden classes become transparent
        hid = self.c.get(f"{self.u}/images/{iid}/classification/overlay.png?hide=" + ",".join(d["counts"].keys()))
        b = cv2.imdecode(np.frombuffer(hid.content, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(int((b[..., 3] > 0).sum()), 0)
        # nothing left to do unless redo is requested
        again = self.c.post(f"{self.u}/image-classification", json={"scope": "all"})
        self.assertEqual(again.status_code, 409)
        self.assertEqual(again.json()["error"]["code"], "nothing_to_do")
        self.assertEqual(self.c.post(f"{self.u}/image-classification", json={"scope": "all", "overwrite": True}).status_code, 202)
        drain(self.ctx)

    def test_unavailable_model_is_refused_up_front_and_clear_works(self):
        self.upload_photos(1)
        r = self.c.post(f"{self.u}/image-classification", json={"model": "onnx"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["error"]["code"], "no_model")
        self.assertEqual(self.c.get(f"{self.u}/images/{self.ctx.db.one('SELECT id FROM images LIMIT 1')['id']}/classification").status_code, 404)
        self.c.post(f"{self.u}/image-classification", json={}); drain(self.ctx)
        self.assertEqual(self.c.get(f"{self.u}/image-classification").json()["classified"], 1)
        self.assertEqual(self.c.delete(f"{self.u}/image-classification").json()["removed"], 1)
        self.assertEqual(self.c.get(f"{self.u}/image-classification").json()["classified"], 0)

    def test_csv_export_of_photo_classes(self):
        self.upload_photos(2)
        self.c.post(f"{self.u}/image-classification", json={}); drain(self.ctx)
        cat = {i["item"]: i for i in self.c.get(f"{self.u}/exports/catalog").json()["items"]}
        self.assertTrue(cat["image_classes"]["available"])
        self.assertFalse(cat["edited_cloud"]["available"])
        r = self.c.post(f"{self.u}/exports", json={"items": ["image_classes", "edited_cloud"]})
        drain(self.ctx)
        m = self.c.get(f"{self.u}/exports/{r.json()['export_id']}").json()
        f = next(f for f in m["files"] if f["name"] == "photo_classification_stats.csv")
        txt = self.c.get(f"{self.u}/exports/{m['export_id']}/files/{f['name']}").text
        self.assertEqual(len(txt.strip().splitlines()), 3)
        self.assertTrue(any(s["item"] == "edited_cloud" for s in m["skipped"]))


class CloudEditTests(TerrainBase):
    def build(self):
        j = self.run_terrain()
        self.assertEqual(j["status"], "succeeded", j)
        meta = self.c.get(f"{self.u}/pointcloud").json()
        self.assertTrue(meta["available"], meta)
        return meta

    def data(self):
        r = self.c.get(f"{self.u}/pointcloud/data?max=5000000")
        m = json.loads(r.headers["X-Meta"])
        n = m["n"]
        cls = np.frombuffer(r.content, np.uint8, n, 12 * n)
        return m, cls.copy()

    def edit(self, idx, cls):
        return self.c.post(f"{self.u}/pointcloud/edit?cls={cls}", content=np.asarray(idx, "<u4").tobytes())

    def test_edit_undo_redo_persist_and_export(self):
        self.build()
        m, c0 = self.data()
        self.assertTrue(m["complete"])
        self.assertTrue(self.c.get(f"{self.u}/pointcloud/edit").json()["available"])
        sel = np.arange(0, 400)
        r = self.edit(sel, 7)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["edited_points"], int((c0[sel] != 7).sum()))
        _, c1 = self.data()
        self.assertTrue((c1[sel] == 7).all())
        self.assertTrue((c1[400:] == c0[400:]).all())
        self.assertEqual(self.c.get(f"{self.u}/pointcloud").json()["classes"]["7"], int((c1 == 7).sum()))
        # a second edit then undo twice, redo once
        self.edit(np.arange(0, 100), 2)
        _, c2 = self.data(); self.assertTrue((c2[:100] == 2).all())
        self.assertEqual(self.c.post(f"{self.u}/pointcloud/edit/undo").status_code, 200)
        _, c3 = self.data(); self.assertTrue((c3 == c1).all())
        self.c.post(f"{self.u}/pointcloud/edit/undo")
        _, c4 = self.data(); self.assertTrue((c4 == c0).all())
        self.assertEqual(self.c.post(f"{self.u}/pointcloud/edit/undo").status_code, 409)
        self.c.post(f"{self.u}/pointcloud/edit/redo")
        _, c5 = self.data(); self.assertTrue((c5 == c1).all())
        # a new edit discards the redo tail
        self.edit([500], 6)
        self.assertEqual(self.c.post(f"{self.u}/pointcloud/edit/redo").status_code, 409)
        # persistence: a fresh read of state (as after a restart) is identical
        _, c6 = self.data()
        self.assertEqual(int(c6[500]), 6)
        info = self.c.get(f"{self.u}/pointcloud/edit").json()
        self.assertTrue(info["can_undo"] and not info["can_redo"])
        # point edits do not touch the elevation products
        import hashlib
        from app.services import products as P
        h = lambda: hashlib.sha256(self.ctx.storage.path(self.pid, P.LAYOUT["dtm"][0]).read_bytes()).hexdigest()
        before = h()
        self.edit(np.arange(10), 7)
        self.assertEqual(before, h())
        # export is an honest SAMPLE
        r = self.c.post(f"{self.u}/exports", json={"items": ["edited_cloud"]}); drain(self.ctx)
        mf = self.c.get(f"{self.u}/exports/{r.json()['export_id']}").json()
        f = next(f for f in mf["files"] if f["item"] == "edited_cloud")
        self.assertTrue(f["sample"])
        self.assertIn("SAMPLE", f["description"])
        from app.services import las as L
        pts = L.read_all(self.ctx.storage.path(self.pid, f"exports/{mf['export_id']}/{f['name']}"))
        self.assertEqual(len(pts["x"]), f["points"])
        self.assertEqual(int((pts["classification"] == 7).sum()), int((c6 == 7).sum()))

    def test_validation(self):
        self.build()
        self.assertEqual(self.edit([0], 99).status_code, 400)
        self.assertEqual(self.edit([], 2).status_code, 400)
        self.assertEqual(self.edit([10 ** 9], 2).status_code, 400)
        self.assertEqual(self.c.post(f"{self.u}/pointcloud/edit?cls=2", content=b"123").status_code, 400)
        self.assertEqual(self.c.post(f"{self.u}/pointcloud/edit?cls=x", content=b"").status_code, 400)

    def test_rebuilt_sample_invalidates_old_edits(self):
        self.build()
        _, c0 = self.data()
        self.edit(np.arange(50), 7)
        f, _ = CE.sample_file(self.ctx, self.pid)
        import os
        st = f.stat(); os.utime(f, (st.st_atime, st.st_mtime + 10))           # same points, "rebuilt" file
        _, c1 = self.data()
        self.assertTrue((c1 == c0).all(), "edits from an older sample must not be applied to a different one")
        self.assertEqual(self.c.get(f"{self.u}/pointcloud/edit").json()["edited_points"], 0)
        r = self.edit(np.arange(5), 2)
        self.assertEqual(r.status_code, 200)

    def test_no_cloud_is_a_clear_409(self):
        r = self.c.post("/api/projects", json={"name": "empty", "epsg": 32616})
        pid = r.json()["id"]
        r = self.c.post(f"/api/projects/{pid}/pointcloud/edit?cls=2", content=np.arange(3, dtype="<u4").tobytes())
        self.assertEqual(r.status_code, 409)
        self.assertFalse(self.c.get(f"/api/projects/{pid}/pointcloud/edit").json()["available"])


if __name__ == "__main__":
    unittest.main()
