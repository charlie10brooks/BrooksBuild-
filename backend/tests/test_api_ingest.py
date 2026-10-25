import hashlib
import tempfile
import time
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from app.api.main import create_app
from app.workers.runner import Worker
from tests.helpers import drain, jpeg_bytes, make_app

CHUNK = 64 * 1024


def upload(c, pid, name, data, batch=None, send=None, key=None, declare_sha=False, complete=True):
    body = {"filename": name, "size": len(data), "chunk_size": CHUNK, "client_key": key}
    if batch: body["batch_id"] = batch
    if declare_sha: body["sha256"] = hashlib.sha256(data).hexdigest()
    r = c.post(f"/api/projects/{pid}/uploads", json=body)
    assert r.status_code == 201, r.text
    st = r.json()
    if st["status"] == "duplicate":
        return st
    uid = st["upload_id"]
    for i in (send if send is not None else st["missing_chunks"]):
        r = c.put(f"/api/projects/{pid}/uploads/{uid}/chunks/{i}", content=data[i * CHUNK:(i + 1) * CHUNK])
        assert r.status_code == 200, r.text
    if complete:
        r = c.post(f"/api/projects/{pid}/uploads/{uid}/complete")
        return {**st, "complete": r}
    return st


class ApiIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_app(Path(self.tmp.name))
        self.c = TestClient(create_app(self.ctx, static_dir=Path(self.tmp.name) / "nostatic"))
        r = self.c.post("/api/projects", json={"name": "Lot 7", "epsg": 32616})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]

    def tearDown(self):
        self.ctx.db.close()
        self.tmp.cleanup()

    def test_project_validation(self):
        self.assertEqual(self.c.post("/api/projects", json={"name": "  "}).status_code, 422)
        self.assertEqual(self.c.post("/api/projects", json={"name": "x", "boundary": {"type": "Point", "coordinates": [0, 0]}}).status_code, 422)
        self.assertEqual(self.c.post("/api/projects", json={"name": "x", "horizontal_units": "furlong"}).status_code, 422)
        self.assertEqual(self.c.get("/api/projects/" + "0" * 32).status_code, 404)
        self.assertEqual(self.c.get("/api/projects/../../etc").status_code, 404)
        b = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]]}
        r = self.c.post("/api/projects", json={"name": "with boundary", "boundary": b})
        self.assertEqual(r.json()["boundary"], b)

    def test_upload_ingest_duplicates_rejects_and_flights(self):
        pid = self.pid
        batch = self.c.post(f"/api/projects/{pid}/batches", json={"label": "Morning flight"}).json()["id"]
        # a 3x3 grid of GPS-tagged images ~ 20 m apart, relative altitude 40 m
        imgs = []
        for k in range(9):
            lat = 39.0 + (k // 3) * 0.00018
            lon = -86.0 + (k % 3) * 0.00023
            imgs.append((f"DJI_{k:04d}.JPG", jpeg_bytes(k, gps=(lat, lon, 250.0), rel_alt=40.0, when=f"2026:05:01 10:00:{k:02d}")))
        for name, data in imgs:
            r = upload(self.c, pid, name, data, batch)
            self.assertEqual(r["complete"].status_code, 202, r["complete"].text)
        drain(self.ctx)       # the originals must be ingested before a declared-checksum duplicate can short-circuit
        # duplicate content under another name, declared sha => short-circuits without sending bytes
        d = upload(self.c, pid, "copy.jpg", imgs[0][1], batch, declare_sha=True)
        self.assertEqual(d["status"], "duplicate")
        # duplicate discovered at ingest (no declared sha)
        upload(self.c, pid, "copy2.jpg", imgs[1][1], batch)
        # not an image
        upload(self.c, pid, "broken.jpg", b"this is not a jpeg" * 100, batch)
        # no GPS
        upload(self.c, pid, "nogps.jpg", jpeg_bytes(99), batch)
        # unsupported type refused at creation
        r = self.c.post(f"/api/projects/{pid}/uploads", json={"filename": "x.exe", "size": 100, "chunk_size": CHUNK})
        self.assertEqual(r.status_code, 415)

        drain(self.ctx)

        s = self.c.get(f"/api/projects/{pid}/uploads/summary").json()
        self.assertEqual(s["by_status"]["complete"]["files"], 10)
        self.assertEqual(s["by_status"]["duplicate"]["files"], 2)
        self.assertEqual(s["by_status"]["rejected"]["files"], 1)
        self.assertEqual(s["needs_attention"], 1)
        # nothing silently discarded: rejected file is kept and explained
        self.assertTrue(list((Path(self.tmp.name) / "projects" / pid / "rejected").glob("*broken.jpg")))
        rej = self.c.get(f"/api/projects/{pid}/uploads?status=rejected").json()["items"]
        self.assertIn("unreadable", rej[0]["error"])

        lst = self.c.get(f"/api/projects/{pid}/images?limit=50").json()
        self.assertEqual(lst["total"], 10)
        withgps = [i for i in lst["items"] if i["lat"] is not None]
        self.assertEqual(len(withgps), 9)
        self.assertAlmostEqual(withgps[0]["lat"], 39.0, delta=0.001)
        self.assertEqual(withgps[0]["georef_status"], "ok")
        self.assertEqual(withgps[0]["rel_alt"], 40.0)
        nog = [i for i in lst["items"] if i["lat"] is None][0]
        self.assertEqual(nog["georef_status"], "missing")
        self.assertTrue(nog["issues"])
        self.assertEqual(self.c.get(f"/api/projects/{pid}/images?georef=missing").json()["total"], 1)

        # thumbnail + original are served
        iid = withgps[0]["id"]
        t = self.c.get(f"/api/projects/{pid}/images/{iid}/thumbnail")
        self.assertEqual(t.status_code, 200)
        self.assertTrue(t.content.startswith(b"\xff\xd8"))
        o = self.c.get(f"/api/projects/{pid}/images/{iid}/original")
        self.assertEqual(hashlib.sha256(o.content).hexdigest(), dict(imgs)[withgps[0]["filename"]] and hashlib.sha256(dict(imgs)[withgps[0]["filename"]]).hexdigest())
        detail = self.c.get(f"/api/projects/{pid}/images/{iid}").json()
        self.assertEqual(detail["metadata"]["make"], "TestCam")

        # flight analysis ran as a job and produced footprints
        fl = self.c.get(f"/api/projects/{pid}/flights").json()
        self.assertGreaterEqual(len(fl["groups"]), 1)
        self.assertEqual(fl["summary"]["with_gps"], 9)
        fp = self.c.get(f"/api/projects/{pid}/footprints").json()
        kinds = [f["properties"]["kind"] for f in fp["features"]]
        self.assertEqual(kinds.count("camera"), 9)
        self.assertEqual(kinds.count("footprint"), 9)
        proj = self.c.get(f"/api/projects/{pid}").json()
        self.assertEqual(proj["image_count"], 10)
        self.assertEqual(proj["images_without_gps"], 1)
        self.assertEqual(proj["suggested_crs"]["epsg"], 32616)

    def test_resume_interrupted_upload(self):
        pid = self.pid
        data = jpeg_bytes(5, size=(900, 700), gps=(39.1, -86.1, 200.0))
        n = -(-len(data) // CHUNK)
        self.assertGreater(n, 2)
        st = upload(self.c, pid, "big.jpg", data, key="big.jpg:1", send=[0], complete=False)   # "browser closed" after chunk 0
        self.assertEqual(st["missing_chunks"], list(range(n)))
        # completing early is refused and names what is missing
        r = self.c.post(f"/api/projects/{pid}/uploads/{st['upload_id']}/complete")
        self.assertEqual(r.status_code, 409)
        # new session: same client key resumes and only the missing chunks are needed
        r = self.c.post(f"/api/projects/{pid}/uploads", json={"filename": "big.jpg", "size": len(data), "chunk_size": CHUNK, "client_key": "big.jpg:1"}).json()
        self.assertTrue(r["resumed"])
        self.assertEqual(r["upload_id"], st["upload_id"])
        self.assertEqual(r["missing_chunks"], list(range(1, n)))
        # corrupt chunk is refused
        bad = self.c.put(f"/api/projects/{pid}/uploads/{r['upload_id']}/chunks/1", content=b"x" * CHUNK, headers={"x-chunk-sha256": "0" * 64})
        self.assertEqual(bad.status_code, 422)
        for i in r["missing_chunks"]:
            self.assertEqual(self.c.put(f"/api/projects/{pid}/uploads/{r['upload_id']}/chunks/{i}", content=data[i * CHUNK:(i + 1) * CHUNK]).status_code, 200)
        self.assertEqual(self.c.post(f"/api/projects/{pid}/uploads/{r['upload_id']}/complete").status_code, 202)
        drain(self.ctx)
        self.assertEqual(self.c.get(f"/api/projects/{pid}/images").json()["total"], 1)

    def test_corrupt_declared_checksum_is_reported_not_ingested(self):
        pid = self.pid
        data = jpeg_bytes(3)
        body = {"filename": "c.jpg", "size": len(data), "chunk_size": CHUNK, "sha256": "ab" * 32}
        st = self.c.post(f"/api/projects/{pid}/uploads", json=body).json()
        self.c.put(f"/api/projects/{pid}/uploads/{st['upload_id']}/chunks/0", content=data)
        self.c.post(f"/api/projects/{pid}/uploads/{st['upload_id']}/complete")
        drain(self.ctx)
        u = self.c.get(f"/api/projects/{pid}/uploads/{st['upload_id']}").json()
        self.assertEqual(u["status"], "failed")
        self.assertIn("checksum", u["error"])
        self.assertEqual(self.c.get(f"/api/projects/{pid}/images").json()["total"], 0)
        jobs = self.c.get(f"/api/projects/{pid}/jobs?kind=ingest_image").json()
        self.assertEqual(jobs[0]["status"], "failed")

    def test_no_image_count_limit(self):
        """The platform never rejects an upload because of how many images already exist."""
        pid = self.pid
        t = time.time()
        rows = [{"id": f"{i:032x}", "project_id": pid, "filename": f"{i}.jpg", "sha256": f"{i:064x}", "size": 1, "rel_path": "x",
                 "created_at": t} for i in range(1, 2501)]
        with self.ctx.db.tx() as c:
            for r in rows:
                c.execute("INSERT INTO images(id,project_id,filename,sha256,size,rel_path,created_at) VALUES(?,?,?,?,?,?,?)",
                          (r["id"], pid, r["filename"], r["sha256"], r["size"], r["rel_path"], r["created_at"]))
        r = self.c.post(f"/api/projects/{pid}/uploads", json={"filename": "n.jpg", "size": 1000, "chunk_size": CHUNK})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.c.get(f"/api/projects/{pid}/images?limit=10&offset=2490").json()["total"], 2500)
        self.assertIsNone(self.c.get("/api/storage").json()["limits"]["max_images"])

    def test_delete_requires_confirmation(self):
        self.assertEqual(self.c.delete(f"/api/projects/{self.pid}").status_code, 400)
        self.assertEqual(self.c.delete(f"/api/projects/{self.pid}?confirm=Lot 7").status_code, 200)
        self.assertEqual(self.c.get(f"/api/projects/{self.pid}").status_code, 404)

    def test_control_point_import(self):
        pid = self.pid
        r = self.c.post(f"/api/projects/{pid}/control-points", json={"csv": "label,x,y,z,kind\nG1,571000,4402000,250.1,gcp\nC1,571010,4402010,251.0,check\n"})
        self.assertEqual(r.status_code, 201)
        pts = self.c.get(f"/api/projects/{pid}/control-points").json()
        self.assertEqual({p["kind"] for p in pts}, {"gcp", "check"})
        bad = self.c.post(f"/api/projects/{pid}/control-points", json={"csv": "G1,abc,2,3\nG2,1,2,3"})
        self.assertEqual(bad.status_code, 422)


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_app(Path(self.tmp.name))
        from app.workers import handlers
        self.calls = []

        def ok_handler(app, ctx):
            ctx.progress(0.5, "half")
            self.calls.append(ctx.id)
            return {"fine": True}

        def bad_handler(app, ctx):
            raise RuntimeError("boom")
        handlers.HANDLERS["t_ok"] = ok_handler
        handlers.HANDLERS["t_bad"] = bad_handler
        self.w = Worker(self.ctx)

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_and_failure_are_recorded_honestly(self):
        q = self.ctx.queue
        a, b = q.enqueue("t_ok", None), q.enqueue("t_bad", None)
        while self.w.run_one():
            pass
        self.assertEqual(q.get(a)["status"], "succeeded")
        jb = q.get(b)
        self.assertEqual(jb["status"], "failed")
        self.assertIn("boom", jb["error"])
        self.assertNotEqual(jb["progress"], 1)
        self.assertIn("Traceback", " ".join(l["message"] for l in q.logs(b)))

    def test_dead_worker_job_is_recovered_not_reported_done(self):
        q = self.ctx.queue
        j = q.enqueue("t_ok", None)
        claimed = q.claim("worker-that-dies")
        self.assertEqual(claimed["id"], j)
        self.assertEqual(q.get(j)["status"], "running")
        self.assertEqual(q.recover_stale(), [])         # heartbeat is fresh
        time.sleep(2.2)                                  # stale_job_seconds=2 in tests
        self.assertEqual(q.recover_stale(), [j])
        self.assertEqual(q.get(j)["status"], "queued")
        self.assertTrue(self.w.run_one())
        self.assertEqual(q.get(j)["status"], "succeeded")
        self.assertEqual(q.get(j)["attempts"], 2)

    def test_attempts_exhausted_fails(self):
        q = self.ctx.queue
        j = q.enqueue("t_ok", None, max_attempts=1)
        q.claim("w")
        time.sleep(2.2)
        q.recover_stale()
        job = q.get(j)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error_code"], "worker_lost")

    def test_cancel_queued_and_retry(self):
        q = self.ctx.queue
        j = q.enqueue("t_ok", None)
        self.assertEqual(q.cancel(j), "cancelled")
        self.assertFalse(self.w.run_one())
        q.retry(j)
        self.assertTrue(self.w.run_one())
        with self.assertRaises(ValueError):
            q.retry(j)           # succeeded jobs are not retryable

    def test_heavy_jobs_serialised_and_pause(self):
        q = self.ctx.queue
        q.enqueue("terrain", None)
        q.enqueue("terrain", None)
        first = q.claim("a")
        self.assertIsNotNone(first)
        self.assertIsNone(q.claim("b"))            # heavy_concurrency=1
        light = q.enqueue("ingest_image", None)
        self.assertEqual(q.claim("b")["id"], light)  # light jobs still flow
        q.set_paused(True)
        q.enqueue("ingest_image", None)
        self.assertIsNone(q.claim("c"))
        q.set_paused(False)
        self.assertIsNotNone(q.claim("c"))

    def test_dedupe_only_merges_with_queued(self):
        q = self.ctx.queue
        a = q.enqueue("analyze_flights", None, dedupe_key="k")
        self.assertEqual(q.enqueue("analyze_flights", None, dedupe_key="k"), a)
        q.claim("w")
        self.assertNotEqual(q.enqueue("analyze_flights", None, dedupe_key="k"), a)


if __name__ == "__main__":
    unittest.main()
