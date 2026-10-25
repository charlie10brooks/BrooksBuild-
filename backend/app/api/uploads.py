"""Resumable chunked uploads. Files are finalised asynchronously by ``ingest_image`` jobs."""
from __future__ import annotations

import tempfile
from typing import Literal, Optional

from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.routing import Route

from ..core.uploadstore import UploadError
from ..db import new_id, now
from .deps import ApiError, body, ctx_of, ok, project_or_404, run

PURPOSE_EXT = {"image": {".jpg", ".jpeg", ".tif", ".tiff"}, "orthophoto": {".tif", ".tiff"}, "dem": {".tif", ".tiff"},
               "lidar": {".las", ".laz"}, "survey_points": {".csv", ".txt"}, "gcp_list": {".txt"}}
_CODE_STATUS = {"not_found": 404, "too_large": 413, "unsupported": 415, "insufficient_storage": 507, "checksum_mismatch": 422,
                "incomplete": 409, "size_mismatch": 422}


def _err(e: UploadError) -> ApiError:
    return ApiError(_CODE_STATUS.get(e.code, 400), str(e), e.code)


class UploadIn(BaseModel):
    filename: str = Field(min_length=1, max_length=500)
    size: int = Field(gt=0)
    chunk_size: int = Field(default=8 * 1024 * 1024)
    sha256: Optional[str] = None
    batch_id: Optional[str] = None
    client_key: Optional[str] = Field(None, max_length=300, description="stable client identity (name+size+mtime) for resume")
    purpose: Literal["image", "orthophoto", "dem", "lidar", "survey_points", "gcp_list"] = "image"


class BatchIn(BaseModel):
    label: Optional[str] = Field(None, max_length=200)


async def create_batch(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, BatchIn)
    bid = new_id()
    n = app.db.one("SELECT COUNT(*) n FROM batches WHERE project_id=?", (pid,))["n"] + 1
    await run(app.db.insert, "batches", {"id": bid, "project_id": pid, "label": b.label or f"Batch {n}", "created_at": now()})
    return ok({"id": bid, "label": b.label or f"Batch {n}"}, 201)


async def list_batches(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    rows = app.db.query("SELECT b.id, b.label, b.created_at, (SELECT COUNT(*) FROM images i WHERE i.batch_id=b.id) images, "
                        "(SELECT COALESCE(SUM(size),0) FROM images i WHERE i.batch_id=b.id) bytes, "
                        "(SELECT COUNT(*) FROM upload_files u WHERE u.batch_id=b.id AND u.status IN ('failed','rejected')) problems "
                        "FROM batches b WHERE b.project_id=? ORDER BY b.created_at", (pid,))
    return ok([dict(r) for r in rows])


async def create_upload(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, UploadIn)
    s = app.settings

    def work():
        if b.batch_id and not app.db.one("SELECT 1 FROM batches WHERE id=? AND project_id=?", (b.batch_id, pid)):
            raise ApiError(404, "batch not found", "not_found")
        if b.client_key:      # resume an interrupted upload of the same file
            r = app.db.one("SELECT upload_id FROM upload_files WHERE project_id=? AND client_key=? AND status IN ('uploading','failed')",
                           (pid, b.client_key))
            if r:
                try:
                    st = app.uploads.status(r["upload_id"])
                    if st["size"] == b.size:
                        return {**st, "resumed": True, "status": "uploading"}
                except UploadError:
                    app.db.execute("DELETE FROM upload_files WHERE upload_id=?", (r["upload_id"],))
        if b.sha256 and b.purpose == "image":
            d = app.db.one("SELECT id, filename FROM images WHERE project_id=? AND sha256=?", (pid, b.sha256.lower()))
            if d:
                uid = new_id()
                app.db.insert("upload_files", {"upload_id": uid, "project_id": pid, "batch_id": b.batch_id, "filename": b.filename[:200],
                                               "size": b.size, "status": "duplicate", "duplicate_of": d["id"],
                                               "error": f"identical to already uploaded file {d['filename']}",
                                               "client_key": b.client_key, "created_at": now(), "updated_at": now()})
                return {"upload_id": uid, "status": "duplicate", "duplicate_of": d["id"], "missing_chunks": [], "total_chunks": 0}
        if s.project_quota_bytes and app.storage.usage(pid) + b.size > s.project_quota_bytes:
            raise ApiError(413, "project storage quota would be exceeded", "quota_exceeded")
        try:
            st = app.uploads.create(pid, b.filename, b.size, b.chunk_size, b.sha256, b.batch_id, s.max_file_bytes or None,
                                    PURPOSE_EXT[b.purpose])
        except UploadError as e:
            raise _err(e)
        app.db.insert("upload_files", {"upload_id": st.id, "project_id": pid, "batch_id": b.batch_id, "filename": st.filename,
                                       "size": b.size, "status": "uploading", "client_key": b.client_key, "purpose": b.purpose,
                                       "created_at": now(), "updated_at": now()})
        return {**app.uploads.status(st.id), "status": "uploading", "resumed": False}

    return ok(await run(work), 201)


def _upload_or_404(app, pid, uid):
    r = app.db.one("SELECT * FROM upload_files WHERE upload_id=? AND project_id=?", (uid, pid))
    if r is None:
        raise ApiError(404, "upload not found", "not_found")
    return r


async def get_upload(request: Request):
    app = ctx_of(request)
    pid, uid = request.path_params["pid"], request.path_params["uid"]
    r = await run(_upload_or_404, app, pid, uid)
    out = dict(r)
    if r["status"] in ("uploading", "failed"):
        try:
            out.update(await run(app.uploads.status, uid))
        except UploadError:
            pass
    return ok(out)


async def put_chunk(request: Request):
    app = ctx_of(request)
    pid, uid = request.path_params["pid"], request.path_params["uid"]
    idx = int(request.path_params["idx"])
    r = await run(_upload_or_404, app, pid, uid)
    if r["status"] not in ("uploading", "failed"):
        raise ApiError(409, f"upload is {r['status']}; chunks are no longer accepted", "bad_state")
    limit = app.settings.max_request_body_bytes
    cl = request.headers.get("content-length")
    if cl and int(cl) > limit:
        raise ApiError(413, f"chunk larger than the {limit} byte limit", "too_large")
    spool = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    n = 0
    try:
        async for block in request.stream():
            n += len(block)
            if n > limit:
                raise ApiError(413, f"chunk larger than the {limit} byte limit", "too_large")
            spool.write(block)
        spool.seek(0)

        def work():
            def blocks():
                while True:
                    b = spool.read(1024 * 1024)
                    if not b:
                        return
                    yield b
            try:
                return app.uploads.put_chunk(uid, idx, blocks(), request.headers.get("x-chunk-sha256"))
            except UploadError as e:
                raise _err(e)
        res = await run(work)
    finally:
        spool.close()
    return ok(res)


async def complete_upload(request: Request):
    """Queue server-side assembly + ingestion. Idempotent; also used to re-queue after a failed ingest."""
    app = ctx_of(request)
    pid, uid = request.path_params["pid"], request.path_params["uid"]
    r = await run(_upload_or_404, app, pid, uid)
    if r["status"] in ("queued", "complete", "duplicate"):
        return ok({"status": r["status"]})
    if r["status"] == "rejected":
        raise ApiError(409, "this file was rejected as an invalid image", "rejected")
    try:
        st = await run(app.uploads.status, uid)
    except UploadError as e:
        raise _err(e)
    if st["missing_chunks"]:
        raise ApiError(409, f"{len(st['missing_chunks'])} chunk(s) still missing", "incomplete", {"missing_chunks": st["missing_chunks"][:200]})

    def work():
        app.db.update("upload_files", {"upload_id": uid}, {"status": "queued", "error": None, "updated_at": now()})
        if r["purpose"] == "image":
            return app.queue.enqueue("ingest_image", pid, {"upload_id": uid, "project_id": pid}, priority=0)
        return app.queue.enqueue("import_asset", pid, {"upload_id": uid, "project_id": pid, "purpose": r["purpose"]}, priority=5)
    jid = await run(work)
    return ok({"status": "queued", "job_id": jid}, 202)


async def abort_upload(request: Request):
    app = ctx_of(request)
    pid, uid = request.path_params["pid"], request.path_params["uid"]
    r = await run(_upload_or_404, app, pid, uid)
    if r["status"] in ("queued", "complete"):
        raise ApiError(409, f"upload is already {r['status']}", "bad_state")
    await run(app.uploads.discard, uid)
    await run(app.db.update, "upload_files", {"upload_id": uid}, {"status": "aborted", "updated_at": now()})
    return ok()


async def list_uploads(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    q = request.query_params
    where, args = ["project_id=?"], [pid]
    if q.get("status"):
        sts = q["status"].split(",")
        where.append("status IN (%s)" % ",".join("?" * len(sts))); args += sts
    if q.get("batch"):
        where.append("batch_id=?"); args.append(q["batch"])
    limit, offset = max(1, min(int(q.get("limit", 200)), 2000)), max(0, int(q.get("offset", 0)))
    sql = " AND ".join(where)
    rows = app.db.query(f"SELECT upload_id,batch_id,filename,size,status,error,image_id,duplicate_of,purpose,updated_at FROM upload_files WHERE {sql} "
                        "ORDER BY updated_at DESC LIMIT ? OFFSET ?", [*args, limit, offset])
    total = app.db.one(f"SELECT COUNT(*) n FROM upload_files WHERE {sql}", args)["n"]
    return ok({"total": total, "items": [dict(r) for r in rows]})


async def upload_summary(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    rows = app.db.query("SELECT status, COUNT(*) n, COALESCE(SUM(size),0) bytes FROM upload_files WHERE project_id=? GROUP BY status", (pid,))
    by = {r["status"]: {"files": r["n"], "bytes": r["bytes"]} for r in rows}
    return ok({"by_status": by, "total_files": sum(v["files"] for v in by.values()),
               "total_bytes": sum(v["bytes"] for v in by.values()),
               "needs_attention": sum(by.get(k, {"files": 0})["files"] for k in ("failed", "rejected"))})


routes = [
    Route("/projects/{pid}/batches", list_batches, methods=["GET"]),
    Route("/projects/{pid}/batches", create_batch, methods=["POST"]),
    Route("/projects/{pid}/uploads", list_uploads, methods=["GET"]),
    Route("/projects/{pid}/uploads", create_upload, methods=["POST"]),
    Route("/projects/{pid}/uploads/summary", upload_summary, methods=["GET"]),
    Route("/projects/{pid}/uploads/{uid}", get_upload, methods=["GET"]),
    Route("/projects/{pid}/uploads/{uid}", abort_upload, methods=["DELETE"]),
    Route("/projects/{pid}/uploads/{uid}/chunks/{idx:int}", put_chunk, methods=["PUT"]),
    Route("/projects/{pid}/uploads/{uid}/complete", complete_upload, methods=["POST"]),
]
