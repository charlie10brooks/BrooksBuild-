"""Classification of the original drone photos (per-photo masks). See services/imgclass.py for what this is and is not."""
from __future__ import annotations

from typing import List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ..core import classes as C
from ..db import loads
from ..services import classify as K
from ..services import imgclass as IC
from .deps import ApiError, body, ctx_of, ok, project_or_404, run


class ClassifyImagesIn(BaseModel):
    scope: Literal["all", "batch", "ids"] = "all"
    batch_id: Optional[str] = None
    ids: Optional[List[str]] = Field(None, max_length=100000)
    model: Literal["heuristic", "learned", "onnx"] = "heuristic"
    max_side: int = Field(1600, ge=256, le=4096)
    overwrite: bool = False


def _classes(proj) -> list:
    return loads(proj["classes_json"], None) or [c.to_dict() for c in C.DEFAULT_CLASSES]


async def state(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        models = [m for m in K.available_models(app, pid)]
        return {**IC.summary(app, pid), "models": models,
                "disclaimer": "Per-photo land-cover labels from a colour-rule heuristic or a model you trained/supplied. Not georeferenced, not elevation "
                              "evidence, and assumes roughly nadir photos (sky and oblique views can be misread)."}
    return ok(await run(work))


async def start(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, ClassifyImagesIn)

    def work():
        if b.scope == "batch" and not b.batch_id:
            raise ApiError(400, "batch_id is required when scope is 'batch'", "bad_request")
        m = next((m for m in K.available_models(app, pid) if m["id"] == b.model), None)
        if not m or not m["available"]:
            raise ApiError(409, (m or {}).get("note") or "That model is not available. Use the heuristic model, or train one from corrections first.", "no_model")
        rows = IC.target_images(app, pid, b.scope, b.batch_id, b.ids, b.overwrite)
        if not rows:
            raise ApiError(409, "No photos to classify" + ("" if b.overwrite else " (all of them already have labels; tick 'redo existing' to replace)"), "nothing_to_do")
        jid = app.queue.enqueue("classify_images", pid, {"project_id": pid, **b.model_dump()})
        return {"job_id": jid, "photos": len(rows)}
    return ok(await run(work), 202)


async def items(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    q = request.query_params

    def work():
        where, args = ["i.project_id=?"], [pid]
        if q.get("batch"):
            where.append("i.batch_id=?"); args.append(q["batch"])
        if q.get("state") == "classified":
            where.append("c.image_id IS NOT NULL")
        elif q.get("state") == "unclassified":
            where.append("c.image_id IS NULL")
        sql = " AND ".join(where)
        j = "FROM images i LEFT JOIN image_classification c ON c.image_id=i.id"
        total = app.db.one(f"SELECT COUNT(*) n {j} WHERE {sql}", args)["n"]
        limit, offset = max(1, min(int(q.get("limit", 60)), 500)), max(0, int(q.get("offset", 0)))
        rows = app.db.query(f"SELECT i.id, i.filename, i.batch_id, i.width, i.height, c.model, c.counts_json, c.unclassified, c.mean_conf, c.created_at c_at "
                            f"{j} WHERE {sql} ORDER BY i.captured_at, i.filename LIMIT ? OFFSET ?", [*args, limit, offset])
        out = []
        for r in rows:
            counts = loads(r["counts_json"], {}) if r["counts_json"] else None
            px = sum(counts.values()) if counts else 0
            out.append({"id": r["id"], "filename": r["filename"], "batch_id": r["batch_id"], "classified": counts is not None, "model": r["model"],
                        "unclassified": r["unclassified"], "mean_confidence": r["mean_conf"],
                        "fractions": {k: v / px for k, v in counts.items()} if px else None})
        return {"total": total, "offset": offset, "limit": limit, "items": out}
    return ok(await run(work))


def _row(app, pid, iid):
    r = app.db.one("SELECT * FROM image_classification WHERE image_id=? AND project_id=?", (iid, pid))
    if r is None:
        raise ApiError(404, "this photo has not been classified", "not_classified")
    return r


async def overlay(request: Request):
    """RGBA PNG (working resolution) coloured with the project's class colours. ?opacity=0..1&hide=1,2"""
    app, pid, iid = ctx_of(request), request.path_params["pid"], request.path_params["iid"]
    proj = await run(project_or_404, app, pid)
    q = request.query_params

    def work():
        _row(app, pid, iid)
        m = cv2.imread(str(IC.mask_path(app, pid, iid)), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ApiError(404, "mask file missing; classify the photo again", "mask_missing")
        hidden = {int(v) for v in q.get("hide", "").split(",") if v.strip().isdigit()}
        lut = C.build_lut(_classes(proj), float(q.get("opacity", 0.6)), hidden)
        rgba = C.colorize(m[..., 2], lut)
        ok_, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=300"})
    return await run(work)


async def detail(request: Request):
    app, pid, iid = ctx_of(request), request.path_params["pid"], request.path_params["iid"]
    await run(project_or_404, app, pid)

    def work():
        r = _row(app, pid, iid)
        return {"image_id": iid, "model": r["model"], "width": r["width"], "height": r["height"], "scale": r["scale"],
                "counts": loads(r["counts_json"], {}), "unclassified": r["unclassified"], "mean_confidence": r["mean_conf"],
                "note": "Labels are in this photo's own (reduced) pixels; not georeferenced and not elevation evidence."}
    return ok(await run(work))


async def clear(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        n = app.db.one("SELECT COUNT(*) n FROM image_classification WHERE project_id=?", (pid,))["n"]
        with app.db.tx() as c:
            c.execute("DELETE FROM image_classification WHERE project_id=?", (pid,))
        d = app.storage.path(pid, IC.DIR)
        if d.exists():
            for f in d.glob("*.png"):
                f.unlink()
        return {"removed": n}
    return ok(await run(work))


routes = [
    Route("/projects/{pid}/image-classification", state),
    Route("/projects/{pid}/image-classification", start, methods=["POST"]),
    Route("/projects/{pid}/image-classification", clear, methods=["DELETE"]),
    Route("/projects/{pid}/image-classification/items", items),
    Route("/projects/{pid}/images/{iid}/classification", detail),
    Route("/projects/{pid}/images/{iid}/classification/overlay.png", overlay),
]
