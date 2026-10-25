"""Orthomosaic tiles and the classification editor API.

Mask tiles are 3-channel PNGs:  R = class id, G = provenance (0 none, 1 manual edit, 2 automatic), B = confidence 0..100.
The client colours them with its own palette, so changing class colours / visibility / opacity never needs a refetch.
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import Route

from ..core import classes as C
from ..core.editing import EditError, EditOp, apply_edit, compute_selection
from ..core.rasterstore import EditTooLarge, RasterStoreError
from ..core.tiling import Window
from ..db import loads
from ..services import classify as K
from ..services import ortho as O
from ..services import products as P
from .deps import ApiError, body, ctx_of, ok, project_or_404, run

TILE = O.TILE


def _png(arr_bgr: np.ndarray) -> bytes:
    ok_, buf = cv2.imencode(".png", arr_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()


def _need_ortho(app, pid):
    p = P.get(app, pid, "orthophoto")
    if not p or not p["exists"]:
        raise ApiError(409, "This project has no orthomosaic yet. Run photogrammetry or import an orthomosaic GeoTIFF first.", "no_orthophoto")
    return p


def _classes(proj) -> list:
    return loads(proj["classes_json"], None) or [c.to_dict() for c in C.DEFAULT_CLASSES]


def _encode(vals: np.ndarray, conf: np.ndarray) -> bytes:
    cls = (vals & K.CLASS_MASK).astype(np.uint8)
    prov = np.where(vals & K.TAG_MANUAL, 1, np.where(vals & K.TAG_AUTO, 2, 0)).astype(np.uint8)
    return _png(np.dstack([conf, prov, cls]))          # cv2 writes BGR, so the PNG's R channel is the class


def _etag_ok(request: Request, etag: str) -> bool:
    return request.headers.get("if-none-match") == etag


def _unit_area(proj, grid) -> Optional[float]:
    """Area of one ortho pixel in m^2 (None when unknown or the CRS is geographic)."""
    if not grid or proj["horizontal_units"] == "degree":
        return None
    f = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200 / 3937}.get(proj["horizontal_units"], 1.0)
    return (grid["res"] * f) ** 2


# ----------------------------------------------------------------------------- ortho
async def ortho_info(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)
    p = await run(P.get, app, pid, "orthophoto")
    if not p or not p["exists"]:
        return ok({"available": False})
    m = p["meta"]
    return ok({"available": True, "width": m["width"], "height": m["height"], "bands": m["bands"], "epsg": m["epsg"],
               "georeferenced": m["georeferenced"], "grid": m["grid"], "horizontal_units": m["horizontal_units"],
               "pyramid": m["pyramid"], "warnings": m.get("warnings", []), "created_at": p["created_at"]})


async def ortho_tile(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    root = app.storage.path(pid, "ortho/pyramid")
    if not (root / "pyramid.json").exists():
        raise ApiError(404, "no orthomosaic", "no_orthophoto")
    pyr = O.OrthoPyramid(root)
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    p = pyr.tile_path(z, x, y)
    if p is None:
        raise ApiError(404, "tile not found", "not_found")
    return FileResponse(p, media_type="image/png" if pyr.ext == "png" else "image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


# ----------------------------------------------------------------------------- classification state
def _info(app, pid, proj):
    prod = _need_ortho(app, pid)
    mask, conf = O.class_rasters(app, pid)
    m = prod["meta"]
    return {"width": mask.width, "height": mask.height, "tile": mask.tile, "version": mask.version, "can_undo": mask.can_undo(),
            "can_redo": mask.can_redo(), "classes": _classes(proj), "epsg": m["epsg"], "georeferenced": m["georeferenced"], "grid": m["grid"],
            "pixel_area_m2": _unit_area(proj, m["grid"]), "horizontal_units": proj["horizontal_units"],
            "models": K.available_models(app, pid), "snapshots": K.list_snapshots(app, pid),
            "notice": "Classification is a visual interpretation of the orthomosaic. It is NOT elevation evidence and does not by itself "
                      "recover ground beneath vegetation."}


async def classification_info(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)
    return ok(await run(_info, app, pid, proj))


async def class_tile(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
    if not 0 <= z <= 2:
        raise ApiError(400, "mask tiles are served for levels 0-2; use /classification/overview when zoomed further out", "bad_level")

    def work():
        _need_ortho(app, pid)
        mask, conf = O.class_rasters(app, pid)
        etag = f'W/"{mask.version}-{z}-{x}-{y}"'
        if _etag_ok(request, etag):
            return Response(status_code=304, headers={"ETag": etag})
        s, span = 2 ** z, TILE * 2 ** z
        if x * span >= mask.width or y * span >= mask.height:
            raise ApiError(404, "tile outside raster", "not_found")
        win = Window(x * span, y * span, min((x + 1) * span, mask.width), min((y + 1) * span, mask.height))
        if win.x0 >= win.x1 or win.y0 >= win.y1:
            raise ApiError(404, "tile outside raster", "not_found")
        v, c = mask.read(win)[::s, ::s], conf.read(win)[::s, ::s]
        return Response(_encode(v, c), media_type="image/png", headers={"ETag": etag, "X-Raster-Version": str(mask.version),
                                                                         "Cache-Control": "private, no-cache"})
    return await run(work)


async def class_overview(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    size = max(256, min(int(request.query_params.get("size", 2048)), 4096))

    def work():
        _need_ortho(app, pid)
        mask, conf = O.class_rasters(app, pid)
        etag = f'W/"ov-{mask.version}-{size}"'
        if _etag_ok(request, etag):
            return Response(status_code=304, headers={"ETag": etag})
        s = max(1, -(-max(mask.width, mask.height) // size))
        out = np.zeros((-(-mask.height // s), -(-mask.width // s)), np.uint8)
        outc = np.zeros_like(out)
        for (ty, tx) in mask.iter_existing_tiles():
            t = mask.read_tile(ty, tx)
            oy, ox = ty * mask.tile, tx * mask.tile
            y0, x0 = -(-oy // s) * s, -(-ox // s) * s
            if y0 >= oy + t.shape[0] or x0 >= ox + t.shape[1]:
                continue
            sub = t[y0 - oy::s, x0 - ox::s]
            out[y0 // s:y0 // s + sub.shape[0], x0 // s:x0 // s + sub.shape[1]] = sub
            ct = conf.read_tile(ty, tx)[y0 - oy::s, x0 - ox::s]
            outc[y0 // s:y0 // s + ct.shape[0], x0 // s:x0 // s + ct.shape[1]] = ct
        return Response(_encode(out, outc), media_type="image/png",
                        headers={"ETag": etag, "X-Scale": str(s), "X-Raster-Version": str(mask.version), "Cache-Control": "private, no-cache"})
    return await run(work)


# ----------------------------------------------------------------------------- editing
class EditIn(BaseModel):
    type: Literal["brush", "polygon", "lasso", "rect", "fill", "similar"]
    class_id: int = Field(ge=0, le=63)
    points: List[List[float]] = Field(default_factory=list, max_length=200_000)
    radius: float = 8.0
    tolerance: float = Field(12.0, ge=0, le=441)
    connectivity: Literal[4, 8] = 4
    rect: Optional[List[float]] = None
    analysis_window: Optional[List[int]] = None
    only_classes: Optional[List[int]] = None
    protect_classes: Optional[List[int]] = None
    min_area: int = Field(0, ge=0)
    fill_holes: bool = False
    smooth_radius: int = Field(0, ge=0, le=20)
    label: Optional[str] = None


def _op(e: EditIn) -> EditOp:
    for p in e.points:
        if len(p) != 2 or not all(np.isfinite(p)):
            raise ApiError(422, "points must be finite [x, y] pairs", "validation_error")
    return EditOp(type=e.type, class_id=e.class_id, points=[(p[0], p[1]) for p in e.points], radius=e.radius, tolerance=e.tolerance,
                  connectivity=e.connectivity, rect=tuple(e.rect) if e.rect else None,
                  analysis_window=Window(*e.analysis_window) if e.analysis_window else None,
                  only_classes=e.only_classes, protect_classes=e.protect_classes, min_area=e.min_area, fill_holes=e.fill_holes,
                  smooth_radius=e.smooth_radius)


def _state(mask, win=None, **extra):
    return {"version": mask.version, "can_undo": mask.can_undo(), "can_redo": mask.can_redo(),
            "window": [win.x0, win.y0, win.x1, win.y1] if win else None, **extra}


async def edit(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    e = await body(request, EditIn)
    op = _op(e)

    def work():
        prod = _need_ortho(app, pid)
        mask, _ = O.class_rasters(app, pid)
        known = {c["id"] for c in _classes(app.db.one("SELECT classes_json FROM projects WHERE id=?", (pid,)))}
        if e.class_id not in known:
            raise ApiError(422, f"class {e.class_id} is not defined in this project", "unknown_class")
        rd = O.ProjectOrthoReader(prod["path"]) if e.type in ("fill", "similar") else None
        try:
            r = apply_edit(op, mask, rd, label=e.label, class_mask=K.CLASS_MASK, tag=K.TAG_MANUAL)
        except EditError as ex:
            raise ApiError(422, str(ex), "edit_error")
        except EditTooLarge as ex:
            raise ApiError(413, str(ex), "edit_too_large")
        finally:
            if rd:
                rd.close()
        return _state(mask, r.window, changed_pixels=r.changed_pixels, truncated=r.truncated, message=r.message)
    return ok(await run(work))


async def select_preview(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    e = await body(request, EditIn)
    op = _op(e)

    def work():
        prod = _need_ortho(app, pid)
        mask, _ = O.class_rasters(app, pid)
        rd = O.ProjectOrthoReader(prod["path"]) if e.type in ("fill", "similar") else None
        try:
            win, sel, trunc = compute_selection(op, mask, rd, K.CLASS_MASK)
        except EditError as ex:
            raise ApiError(422, str(ex), "edit_error")
        finally:
            if rd:
                rd.close()
        if win is None:
            return Response(status_code=204)
        return Response(_png((sel.astype(np.uint8) * 255)), media_type="image/png",
                        headers={"X-Window": f"{win.x0},{win.y0},{win.x1},{win.y1}", "X-Selected-Pixels": str(int(sel.sum())),
                                 "X-Truncated": "1" if trunc else "0"})
    return await run(work)


async def undo(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        _need_ortho(app, pid)
        mask, _ = O.class_rasters(app, pid)
        return _state(mask, mask.undo())
    return ok(await run(work))


async def redo(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        _need_ortho(app, pid)
        mask, _ = O.class_rasters(app, pid)
        return _state(mask, mask.redo())
    return ok(await run(work))


async def sample(request: Request):
    """Inspect one pixel: colour, class, provenance, confidence, map coordinates, and the colour-similar segment around it."""
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)
    q = request.query_params
    try:
        x, y = int(float(q["x"])), int(float(q["y"]))
        tol = float(q.get("tolerance", 12))
    except (KeyError, ValueError):
        raise ApiError(400, "x and y are required", "bad_request")

    def work():
        prod = _need_ortho(app, pid)
        mask, conf = O.class_rasters(app, pid)
        if not (0 <= x < mask.width and 0 <= y < mask.height):
            raise ApiError(404, "outside the raster", "not_found")
        rd = O.ProjectOrthoReader(prod["path"])
        try:
            rgb, valid = rd.read_rgb_valid(Window(x, y, x + 1, y + 1))
            seg = None
            if q.get("segment") == "1":
                op = EditOp("fill", 0, [(x + .5, y + .5)], tolerance=tol)
                win, sel, trunc = compute_selection(op, mask, rd, K.CLASS_MASK)
                if win is not None:
                    ys, xs = np.nonzero(sel)
                    seg = {"pixels": int(sel.sum()), "truncated": trunc, "bbox": [int(xs.min() + win.x0), int(ys.min() + win.y0),
                                                                                  int(xs.max() + win.x0 + 1), int(ys.max() + win.y0 + 1)]}
        finally:
            rd.close()
        v = int(mask.read(Window(x, y, x + 1, y + 1))[0, 0])
        cid = v & K.CLASS_MASK
        cls = next((c for c in _classes(proj) if c["id"] == cid), None)
        g = prod["meta"]["grid"]
        mapxy = None
        if g:
            mapxy = {"x": g["x0"] + (x + .5) * g["res"], "y": g["y0"] - (y + .5) * g["res"], "epsg": prod["meta"]["epsg"]}
        return {"x": x, "y": y, "rgb": [int(c) for c in rgb[0, 0]], "valid": bool(valid[0, 0]), "class_id": cid, "class_name": cls["name"] if cls else None,
                "provenance": "manual" if v & K.TAG_MANUAL else ("automatic" if v & K.TAG_AUTO else "none"),
                "confidence": (int(conf.read(Window(x, y, x + 1, y + 1))[0, 0]) / 100.0) if v & K.TAG_AUTO else None,
                "map": mapxy, "segment": seg}
    return ok(await run(work))


async def class_stats(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)

    def work():
        prod = _need_ortho(app, pid)
        mask, _ = O.class_rasters(app, pid)
        h = mask.histogram()
        unit = _unit_area(proj, prod["meta"]["grid"])
        total = mask.width * mask.height
        rows = []
        for c in _classes(proj):
            cid = c["id"]
            n = {"none": 0, "manual": 0, "automatic": 0}
            for v, cnt in h.items():
                if v & K.CLASS_MASK == cid:
                    n["manual" if v & K.TAG_MANUAL else ("automatic" if v & K.TAG_AUTO else "none")] += cnt
            px = sum(n.values())
            rows.append({"class_id": cid, "name": c["name"], "color": c["color"], "vegetation": c.get("vegetation", False), "pixels": px,
                         "percent": 100.0 * px / total if total else 0, "area_m2": px * unit if unit else None,
                         "area_acres": px * unit / 4046.8564224 if unit else None, "manual_pixels": n["manual"],
                         "automatic_pixels": n["automatic"]})
        classified = sum(r["pixels"] for r in rows if r["class_id"] != 0)
        return {"total_pixels": total, "classified_pixels": classified, "pixel_area_m2": unit, "classes": rows, "version": mask.version}
    return ok(await run(work))


# ----------------------------------------------------------------------------- jobs & snapshots
class AutoIn(BaseModel):
    model: Literal["heuristic", "learned", "onnx"] = "heuristic"
    overwrite: Literal["unclassified", "auto", "all"] = "auto"
    region: Optional[dict] = None


async def auto_classify(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    a = await body(request, AutoIn)
    if a.region is not None:
        t = a.region.get("type")
        if t not in ("rect", "polygon", "lasso") or (t == "rect" and len(a.region.get("rect", [])) != 4) or \
           (t != "rect" and len(a.region.get("points", [])) < 3):
            raise ApiError(422, "region must be {type:'rect',rect:[x0,y0,x1,y1]} or {type:'polygon',points:[[x,y],...]}", "validation_error")

    def work():
        _need_ortho(app, pid)
        busy = app.db.one("SELECT id FROM jobs WHERE project_id=? AND kind IN ('classify','train_classifier') AND status IN ('queued','running')", (pid,))
        if busy:
            raise ApiError(409, "a classification job is already queued or running for this project", "job_active")
        return app.queue.enqueue("classify", pid, {"project_id": pid, **a.model_dump()})
    return ok({"job_id": await run(work)}, 202)


async def train(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        _need_ortho(app, pid)
        busy = app.db.one("SELECT id FROM jobs WHERE project_id=? AND kind IN ('classify','train_classifier') AND status IN ('queued','running')", (pid,))
        if busy:
            raise ApiError(409, "a classification job is already queued or running for this project", "job_active")
        return app.queue.enqueue("train_classifier", pid, {"project_id": pid})
    return ok({"job_id": await run(work)}, 202)


class SnapIn(BaseModel):
    label: str = Field("manual snapshot", max_length=80)


async def snapshots(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(K.list_snapshots, app, pid))


async def snapshot_create(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, SnapIn)
    await run(_need_ortho, app, pid)
    return ok(await run(K.take_snapshot, app, pid, b.label), 201)


async def snapshot_restore(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    try:
        r = await run(K.restore_snapshot, app, pid, request.path_params["sid"])
    except Exception as ex:
        from ..jobs import JobFailure
        if isinstance(ex, JobFailure):
            raise ApiError(404, str(ex), "not_found")
        raise
    mask, _ = O.class_rasters(app, pid)
    return ok({**r, "version": mask.version})


routes = [
    Route("/projects/{pid}/ortho", ortho_info),
    Route("/projects/{pid}/ortho/tiles/{z:int}/{x:int}/{y:int}", ortho_tile),
    Route("/projects/{pid}/classification", classification_info),
    Route("/projects/{pid}/classification/tiles/{z:int}/{x:int}/{y:int}", class_tile),
    Route("/projects/{pid}/classification/overview", class_overview),
    Route("/projects/{pid}/classification/edit", edit, methods=["POST"]),
    Route("/projects/{pid}/classification/select-preview", select_preview, methods=["POST"]),
    Route("/projects/{pid}/classification/undo", undo, methods=["POST"]),
    Route("/projects/{pid}/classification/redo", redo, methods=["POST"]),
    Route("/projects/{pid}/classification/sample", sample),
    Route("/projects/{pid}/classification/stats", class_stats),
    Route("/projects/{pid}/classification/auto", auto_classify, methods=["POST"]),
    Route("/projects/{pid}/classification/train", train, methods=["POST"]),
    Route("/projects/{pid}/classification/snapshots", snapshots),
    Route("/projects/{pid}/classification/snapshots", snapshot_create, methods=["POST"]),
    Route("/projects/{pid}/classification/snapshots/{sid}/restore", snapshot_restore, methods=["POST"]),
]
