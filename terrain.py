"""Terrain products, viewers' data feeds, site-planning tools, project features."""
from __future__ import annotations

import csv
import io
import json
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ..core import dtm as D
from ..core import planning as PL
from ..core import terrain_analysis as TA
from ..core.vectorize import mask_to_polygons
from ..db import dumps, loads, new_id, now
from ..services import ortho as O
from ..services import products as P
from ..services import render as R
from .deps import DISCLAIMER, ApiError, body, ctx_of, ok, project_or_404, run

LAYERS = ("dsm", "dtm", "ndsm", "confidence", "evidence", "source", "odm_dsm", "odm_dtm")


def _prod(app, pid, name, required=True):
    p = P.get(app, pid, name)
    if (not p or not p["exists"]) and required:
        raise ApiError(404, f"{name!r} is not available for this project yet" + (". Run terrain reconstruction first." if name in ("dtm", "ndsm", "confidence", "evidence") else "."), "no_product")
    return p if p and p["exists"] else None


def _pt(v, what="point"):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(a, (int, float)) and np.isfinite(a) for a in v)):
        raise ValueError(f"{what} must be a finite [x, y] pair")
    return (float(v[0]), float(v[1]))


class PointList(BaseModel):
    points: List[List[float]] = Field(min_length=1, max_length=5000)

    @field_validator("points")
    @classmethod
    def _v(cls, v):
        return [list(_pt(p)) for p in v]


# ----------------------------------------------------------------------------- products & terrain state
async def products(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    out = await run(P.list_products, app, pid)
    for p in out:
        p["surface_kind"] = R.KIND.get(p["name"])
        p["surface_text"] = R.KIND_TEXT.get(R.KIND.get(p["name"], ""), None)
        if p["name"] in ("terrain_summary",):
            p["meta"] = {"warnings": p["meta"].get("warnings"), "coverage": p["meta"].get("coverage")}
    return ok(out)


async def terrain_state(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)

    def work():
        summ = P.get(app, pid, "terrain_summary")
        srcs = [{**dict(r), "meta": loads(r["meta_json"], {})} for r in
                app.db.query("SELECT id,name,kind,priority,z_offset,trust,enabled,meta_json FROM elevation_sources WHERE project_id=? ORDER BY priority DESC", (pid,))]
        for s in srcs:
            s.pop("meta_json", None)
            s["enabled"] = bool(s["enabled"])
        have = {n: bool(P.get(app, pid, n) and P.get(app, pid, n)["exists"]) for n in LAYERS + ("pointcloud", "classified_cloud", "orthophoto")}
        return {"available": have["dtm"], "layers": have, "summary": summ["meta"] if summ else None, "sources": srcs,
                "epsg": proj["epsg"], "horizontal_units": proj["horizontal_units"], "vertical_units": proj["vertical_units"],
                "pdal_available": bool(__import__("shutil").which(app.settings.pdal_bin)),
                "notice": "The DTM is an ESTIMATE of bare earth. Where evidence is 'interpolated' or 'external' it was not observed by this "
                          "project's photogrammetry. " + DISCLAIMER,
                "dense_canopy_advice": (summ["meta"].get("confidence", {}).get("advice") if summ else None)}
    return ok(await run(work))


class GroundIn(BaseModel):
    engine: Literal["pdal_smrf", "pdal_pmf", "builtin_pmf"] = "pdal_smrf"
    smrf_cell: float = Field(1.0, gt=0, le=50)
    smrf_slope: float = Field(0.15, ge=0, le=5)
    smrf_window: float = Field(18.0, gt=0, le=500)
    smrf_threshold: float = Field(0.5, ge=0, le=20)
    smrf_scalar: float = Field(1.25, ge=0, le=20)
    outlier_removal: bool = True


class InterpIn(BaseModel):
    method: Literal["local_plane", "idw", "tin", "tps", "nearest"] = "local_plane"
    k: int = Field(16, ge=3, le=200)
    power: float = Field(2.0, gt=0, le=10)
    max_radius_m: float = Field(30.0, gt=0, le=1000)
    smoothing: float = Field(0.0, ge=0, le=1000)


class ConfIn(BaseModel):
    high_dist_m: float = Field(2.0, gt=0)
    moderate_dist_m: float = Field(6.0, gt=0)
    low_dist_m: float = Field(15.0, gt=0)


class TerrainRunIn(BaseModel):
    resolution_m: float = Field(0.5, gt=0, le=50)
    ground: GroundIn = GroundIn()
    interp: InterpIn = InterpIn()
    confidence: ConfIn = ConfIn()
    canopy_clearance_m: float = Field(1.0, ge=0, le=20)
    use_classification: bool = True


async def terrain_run(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, TerrainRunIn)
    c = b.confidence
    if not (c.high_dist_m <= c.moderate_dist_m <= c.low_dist_m):
        raise ApiError(422, "confidence distances must satisfy high <= moderate <= low", "validation_error")
    params = {"resolution_m": b.resolution_m, "canopy_clearance_m": b.canopy_clearance_m, "use_classification": b.use_classification}
    gd = b.ground.model_dump()
    gd = {k: v for k, v in gd.items()}

    def work():
        if not (P.get(app, pid, "pointcloud") or {}).get("exists"):
            raise ApiError(409, "No point cloud yet. Run photogrammetry (OpenDroneMap) first.", "no_pointcloud")
        busy = app.db.one("SELECT id FROM jobs WHERE project_id=? AND kind='terrain' AND status IN ('queued','running')", (pid,))
        if busy:
            raise ApiError(409, "A terrain job is already queued or running for this project.", "job_active")
        return app.queue.enqueue("terrain", pid, {"project_id": pid, "use_classification": b.use_classification,
                                                   "params": {"resolution_m": b.resolution_m, "canopy_clearance_m": b.canopy_clearance_m,
                                                              "ground": gd, "interp": b.interp.model_dump(), "confidence": c.model_dump()}})
    return ok({"job_id": await run(work)}, 202)


# ----------------------------------------------------------------------------- elevation sources
class SourcePatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    priority: Optional[int] = Field(None, ge=0, le=1000)
    z_offset: Optional[float] = Field(None, ge=-1000, le=1000)
    trust: Optional[Literal["high", "moderate", "low"]] = None
    enabled: Optional[bool] = None


async def source_patch(request: Request):
    app = ctx_of(request)
    pid, sid = request.path_params["pid"], int(request.path_params["sid"])
    await run(project_or_404, app, pid)
    b = await body(request, SourcePatch)
    vals = {k: (int(v) if isinstance(v, bool) else v) for k, v in b.model_dump(exclude_unset=True).items()}
    n = await run(app.db.update, "elevation_sources", {"project_id": pid, "id": sid}, vals)
    if not n:
        raise ApiError(404, "source not found", "not_found")
    return ok()


async def source_delete(request: Request):
    app = ctx_of(request)
    pid, sid = request.path_params["pid"], int(request.path_params["sid"])
    await run(project_or_404, app, pid)
    await run(app.db.execute, "DELETE FROM elevation_sources WHERE project_id=? AND id=?", (pid, sid))
    return ok()


# ----------------------------------------------------------------------------- rendering
def _bbox(q) -> tuple:
    try:
        b = tuple(float(v) for v in q["bbox"].split(","))
        w, h = int(q.get("w", 512)), int(q.get("h", 512))
    except (KeyError, ValueError):
        raise ApiError(400, "bbox=minx,miny,maxx,maxy and w,h are required", "bad_request")
    if len(b) != 4 or not (b[2] > b[0] and b[3] > b[1]) or not all(np.isfinite(b)):
        raise ApiError(400, "bbox must be minx,miny,maxx,maxy with max > min", "bad_request")
    if not (1 <= w <= 4096 and 1 <= h <= 4096):
        raise ApiError(400, "w and h must be between 1 and 4096", "bad_request")
    return b, w, h


async def render(request: Request):
    app = ctx_of(request)
    pid, layer = request.path_params["pid"], request.path_params["layer"]
    await run(project_or_404, app, pid)
    if layer not in LAYERS:
        raise ApiError(404, f"unknown layer {layer!r}", "not_found")
    q = request.query_params
    bbox, w, h = _bbox(q)
    ramp = q.get("ramp", "terrain")
    zex = float(q.get("zexag", 1.0))
    vmin = float(q["vmin"]) if q.get("vmin") not in (None, "") else None
    vmax = float(q["vmax"]) if q.get("vmax") not in (None, "") else None
    op = min(max(float(q.get("opacity", 1.0)), 0.0), 1.0)

    def work():
        p = _prod(app, pid, layer)
        try:
            img = R.render_layer(p["path"], layer, bbox, w, h, ramp, vmin, vmax, q.get("hillshade") == "1", zex, op)
        except ValueError as e:
            raise ApiError(400, str(e), "bad_request")
        kind = R.KIND[layer]
        return Response(R.png_rgba(img), media_type="image/png",
                        headers={"X-Surface-Kind": kind, "X-Surface-Label": R.KIND_TEXT[kind][:120], "Cache-Control": "private, max-age=60",
                                 "ETag": f'W/"{p["created_at"]}-{layer}"'})
    return await run(work)


async def layer_stats(request: Request):
    app = ctx_of(request)
    pid, layer = request.path_params["pid"], request.path_params["layer"]
    await run(project_or_404, app, pid)
    if layer not in LAYERS:
        raise ApiError(404, "unknown layer", "not_found")

    def work():
        p = _prod(app, pid, layer)
        arr, grid, _ = R.RASTERS.get(p["path"])
        kind = R.KIND[layer]
        return {"layer": layer, "kind": kind, "label": R.KIND_TEXT[kind], "grid": grid.to_dict(), "stats": R.raster_stats(p["path"]),
                "units": app.db.one("SELECT vertical_units FROM projects WHERE id=?", (pid,))["vertical_units"], "created_at": p["created_at"]}
    return ok(await run(work))


async def ortho_render(request: Request):
    """Mosaic of pyramid tiles for a map-coordinate window (used as the texture in the 3D viewer)."""
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    bbox, w, h = _bbox(request.query_params)

    def work():
        import cv2
        prod = P.get(app, pid, "orthophoto")
        if not prod or not prod["exists"] or not prod["meta"].get("grid"):
            raise ApiError(404, "no georeferenced orthomosaic", "no_orthophoto")
        g = prod["meta"]["grid"]
        pyr = O.OrthoPyramid(app.storage.path(pid, "ortho/pyramid"))
        T = pyr.info["tile"]
        # pixel window in level-0 coordinates
        c0, c1 = (bbox[0] - g["x0"]) / g["res"], (bbox[2] - g["x0"]) / g["res"]
        r0, r1 = (g["y0"] - bbox[3]) / g["res"], (g["y0"] - bbox[1]) / g["res"]
        z = int(max(0, min(pyr.info["levels"] - 1, np.floor(np.log2(max((c1 - c0) / w, (r1 - r0) / h, 1.0))))))
        s = 2 ** z
        tx0, tx1 = int(max(0, np.floor(c0 / s / T))), int(min(np.ceil(g["width"] / s / T) - 1, np.floor(c1 / s / T)))
        ty0, ty1 = int(max(0, np.floor(r0 / s / T))), int(min(np.ceil(g["height"] / s / T) - 1, np.floor(r1 / s / T)))
        if tx1 < tx0 or ty1 < ty0:
            return Response(R.png_rgba(np.zeros((h, w, 4), np.uint8)), media_type="image/png")
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 400:
            raise ApiError(400, "window too large for this output size", "bad_request")
        canvas = np.zeros(((ty1 - ty0 + 1) * T, (tx1 - tx0 + 1) * T, 4), np.uint8)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                pth = pyr.tile_path(z, tx, ty)
                if pth:
                    t = O._tile_decode(pth)
                    if t.shape[2] == 3:
                        t = np.dstack([t, np.full(t.shape[:2], 255, np.uint8)])
                    canvas[(ty - ty0) * T:(ty - ty0) * T + t.shape[0], (tx - tx0) * T:(tx - tx0) * T + t.shape[1]] = t
        # sample canvas for each output pixel
        xs = bbox[0] + (np.arange(w) + .5) * (bbox[2] - bbox[0]) / w
        ys = bbox[3] - (np.arange(h) + .5) * (bbox[3] - bbox[1]) / h
        cc = np.floor((xs - g["x0"]) / g["res"] / s - tx0 * T).astype(int)
        rr = np.floor((g["y0"] - ys) / g["res"] / s - ty0 * T).astype(int)
        inside = (rr[:, None] >= 0) & (rr[:, None] < canvas.shape[0]) & (cc[None, :] >= 0) & (cc[None, :] < canvas.shape[1])
        out = canvas[np.clip(rr, 0, canvas.shape[0] - 1)[:, None], np.clip(cc, 0, canvas.shape[1] - 1)[None, :]]
        out[~inside] = 0
        return Response(R.png_rgba(out), media_type="image/png", headers={"Cache-Control": "private, max-age=60"})
    return await run(work)


async def cloud_edit_info(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(CE.info, app, pid))


def _ce_call(fn, *a):
    from ..services import cloudedit as CE
    try:
        return fn(*a)
    except CE.CloudEditError as e:
        raise ApiError(e.status, str(e), e.code)


async def cloud_edit(request: Request):
    """Body: raw little-endian uint32 point indices (into the display sample, in the order /pointcloud/data returned them). ?cls=<ASPRS class>."""
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    try:
        cls = int(request.query_params.get("cls", ""))
    except ValueError:
        raise ApiError(400, "cls must be an integer class code", "bad_class")
    raw = await request.body()
    if len(raw) % 4 or len(raw) > 4 * 6_000_000:
        raise ApiError(400, "body must be uint32 indices (at most 6 million)", "bad_body")
    idx = np.frombuffer(raw, dtype="<u4")
    return ok(await run(_ce_call, CE.edit, app, pid, idx, cls))


async def cloud_edit_undo(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_ce_call, CE.undo, app, pid))


async def cloud_edit_redo(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_ce_call, CE.redo, app, pid))


async def cloud_edit_reset(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(CE.reset, app, pid))


async def heightmap(request: Request):
    """Decimated float32 grid (row-major, NaN = no data) for the 3D viewer. Header X-Grid describes it."""
    app = ctx_of(request)
    pid, layer = request.path_params["pid"], request.path_params["layer"]
    await run(project_or_404, app, pid)
    if layer not in ("dsm", "dtm", "ndsm", "odm_dsm", "odm_dtm"):
        raise ApiError(404, "unknown layer", "not_found")
    maxdim = max(32, min(int(request.query_params.get("maxdim", 400)), 1200))

    def work():
        p = _prod(app, pid, layer)
        arr, grid, nodata = R.RASTERS.get(p["path"])
        s = max(1, int(np.ceil(max(arr.shape) / maxdim)))
        sub = arr[::s, ::s].astype("<f4").copy()
        if nodata is not None and not np.isnan(nodata):
            sub[sub == nodata] = np.nan
        kind = R.KIND[layer]
        meta = {"x0": grid.x0, "y0": grid.y0, "res": grid.res * s, "width": int(sub.shape[1]), "height": int(sub.shape[0]), "stride": s,
                "native_res": grid.res, "kind": kind, "label": R.KIND_TEXT[kind]}
        return Response(sub.tobytes(), media_type="application/octet-stream", headers={"X-Grid": json.dumps(meta), "Cache-Control": "private, max-age=60",
                                                                                         "Access-Control-Expose-Headers": "X-Grid"})
    return await run(work)


async def inspect(request: Request):
    """Everything known at one map coordinate: observed surface, estimated ground, evidence, confidence, land cover."""
    app = ctx_of(request)
    pid = request.path_params["pid"]
    proj = await run(project_or_404, app, pid)
    try:
        x, y = float(request.query_params["x"]), float(request.query_params["y"])
    except (KeyError, ValueError):
        raise ApiError(400, "x and y (project CRS) are required", "bad_request")

    def work():
        out: Dict[str, object] = {"x": x, "y": y, "epsg": proj["epsg"], "vertical_units": proj["vertical_units"]}
        for layer in ("dsm", "dtm", "ndsm"):
            p = P.get(app, pid, layer)
            out[layer] = R.sample_point(p["path"], x, y) if p and p["exists"] else None
        ev = conf = src = None
        pe, pc, ps = P.get(app, pid, "evidence"), P.get(app, pid, "confidence"), P.get(app, pid, "source")
        if pe and pe["exists"]:
            ev = R.sample_point(pe["path"], x, y, False)
        if pc and pc["exists"]:
            conf = R.sample_point(pc["path"], x, y, False)
        if ps and ps["exists"]:
            src = R.sample_point(ps["path"], x, y, False)
        out["evidence"] = {0: "none", 1: "measured ground", 2: "interpolated", 3: "external model"}.get(int(ev)) if ev is not None else None
        out["confidence"] = D.CONF_NAMES.get(int(conf)) if conf is not None else None
        sname = None
        if src is not None:
            if int(src) == 1:
                sname = "Photogrammetry ground points"
            else:
                r = app.db.one("SELECT name FROM elevation_sources WHERE project_id=? AND id=?", (pid, int(src)))
                sname = r["name"] if r else None
        out["source"] = sname
        if out["dtm"] is not None:
            out["note"] = ("Ground elevation here was MEASURED." if ev == 1 else
                           "Ground elevation here is an ESTIMATE (" + str(out["evidence"]) + "), not directly observed.")
        # land cover at this location
        prod = P.get(app, pid, "orthophoto")
        if prod and prod["exists"] and prod["meta"].get("grid"):
            g = prod["meta"]["grid"]
            px, py = int((x - g["x0"]) / g["res"]), int((g["y0"] - y) / g["res"])
            if 0 <= px < g["width"] and 0 <= py < g["height"]:
                from ..core.tiling import Window
                mask, _ = O.class_rasters(app, pid)
                v = int(mask.read(Window(px, py, px + 1, py + 1))[0, 0]) & 0x3F
                cls = next((c for c in loads(proj["classes_json"], []) if c["id"] == v), None)
                out["land_cover"] = cls["name"] if cls else None
                out["ortho_pixel"] = [px, py]
        return out
    return ok(await run(work))


async def sample(request: Request):
    """Batch elevation sampling for measurement tools: values of dsm/dtm at many points."""
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, PointList)

    def work():
        res = {}
        for layer in ("dsm", "dtm"):
            p = P.get(app, pid, layer)
            res[layer] = [R.sample_point(p["path"], x, y) for x, y in b.points] if p and p["exists"] else None
        return {"points": b.points, **res}
    return ok(await run(work))


# ----------------------------------------------------------------------------- planning
class ProfileIn(BaseModel):
    line: List[List[float]] = Field(min_length=2, max_length=2000)
    spacing: Optional[float] = Field(None, gt=0)
    format: Literal["json", "csv"] = "json"

    @field_validator("line")
    @classmethod
    def _v(cls, v):
        return [list(_pt(p)) for p in v]


def _dem(app, pid):
    d = _prod(app, pid, "dtm")
    arr, grid, _ = R.RASTERS.get(d["path"])
    c = _prod(app, pid, "confidence", False)
    s = _prod(app, pid, "dsm", False)
    conf = R.RASTERS.get(c["path"])[0] if c else None
    dsm = R.RASTERS.get(s["path"])[0] if s else None
    return arr, grid, conf, dsm


async def profile(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, ProfileIn)

    def work():
        dem, grid, conf, dsm = _dem(app, pid)
        prof = PL.elevation_profile(dem, grid, [tuple(p) for p in b.line], b.spacing, conf, dsm)
        prof["disclaimer"] = DISCLAIMER
        return prof
    prof = await run(work)
    cols = _profile_columns(prof)
    if b.format == "csv":
        buf = io.StringIO()
        buf.write("# " + DISCLAIMER + "\n")
        buf.write("# ground_z is an ESTIMATE of bare earth; surface_z (if present) is the OBSERVED surface (canopy/structure tops). Units: project units.\n")
        w = csv.writer(buf)
        w.writerow(list(cols))
        for row in zip(*cols.values()):
            w.writerow(["" if v is None else v for v in row])
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="elevation_profile.csv"'})
    scalars = {k: v for k, v in prof.items() if not isinstance(v, np.ndarray)}
    return ok({**scalars, "columns": cols, "samples": len(next(iter(cols.values()))), "disclaimer": DISCLAIMER})


def _clean(a) -> list:
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in np.asarray(a).tolist()]


def _profile_columns(prof: dict) -> dict:
    cols = {"distance": _clean(np.round(prof["distance_m"], 4)), "x": _clean(np.round(prof["x"], 4)), "y": _clean(np.round(prof["y"], 4)),
            "ground_z": _clean(np.round(prof["ground_z"], 4))}
    if "surface_z" in prof:
        cols["surface_z"] = _clean(np.round(prof["surface_z"], 4))
        cols["canopy_height"] = _clean(np.round(prof["canopy_height"], 4))
    if "confidence" in prof:
        cols["confidence"] = [D.CONF_NAMES[int(c)] for c in prof["confidence"]]
    return cols


class FootprintIn(BaseModel):
    polygon: List[List[float]] = Field(min_length=3, max_length=5000)
    steep_deg: float = Field(15.0, gt=0, le=90)

    @field_validator("polygon")
    @classmethod
    def _v(cls, v):
        return [list(_pt(p)) for p in v]


async def footprint_report(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, FootprintIn)

    def work():
        dem, grid, conf, dsm = _dem(app, pid)
        rep = PL.footprint_report(dem, grid, [tuple(p) for p in b.polygon], conf, dsm, b.steep_deg)
        rep["area_m2"] = rep.get("area_map_units2")
        rep["footprint_area_note"] = "Plan-view (horizontal) area in squared project horizontal units."
        return rep
    return ok(await run(work))


class SurfaceIn(BaseModel):
    type: Literal["flat", "plane", "three_points"] = "flat"
    z: Optional[float] = None
    x_ref: float = 0.0
    y_ref: float = 0.0
    dzdx: float = 0.0
    dzdy: float = 0.0
    points: Optional[List[List[float]]] = None


class CutFillIn(BaseModel):
    polygon: List[List[float]] = Field(min_length=3, max_length=5000)
    surface: SurfaceIn

    @field_validator("polygon")
    @classmethod
    def _v(cls, v):
        return [list(_pt(p)) for p in v]


async def cut_fill(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, CutFillIn)
    s = b.surface
    try:
        if s.type == "flat":
            if s.z is None:
                raise ValueError("flat surface needs z (proposed finished elevation)")
            surf = PL.ProposedSurface.flat(s.z)
        elif s.type == "plane":
            if s.z is None:
                raise ValueError("plane needs z at the reference point")
            surf = PL.ProposedSurface(s.z, s.x_ref, s.y_ref, s.dzdx, s.dzdy)
        else:
            if not s.points or len(s.points) != 3 or any(len(p) != 3 for p in s.points):
                raise ValueError("three_points needs three [x, y, z] points")
            surf = PL.ProposedSurface.from_three_points(*s.points)
    except ValueError as e:
        raise ApiError(422, str(e), "validation_error")

    def work():
        dem, grid, conf, _ = _dem(app, pid)
        summ = P.get(app, pid, "terrain_summary")
        cv = (summ["meta"].get("cross_validation") or {}) if summ else {}
        res = PL.cut_fill(dem, grid, [tuple(p) for p in b.polygon], surf, conf, error_rmse_m=cv.get("overall_rmse_m"))
        res["vertical_units"] = app.db.one("SELECT vertical_units FROM projects WHERE id=?", (pid,))["vertical_units"]
        return res
    return ok(await run(work))


_drain_cache: Dict[tuple, dict] = {}


async def drainage(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    min_area = float(request.query_params.get("min_area", 500))

    def work():
        p = _prod(app, pid, "dtm")
        key = (str(p["path"]), p["path"].stat().st_mtime_ns, min_area)
        if key not in _drain_cache:
            dem, grid, _ = R.RASTERS.get(p["path"])
            if dem.size > 25_000_000:
                raise ApiError(413, "terrain is too large for interactive drainage analysis; use a coarser terrain resolution", "too_large")
            a = TA.analyse_drainage(dem, grid, min_area)
            polys = mask_to_polygons(a["depressions"]["mask"], grid, min_cells=2)
            _drain_cache.clear()
            _drain_cache[key] = {
                "channels": {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "properties": {"upstream_area_m2": s["upstream_area_m2"]}, "geometry": {"type": "LineString", "coordinates": s["coords"]}}
                    for s in a["segments"]]},
                "depressions": {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": g} for g in polys]},
                "depression_summary": {k: v for k, v in a["depressions"].items() if k != "mask"},
                "note": "Potential flow paths computed from the ESTIMATED terrain (D8, depressions filled). Under dense canopy these are indicative only."}
        return _drain_cache[key]
    return ok(await run(work))


async def steep(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    thr = float(request.query_params.get("degrees", 15))

    def work():
        p = _prod(app, pid, "dtm")
        dem, grid, _ = R.RASTERS.get(p["path"])
        sl, _ = TA.slope_aspect(dem, grid.res)
        m = np.nan_to_num(sl, nan=0) > thr
        polys = mask_to_polygons(m, grid, min_cells=max(4, int(25 / grid.cell_area)), simplify_cells=1.0)
        return {"threshold_degrees": thr, "area_m2": float(m.sum() * grid.cell_area), "fraction": float(m.sum() / np.isfinite(dem).sum()) if np.isfinite(dem).any() else 0,
                "polygons": {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"slope_over_deg": thr}, "geometry": g} for g in polys]},
                "note": "Slopes from the ESTIMATED bare-earth surface. " + DISCLAIMER}
    return ok(await run(work))


# ----------------------------------------------------------------------------- features (boundary, footprints, profiles, measurements)
class FeatureIn(BaseModel):
    kind: Literal["boundary", "footprint", "profile", "measurement", "note"]
    name: Optional[str] = Field(None, max_length=200)
    geometry: dict
    props: dict = Field(default_factory=dict)

    @field_validator("geometry")
    @classmethod
    def _g(cls, g):
        if g.get("type") not in ("Point", "LineString", "Polygon", "MultiPolygon"):
            raise ValueError("geometry must be a GeoJSON Point, LineString, Polygon or MultiPolygon")
        if "coordinates" not in g:
            raise ValueError("geometry needs coordinates")
        return g


def _feat(r) -> dict:
    return {"id": r["id"], "kind": r["kind"], "name": r["name"], "geometry": loads(r["geometry_json"]), "props": loads(r["props_json"], {}),
            "updated_at": r["updated_at"]}


async def features_list(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    k = request.query_params.get("kind")
    rows = app.db.query("SELECT * FROM features WHERE project_id=?" + (" AND kind=?" if k else "") + " ORDER BY created_at",
                        (pid, k) if k else (pid,))
    return ok([_feat(r) for r in rows])


async def features_create(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, FeatureIn)
    fid, t = new_id(), now()
    await run(app.db.insert, "features", {"id": fid, "project_id": pid, "kind": b.kind, "name": b.name, "geometry_json": dumps(b.geometry),
                                          "props_json": dumps(b.props), "created_at": t, "updated_at": t})
    return ok(_feat(app.db.one("SELECT * FROM features WHERE id=?", (fid,))), 201)


async def features_update(request: Request):
    app = ctx_of(request)
    pid, fid = request.path_params["pid"], request.path_params["fid"]
    await run(project_or_404, app, pid)
    b = await body(request, FeatureIn)
    n = await run(app.db.update, "features", {"id": fid, "project_id": pid},
                  {"kind": b.kind, "name": b.name, "geometry_json": dumps(b.geometry), "props_json": dumps(b.props), "updated_at": now()})
    if not n:
        raise ApiError(404, "feature not found", "not_found")
    return ok(_feat(app.db.one("SELECT * FROM features WHERE id=?", (fid,))))


async def features_delete(request: Request):
    app = ctx_of(request)
    pid, fid = request.path_params["pid"], request.path_params["fid"]
    await run(project_or_404, app, pid)
    await run(app.db.execute, "DELETE FROM features WHERE id=? AND project_id=?", (fid, pid))
    return ok()


# ----------------------------------------------------------------------------- point cloud feed
def _cloud_file(app, pid):
    w = app.storage.path(pid, "terrain")
    for n, kind in (("viewer_cloud.npz", "classified"), ("viewer_cloud_raw.npz", "raw")):
        if (w / n).exists():
            return w / n, kind
    return None, None


async def cloud_meta(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        f, kind = _cloud_file(app, pid)
        has_cloud = bool((P.get(app, pid, "pointcloud") or {}).get("exists"))
        if not f:
            return {"available": False, "has_pointcloud": has_cloud,
                    "hint": "Build a display sample of the point cloud first." if has_cloud else "No point cloud yet; run photogrammetry."}
        with np.load(f, allow_pickle=False) as z:
            n = int(len(z["x"]))
            from ..services import cloudedit as CE
            cls = np.bincount(CE.apply_overrides(app, pid, z["cls"]), minlength=32)
            return {"available": True, "kind": kind, "points": n, "total_points": int(z["total"]), "subsampled": n < int(z["total"]),
                    "origin": [float(v) for v in z["origin"]], "has_rgb": "rgb" in z.files, "epsg": int(z["epsg"]) or None,
                    "classes": {str(i): int(c) for i, c in enumerate(cls) if c},
                    "class_names": {"0": "never classified", "1": "unclassified", "2": "ground", "3": "low vegetation", "4": "medium vegetation",
                                    "5": "high vegetation", "6": "building", "7": "noise"},
                    "note": ("Classified: ground via SMRF/PMF; vegetation by height above the ESTIMATED ground." if kind == "classified"
                             else "Raw cloud, not ground-classified yet. Run terrain reconstruction.")}
    return ok(await run(work))


async def cloud_data(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    maxp = max(1000, min(int(request.query_params.get("max", 1_500_000)), 5_000_000))

    def work():
        f, kind = _cloud_file(app, pid)
        if not f:
            raise ApiError(404, "no display point cloud yet", "no_cloud")
        with np.load(f, allow_pickle=False) as z:
            n = len(z["x"])
            idx = slice(None) if n <= maxp else np.sort(np.random.default_rng(1).choice(n, maxp, replace=False))
            from ..services import cloudedit as CE
            full_cls = CE.apply_overrides(app, pid, z["cls"])
            x, y, zz = (np.ascontiguousarray(z[k][idx]) for k in ("x", "y", "z"))
            c = np.ascontiguousarray(full_cls[idx])
            m = len(x)
            pad = (-m) % 4
            parts = [x.astype("<f4").tobytes(), y.astype("<f4").tobytes(), zz.astype("<f4").tobytes(), c.tobytes() + b"\x00" * pad]
            if "rgb" in z.files:
                parts.append(np.ascontiguousarray(z["rgb"][idx]).tobytes())
            meta = {"n": int(m), "origin": [float(v) for v in z["origin"]], "has_rgb": "rgb" in z.files, "kind": kind, "complete": bool(m == n)}
        return Response(b"".join(parts), media_type="application/octet-stream",
                        headers={"X-Meta": json.dumps(meta), "Access-Control-Expose-Headers": "X-Meta", "Cache-Control": "private, max-age=60"})
    return await run(work)


async def cloud_edit_info(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(CE.info, app, pid))


def _ce_call(fn, *a):
    from ..services import cloudedit as CE
    try:
        return fn(*a)
    except CE.CloudEditError as e:
        raise ApiError(e.status, str(e), e.code)


async def cloud_edit(request: Request):
    """Body: raw little-endian uint32 point indices (into the display sample, in the order /pointcloud/data returned them). ?cls=<ASPRS class>."""
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    try:
        cls = int(request.query_params.get("cls", ""))
    except ValueError:
        raise ApiError(400, "cls must be an integer class code", "bad_class")
    raw = await request.body()
    if len(raw) % 4 or len(raw) > 4 * 6_000_000:
        raise ApiError(400, "body must be uint32 indices (at most 6 million)", "bad_body")
    idx = np.frombuffer(raw, dtype="<u4")
    return ok(await run(_ce_call, CE.edit, app, pid, idx, cls))


async def cloud_edit_undo(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_ce_call, CE.undo, app, pid))


async def cloud_edit_redo(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_ce_call, CE.redo, app, pid))


async def cloud_edit_reset(request: Request):
    from ..services import cloudedit as CE
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(CE.reset, app, pid))


async def cloud_build(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        if not (P.get(app, pid, "pointcloud") or {}).get("exists"):
            raise ApiError(409, "No point cloud yet", "no_pointcloud")
        return app.queue.enqueue("viewer_cloud", pid, {"project_id": pid}, dedupe_key=f"vc:{pid}")
    return ok({"job_id": await run(work)}, 202)


routes = [
    Route("/projects/{pid}/products", products),
    Route("/projects/{pid}/terrain", terrain_state),
    Route("/projects/{pid}/terrain/run", terrain_run, methods=["POST"]),
    Route("/projects/{pid}/elevation-sources/{sid:int}", source_patch, methods=["PATCH"]),
    Route("/projects/{pid}/elevation-sources/{sid:int}", source_delete, methods=["DELETE"]),
    Route("/projects/{pid}/terrain/render/{layer}", render),
    Route("/projects/{pid}/terrain/stats/{layer}", layer_stats),
    Route("/projects/{pid}/terrain/heightmap/{layer}", heightmap),
    Route("/projects/{pid}/terrain/inspect", inspect),
    Route("/projects/{pid}/terrain/sample", sample, methods=["POST"]),
    Route("/projects/{pid}/ortho/render", ortho_render),
    Route("/projects/{pid}/planning/profile", profile, methods=["POST"]),
    Route("/projects/{pid}/planning/footprint", footprint_report, methods=["POST"]),
    Route("/projects/{pid}/planning/cut-fill", cut_fill, methods=["POST"]),
    Route("/projects/{pid}/planning/drainage", drainage),
    Route("/projects/{pid}/planning/steep", steep),
    Route("/projects/{pid}/features", features_list),
    Route("/projects/{pid}/features", features_create, methods=["POST"]),
    Route("/projects/{pid}/features/{fid}", features_update, methods=["PUT"]),
    Route("/projects/{pid}/features/{fid}", features_delete, methods=["DELETE"]),
    Route("/projects/{pid}/pointcloud", cloud_meta),
    Route("/projects/{pid}/pointcloud/data", cloud_data),
    Route("/projects/{pid}/pointcloud/build", cloud_build, methods=["POST"]),
    Route("/projects/{pid}/pointcloud/edit", cloud_edit_info),
    Route("/projects/{pid}/pointcloud/edit", cloud_edit, methods=["POST"]),
    Route("/projects/{pid}/pointcloud/edit/undo", cloud_edit_undo, methods=["POST"]),
    Route("/projects/{pid}/pointcloud/edit/redo", cloud_edit_redo, methods=["POST"]),
    Route("/projects/{pid}/pointcloud/edit/reset", cloud_edit_reset, methods=["POST"]),
]
