"""Terrain reconstruction job: photogrammetry cloud (+ optional LiDAR / survey / external DEM) -> DSM, DTM, confidence, evidence.

Separation of concerns (critical):
* ELEVATION VALUES come only from point/raster evidence (``terrain_pipeline``).
* The visual land-cover mask is used ONLY as a safety check that demotes suspicious "ground" under mapped canopy
  (and to label vegetation/buildings in the output cloud). Painting trees as soil never creates ground evidence.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from ..context import AppContext
from ..core import dtm as D
from ..core.dtm import GridSpec
from ..core.planning import bilinear_sample
from ..core.tiling import Window
from ..db import dumps, loads, now
from ..jobs import JobContext, JobFailure
from . import classify as K
from . import geotiff as gt
from . import las as L
from . import ortho as O
from . import products as P
from . import groundfilter as G
from . import terrain_pipeline as T

OUTPUTS = {"dsm": "dsm", "dtm": "dtm", "ndsm": "ndsm", "confidence": "confidence", "evidence": "evidence", "source": "source"}


def _dc(cls, d: Optional[dict], what: str):
    d = d or {}
    names = {f.name for f in dataclasses.fields(cls)}
    bad = sorted(set(d) - names)
    if bad:
        raise JobFailure(f"unknown {what} setting(s): {', '.join(bad)}", "bad_request")
    return cls(**d)


def build_params(p: dict) -> T.TerrainParams:
    ground = _dc(G.GroundFilterParams, p.get("ground"), "ground-filter")
    interp = _dc(D.InterpParams, p.get("interp"), "interpolation")
    conf = _dc(D.ConfidenceParams, p.get("confidence"), "confidence")
    try:
        ground.validate()
    except ValueError as e:
        raise JobFailure(str(e), "bad_request")
    if interp.method not in ("local_plane", "idw", "tin", "tps", "nearest"):
        raise JobFailure(f"interpolation method must be one of local_plane, idw, tin, tps, nearest", "bad_request")
    tp = T.TerrainParams(resolution_m=float(p.get("resolution_m", 0.5)), ground=ground, interp=interp, confidence=conf,
                         canopy_clearance_m=float(p.get("canopy_clearance_m", 1.0)), stat=p.get("stat", "median"),
                         photo_priority=int(p.get("photo_priority", 10)))
    if tp.resolution_m <= 0:
        raise JobFailure("resolution_m must be positive", "bad_request")
    return tp


def mask_to_grid(app: AppContext, pid: str, grid: GridSpec, class_ids: set, block_rows: int = 256) -> Optional[np.ndarray]:
    """Boolean array on ``grid``: True where the classification mask (any provenance) holds one of ``class_ids``.
    None if there is no georeferenced orthomosaic/mask to sample."""
    prod = P.get(app, pid, "orthophoto")
    if not prod or not prod["exists"] or not prod["meta"].get("georeferenced") or not prod["meta"].get("grid"):
        return None
    og = prod["meta"]["grid"]
    mask, _ = O.class_rasters(app, pid)
    out = np.zeros(grid.shape, bool)
    ids = np.array(sorted(class_ids), np.uint8)
    for r0 in range(0, grid.height, block_rows):
        r1 = min(r0 + block_rows, grid.height)
        cols = np.arange(grid.width)
        rows = np.arange(r0, r1)
        cx, cy = grid.cell_centers_xy(rows[:, None] + 0 * cols[None, :], cols[None, :] + 0 * rows[:, None])
        oc = np.floor((cx - og["x0"]) / og["res"]).astype(np.int64)
        orr = np.floor((og["y0"] - cy) / og["res"]).astype(np.int64)
        inside = (oc >= 0) & (oc < og["width"]) & (orr >= 0) & (orr < og["height"])
        if not inside.any():
            continue
        win = Window(int(oc[inside].min()), int(orr[inside].min()), int(oc[inside].max()) + 1, int(orr[inside].max()) + 1)
        sub = mask.read(win) & K.CLASS_MASK
        hit = np.zeros(oc.shape, bool)
        hit[inside] = np.isin(sub[orr[inside] - win.y0, oc[inside] - win.x0], ids)
        out[r0:r1] = hit
    return out


def _class_ids(app: AppContext, pid: str, vegetation: bool) -> set:
    cl = loads(app.db.one("SELECT classes_json FROM projects WHERE id=?", (pid,))["classes_json"], [])
    if vegetation:
        return {int(c["id"]) for c in cl if c.get("vegetation")}
    return {int(c["id"]) for c in cl if c.get("key") == "building"}


def _has_pixels(app, pid, ids: set) -> bool:
    try:
        mask, _ = O.class_rasters(app, pid)
    except Exception:
        return False
    h = mask.histogram()
    return any((v & K.CLASS_MASK) in ids and n > 0 for v, n in h.items())


def checkpoint_residuals(app: AppContext, pid: str, dtm: np.ndarray, grid: GridSpec, conf: np.ndarray, evidence: np.ndarray) -> dict:
    rows = app.db.query("SELECT id,label,x,y,z FROM control_points WHERE project_id=? AND kind='check' AND z IS NOT NULL", (pid,))
    if not rows:
        return {"n": 0, "note": "No check points with elevations were provided, so vertical accuracy is NOT independently validated."}
    xs, ys = np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows])
    zs = np.array([r["z"] for r in rows])
    est = bilinear_sample(dtm, grid, xs, ys)
    rr, cc = np.clip(np.floor(grid.xy_to_rc(xs, ys)[0]).astype(int), 0, grid.height - 1), np.clip(np.floor(grid.xy_to_rc(xs, ys)[1]).astype(int), 0, grid.width - 1)
    names = {0: "unknown", 1: "low", 2: "moderate", 3: "high"}
    ev = {0: "none", 1: "measured", 2: "interpolated", 3: "external"}
    res, used = [], []
    with app.db.tx() as c:
        for r, e, i, j in zip(rows, est, rr, cc):
            if np.isnan(e):
                c.execute("UPDATE control_points SET residual_json=? WHERE id=?", (dumps({"status": "outside_or_no_estimate"}), r["id"]))
                continue
            d = float(e - r["z"])
            used.append(d)
            info = {"status": "ok", "dtm_z": float(e), "dz": d, "confidence": names[int(conf[i, j])], "evidence": ev[int(evidence[i, j])]}
            res.append(info)
            c.execute("UPDATE control_points SET residual_json=? WHERE id=?", (dumps(info), r["id"]))
    if not used:
        return {"n": 0, "checked": len(rows), "note": "None of the check points fall on the estimated terrain."}
    a = np.array(used)
    return {"n": int(len(a)), "checked": len(rows), "mean_dz": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())), "max_abs": float(np.abs(a).max()),
            "note": "dz = estimated terrain minus check-point elevation (check points should lie on bare ground)."}


def build_viewer_cloud(src: Path, dst: Path, target: int = 1_500_000, seed: int = 3) -> dict:
    """Uniform random subsample (for display only) stored as compact arrays relative to a local origin."""
    h = L.read_header(src)
    p = min(1.0, target / max(h.count, 1))
    rng = np.random.default_rng(seed)
    xs, ys, zs, cs, rgbs = [], [], [], [], []
    has_rgb = False
    for ch in L.iter_points(src):
        keep = rng.random(len(ch["x"])) < p if p < 1 else np.ones(len(ch["x"]), bool)
        xs.append(ch["x"][keep]); ys.append(ch["y"][keep]); zs.append(ch["z"][keep]); cs.append(ch["classification"][keep])
        if "rgb" in ch:
            has_rgb = True
            rgbs.append((ch["rgb"][keep] >> 8).astype(np.uint8) if ch["rgb"].max() > 255 else ch["rgb"][keep].astype(np.uint8))
    x, y, z, c = (np.concatenate(a) for a in (xs, ys, zs, cs))
    if len(x) == 0:
        raise JobFailure("point cloud has no points", "empty_cloud")
    origin = [float(h.mins[0]), float(h.mins[1]), float(h.mins[2])]
    arrays = dict(x=(x - origin[0]).astype(np.float32), y=(y - origin[1]).astype(np.float32), z=(z - origin[2]).astype(np.float32),
                  cls=c.astype(np.uint8), origin=np.array(origin), total=np.array(h.count), epsg=np.array(h.epsg or 0))
    if has_rgb and rgbs:
        arrays["rgb"] = np.concatenate(rgbs)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dst)
    return {"points": int(len(x)), "total_points": int(h.count), "subsampled": p < 1, "has_rgb": has_rgb}


def handle_terrain(app: AppContext, ctx: JobContext) -> dict:
    pid = ctx.params["project_id"]
    proj = app.db.one("SELECT epsg, vertical_epsg, horizontal_units, vertical_units FROM projects WHERE id=?", (pid,))
    cloud = P.get(app, pid, "pointcloud")
    if not cloud or not cloud["exists"]:
        raise JobFailure("No point cloud is available. Run photogrammetry (OpenDroneMap) first, or import a point cloud.", "missing_product")
    params = build_params(ctx.params.get("params") or {})
    if proj["horizontal_units"] != "metre" or proj["vertical_units"] != "metre":
        ctx.log("project units are not metres: the terrain pipeline works in metres; inputs are assumed to be metre-based", "warning")
    srcs = []
    for r in app.db.query("SELECT * FROM elevation_sources WHERE project_id=? AND enabled=1 ORDER BY id", (pid,)):
        srcs.append(T.SourceSpec(r["id"], r["name"], r["kind"], app.storage.path(pid, r["rel_path"]), r["priority"], r["z_offset"], r["trust"],
                                 loads(r["meta_json"], {})))
    if srcs:
        ctx.log("supplemental sources (priority order): " + ", ".join(f"{s.name}[{s.kind},p{s.priority},z{s.z_offset:+g}]" for s in sorted(srcs, key=lambda s: -s.priority)))
    veg_ids, bld_ids = _class_ids(app, pid, True), _class_ids(app, pid, False)
    use_cls = ctx.params.get("use_classification", True)
    veg_fn = bld_fn = None
    if use_cls and _has_pixels(app, pid, veg_ids):
        veg_fn = lambda g: mask_to_grid(app, pid, g, veg_ids)
        ctx.log("using the vegetation classification only to flag suspect 'ground' under canopy (not as elevation)")
        if _has_pixels(app, pid, bld_ids):
            bld_fn = lambda g: mask_to_grid(app, pid, g, bld_ids)
    elif use_cls:
        ctx.log("no vegetation pixels classified yet; canopy contamination of 'ground' cannot be checked", "warning")
    odm_dsm = P.get(app, pid, "odm_dsm")
    work = app.storage.ensure(pid, "terrain")
    run_dir = work / "run_tmp"
    shutil.rmtree(run_dir, ignore_errors=True)
    res = T.run_terrain(cloud["path"], run_dir, proj["epsg"], params, srcs, veg_fn, bld_fn,
                        odm_dsm["path"] if odm_dsm and odm_dsm["exists"] else None, proj["vertical_units"], proj["horizontal_units"],
                        proj["vertical_epsg"], app.settings.pdal_bin, ctx.log, ctx.progress, ctx.cancelled)
    ctx.progress(0.9, "registering products")
    # publish outputs
    prod_dir = app.storage.ensure(pid, "products")
    meta = {"grid": res.grid.to_dict(), "epsg": proj["epsg"], "vertical_units": proj["vertical_units"], "job_id": ctx.id}
    for name, key in OUTPUTS.items():
        src = Path(res.paths[key])
        dst = app.storage.path(pid, P.LAYOUT[name][0])
        os.replace(src, dst)
        side = Path(str(src) + ".meta.json")
        if side.exists():
            os.replace(side, Path(str(dst) + ".meta.json"))
    cc = app.storage.path(pid, P.LAYOUT["classified_cloud"][0])
    os.replace(res.paths["classified_cloud"], cc)
    for name in OUTPUTS:
        P.register(app, pid, name, P.LAYOUT[name][0], "raster", {**meta, "label": {
            "dsm": "Observed surface (canopy and structure tops)", "dtm": "ESTIMATED bare-earth surface",
            "ndsm": "Object height above estimated ground", "confidence": "Terrain confidence (0 unknown, 1 low, 2 moderate, 3 high)",
            "evidence": "Evidence (0 none, 1 measured ground, 2 interpolated, 3 external model)", "source": "Elevation source id"}[name]}, ctx.id)
    P.register(app, pid, "classified_cloud", P.LAYOUT["classified_cloud"][0], "pointcloud", {"labels": res.summary["point_labels"]}, ctx.id)
    # check-point residuals (independent validation, when provided)
    dtm = gt.read_full(app.storage.path(pid, P.LAYOUT["dtm"][0]))
    confr = gt.read_full(app.storage.path(pid, P.LAYOUT["confidence"][0]))
    evr = gt.read_full(app.storage.path(pid, P.LAYOUT["evidence"][0]))
    res.summary["check_points"] = checkpoint_residuals(app, pid, dtm, res.grid, confr, evr)
    res.summary["finished_at"] = now()
    (work / "summary.json").write_text(json.dumps(res.summary, indent=2, default=str))
    P.register(app, pid, "terrain_summary", "terrain/summary.json", "report", res.summary, ctx.id)
    try:
        vc = build_viewer_cloud(cc, work / "viewer_cloud.npz")
        res.summary["viewer_cloud"] = vc
    except Exception as e:                     # display aid only; never fail the terrain run over it
        ctx.log(f"viewer cloud not built: {e}", "warning")
    shutil.rmtree(run_dir, ignore_errors=True)
    ctx.progress(1.0, "done")
    return res.summary


def handle_viewer_cloud(app: AppContext, ctx: JobContext) -> dict:
    pid = ctx.params["project_id"]
    cloud = P.require(app, pid, "pointcloud")
    work = app.storage.ensure(pid, "terrain")
    src = cloud["path"]
    if src.suffix.lower() == ".laz":
        src = L.ensure_las(src, work / "tmp", app.settings.pdal_bin)
    ctx.progress(0.1, "sampling point cloud")
    r = build_viewer_cloud(src, work / "viewer_cloud_raw.npz")
    return r


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["terrain"] = handle_terrain
HANDLERS["viewer_cloud"] = handle_viewer_cloud
