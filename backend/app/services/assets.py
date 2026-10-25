"""Import of non-image uploads: orthomosaic, LiDAR, DEM, survey points, GCP list. Runs in a worker."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from ..context import AppContext
from ..core.uploadstore import UploadError
from ..db import dumps, loads, now
from ..jobs import JobContext, JobFailure
from . import geotiff as gt
from . import las as L
from . import products as P
from .terrain_pipeline import TerrainError, load_survey_csv


def next_source_id(app: AppContext, pid: str) -> int:
    used = {r["id"] for r in app.db.query("SELECT id FROM elevation_sources WHERE project_id=?", (pid,))}
    for i in range(2, 256):
        if i not in used:
            return i
    raise JobFailure("a project can hold at most 254 supplemental elevation sources", "too_many_sources")


def _set(app, uid, status, error=None, **kw):
    app.db.update("upload_files", {"upload_id": uid}, {"status": status, "error": error, "updated_at": now(), **kw})


def validate_gcp_list(text: str) -> dict:
    """ODM gcp_list.txt: first line CRS (EPSG:xxxx / proj string), then ``x y z im_x im_y image_name [label]`` per observation."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 2:
        raise JobFailure("gcp_list.txt needs a CRS line followed by observation lines", "bad_gcp")
    crs = lines[0]
    if not (crs.upper().startswith("EPSG:") or crs.startswith("+proj") or crs.upper() == "WGS84"):
        raise JobFailure(f"first line must be the CRS (e.g. 'EPSG:32616' or '+proj=...'), got {crs[:40]!r}", "bad_gcp")
    obs, names = 0, set()
    for n, l in enumerate(lines[1:], 2):
        parts = l.split()
        if len(parts) < 6:
            raise JobFailure(f"line {n}: expected 'x y z im_x im_y image_name', got {len(parts)} columns", "bad_gcp")
        try:
            [float(v) for v in parts[:5]]
        except ValueError:
            raise JobFailure(f"line {n}: coordinates must be numeric", "bad_gcp")
        obs += 1
        names.add(parts[5])
    if obs < 3:
        raise JobFailure("at least 3 GCP observations are needed (5+ recommended, seen in 3+ images each)", "bad_gcp")
    return {"crs": crs, "observations": obs, "images": len(names)}


def import_asset(app: AppContext, ctx: JobContext) -> dict:
    p = ctx.params
    uid, pid, purpose = p["upload_id"], p["project_id"], p["purpose"]
    row = app.db.one("SELECT * FROM upload_files WHERE upload_id=?", (uid,))
    if row is None:
        raise JobFailure("upload record not found", "not_found")
    if row["status"] in ("complete", "rejected"):
        return {"status": row["status"]}
    proj = app.db.one("SELECT epsg FROM projects WHERE id=?", (pid,))
    name = row["filename"]
    sub = {"orthophoto": "source", "dem": "source/dem", "lidar": "source/lidar", "survey_points": "source/survey", "gcp_list": "odm"}[purpose]
    dest_rel = f"{sub}/{uid[:8]}_{name}" if purpose != "gcp_list" else "odm/gcp_list.txt"
    dest = app.storage.path(pid, dest_rel)
    try:
        info = app.uploads.assemble(uid, dest)
    except UploadError as e:
        _set(app, uid, "failed", str(e))
        raise JobFailure(str(e), e.code)
    ctx.log(f"assembled {name} ({info['size']} bytes) sha256={info['sha256'][:12]}")

    def reject(msg: str, code: str):
        rej = app.storage.ensure(pid, "rejected")
        os.replace(dest, rej / f"{uid}_{name}")
        app.uploads.discard(uid)
        _set(app, uid, "rejected", msg)
        ctx.log(f"rejected: {msg} (kept in rejected/)", "error")
        return {"status": "rejected", "error": msg, "code": code}

    try:
        if purpose == "orthophoto":
            gi = gt.read_georef(dest)
            if gi.bands < 3:
                return reject(f"needs at least 3 bands (RGB); file has {gi.bands}", "bad_ortho")
            if proj["epsg"] and gi.epsg and gi.epsg != proj["epsg"]:
                return reject(f"orthomosaic is EPSG:{gi.epsg} but the project CRS is EPSG:{proj['epsg']}; reproject first", "crs_mismatch")
            jid = app.queue.enqueue("prepare_ortho", pid, {"project_id": pid, "rel_path": dest_rel})
            meta = {"queued_job": jid}
        elif purpose == "dem":
            gi = gt.read_georef(dest)
            if not gi.georeferenced:
                return reject("DEM has no georeferencing (no CRS / geotransform); it cannot be placed on the project grid", "no_georef")
            if gi.bands != 1:
                return reject(f"DEM must be single-band; file has {gi.bands} bands", "bad_dem")
            if proj["epsg"] and gi.epsg != proj["epsg"]:
                return reject(f"DEM is EPSG:{gi.epsg} but the project CRS is EPSG:{proj['epsg']}; reproject it first "
                              f"(gdalwarp -t_srs EPSG:{proj['epsg']})", "crs_mismatch")
            sid = next_source_id(app, pid)
            meta = {"epsg": gi.epsg, "width": gi.width, "height": gi.height, "res": gi.grid.res if gi.grid else None,
                    "vertical_units": gi.vertical_units, "nodata": gi.nodata,
                    "vertical_datum_note": "Vertical datum is not verified. Set a z offset if this DEM uses a different datum than the photogrammetry."}
            app.db.insert("elevation_sources", {"id": sid, "project_id": pid, "name": name, "kind": "external_dem", "rel_path": dest_rel,
                                                "priority": 20, "trust": "moderate", "meta_json": dumps(meta), "created_at": now()})
            meta["source_id"] = sid
        elif purpose == "lidar":
            h = L.read_header(dest)
            warn = None
            if h.epsg and proj["epsg"] and h.epsg != proj["epsg"]:
                return reject(f"LiDAR is EPSG:{h.epsg} but the project CRS is EPSG:{proj['epsg']}; reproject it first "
                              "(pdal translate --filters.reprojection)", "crs_mismatch")
            if not h.epsg:
                warn = "LiDAR file has no CRS; it is assumed to be in the project CRS. Verify this."
                ctx.log(warn, "warning")
            sid = next_source_id(app, pid)
            meta = {"points": h.count, "bounds": [*h.mins[:2], *h.maxs[:2]], "z_range": [h.mins[2], h.maxs[2]], "epsg": h.epsg,
                    "compressed": h.compressed, "point_format": h.point_format, "warning": warn}
            app.db.insert("elevation_sources", {"id": sid, "project_id": pid, "name": name, "kind": "lidar", "rel_path": dest_rel,
                                                "priority": 30, "trust": "high", "meta_json": dumps(meta), "created_at": now()})
            meta["source_id"] = sid
        elif purpose == "survey_points":
            try:
                x, y, z = load_survey_csv(dest)
            except TerrainError as e:
                return reject(str(e), "bad_survey")
            sid = next_source_id(app, pid)
            meta = {"points": int(len(x)), "bounds": [float(x.min()), float(y.min()), float(x.max()), float(y.max())],
                    "z_range": [float(z.min()), float(z.max())]}
            app.db.insert("elevation_sources", {"id": sid, "project_id": pid, "name": name, "kind": "survey", "rel_path": dest_rel,
                                                "priority": 40, "trust": "high", "meta_json": dumps(meta), "created_at": now()})
            meta["source_id"] = sid
        else:   # gcp_list
            try:
                meta = validate_gcp_list(dest.read_text())
            except JobFailure as e:
                return reject(str(e), e.code)
            P.register(app, pid, "gcp_list", dest_rel, "text", meta)
    except (gt.GeoTiffError, L.LasError) as e:
        return reject(f"unreadable {purpose}: {e}", "bad_file")
    app.uploads.discard(uid)
    _set(app, uid, "complete", None)
    ctx.log(f"imported {purpose}: {meta}")
    return {"status": "complete", "purpose": purpose, **meta}
