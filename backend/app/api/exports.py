"""Exports (deliverables) and the photogrammetry (OpenDroneMap) controls."""
from __future__ import annotations

import json
import shutil
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Route

from ..db import loads
from ..services import exports as X
from ..services import odm as ODM
from ..services import products as P
from .deps import DISCLAIMER, ApiError, body, ctx_of, ok, project_or_404, run


class ExportIn(BaseModel):
    items: Union[Literal["all"], List[str]] = Field(default_factory=lambda: list(X.DEFAULT_ITEMS))
    laz: bool = False
    contour_interval: Optional[float] = Field(None, gt=0, le=1000, description="CAD contour interval in vertical units (default: automatic)")
    label: Optional[str] = Field(None, max_length=120)


async def exports_catalog(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    need = {"orthophoto": "orthophoto", "classification": "orthophoto", "class_stats": "orthophoto", "class_polygons": "orthophoto",
            "dsm": "dsm", "dtm": "dtm", "confidence": "confidence", "evidence": "evidence", "pointcloud": "pointcloud",
            "classified_cloud": "classified_cloud"}

    def work():
        have = {r["name"] for r in P.list_products(app, pid) if r["exists"]}
        n_img = app.db.one("SELECT COUNT(*) n FROM images WHERE project_id=?", (pid,))["n"]
        out = []
        for i in X.ITEMS:
            if i == "originals":
                avail, why = n_img > 0, "no images" if not n_img else ""
            elif i in need:
                avail, why = need[i] in have, f"needs the {need[i]} product" if need[i] not in have else ""
            elif i == "edited_cloud":
                from ..services import cloudedit as CE
                ci = CE.info(app, pid)
                avail, why = bool(ci.get("available") and ci.get("edited_points")), "" if (ci.get("available") and ci.get("edited_points")) else "no manual point edits yet"
            elif i == "image_classes":
                avail = bool(app.db.one("SELECT 1 FROM image_classification WHERE project_id=?", (pid,)))
                why = "" if avail else "classify some drone photos first"
            elif i in ("profiles", "footprints"):
                kind = "profile" if i == "profiles" else "footprint"
                avail = bool(app.db.one("SELECT 1 FROM features WHERE project_id=? AND kind=?", (pid, kind))) and "dtm" in have
                why = "" if avail else f"draw a {kind} and run the terrain job"
            elif i == "cad_dxf":
                has_feat = bool(app.db.one("SELECT 1 FROM features WHERE project_id=?", (pid,))) or bool(app.db.one("SELECT boundary_geojson b FROM projects WHERE id=?", (pid,))["b"]) \
                    or bool(app.db.one("SELECT 1 FROM control_points WHERE project_id=?", (pid,)))
                avail = "dtm" in have or has_feat
                why = "" if avail else "needs a terrain model, a boundary, drawn features or control points"
            elif i == "features":
                avail = bool(app.db.one("SELECT 1 FROM features WHERE project_id=?", (pid,))) or bool(
                    app.db.one("SELECT boundary_geojson b FROM projects WHERE id=?", (pid,))["b"])
                why = "" if avail else "nothing drawn yet"
            else:
                avail, why = True, ""
            out.append({"item": i, "available": avail, "reason": why, "default": i in X.DEFAULT_ITEMS})
        return out
    return ok({"items": await run(work), "disclaimer": DISCLAIMER})


async def exports_create(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, ExportIn)
    if b.items != "all":
        bad = [i for i in b.items if i not in X.ITEMS]
        if bad:
            raise ApiError(422, f"unknown export item(s): {', '.join(bad)}", "validation_error", {"allowed": list(X.ITEMS)})
        if not b.items:
            raise ApiError(422, "choose at least one item", "validation_error")
    jid = await run(app.queue.enqueue, "export", pid, {"project_id": pid, "items": b.items, "laz": b.laz, "label": b.label, "contour_interval": b.contour_interval})
    return ok({"job_id": jid, "export_id": jid[:16]}, 202)


async def exports_list(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(X.list_exports, app, pid))


def _manifest(app, pid, eid):
    try:
        d = X.export_dir(app, pid, eid)
    except ValueError:
        raise ApiError(404, "export not found", "not_found")
    m = d / "manifest.json"
    if not m.exists():
        raise ApiError(404, "export not found", "not_found")
    return d, json.loads(m.read_text())


async def exports_get(request: Request):
    app = ctx_of(request)
    pid, eid = request.path_params["pid"], request.path_params["eid"]
    await run(project_or_404, app, pid)
    _, m = await run(_manifest, app, pid, eid)
    return ok(m)


async def exports_file(request: Request):
    app = ctx_of(request)
    pid, eid, name = request.path_params["pid"], request.path_params["eid"], request.path_params["name"]
    await run(project_or_404, app, pid)
    d, m = await run(_manifest, app, pid, eid)
    if name not in {f["name"] for f in m["files"]}:           # only files the manifest lists: no path traversal possible
        raise ApiError(404, "file not in this export", "not_found")
    p = d / name
    if not p.is_file():
        raise ApiError(404, "file missing on disk", "not_found")
    return FileResponse(p, filename=p.name)


async def exports_delete(request: Request):
    app = ctx_of(request)
    pid, eid = request.path_params["pid"], request.path_params["eid"]
    await run(project_or_404, app, pid)
    d, _ = await run(_manifest, app, pid, eid)
    await run(shutil.rmtree, d)
    return ok({"deleted": eid})


# ------------------------------------------------------------------------------------------ photogrammetry
class OdmIn(BaseModel):
    preset: Literal["draft", "balanced", "high"] = "balanced"
    options: dict = Field(default_factory=dict)
    exclude_images: List[str] = Field(default_factory=list, max_length=1_000_000)
    allow_ungeoreferenced: bool = False
    build_terrain: bool = Field(False, description="after photogrammetry, continue automatically into terrain reconstruction (3D model)")
    ground_engine: Optional[Literal["pdal_smrf", "pdal_pmf", "builtin_pmf"]] = None
    auto_label: bool = Field(False, description="after the textured 3D model is prepared, label it automatically from its own photos")


async def photogrammetry_state(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        n = app.db.one("SELECT COUNT(*) n, SUM(lat IS NOT NULL) g FROM images WHERE project_id=?", (pid,))
        gcp = P.get(app, pid, "gcp_list")
        docker = {"available": bool(shutil.which(app.settings.docker_bin)), "bin": app.settings.docker_bin, "image": app.settings.odm_image}
        if not docker["available"]:
            docker["hint"] = ODM.DOCKER_HINT
        last = app.db.one("SELECT id,status,progress,message,error,error_code,created_at,finished_at FROM jobs WHERE project_id=? AND kind='odm' "
                          "ORDER BY created_at DESC LIMIT 1", (pid,))
        q = P.get(app, pid, "odm_quality")
        problems = []
        if n["n"] < 3:
            problems.append(f"At least 3 images are needed (have {n['n']}).")
        if (n["g"] or 0) == 0 and not (gcp and gcp["exists"]):
            problems.append("No image has GPS and no GCP list is present: results could not be georeferenced.")
        return {"images": n["n"], "images_with_gps": n["g"] or 0, "gcp_list": bool(gcp and gcp["exists"]), "docker": docker, "presets": ODM.PRESETS,
                "allowed_options": {k: (list(v) if isinstance(v, tuple) else v.__name__) for k, v in ODM.ALLOWED.items()},
                "last_job": dict(last) if last else None, "quality": q["meta"] if q and q["exists"] else None, "blocking": problems,
                "resume_note": "A failed run is resumed by ODM from its last completed stage when retried. Running jobs cannot be paused; "
                               "cancel stops the container and a later run resumes."}
    return ok(await run(work))


async def photogrammetry_start(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, OdmIn)

    def work():
        busy = app.db.one("SELECT id FROM jobs WHERE project_id=? AND kind='odm' AND status IN ('queued','running')", (pid,))
        if busy:
            raise ApiError(409, "A photogrammetry job is already queued or running for this project.", "job_active", {"job_id": busy["id"]})
        try:
            ODM.build_options(b.preset, b.options, False)          # fail fast on invalid options
        except Exception as e:                                       # noqa: BLE001
            raise ApiError(422, str(e), "validation_error")
        params = {"project_id": pid, **b.model_dump(exclude={"build_terrain", "ground_engine"})}
        note = None
        if b.build_terrain:
            engine = b.ground_engine or ("pdal_smrf" if shutil.which(app.settings.pdal_bin) else "builtin_pmf")
            if engine.startswith("pdal") and not shutil.which(app.settings.pdal_bin):
                raise ApiError(409, "PDAL is not installed here, so the PDAL ground filters cannot run. Choose the built-in filter (less accurate on "
                               "steep or wide terrain) or install PDAL.", "no_pdal")
            params["terrain_chain"] = {"ground": {"engine": engine}, "resolution_m": 0.5}
            if engine == "builtin_pmf":
                note = "PDAL was not found, so the built-in ground filter will be used. It is less accurate than PDAL's SMRF on steep or wide terrain."
        return {"job_id": app.queue.enqueue("odm", pid, params, max_attempts=2), "ground_engine": params.get("terrain_chain", {}).get("ground", {}).get("engine"), "note": note}
    return ok(await run(work), 202)


routes = [
    Route("/projects/{pid}/exports/catalog", exports_catalog),
    Route("/projects/{pid}/exports", exports_create, methods=["POST"]),
    Route("/projects/{pid}/exports", exports_list, methods=["GET"]),
    Route("/projects/{pid}/exports/{eid}", exports_get, methods=["GET"]),
    Route("/projects/{pid}/exports/{eid}", exports_delete, methods=["DELETE"]),
    Route("/projects/{pid}/exports/{eid}/files/{name:path}", exports_file, methods=["GET"]),
    Route("/projects/{pid}/photogrammetry", photogrammetry_state, methods=["GET"]),
    Route("/projects/{pid}/photogrammetry", photogrammetry_start, methods=["POST"]),
]
