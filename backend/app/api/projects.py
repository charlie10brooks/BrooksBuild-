"""Projects, images, flights, control points, storage."""
from __future__ import annotations

import csv
import io
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from ..core import classes as C
from ..core import flight as F
from ..db import dumps, loads, new_id, now
from ..services import flights as FS
from .deps import ApiError, body, ctx_of, ok, project_dict, project_or_404, run

Units = Literal["metre", "foot", "us_survey_foot"]


def validate_boundary(g) -> Optional[dict]:
    """Accept Polygon / MultiPolygon / Feature / FeatureCollection(of polygons) in lon/lat or project CRS. Returns a geometry."""
    if g is None:
        return None
    if isinstance(g, dict) and g.get("type") == "FeatureCollection":
        geoms = [f.get("geometry") for f in g.get("features", []) if f.get("geometry")]
        if not geoms:
            raise ValueError("boundary FeatureCollection has no geometry")
        g = geoms[0]
    if isinstance(g, dict) and g.get("type") == "Feature":
        g = g.get("geometry")
    if not isinstance(g, dict) or g.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("boundary must be a GeoJSON Polygon or MultiPolygon")
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    n = 0
    for poly in polys:
        if not poly or len(poly[0]) < 4:
            raise ValueError("each boundary ring needs at least 4 positions (closed ring)")
        for ring in poly:
            for p in ring:
                if len(p) < 2 or not all(isinstance(v, (int, float)) for v in p[:2]):
                    raise ValueError("boundary positions must be numeric [x, y]")
                n += 1
    if n > 200_000:
        raise ValueError("boundary has too many vertices")
    return g


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    address: Optional[str] = Field(None, max_length=500)
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    epsg: Optional[int] = Field(None, ge=1024, le=69999)
    vertical_epsg: Optional[int] = Field(None, ge=1024, le=69999)
    horizontal_units: Units = "metre"
    vertical_units: Units = "metre"
    unit_system: Literal["metric", "imperial"] = "metric"
    boundary: Optional[dict] = None

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v

    @field_validator("boundary")
    @classmethod
    def _b(cls, v):
        return validate_boundary(v)


class ProjectPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    address: Optional[str] = Field(None, max_length=500)
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    epsg: Optional[int] = Field(None, ge=1024, le=69999)
    vertical_epsg: Optional[int] = Field(None, ge=1024, le=69999)
    horizontal_units: Optional[Units] = None
    vertical_units: Optional[Units] = None
    unit_system: Optional[Literal["metric", "imperial"]] = None
    boundary: Optional[dict] = None
    clear_boundary: bool = False
    classes: Optional[List[dict]] = None

    @field_validator("boundary")
    @classmethod
    def _b(cls, v):
        return validate_boundary(v)

    @field_validator("classes")
    @classmethod
    def _c(cls, v):
        if v is None:
            return v
        seen = set()
        for c in v:
            try:
                cid = int(c["id"])
                C.hex_to_rgb(c["color"])
            except Exception:
                raise ValueError("each class needs an integer id and a #RRGGBB color")
            if not 0 <= cid <= 63 or cid in seen:
                raise ValueError("class ids must be unique integers 0..63 (higher bits of the mask hold provenance tags)")
            seen.add(cid)
            if not str(c.get("name", "")).strip():
                raise ValueError("each class needs a name")
        if 0 not in seen:
            raise ValueError("class 0 (unknown) is required")
        return v


def _suggest_epsg(app, pid: str) -> Optional[dict]:
    r = app.db.one("SELECT AVG(lon) lon, AVG(lat) lat, COUNT(*) n FROM images WHERE project_id=? AND lat IS NOT NULL AND lon IS NOT NULL", (pid,))
    if r and r["n"]:
        return {"epsg": F.suggest_utm_epsg(r["lon"], r["lat"]), "basis": f"UTM zone for the mean GPS position of {r['n']} images"}
    return None


def project_summary(app, r) -> dict:
    pid = r["id"]
    counts = app.db.one("SELECT COUNT(*) n, COALESCE(SUM(size),0) bytes, SUM(lat IS NULL OR lon IS NULL) no_gps FROM images WHERE project_id=?", (pid,))
    jobs = app.db.one("SELECT SUM(status='running') running, SUM(status='queued') queued, SUM(status='failed') failed FROM jobs "
                      "WHERE project_id=? AND kind NOT IN ('ingest_image','analyze_flights')", (pid,))
    prods = [x["name"] for x in app.db.query("SELECT name FROM products WHERE project_id=?", (pid,))]
    return project_dict(r, {"image_count": counts["n"], "image_bytes": counts["bytes"], "images_without_gps": counts["no_gps"] or 0,
                            "jobs": {k: jobs[k] or 0 for k in ("running", "queued", "failed")}, "products": prods,
                            "suggested_crs": _suggest_epsg(app, pid)})


# ---------------------------------------------------------------- handlers
async def list_projects(request: Request):
    app = ctx_of(request)
    rows = await run(app.db.query, "SELECT * FROM projects ORDER BY updated_at DESC")
    return ok([await run(project_summary, app, r) for r in rows])


async def create_project(request: Request):
    app = ctx_of(request)
    p = await body(request, ProjectIn)
    pid, t = new_id(), now()
    classes = [c.to_dict() for c in C.DEFAULT_CLASSES]
    await run(app.db.insert, "projects", {
        "id": pid, "name": p.name, "address": p.address, "lat": p.lat, "lon": p.lon, "epsg": p.epsg, "vertical_epsg": p.vertical_epsg,
        "horizontal_units": p.horizontal_units, "vertical_units": p.vertical_units, "unit_system": p.unit_system,
        "boundary_geojson": dumps(p.boundary) if p.boundary else None, "classes_json": dumps(classes), "created_at": t, "updated_at": t})
    app.storage.ensure(pid)
    return ok(project_summary(app, app.db.one("SELECT * FROM projects WHERE id=?", (pid,))), 201)


async def get_project(request: Request):
    app = ctx_of(request)
    r = await run(project_or_404, app, request.path_params["pid"])
    s = await run(project_summary, app, r)
    s["storage_bytes"] = await run(app.storage.usage, r["id"])
    return ok(s)


async def patch_project(request: Request):
    app = ctx_of(request)
    r = await run(project_or_404, app, request.path_params["pid"])
    p = await body(request, ProjectPatch)
    vals = {k: v for k, v in p.model_dump(exclude_unset=True).items() if k not in ("boundary", "clear_boundary", "classes")}
    if p.boundary is not None:
        vals["boundary_geojson"] = dumps(p.boundary)
    if p.clear_boundary:
        vals["boundary_geojson"] = None
    if p.classes is not None:
        vals["classes_json"] = dumps(p.classes)
    if ("epsg" in vals or "horizontal_units" in vals) and app.db.one("SELECT 1 FROM products WHERE project_id=?", (r["id"],)):
        # Changing the CRS label does not reproject anything; make that explicit instead of silently diverging.
        if vals.get("epsg", r["epsg"]) != r["epsg"]:
            raise ApiError(409, "This project already has processed products in its current CRS. Changing the CRS label would "
                                "mislabel them. Create a new project or reprocess instead.", "crs_locked")
    vals["updated_at"] = now()
    await run(app.db.update, "projects", {"id": r["id"]}, vals)
    return ok(project_summary(app, app.db.one("SELECT * FROM projects WHERE id=?", (r["id"],))))


async def delete_project(request: Request):
    app = ctx_of(request)
    r = await run(project_or_404, app, request.path_params["pid"])
    if request.query_params.get("confirm") != r["name"]:
        raise ApiError(400, "pass ?confirm=<exact project name> to delete a project and all its data", "confirmation_required")
    busy = app.db.one("SELECT 1 FROM jobs WHERE project_id=? AND status='running'", (r["id"],))
    if busy:
        raise ApiError(409, "cancel running jobs first", "jobs_running")
    await run(app.db.execute, "DELETE FROM projects WHERE id=?", (r["id"],))
    await run(app.storage.delete_project, r["id"])
    return ok()


# ---------------------------------------------------------------- images
def _img(r, brief=True) -> dict:
    d = dict(r)
    d["issues"] = loads(d.pop("issues_json"), [])
    meta = loads(d.pop("meta_json"), {})
    d["overlap"] = loads(d.pop("overlap_json"), None)
    d.pop("rel_path", None)
    d.pop("thumb_path", None)
    d["has_thumbnail"] = True
    if not brief:
        d["metadata"] = meta
    return d


async def list_images(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    q = request.query_params
    where, args = ["project_id=?"], [pid]
    if q.get("flight") is not None and q.get("flight") != "":
        where.append("flight_group=?"); args.append(int(q["flight"]))
    if q.get("georef"):
        where.append("georef_status=?"); args.append(q["georef"])
    if q.get("batch"):
        where.append("batch_id=?"); args.append(q["batch"])
    if q.get("poor_overlap") == "1":
        where.append("overlap_json LIKE '%\"poor\":true%'")
    if q.get("q"):
        where.append("filename LIKE ?"); args.append("%" + q["q"].replace("%", "") + "%")
    limit = max(1, min(int(q.get("limit", 100)), 1000))
    offset = max(0, int(q.get("offset", 0)))
    sql = " AND ".join(where)
    total = app.db.one(f"SELECT COUNT(*) n FROM images WHERE {sql}", args)["n"]
    rows = app.db.query(f"SELECT * FROM images WHERE {sql} ORDER BY captured_at, filename LIMIT ? OFFSET ?", [*args, limit, offset])
    return ok({"total": total, "offset": offset, "limit": limit, "items": [_img(r) for r in rows]})


def _image_or_404(app, pid, iid):
    r = app.db.one("SELECT * FROM images WHERE id=? AND project_id=?", (iid, pid))
    if r is None:
        raise ApiError(404, "image not found", "not_found")
    return r


async def get_image(request: Request):
    app = ctx_of(request)
    r = await run(_image_or_404, app, request.path_params["pid"], request.path_params["iid"])
    return ok(_img(r, brief=False))


async def image_thumbnail(request: Request):
    app = ctx_of(request)
    r = await run(_image_or_404, app, request.path_params["pid"], request.path_params["iid"])
    if not r["thumb_path"]:
        raise ApiError(404, "no thumbnail for this image", "no_thumbnail")
    return FileResponse(app.storage.path(r["project_id"], r["thumb_path"]), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


async def image_original(request: Request):
    app = ctx_of(request)
    r = await run(_image_or_404, app, request.path_params["pid"], request.path_params["iid"])
    return FileResponse(app.storage.path(r["project_id"], r["rel_path"]), filename=r["filename"])


# ---------------------------------------------------------------- flights
async def flights(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    r = await run(project_or_404, app, pid)
    rows = app.db.query("SELECT flight_group g, COUNT(*) n, MIN(captured_at) t0, MAX(captured_at) t1, SUM(lat IS NULL OR lon IS NULL) no_gps, "
                        "SUM(overlap_json LIKE '%\"poor\":true%') poor, AVG(rel_alt) alt FROM images WHERE project_id=? GROUP BY flight_group ORDER BY g", (pid,))
    return ok({"groups": [{"group": x["g"], "label": "Unassigned (no capture time)" if x["g"] == 0 else f"Flight {x['g']}", "images": x["n"],
                           "start": x["t0"], "end": x["t1"], "images_without_gps": x["no_gps"] or 0, "poor_overlap": x["poor"] or 0,
                           "mean_relative_altitude_m": x["alt"]} for x in rows],
               "summary": loads(r["settings_json"], {}).get("flight_summary")})


async def footprints(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    return JSONResponse(await run(FS.footprints_geojson, app, pid), media_type="application/geo+json")


async def reanalyze(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    jid = await run(app.queue.enqueue, "analyze_flights", pid, {"project_id": pid}, 0, f"analyze:{pid}")
    return ok({"job_id": jid}, 202)


# ---------------------------------------------------------------- control points
_HEADER_X = {"x", "easting", "east", "e", "lon", "lng", "longitude"}
_CP_KIND = {"gcp": "gcp", "control": "gcp", "check": "check", "checkpoint": "check", "cp": "check"}


def parse_control_csv(text: str, default_kind: str = "gcp") -> List[dict]:
    """label,x,y,z[,kind]. Header optional. Rejects rows it cannot parse instead of skipping them silently."""
    rows = list(csv.reader(io.StringIO(text.strip())))
    out, errors = [], []
    for n, row in enumerate(rows, 1):
        row = [c.strip() for c in row]
        if not row or all(not c for c in row) or row[0].startswith("#"):
            continue
        try:
            x, y = float(row[1]), float(row[2])
            z = float(row[3]) if len(row) > 3 and row[3] != "" else None
        except (ValueError, IndexError):
            if n == 1 and len(row) > 1 and row[1].lower() in _HEADER_X:      # a recognisable header row only
                continue
            errors.append(f"line {n}: expected label,x,y,z[,kind]")
            continue
        kind = _CP_KIND.get(row[4].lower()) if len(row) > 4 and row[4] else default_kind
        if kind is None:
            errors.append(f"line {n}: kind must be gcp or check")
            continue
        out.append({"label": row[0][:100] or f"P{n}", "x": x, "y": y, "z": z, "kind": kind})
    if errors:
        raise ApiError(422, "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""), "bad_csv", errors)
    if not out:
        raise ApiError(422, "no control points found", "bad_csv")
    return out


class ControlImport(BaseModel):
    csv: str = Field(max_length=5_000_000)
    kind: Literal["gcp", "check"] = "gcp"
    replace: bool = False


async def import_control(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await body(request, ControlImport)
    pts = parse_control_csv(b.csv, b.kind)

    def work():
        with app.db.tx() as c:
            if b.replace:
                c.execute("DELETE FROM control_points WHERE project_id=?", (pid,))
            for p in pts:
                c.execute("INSERT INTO control_points(id,project_id,kind,label,x,y,z,created_at) VALUES(?,?,?,?,?,?,?,?)",
                          (new_id(), pid, p["kind"], p["label"], p["x"], p["y"], p["z"], now()))
    await run(work)
    return ok({"imported": len(pts)}, 201)


async def list_control(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok([{**dict(r), "residual": loads(r["residual_json"], None)} for r in
               app.db.query("SELECT id,kind,label,x,y,z,residual_json FROM control_points WHERE project_id=? ORDER BY label", (pid,))])


async def delete_control(request: Request):
    app = ctx_of(request)
    pid = request.path_params["pid"]
    await run(project_or_404, app, pid)
    await run(app.db.execute, "DELETE FROM control_points WHERE id=? AND project_id=?", (request.path_params["cid"], pid))
    return ok()


# ---------------------------------------------------------------- storage
async def storage_info(request: Request):
    import shutil
    app = ctx_of(request)
    du = shutil.disk_usage(app.settings.data_dir)
    projs = [{"id": r["id"], "name": r["name"], "bytes": app.storage.usage(r["id"])} for r in app.db.query("SELECT id,name FROM projects")]
    up = sum(f.stat().st_size for f in (app.settings.data_dir / "uploads").rglob("*") if f.is_file())
    return ok({"disk_total": du.total, "disk_free": du.free, "min_free_bytes": app.settings.min_free_bytes,
               "pending_upload_bytes": up, "projects": projs,
               "limits": {"max_file_bytes": app.settings.max_file_bytes or None, "project_quota_bytes": app.settings.project_quota_bytes or None,
                          "max_images": None, "max_image_megapixels": app.settings.max_image_megapixels},
               "low_space": du.free < app.settings.min_free_bytes * 2})


routes = [
    Route("/projects", list_projects, methods=["GET"]),
    Route("/projects", create_project, methods=["POST"]),
    Route("/projects/{pid}", get_project, methods=["GET"]),
    Route("/projects/{pid}", patch_project, methods=["PATCH"]),
    Route("/projects/{pid}", delete_project, methods=["DELETE"]),
    Route("/projects/{pid}/images", list_images, methods=["GET"]),
    Route("/projects/{pid}/images/{iid}", get_image, methods=["GET"]),
    Route("/projects/{pid}/images/{iid}/thumbnail", image_thumbnail, methods=["GET"]),
    Route("/projects/{pid}/images/{iid}/original", image_original, methods=["GET"]),
    Route("/projects/{pid}/flights", flights, methods=["GET"]),
    Route("/projects/{pid}/footprints", footprints, methods=["GET"]),
    Route("/projects/{pid}/flights/analyze", reanalyze, methods=["POST"]),
    Route("/projects/{pid}/control-points", list_control, methods=["GET"]),
    Route("/projects/{pid}/control-points", import_control, methods=["POST"]),
    Route("/projects/{pid}/control-points/{cid}", delete_control, methods=["DELETE"]),
    Route("/storage", storage_info, methods=["GET"]),
]


# ---------------------------------------------------------------- dev mode sample
async def create_sample(request: Request):
    app = ctx_of(request)
    if not app.settings.dev_mode:
        raise ApiError(403, "sample projects are only available in development mode (TERRASCOPE_DEV_MODE=1)", "dev_mode_off")
    from ..services import samples
    pid = await run(samples.create_sample_project, app)
    return ok(project_summary(app, app.db.one("SELECT * FROM projects WHERE id=?", (pid,))), 201)


routes.append(Route("/dev/sample-project", create_sample, methods=["POST"]))
