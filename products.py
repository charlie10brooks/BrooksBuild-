"""Registry of processing products (rasters, clouds...) per project, plus the standard project directory layout."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import AppContext
from ..db import dumps, loads, now

# canonical product names -> (relative path, kind)
LAYOUT = {
    "orthophoto": ("products/orthophoto.tif", "raster"),
    "dsm": ("products/dsm.tif", "raster"),
    "dtm": ("products/dtm.tif", "raster"),
    "ndsm": ("products/ndsm.tif", "raster"),
    "confidence": ("products/terrain_confidence.tif", "raster"),
    "evidence": ("products/terrain_evidence.tif", "raster"),
    "source": ("products/elevation_source.tif", "raster"),
    "classification": ("products/classification.tif", "raster"),
    "pointcloud": ("products/pointcloud.las", "pointcloud"),
    "classified_cloud": ("products/pointcloud_classified.las", "pointcloud"),
}
REQUIRES_MIN = 1


def register(app: AppContext, pid: str, name: str, rel_path: str, kind: str, meta: Optional[dict] = None,
             job_id: Optional[str] = None) -> None:
    with app.db.tx() as c:
        c.execute("INSERT INTO products(project_id,name,rel_path,kind,meta_json,job_id,created_at) VALUES(?,?,?,?,?,?,?) "
                  "ON CONFLICT(project_id,name) DO UPDATE SET rel_path=excluded.rel_path, kind=excluded.kind, "
                  "meta_json=excluded.meta_json, job_id=excluded.job_id, created_at=excluded.created_at",
                  (pid, name, rel_path, kind, dumps(meta or {}), job_id, now()))


def get(app: AppContext, pid: str, name: str) -> Optional[Dict[str, Any]]:
    r = app.db.one("SELECT * FROM products WHERE project_id=? AND name=?", (pid, name))
    if not r:
        return None
    p = app.storage.path(pid, r["rel_path"])
    return {"name": r["name"], "path": p, "rel_path": r["rel_path"], "kind": r["kind"], "meta": loads(r["meta_json"], {}),
            "exists": p.exists(), "created_at": r["created_at"], "job_id": r["job_id"]}


def require(app: AppContext, pid: str, name: str) -> Dict[str, Any]:
    from ..jobs import JobFailure
    p = get(app, pid, name)
    if not p or not p["exists"]:
        raise JobFailure(f"product {name!r} is not available for this project yet", "missing_product")
    return p


def list_products(app: AppContext, pid: str) -> List[Dict[str, Any]]:
    out = []
    for r in app.db.query("SELECT name FROM products WHERE project_id=? ORDER BY name", (pid,)):
        p = get(app, pid, r["name"])
        if p:
            out.append({"name": p["name"], "kind": p["kind"], "exists": p["exists"], "meta": p["meta"], "created_at": p["created_at"],
                        "size": p["path"].stat().st_size if p["exists"] else None})
    return out
