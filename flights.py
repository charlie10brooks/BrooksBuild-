"""Flight grouping, overlap estimation and footprints for a project's images."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..context import AppContext
from ..core import flight as F
from ..db import dumps, loads, now
from ..jobs import JobContext


def _geo(r) -> F.ImageGeo:
    meta = loads(r["meta_json"], {})
    return F.ImageGeo(r["id"], r["lon"], r["lat"], r["captured_at"], r["rel_alt"], meta.get("focal35_mm"),
                      meta.get("flight_yaw") or meta.get("gimbal_yaw"), r["width"], r["height"], meta.get("gimbal_pitch"))


def analyze_project(app: AppContext, pid: str, poor_threshold: float = 0.6) -> Dict[str, Any]:
    rows = app.db.query("SELECT id, lon, lat, captured_at, rel_alt, width, height, meta_json FROM images WHERE project_id=?", (pid,))
    imgs = [_geo(r) for r in rows]
    groups = F.group_flights(imgs)
    overlap, summary = F.analyse_overlap(imgs, poor_threshold)
    with app.db.tx() as c:
        for i in imgs:
            o = overlap[i.id]
            c.execute("UPDATE images SET flight_group=?, overlap_json=? WHERE id=?",
                      (groups.get(i.id, 0), dumps(o.__dict__), i.id))
        proj = c.execute("SELECT settings_json FROM projects WHERE id=?", (pid,)).fetchone()
        if proj is not None:
            st = loads(proj["settings_json"], {})
            summary["flight_groups"] = sorted(set(groups.values()) - {0})
            summary["ungrouped"] = sum(1 for g in groups.values() if g == 0)
            summary["computed_at"] = now()
            st["flight_summary"] = summary
            c.execute("UPDATE projects SET settings_json=?, updated_at=? WHERE id=?", (dumps(st), now(), pid))
    return summary


def footprints_geojson(app: AppContext, pid: str) -> Dict[str, Any]:
    """FeatureCollection of camera positions (Point) and image footprints (Polygon)."""
    rows = app.db.query("SELECT * FROM images WHERE project_id=? AND lat IS NOT NULL AND lon IS NOT NULL", (pid,))
    feats: List[dict] = []
    if not rows:
        return {"type": "FeatureCollection", "features": feats}
    lon0 = float(np.mean([r["lon"] for r in rows]))
    lat0 = float(np.mean([r["lat"] for r in rows]))
    for r in rows:
        ov = loads(r["overlap_json"], None) or {}
        props = {"id": r["id"], "filename": r["filename"], "flight_group": r["flight_group"], "georef_status": r["georef_status"],
                 "poor_overlap": bool(ov.get("poor")), "best_overlap": ov.get("best"), "rel_alt": r["rel_alt"]}
        feats.append({"type": "Feature", "properties": {**props, "kind": "camera"},
                      "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"], r["alt_msl"]] if r["alt_msl"] is not None else [r["lon"], r["lat"]]}})
        g = F.footprint_geojson(_geo(r), lon0, lat0)
        if g:
            feats.append({"type": "Feature", "properties": {**props, "kind": "footprint"}, "geometry": g})
    return {"type": "FeatureCollection", "features": feats}


def handle_analyze(app: AppContext, ctx: JobContext) -> dict:
    ctx.progress(0.1, "analysing flights")
    s = analyze_project(app, ctx.params["project_id"])
    ctx.log(f"{s['images']} images, {s['with_gps']} with GPS, {s['poor_overlap']} with poor estimated overlap")
    return s
