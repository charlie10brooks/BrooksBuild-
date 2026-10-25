"""Development-mode sample project built from the synthetic site generator.

The data is **synthetic and labelled as such everywhere** (project name, flag, notices). It exists so the editor, terrain
pipeline and exports can be exercised without drone imagery or OpenDroneMap. The synthetic ground truth is deliberately
NOT stored in the project, so the UI can never show it as if it were an estimate.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..context import AppContext
from ..core import classes as C
from ..db import dumps, new_id, now
from . import geotiff as gt
from . import las as L
from . import products as P
from .synthetic import SYNTHETIC_NOTICE, make_site


def create_sample_project(app: AppContext, width_m: float = 120.0, height_m: float = 90.0, res: float = 0.25, seed: int = 7) -> str:
    site = make_site(width_m, height_m, res, seed)
    pid, t = new_id(), now()
    app.db.insert("projects", {
        "id": pid, "name": "SAMPLE (synthetic) wooded lot", "address": SYNTHETIC_NOTICE, "lat": 39.76, "lon": -86.16, "epsg": site.epsg,
        "horizontal_units": "metre", "vertical_units": "metre", "unit_system": "metric",
        "classes_json": dumps([c.to_dict() for c in C.DEFAULT_CLASSES]), "settings_json": dumps({"synthetic": True, "notice": SYNTHETIC_NOTICE}),
        "is_sample": 1, "created_at": t, "updated_at": t})
    # a rectangular property boundary inside the site (project CRS coordinates)
    b = site.grid.bounds()
    m = 8.0
    ring = [[b[0] + m, b[1] + m], [b[2] - m, b[1] + m], [b[2] - m, b[3] - m], [b[0] + m, b[3] - m], [b[0] + m, b[1] + m]]
    app.db.update("projects", {"id": pid}, {"boundary_geojson": dumps({"type": "Polygon", "coordinates": [ring]})})
    app.storage.ensure(pid, "source")
    gt.write_geotiff(app.storage.path(pid, "source/sample_orthophoto.tif"), site.rgb, site.grid, site.epsg, description=SYNTHETIC_NOTICE)
    pts = site.points(density=3.0)
    cloud = app.storage.path(pid, P.LAYOUT["pointcloud"][0])
    cloud.parent.mkdir(parents=True, exist_ok=True)
    L.write_las(cloud, pts["x"], pts["y"], pts["z"], epsg=site.epsg)
    P.register(app, pid, "pointcloud", P.LAYOUT["pointcloud"][0], "pointcloud",
               {"synthetic": True, "points": int(len(pts["x"])), "notice": SYNTHETIC_NOTICE})
    app.queue.enqueue("prepare_ortho", pid, {"project_id": pid, "rel_path": "source/sample_orthophoto.tif"})
    return pid


def handle_noop(app, ctx):  # pragma: no cover
    return {}
