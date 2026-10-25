"""ASGI application factory.  ``uvicorn app.api.main:app``"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..context import AppContext
from ..config import Settings
from . import jobs_api, projects, uploads
from .deps import ApiError, ctx_of, ok, run

log = logging.getLogger("terrascope.api")
VERSION = "0.1.0"


def dependency_report(app: AppContext) -> dict:
    s = app.settings
    pdal = shutil.which(s.pdal_bin)
    docker = shutil.which(s.docker_bin)
    docker_running = None
    if docker:
        import subprocess
        try:
            docker_running = subprocess.run([docker, "info"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            docker_running = False
    return {
        "pdal": {"available": bool(pdal), "path": pdal, "needed_for": "SMRF/PMF ground filtering and LAZ conversion",
                 "install": "conda install -c conda-forge pdal   |   apt install pdal   |   docker image pdal/pdal"},
        "docker": {"available": bool(docker), "daemon_running": docker_running, "needed_for": "OpenDroneMap photogrammetry",
                   "install": "Install Docker Engine and start the daemon: https://docs.docker.com/engine/install/"},
        "opendronemap": {"image": s.odm_image, "needed_for": "orthomosaic, point cloud, DSM from drone images",
                         "install": f"docker pull {s.odm_image}  (several GB; needs the Docker daemon)"},
        "segmentation_model": {"configured": bool(s.model_path), "path": s.model_path or None,
                               "note": "Optional ONNX model. Without it the built-in colour/texture heuristic is used."},
    }


async def health(request: Request):
    app = ctx_of(request)
    du = shutil.disk_usage(app.settings.data_dir)
    return ok({"status": "ok", "version": VERSION, "dev_mode": app.settings.dev_mode, "disk_free": du.free,
               "queue": await run(app.queue.stats)})


async def system(request: Request):
    app = ctx_of(request)
    return ok({"version": VERSION, "dependencies": await run(dependency_report, app), "dev_mode": app.settings.dev_mode})


def create_app(ctx: Optional[AppContext] = None, static_dir: Optional[Path] = None) -> Starlette:
    ctx = ctx or AppContext(Settings())

    async def api_error(request: Request, exc: ApiError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, status_code=exc.status)

    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}}, status_code=500)

    async def not_found(request: Request, exc):
        return JSONResponse({"error": {"code": "not_found", "message": "not found"}}, status_code=404)

    api_routes = [Route("/health", health), Route("/system", system), *projects.routes, *uploads.routes, *jobs_api.routes]
    try:
        from . import editor
        api_routes += editor.routes
    except ImportError:
        pass
    try:
        from . import imgclass
        api_routes += imgclass.routes
        from . import mesh
        api_routes += mesh.routes
    except ImportError:
        pass
    try:
        from . import terrain
        api_routes += terrain.routes
    except ImportError:
        pass
    try:
        from . import exports as exports_api
        api_routes += exports_api.routes
    except ImportError:
        pass
    routes = [Mount("/api", routes=api_routes)]
    static_dir = static_dir or Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if static_dir.exists():
        routes.append(Mount("/", app=StaticFiles(directory=str(static_dir), html=True), name="ui"))
    app = Starlette(routes=routes, exception_handlers={ApiError: api_error, 404: not_found, Exception: unhandled})
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["X-Raster-Version"])
    app.state.ctx = ctx
    return app


def app_factory() -> Starlette:   # uvicorn --factory app.api.main:app_factory
    return create_app()
