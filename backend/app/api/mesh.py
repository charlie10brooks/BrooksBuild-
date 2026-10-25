"""Photo-textured 3D model (from OpenDroneMap) for the browser viewer. See services/meshprep.py."""
from __future__ import annotations

import re

from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Route

from ..services import meshprep as M
from .deps import ApiError, ctx_of, ok, project_or_404, run


async def state(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(M.info, app, pid))


async def build(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        if not M.source_obj(app, pid):
            raise ApiError(409, "There is no textured 3D model from OpenDroneMap to prepare. Run the 'balanced' or 'high' preset (the 'draft' "
                                "preset skips it).", "no_textured_model")
        return app.queue.enqueue("mesh_prep", pid, {"project_id": pid}, dedupe_key=f"mesh_prep:{pid}")
    return ok({"job_id": await run(work)}, status=202)


async def data(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    p = app.storage.path(pid, f"{M.DIR}/mesh.bin")
    if not p.exists():
        raise ApiError(404, "no textured model prepared yet", "not_found")
    return FileResponse(p, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


async def texture(request: Request):
    app, pid, name = ctx_of(request), request.path_params["pid"], request.path_params["name"]
    await run(project_or_404, app, pid)
    if not re.fullmatch(r"tex_\d{1,4}\.jpg", name):
        raise ApiError(404, "no such texture", "not_found")
    p = app.storage.path(pid, f"{M.DIR}/{name}")
    if not p.exists():
        raise ApiError(404, "no such texture", "not_found")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


routes = [
    Route("/projects/{pid}/mesh", state),
    Route("/projects/{pid}/mesh/build", build, methods=["POST"]),
    Route("/projects/{pid}/mesh/data", data),
    Route("/projects/{pid}/mesh/texture/{name}", texture),
]
