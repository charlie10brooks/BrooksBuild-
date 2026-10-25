"""Photo-textured 3D model (from OpenDroneMap) for the browser viewer. See services/meshprep.py."""
from __future__ import annotations

import re

from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Route

import numpy as np
from starlette.responses import Response

from ..services import meshclass as MC
from ..services import meshprep as M
from .deps import ApiError, ctx_of, ok, project_or_404, run


def _wrap(fn, *a):
    try:
        return fn(*a)
    except MC.MeshClassError as e:
        raise ApiError(e.status, e.message, e.code)


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


async def labels_state(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(MC.info, app, pid))


async def labels_data(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    b = await run(_wrap, MC.labels_bytes, app, pid)
    return Response(b, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


async def labels_seed(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_wrap, MC.seed, app, pid))


async def labels_edit(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    try:
        cls = int(request.query_params.get("cls", ""))
    except ValueError:
        raise ApiError(400, "cls must be a class id", "bad_request")
    raw = await request.body()
    if len(raw) % 4 or len(raw) == 0 or len(raw) > 4 * 20_000_000:
        raise ApiError(400, "body must be uint32 little-endian vertex indices", "bad_request")
    idx = np.frombuffer(raw, dtype="<u4")
    return ok(await run(_wrap, MC.edit, app, pid, idx, cls))


async def labels_undo(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_wrap, MC.undo, app, pid))


async def labels_redo(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_wrap, MC.redo, app, pid))


async def labels_clear(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)
    return ok(await run(_wrap, MC.clear, app, pid))


async def labels_auto(request: Request):
    app, pid = ctx_of(request), request.path_params["pid"]
    await run(project_or_404, app, pid)

    def work():
        if not (app.storage.path(pid, f"{M.DIR}/mesh.bin")).exists():
            raise ApiError(409, "There is no prepared photo-textured model to label yet.", "no_mesh")
        return app.queue.enqueue("classify_mesh", pid, {"project_id": pid}, dedupe_key=f"classify_mesh:{pid}")
    return ok({"job_id": await run(work)}, status=202)


routes = [
    Route("/projects/{pid}/mesh", state),
    Route("/projects/{pid}/mesh/build", build, methods=["POST"]),
    Route("/projects/{pid}/mesh/data", data),
    Route("/projects/{pid}/mesh/texture/{name}", texture),
    Route("/projects/{pid}/mesh/labels", labels_state),
    Route("/projects/{pid}/mesh/labels/data", labels_data),
    Route("/projects/{pid}/mesh/labels/auto", labels_auto, methods=["POST"]),
    Route("/projects/{pid}/mesh/labels/seed", labels_seed, methods=["POST"]),
    Route("/projects/{pid}/mesh/labels/edit", labels_edit, methods=["POST"]),
    Route("/projects/{pid}/mesh/labels/undo", labels_undo, methods=["POST"]),
    Route("/projects/{pid}/mesh/labels/redo", labels_redo, methods=["POST"]),
    Route("/projects/{pid}/mesh/labels/clear", labels_clear, methods=["POST"]),
]
