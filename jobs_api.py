"""Job monitoring and queue management."""
from __future__ import annotations

from starlette.requests import Request
from starlette.routing import Route

from .deps import ApiError, ctx_of, ok, project_or_404, run

NOISY = ["ingest_image", "analyze_flights"]


def _public(j: dict) -> dict:
    j = dict(j)
    j["params"] = {k: v for k, v in j["params"].items() if not k.startswith("_")}
    return j


async def list_jobs(request: Request):
    app = ctx_of(request)
    q = request.query_params
    pid = request.path_params.get("pid")
    if pid:
        await run(project_or_404, app, pid)
    kinds = q["kind"].split(",") if q.get("kind") else None
    exclude = None if (kinds or q.get("include_ingest") == "1") else NOISY
    jobs = await run(app.queue.list, pid, kinds, exclude, q.get("status"), int(q.get("limit", 100)))
    return ok([_public(j) for j in jobs])


async def get_job(request: Request):
    app = ctx_of(request)
    j = await run(app.queue.get, request.path_params["jid"])
    if not j:
        raise ApiError(404, "job not found", "not_found")
    return ok(_public(j))


async def job_logs(request: Request):
    app = ctx_of(request)
    jid = request.path_params["jid"]
    if not await run(app.queue.get, jid):
        raise ApiError(404, "job not found", "not_found")
    return ok(await run(app.queue.logs, jid, int(request.query_params.get("after", 0)), int(request.query_params.get("limit", 500))))


async def cancel_job(request: Request):
    app = ctx_of(request)
    try:
        st = await run(app.queue.cancel, request.path_params["jid"])
    except KeyError:
        raise ApiError(404, "job not found", "not_found")
    return ok({"status": st})


async def retry_job(request: Request):
    app = ctx_of(request)
    jid = request.path_params["jid"]
    try:
        await run(app.queue.retry, jid)
    except KeyError:
        raise ApiError(404, "job not found", "not_found")
    except ValueError as e:
        raise ApiError(409, str(e), "bad_state")
    j = app.queue.get(jid)
    if j and j["kind"] in ("ingest_image", "import_asset"):
        app.db.update("upload_files", {"upload_id": j["params"].get("upload_id")}, {"status": "queued", "error": None})
    return ok({"status": "queued"})


async def queue_stats(request: Request):
    return ok(await run(ctx_of(request).queue.stats))


async def queue_pause(request: Request):
    await run(ctx_of(request).queue.set_paused, True)
    return ok({"paused": True})


async def queue_resume(request: Request):
    await run(ctx_of(request).queue.set_paused, False)
    return ok({"paused": False})


routes = [
    Route("/jobs", list_jobs, methods=["GET"]),
    Route("/jobs/{jid}", get_job, methods=["GET"]),
    Route("/jobs/{jid}/logs", job_logs, methods=["GET"]),
    Route("/jobs/{jid}/cancel", cancel_job, methods=["POST"]),
    Route("/jobs/{jid}/retry", retry_job, methods=["POST"]),
    Route("/queue", queue_stats, methods=["GET"]),
    Route("/queue/pause", queue_pause, methods=["POST"]),
    Route("/queue/resume", queue_resume, methods=["POST"]),
    Route("/projects/{pid}/jobs", list_jobs, methods=["GET"]),
]
