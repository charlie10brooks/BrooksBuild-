"""Shared API helpers: error type, JSON/validation helpers, project lookup."""
from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..context import AppContext
from ..db import loads, sanitize

T = TypeVar("T", bound=BaseModel)

DISCLAIMER = ("Preliminary planning estimate. Not suitable for final foundation design, permitting or construction staking "
              "without validation by a licensed surveyor / engineer.")


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str = "error", details: Any = None):
        super().__init__(message)
        self.status, self.message, self.code, self.details = status, message, code, details


def ctx_of(request: Request) -> AppContext:
    return request.app.state.ctx


async def run(fn: Callable, *a, **k):
    """Run blocking work (SQLite, file I/O, numpy) off the event loop."""
    return await asyncio.get_running_loop().run_in_executor(None, partial(fn, *a, **k))


async def body(request: Request, model: Type[T]) -> T:
    try:
        raw = await request.body()
        data = json.loads(raw) if raw else {}
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "request body is not valid JSON", "bad_json")
    try:
        return model.model_validate(data)
    except ValidationError as e:
        errs = [{"field": ".".join(str(p) for p in x["loc"]), "message": x["msg"]} for x in e.errors()]
        raise ApiError(422, "; ".join(f"{x['field']}: {x['message']}" for x in errs)[:500], "validation_error", errs)


def project_or_404(app: AppContext, pid: str):
    try:
        app.storage.project_root(pid)
    except Exception:
        raise ApiError(404, "project not found", "not_found")
    r = app.db.one("SELECT * FROM projects WHERE id=?", (pid,))
    if r is None:
        raise ApiError(404, "project not found", "not_found")
    return r


def project_dict(r, extra: Optional[dict] = None) -> dict:
    d = dict(r)
    d["boundary"] = loads(d.pop("boundary_geojson"), None)
    d["classes"] = loads(d.pop("classes_json"), None)
    d["settings"] = loads(d.pop("settings_json"), {})
    d["is_sample"] = bool(d["is_sample"])
    if extra:
        d.update(extra)
    return d


class SafeJSONResponse(JSONResponse):
    """JSON that can never fail on NaN/Infinity (they become null)."""

    def render(self, content: Any) -> bytes:
        return super().render(sanitize(content))


def ok(data: Any = None, status: int = 200, **kw) -> JSONResponse:
    return SafeJSONResponse(data if data is not None else {"ok": True}, status_code=status, **kw)
