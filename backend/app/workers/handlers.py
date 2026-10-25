"""Job-kind -> handler registry. Handlers get ``(app, ctx)`` and return a JSON-serialisable result or raise."""
from __future__ import annotations

from typing import Callable, Dict

from ..context import AppContext
from ..jobs import JobContext

HANDLERS: Dict[str, Callable[[AppContext, JobContext], dict]] = {}


def register(kind: str):
    def deco(fn):
        HANDLERS[kind] = fn
        return fn
    return deco


def load_all() -> Dict[str, Callable]:
    """Import handler modules for their registration side effects (lazy so the API can start without heavy deps)."""
    from ..services import ingest, flights
    HANDLERS.setdefault("ingest_image", ingest.ingest_image)
    HANDLERS.setdefault("analyze_flights", flights.handle_analyze)
    from ..services import assets
    HANDLERS.setdefault("import_asset", assets.import_asset)
    for mod in ("ortho", "classify", "odm", "terrain_job", "exports", "samples", "imgclass", "meshprep"):
        try:
            __import__(f"{__package__.rsplit('.', 1)[0]}.services.{mod}")
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(f"services.{mod}"):
                continue
            raise
    return HANDLERS
