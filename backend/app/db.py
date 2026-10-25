"""SQLite persistence (WAL mode) shared by the API process and worker processes.

The schema is portable SQL; ``deploy/postgres_schema.sql`` is the PostgreSQL/PostGIS equivalent. That file is
**not exercised by the test-suite in this repository** (no PostgreSQL server was available when it was written).
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, address TEXT, lat REAL, lon REAL,
    epsg INTEGER, vertical_epsg INTEGER, horizontal_units TEXT NOT NULL DEFAULT 'metre',
    vertical_units TEXT NOT NULL DEFAULT 'metre', unit_system TEXT NOT NULL DEFAULT 'metric',
    boundary_geojson TEXT, classes_json TEXT, settings_json TEXT NOT NULL DEFAULT '{}',
    is_sample INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    batch_id TEXT, filename TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
    rel_path TEXT NOT NULL, thumb_path TEXT, width INTEGER, height INTEGER,
    lat REAL, lon REAL, alt_msl REAL, rel_alt REAL, captured_at TEXT, make TEXT, model TEXT,
    flight_group INTEGER NOT NULL DEFAULT 0, georef_status TEXT NOT NULL DEFAULT 'missing',
    issues_json TEXT NOT NULL DEFAULT '[]', meta_json TEXT NOT NULL DEFAULT '{}',
    overlap_json TEXT, created_at REAL NOT NULL, UNIQUE (project_id, sha256));
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id, flight_group);
CREATE TABLE IF NOT EXISTS image_classification (
    image_id TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, scale REAL NOT NULL, counts_json TEXT NOT NULL,
    unclassified REAL NOT NULL, mean_conf REAL, job_id TEXT, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_imgcls_project ON image_classification(project_id);
CREATE TABLE IF NOT EXISTS upload_files (
    upload_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    batch_id TEXT, filename TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL,
    error TEXT, image_id TEXT, duplicate_of TEXT, client_key TEXT, purpose TEXT NOT NULL DEFAULT 'image', created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_upfiles_project ON upload_files(project_id, status);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id) ON DELETE CASCADE, kind TEXT NOT NULL,
    status TEXT NOT NULL, params_json TEXT NOT NULL DEFAULT '{}', priority INTEGER NOT NULL DEFAULT 0,
    progress REAL NOT NULL DEFAULT 0, message TEXT, result_json TEXT, error TEXT, error_code TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
    cancel_requested INTEGER NOT NULL DEFAULT 0, worker_id TEXT, heartbeat_at REAL,
    run_after REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL, started_at REAL, finished_at REAL);
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at);
CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts REAL NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_logs_job ON job_logs(job_id, id);
CREATE TABLE IF NOT EXISTS products (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, name TEXT NOT NULL,
    rel_path TEXT NOT NULL, kind TEXT NOT NULL, meta_json TEXT NOT NULL DEFAULT '{}', job_id TEXT,
    created_at REAL NOT NULL, PRIMARY KEY (project_id, name));
CREATE TABLE IF NOT EXISTS control_points (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, label TEXT NOT NULL, x REAL NOT NULL, y REAL NOT NULL, z REAL,
    residual_json TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS elevation_sources (
    id INTEGER NOT NULL, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL, kind TEXT NOT NULL, rel_path TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 20,
    z_offset REAL NOT NULL DEFAULT 0, trust TEXT NOT NULL DEFAULT 'moderate', enabled INTEGER NOT NULL DEFAULT 1,
    meta_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, PRIMARY KEY (project_id, id));
CREATE TABLE IF NOT EXISTS features (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, name TEXT, geometry_json TEXT NOT NULL, props_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_features_project ON features(project_id, kind);
"""


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)       # executescript commits implicitly, so it runs outside tx()
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Write transaction (BEGIN IMMEDIATE so concurrent claimers serialise instead of deadlocking)."""
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def query(self, sql: str, args: Sequence[Any] = ()) -> List[sqlite3.Row]:
        return self._conn().execute(sql, args).fetchall()

    def one(self, sql: str, args: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        return self._conn().execute(sql, args).fetchone()

    def execute(self, sql: str, args: Sequence[Any] = ()) -> int:
        with self.tx() as c:
            return c.execute(sql, args).rowcount

    def insert(self, table: str, row: Dict[str, Any], or_ignore: bool = False) -> bool:
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
        with self.tx() as c:
            return c.execute(f"{verb} INTO {table}({cols}) VALUES({qs})", list(row.values())).rowcount > 0

    def update(self, table: str, key: Dict[str, Any], values: Dict[str, Any]) -> int:
        if not values:
            return 0
        sets = ",".join(f"{k}=?" for k in values)
        where = " AND ".join(f"{k}=?" for k in key)
        with self.tx() as c:
            return c.execute(f"UPDATE {table} SET {sets} WHERE {where}", [*values.values(), *key.values()]).rowcount

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


def sanitize(o: Any) -> Any:
    """Replace NaN/Infinity (not valid JSON) with None, recursively. Raster nodata values are often NaN."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):          # numpy scalar
        try:
            return sanitize(o.item())
        except Exception:
            return o
    return o


def dumps(o: Any) -> str:
    return json.dumps(sanitize(o), separators=(",", ":"), default=str)


def loads(s: Optional[str], default: Any = None) -> Any:
    if s is None or s == "":
        return default
    return json.loads(s)
