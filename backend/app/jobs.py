"""Persistent job queue (database-backed; works across API and worker processes).

Guarantees
* A job is only ``succeeded`` if its handler returned normally. Any exception => ``failed`` with the reason stored.
* Jobs claimed by a worker that stops heart-beating are re-queued (or failed once attempts are exhausted), so a
  killed worker never leaves a job "running" forever and never reports it as done.
* Cancellation is cooperative (``ctx.cancelled()``); handlers that run subprocesses must poll it and terminate them.
* Heavy kinds (photogrammetry, terrain, classification) are limited separately from light ones (ingest).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .db import Database, dumps, loads, new_id, now

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}
HEAVY_KINDS = {"odm", "terrain", "classify", "train_classifier", "prepare_ortho", "export", "viewer_cloud", "classify_images", "mesh_prep", "classify_mesh"}
LIGHT_KINDS = {"ingest_image", "analyze_flights", "import_asset"}
MAX_LOG_LINES_PER_JOB = 20000


class JobFailure(Exception):
    """A failure whose message is safe and useful to show to the user."""

    def __init__(self, message: str, code: str = "failed", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class JobCancelled(Exception):
    pass


def job_row(r) -> Dict[str, Any]:
    d = dict(r)
    d["params"] = loads(d.pop("params_json"), {})
    d["result"] = loads(d.pop("result_json"), None)
    d["cancel_requested"] = bool(d["cancel_requested"])
    d["eta_seconds"] = None
    if d["status"] == RUNNING and d["started_at"] and d["progress"] and d["progress"] > 0.03:
        el = now() - d["started_at"]
        d["eta_seconds"] = max(0.0, el * (1 - d["progress"]) / d["progress"])
    return d


class JobQueue:
    def __init__(self, db: Database, heavy_concurrency: int = 1, stale_seconds: int = 90, max_attempts: int = 3):
        self.db = db
        self.heavy_concurrency = heavy_concurrency
        self.stale_seconds = stale_seconds
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------ producers
    def enqueue(self, kind: str, project_id: Optional[str], params: Optional[dict] = None, priority: int = 0,
                dedupe_key: Optional[str] = None, max_attempts: Optional[int] = None, delay: float = 0.0) -> str:
        """Create a queued job. With ``dedupe_key`` an already *queued* (not yet started) job of the same key is
        returned instead; a running job may already have read its inputs, so it never absorbs a new request."""
        params = dict(params or {})
        if dedupe_key:
            params["_dedupe"] = dedupe_key
        with self.db.tx() as c:
            if dedupe_key:
                r = c.execute("SELECT id FROM jobs WHERE kind=? AND project_id IS ? AND status='queued' "
                              "AND params_json LIKE ?", (kind, project_id, f'%"_dedupe":"{dedupe_key}"%')).fetchone()
                if r:
                    return r["id"]
            jid = new_id()
            c.execute("INSERT INTO jobs(id,project_id,kind,status,params_json,priority,max_attempts,run_after,created_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?)", (jid, project_id, kind, QUEUED, dumps(params), priority,
                                                    max_attempts or self.max_attempts, now() + delay, now()))
        return jid

    # ------------------------------------------------------------------ worker side
    def is_paused(self) -> bool:
        r = self.db.one("SELECT value FROM meta WHERE key='queue_paused'")
        return bool(r and r["value"] == "1")

    def set_paused(self, paused: bool) -> None:
        self.db.execute("INSERT INTO meta(key,value) VALUES('queue_paused',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        ("1" if paused else "0",))

    def claim(self, worker_id: str, kinds: Optional[set] = None) -> Optional[Dict[str, Any]]:
        if self.is_paused():
            return None
        t = now()
        with self.db.tx() as c:
            running_heavy = c.execute(
                "SELECT COUNT(*) n FROM jobs WHERE status='running' AND kind IN (%s)" % ",".join("?" * len(HEAVY_KINDS)),
                tuple(HEAVY_KINDS)).fetchone()["n"]
            exclude = set()
            if running_heavy >= self.heavy_concurrency:
                exclude |= HEAVY_KINDS
            allowed = kinds
            q = "SELECT id FROM jobs WHERE status='queued' AND run_after<=?"
            args: list = [t]
            if allowed:
                q += " AND kind IN (%s)" % ",".join("?" * len(allowed))
                args += list(allowed)
            if exclude:
                q += " AND kind NOT IN (%s)" % ",".join("?" * len(exclude))
                args += list(exclude)
            q += " ORDER BY priority DESC, created_at ASC LIMIT 1"
            r = c.execute(q, args).fetchone()
            if not r:
                return None
            c.execute("UPDATE jobs SET status='running', worker_id=?, started_at=?, heartbeat_at=?, attempts=attempts+1, "
                      "message='starting', error=NULL, error_code=NULL, progress=CASE WHEN attempts=0 THEN 0 ELSE progress END "
                      "WHERE id=?", (worker_id, t, t, r["id"]))
            row = c.execute("SELECT * FROM jobs WHERE id=?", (r["id"],)).fetchone()
        return job_row(row)

    def heartbeat(self, job_id: str, progress: Optional[float] = None, message: Optional[str] = None) -> None:
        sets, args = ["heartbeat_at=?"], [now()]
        if progress is not None:
            sets.append("progress=?")
            args.append(min(max(float(progress), 0.0), 1.0))
        if message is not None:
            sets.append("message=?")
            args.append(message[:500])
        self.db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=? AND status='running'", [*args, job_id])

    def is_cancel_requested(self, job_id: str) -> bool:
        r = self.db.one("SELECT cancel_requested, status FROM jobs WHERE id=?", (job_id,))
        return bool(r and (r["cancel_requested"] or r["status"] == CANCELLED))

    def log(self, job_id: str, message: str, level: str = "info") -> None:
        with self.db.tx() as c:
            n = c.execute("SELECT COUNT(*) n FROM job_logs WHERE job_id=?", (job_id,)).fetchone()["n"]
            if n < MAX_LOG_LINES_PER_JOB:
                c.execute("INSERT INTO job_logs(job_id,ts,level,message) VALUES(?,?,?,?)", (job_id, now(), level, message[:4000]))
            elif n == MAX_LOG_LINES_PER_JOB:
                c.execute("INSERT INTO job_logs(job_id,ts,level,message) VALUES(?,?,?,?)",
                          (job_id, now(), "warning", "log limit reached; further lines dropped"))

    def succeed(self, job_id: str, result: Optional[dict] = None) -> None:
        self.db.execute("UPDATE jobs SET status='succeeded', progress=1, message='done', result_json=?, finished_at=?, "
                        "error=NULL, error_code=NULL WHERE id=? AND status='running'", (dumps(result or {}), now(), job_id))

    def fail(self, job_id: str, error: str, code: str = "failed", retryable: bool = False) -> str:
        """Mark failed, or re-queue with back-off if retryable and attempts remain. Returns the new status."""
        with self.db.tx() as c:
            r = c.execute("SELECT attempts, max_attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
            if r and retryable and r["attempts"] < r["max_attempts"]:
                delay = min(300, 5 * 2 ** (r["attempts"] - 1))
                c.execute("UPDATE jobs SET status='queued', run_after=?, message=?, worker_id=NULL WHERE id=?",
                          (now() + delay, f"retrying after error: {error[:200]}", job_id))
                return QUEUED
            c.execute("UPDATE jobs SET status='failed', error=?, error_code=?, message='failed', finished_at=? WHERE id=?",
                      (error[:4000], code, now(), job_id))
        return FAILED

    def mark_cancelled(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET status='cancelled', message='cancelled', finished_at=? WHERE id=?", (now(), job_id))

    # ------------------------------------------------------------------ operators
    def cancel(self, job_id: str) -> str:
        with self.db.tx() as c:
            r = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not r:
                raise KeyError(job_id)
            if r["status"] == QUEUED:
                c.execute("UPDATE jobs SET status='cancelled', message='cancelled before start', finished_at=? WHERE id=?",
                          (now(), job_id))
                return CANCELLED
            if r["status"] == RUNNING:
                c.execute("UPDATE jobs SET cancel_requested=1, message='cancelling...' WHERE id=?", (job_id,))
                return "cancelling"
            return r["status"]

    def retry(self, job_id: str) -> str:
        """Re-queue a failed/cancelled job (keeps params, logs and any partial outputs the handler can resume from)."""
        with self.db.tx() as c:
            r = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not r:
                raise KeyError(job_id)
            if r["status"] not in (FAILED, CANCELLED):
                raise ValueError(f"only failed or cancelled jobs can be retried (job is {r['status']})")
            c.execute("UPDATE jobs SET status='queued', cancel_requested=0, error=NULL, error_code=NULL, attempts=0, "
                      "run_after=0, finished_at=NULL, message='re-queued by user', worker_id=NULL WHERE id=?", (job_id,))
        return job_id

    def recover_stale(self) -> List[str]:
        """Re-queue (or fail) running jobs whose worker stopped heart-beating. Safe to call from any process."""
        cutoff = now() - self.stale_seconds
        out = []
        with self.db.tx() as c:
            for r in c.execute("SELECT id, attempts, max_attempts, cancel_requested FROM jobs WHERE status='running' "
                               "AND heartbeat_at < ?", (cutoff,)).fetchall():
                if r["cancel_requested"]:
                    c.execute("UPDATE jobs SET status='cancelled', message='cancelled (worker lost)', finished_at=? WHERE id=?",
                              (now(), r["id"]))
                elif r["attempts"] < r["max_attempts"]:
                    c.execute("UPDATE jobs SET status='queued', worker_id=NULL, message='worker lost; re-queued' WHERE id=?", (r["id"],))
                else:
                    c.execute("UPDATE jobs SET status='failed', error='worker stopped responding and the job has used all "
                              "its attempts', error_code='worker_lost', finished_at=?, message='failed' WHERE id=?", (now(), r["id"]))
                c.execute("INSERT INTO job_logs(job_id,ts,level,message) VALUES(?,?,?,?)",
                          (r["id"], now(), "warning", "worker stopped responding; job recovered by queue monitor"))
                out.append(r["id"])
        return out

    # ------------------------------------------------------------------ queries
    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        r = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return job_row(r) if r else None

    def list(self, project_id: Optional[str] = None, kinds: Optional[List[str]] = None, exclude_kinds: Optional[List[str]] = None,
             status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        q, a = "SELECT * FROM jobs WHERE 1=1", []
        if project_id:
            q += " AND project_id=?"
            a.append(project_id)
        if kinds:
            q += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            a += kinds
        if exclude_kinds:
            q += " AND kind NOT IN (%s)" % ",".join("?" * len(exclude_kinds))
            a += exclude_kinds
        if status:
            q += " AND status=?"
            a.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        a.append(max(1, min(limit, 1000)))
        return [job_row(r) for r in self.db.query(q, a)]

    def logs(self, job_id: str, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT id, ts, level, message FROM job_logs WHERE job_id=? AND id>? "
                                               "ORDER BY id LIMIT ?", (job_id, after, min(limit, 5000)))]

    def stats(self) -> Dict[str, Any]:
        by = {r["status"]: r["n"] for r in self.db.query("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
        kinds = {f"{r['kind']}:{r['status']}": r["n"] for r in self.db.query(
            "SELECT kind, status, COUNT(*) n FROM jobs WHERE status IN ('queued','running') GROUP BY kind, status")}
        return {"by_status": by, "active_by_kind": kinds, "paused": self.is_paused()}


@dataclass
class JobContext:
    """Passed to handlers. Throttles DB writes so tight loops can call ``progress`` freely."""
    queue: JobQueue
    job: Dict[str, Any]
    _last: float = 0.0

    @property
    def id(self) -> str:
        return self.job["id"]

    @property
    def params(self) -> dict:
        return self.job["params"]

    def log(self, msg: str, level: str = "info") -> None:
        self.queue.log(self.id, msg, level)

    def progress(self, frac: float, msg: Optional[str] = None) -> None:
        t = time.monotonic()
        if t - self._last >= 0.5 or frac >= 1.0:
            self._last = t
            self.queue.heartbeat(self.id, frac, msg)

    def cancelled(self) -> bool:
        return self.queue.is_cancel_requested(self.id)

    def check_cancel(self) -> None:
        if self.cancelled():
            raise JobCancelled()
