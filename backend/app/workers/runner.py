"""Worker process: ``python -m app.workers.runner``.

Runs ``worker_concurrency`` threads that claim jobs from the database. A heartbeat thread keeps claimed jobs alive and a
monitor re-queues jobs from dead workers. Killing this process at any time is safe: its jobs are recovered.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from typing import Optional

from ..config import Settings
from ..context import AppContext
from ..jobs import JobCancelled, JobContext, JobFailure
from ..services.groundfilter import Cancelled as GroundCancelled, GroundFilterError
from ..services.las import MissingDependency
from .handlers import HANDLERS, load_all

log = logging.getLogger("terrascope.worker")


class Worker:
    def __init__(self, app: AppContext, worker_id: Optional[str] = None):
        self.app = app
        self.id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.stop = threading.Event()
        self._active: dict = {}
        self._lock = threading.Lock()
        load_all()

    # -------------------------------------------------------------- single job
    def run_one(self, kinds=None) -> bool:
        job = self.app.queue.claim(self.id, kinds)
        if not job:
            return False
        q = self.app.queue
        ctx = JobContext(q, job)
        with self._lock:
            self._active[job["id"]] = time.monotonic()
        ctx.log(f"started on {self.id} (attempt {job['attempts']})")
        handler = HANDLERS.get(job["kind"])
        try:
            if handler is None:
                raise JobFailure(f"no handler registered for job kind {job['kind']!r}", "no_handler")
            result = handler(self.app, ctx)
            if ctx.cancelled():
                q.mark_cancelled(job["id"])
                ctx.log("cancelled")
            else:
                q.succeed(job["id"], result if isinstance(result, dict) else {"result": result})
                ctx.log("finished")
        except (JobCancelled, GroundCancelled):
            q.mark_cancelled(job["id"])
            ctx.log("cancelled by user")
        except JobFailure as e:
            ctx.log(str(e), "error")
            st = q.fail(job["id"], str(e), e.code, e.retryable)
            if st == "queued":
                ctx.log("will retry automatically", "warning")
        except MissingDependency as e:
            ctx.log(str(e), "error")
            q.fail(job["id"], str(e), "missing_dependency", False)
        except GroundFilterError as e:
            ctx.log(str(e), "error")
            q.fail(job["id"], str(e), "processing_error", False)
        except Exception as e:  # unexpected: record traceback, mark failed (never "succeeded")
            tb = traceback.format_exc()
            ctx.log(tb, "error")
            q.fail(job["id"], f"{type(e).__name__}: {e}", "internal_error", retryable=isinstance(e, (OSError, TimeoutError)))
        finally:
            with self._lock:
                self._active.pop(job["id"], None)
        return True

    # -------------------------------------------------------------- loops
    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.app.settings.heartbeat_seconds):
            with self._lock:
                ids = list(self._active)
            for jid in ids:
                self.app.queue.heartbeat(jid)

    def _monitor_loop(self) -> None:
        while not self.stop.wait(max(5, self.app.settings.stale_job_seconds // 3)):
            try:
                rec = self.app.queue.recover_stale()
                if rec:
                    log.warning("recovered %d stale job(s)", len(rec))
            except Exception:
                log.exception("monitor failed")

    def _thread_loop(self) -> None:
        while not self.stop.is_set():
            try:
                if not self.run_one():
                    self.stop.wait(self.app.settings.poll_seconds)
            except Exception:
                log.exception("worker loop error")
                self.stop.wait(2)

    def run_forever(self) -> None:
        self.app.queue.recover_stale()
        ts = [threading.Thread(target=self._heartbeat_loop, daemon=True), threading.Thread(target=self._monitor_loop, daemon=True)]
        ts += [threading.Thread(target=self._thread_loop, daemon=True) for _ in range(max(1, self.app.settings.worker_concurrency))]
        for t in ts:
            t.start()
        log.info("worker %s up with %d slots", self.id, self.app.settings.worker_concurrency)
        try:
            while not self.stop.is_set():
                self.stop.wait(1)
        finally:
            self.stop.set()
            for t in ts:
                t.join(timeout=5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    w = Worker(AppContext(Settings()))
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: w.stop.set())
    w.run_forever()


if __name__ == "__main__":
    main()
