"""Application context: one object wiring settings, database, queue, storage and the upload store."""
from __future__ import annotations

from .config import Settings
from .core.uploadstore import UploadStore
from .db import Database
from .jobs import JobQueue
from .storage import LocalStorage


class AppContext:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.db = Database(self.settings.db_path)
        self.queue = JobQueue(self.db, self.settings.heavy_concurrency, self.settings.stale_job_seconds,
                              self.settings.max_attempts)
        self.storage = LocalStorage(self.settings.data_dir)
        self.uploads = UploadStore(self.settings.data_dir / "uploads", self.settings.min_free_bytes)
