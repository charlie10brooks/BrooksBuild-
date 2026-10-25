"""Runtime configuration, read from environment variables (prefix ``TERRASCOPE_``).

There is deliberately **no limit on the number of images** per project. Limits that exist are byte/resource based and
all configurable; 0 means "no limit".
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    v = os.environ.get(f"TERRASCOPE_{name}")
    return int(v) if v not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(f"TERRASCOPE_{name}")
    return default if v in (None, "") else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("TERRASCOPE_DATA_DIR", "./data")).resolve())
    # byte-based safeguards (0 = unlimited)
    max_file_bytes: int = field(default_factory=lambda: _int("MAX_FILE_BYTES", 8 * 1024 ** 3))
    project_quota_bytes: int = field(default_factory=lambda: _int("PROJECT_QUOTA_BYTES", 0))
    min_free_bytes: int = field(default_factory=lambda: _int("MIN_FREE_BYTES", 2 * 1024 ** 3))
    max_image_megapixels: int = field(default_factory=lambda: _int("MAX_IMAGE_MEGAPIXELS", 400))
    max_request_body_bytes: int = field(default_factory=lambda: _int("MAX_CHUNK_BYTES", 64 * 1024 ** 2))
    # job system
    worker_concurrency: int = field(default_factory=lambda: _int("WORKER_CONCURRENCY", 2))
    heavy_concurrency: int = field(default_factory=lambda: _int("HEAVY_CONCURRENCY", 1))
    stale_job_seconds: int = field(default_factory=lambda: _int("STALE_JOB_SECONDS", 90))
    heartbeat_seconds: int = field(default_factory=lambda: _int("HEARTBEAT_SECONDS", 10))
    max_attempts: int = field(default_factory=lambda: _int("MAX_ATTEMPTS", 3))
    poll_seconds: float = 0.5
    stale_upload_seconds: int = field(default_factory=lambda: _int("STALE_UPLOAD_SECONDS", 7 * 24 * 3600))
    # external tools
    pdal_bin: str = field(default_factory=lambda: os.environ.get("TERRASCOPE_PDAL_BIN", "pdal"))
    docker_bin: str = field(default_factory=lambda: os.environ.get("TERRASCOPE_DOCKER_BIN", "docker"))
    odm_image: str = field(default_factory=lambda: os.environ.get("TERRASCOPE_ODM_IMAGE", "opendronemap/odm:latest"))
    odm_extra_args: str = field(default_factory=lambda: os.environ.get("TERRASCOPE_ODM_ARGS", ""))
    dev_mode: bool = field(default_factory=lambda: _bool("DEV_MODE", False))
    model_path: str = field(default_factory=lambda: os.environ.get("TERRASCOPE_SEGMENTATION_MODEL", ""))

    def ensure_dirs(self) -> None:
        for sub in ("projects", "uploads", "tmp"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "terrascope.sqlite3"
