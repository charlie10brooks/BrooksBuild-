"""Storage abstraction.

Every project lives under ``<data>/projects/<project_id>/``. All paths handed to callers are resolved through
``Storage.path`` which refuses traversal outside the project directory (project-level isolation).

Native tools (PDAL, ODM, tifffile) need real file paths, so workers always operate on a *local* working copy. The
``S3Storage`` placeholder documents the intended extension point; it is **not implemented**.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterator, Protocol

_PID = re.compile(r"^[0-9a-f]{32}$")


class StorageError(Exception):
    pass


class Storage(Protocol):
    def project_root(self, project_id: str) -> Path: ...
    def path(self, project_id: str, rel: str) -> Path: ...
    def usage(self, project_id: str) -> int: ...
    def delete_project(self, project_id: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root) / "projects"
        self.root.mkdir(parents=True, exist_ok=True)

    def project_root(self, project_id: str) -> Path:
        if not _PID.match(project_id or ""):
            raise StorageError("invalid project id")
        return self.root / project_id

    def path(self, project_id: str, rel: str) -> Path:
        base = self.project_root(project_id).resolve()
        p = (base / rel).resolve()
        if p != base and base not in p.parents:
            raise StorageError("path escapes the project directory")
        return p

    def ensure(self, project_id: str, rel: str = "") -> Path:
        p = self.path(project_id, rel)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, project_id: str, rel: str) -> bool:
        return self.path(project_id, rel).exists()

    def usage(self, project_id: str) -> int:
        total = 0
        base = self.project_root(project_id)
        if not base.exists():
            return 0
        for dp, _, files in os.walk(base):
            for f in files:
                try:
                    total += os.lstat(os.path.join(dp, f)).st_size
                except OSError:
                    pass
        return total

    def iter_files(self, project_id: str, rel: str = "") -> Iterator[Path]:
        for dp, _, files in os.walk(self.path(project_id, rel)):
            for f in files:
                yield Path(dp) / f

    def delete_project(self, project_id: str) -> None:
        shutil.rmtree(self.project_root(project_id), ignore_errors=True)


class S3Storage:  # pragma: no cover - documented placeholder
    """NOT IMPLEMENTED. Intended: originals/exports in an S3-compatible bucket, workers staging local copies."""

    def __init__(self, *a, **k):
        raise NotImplementedError("S3-compatible storage is not implemented yet; use LocalStorage. See docs/ARCHITECTURE.md")
