"""Resumable chunked upload store (framework-free).

Protocol: create -> put_chunk(index) any number of times, in any order, retried freely -> assemble.
State lives on disk, so uploads survive process restarts and browser closes; a client resumes by asking
which chunk indices were received. Chunks are written atomically and verified; a file is only released
after its SHA-256 (and, if declared, the client's SHA-256) checks out.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MAX_CHUNK_BYTES = 64 * 1024 * 1024
MIN_CHUNK_BYTES = 64 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}
_SAFE = re.compile(r"[^A-Za-z0-9._ \-()+]")


class UploadError(Exception):
    def __init__(self, message: str, code: str = "bad_request"):
        super().__init__(message)
        self.code = code


def sanitize_filename(name: str) -> str:
    """Basename only, no control characters or separators, bounded length. Never used to build paths."""
    base = os.path.basename(name.replace("\\", "/")).strip().strip(".")
    base = _SAFE.sub("_", base)
    if not base:
        raise UploadError("empty filename")
    if len(base) > 200:
        stem, dot, ext = base.rpartition(".")
        base = (stem[:190] + dot + ext[:8]) if dot else base[:200]
    return base


def sha256_file(path: Path, bufsize: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class UploadState:
    id: str
    project_id: str
    filename: str
    size: int
    chunk_size: int
    total_chunks: int
    declared_sha256: Optional[str]
    created_at: float
    batch_id: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class UploadStore:
    def __init__(self, root: Path, min_free_bytes: int = 2 * 1024 ** 3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = min_free_bytes

    def _dir(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise UploadError("invalid upload id", "not_found")
        return self.root / upload_id

    def create(self, project_id: str, filename: str, size: int, chunk_size: int,
               declared_sha256: Optional[str] = None, batch_id: Optional[str] = None,
               max_file_bytes: Optional[int] = None, allowed_ext: Optional[Iterable[str]] = None) -> UploadState:
        name = sanitize_filename(filename)
        ext = Path(name).suffix.lower()
        allowed = set(allowed_ext) if allowed_ext is not None else ALLOWED_EXTENSIONS
        if ext not in allowed:
            raise UploadError(f"unsupported file type {ext or '(none)'}; accepted: {', '.join(sorted(allowed))}", "unsupported")
        if size <= 0:
            raise UploadError("file is empty")
        if max_file_bytes is not None and size > max_file_bytes:
            raise UploadError(f"file is {size / 1e9:.2f} GB, above the configured per-file limit of "
                              f"{max_file_bytes / 1e9:.2f} GB", "too_large")
        if not MIN_CHUNK_BYTES <= chunk_size <= MAX_CHUNK_BYTES:
            raise UploadError(f"chunk_size must be between {MIN_CHUNK_BYTES} and {MAX_CHUNK_BYTES} bytes")
        if declared_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", declared_sha256):
            raise UploadError("sha256 must be 64 hex characters")
        free = shutil.disk_usage(self.root).free
        if free - size < self.min_free_bytes:
            raise UploadError(f"not enough free disk space ({free / 1e9:.1f} GB free; this file needs "
                              f"{size / 1e9:.2f} GB plus a {self.min_free_bytes / 1e9:.1f} GB reserve)", "insufficient_storage")
        st = UploadState(uuid.uuid4().hex, project_id, name, size, chunk_size, math.ceil(size / chunk_size),
                         declared_sha256.lower() if declared_sha256 else None, time.time(), batch_id)
        d = self._dir(st.id)
        (d / "chunks").mkdir(parents=True)
        self._write_state(d, st)
        return st

    @staticmethod
    def _write_state(d: Path, st: UploadState) -> None:
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(st.to_dict()))
        os.replace(tmp, d / "state.json")

    def state(self, upload_id: str) -> UploadState:
        p = self._dir(upload_id) / "state.json"
        if not p.exists():
            raise UploadError("upload not found", "not_found")
        return UploadState(**json.loads(p.read_text()))

    def received(self, upload_id: str) -> List[int]:
        d = self._dir(upload_id) / "chunks"
        if not d.exists():
            raise UploadError("upload not found", "not_found")
        return sorted(int(f.stem) for f in d.glob("*.part"))

    def expected_chunk_size(self, st: UploadState, index: int) -> int:
        return st.chunk_size if index < st.total_chunks - 1 else st.size - st.chunk_size * (st.total_chunks - 1)

    def put_chunk(self, upload_id: str, index: int, stream: Iterable[bytes],
                  expected_sha256: Optional[str] = None) -> dict:
        """Write one chunk from an iterable of byte blocks. Idempotent; verifies length and checksum."""
        st = self.state(upload_id)
        if not 0 <= index < st.total_chunks:
            raise UploadError(f"chunk index {index} out of range 0..{st.total_chunks - 1}")
        want = self.expected_chunk_size(st, index)
        d = self._dir(upload_id) / "chunks"
        fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".tmp")
        h, n = hashlib.sha256(), 0
        try:
            with os.fdopen(fd, "wb") as f:
                for block in stream:
                    n += len(block)
                    if n > want:
                        raise UploadError(f"chunk {index} is larger than the expected {want} bytes")
                    h.update(block)
                    f.write(block)
                f.flush()
                os.fsync(f.fileno())
            if n != want:
                raise UploadError(f"chunk {index} has {n} bytes, expected {want}")
            digest = h.hexdigest()
            if expected_sha256 and digest != expected_sha256.lower():
                raise UploadError(f"chunk {index} checksum mismatch; resend it", "checksum_mismatch")
            os.replace(tmp, d / f"{index:08d}.part")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return {"index": index, "sha256": digest, "received": len(self.received(upload_id)), "total": st.total_chunks}

    def status(self, upload_id: str) -> dict:
        st = self.state(upload_id)
        got = self.received(upload_id)
        return {"upload_id": st.id, "filename": st.filename, "size": st.size, "chunk_size": st.chunk_size,
                "total_chunks": st.total_chunks, "received_chunks": got,
                "missing_chunks": sorted(set(range(st.total_chunks)) - set(got)),
                "bytes_received": sum(self.expected_chunk_size(st, i) for i in got)}

    def assemble(self, upload_id: str, dest: Path) -> dict:
        """Concatenate chunks (streaming) into ``dest``; returns {sha256, size}. Fails if incomplete or corrupt."""
        st = self.state(upload_id)
        missing = sorted(set(range(st.total_chunks)) - set(self.received(upload_id)))
        if missing:
            raise UploadError(f"{len(missing)} chunk(s) missing, e.g. {missing[:5]}", "incomplete")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".assemble")
        h, total = hashlib.sha256(), 0
        try:
            with os.fdopen(fd, "wb") as out:
                for i in range(st.total_chunks):
                    with open(self._dir(upload_id) / "chunks" / f"{i:08d}.part", "rb") as c:
                        while True:
                            b = c.read(4 * 1024 * 1024)
                            if not b:
                                break
                            h.update(b)
                            total += len(b)
                            out.write(b)
                out.flush()
                os.fsync(out.fileno())
            if total != st.size:
                raise UploadError(f"assembled {total} bytes, expected {st.size}", "size_mismatch")
            digest = h.hexdigest()
            if st.declared_sha256 and digest != st.declared_sha256:
                raise UploadError("file checksum does not match the checksum declared by the client; "
                                  "the upload was corrupted in transit", "checksum_mismatch")
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return {"sha256": digest, "size": total}

    def discard(self, upload_id: str) -> None:
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    def stale(self, older_than_s: float) -> List[str]:
        now, out = time.time(), []
        for d in self.root.iterdir():
            sf = d / "state.json"
            if sf.exists() and now - json.loads(sf.read_text())["created_at"] > older_than_s:
                out.append(d.name)
        return out
