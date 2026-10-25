"""Image ingestion: runs in a worker, never in an HTTP request.

assemble chunks -> checksum -> duplicate check -> validate/parse metadata -> thumbnail -> persist.
Every outcome is recorded in ``upload_files``; nothing is discarded silently (rejected files are kept in ``rejected/``).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image

from ..context import AppContext
from ..core import exif
from ..core.uploadstore import UploadError
from ..db import dumps, new_id, now
from ..jobs import JobContext, JobFailure

THUMB_PX = 320


def make_thumbnail(src: Path, dst: Path, size: int = THUMB_PX) -> None:
    prev = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (size * 2, size * 2))      # decode at reduced size: no full-resolution load
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                import numpy as np
                a = np.asarray(im).astype("float32")
                hi = a.max() or 1.0
                im = Image.fromarray((a / hi * 255).astype("uint8"))
            elif im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail((size, size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").save(dst, "JPEG", quality=80)
    finally:
        Image.MAX_IMAGE_PIXELS = prev


def _set_status(app: AppContext, upload_id: str, status: str, error: Optional[str] = None, **extra) -> None:
    app.db.update("upload_files", {"upload_id": upload_id},
                  {"status": status, "error": error, "updated_at": now(), **extra})


def ingest_image(app: AppContext, ctx: JobContext) -> dict:
    p = ctx.params
    upload_id, pid = p["upload_id"], p["project_id"]
    row = app.db.one("SELECT * FROM upload_files WHERE upload_id=?", (upload_id,))
    if row is None:
        raise JobFailure("upload record not found", "not_found")
    if row["status"] in ("complete", "duplicate", "rejected"):
        ctx.log(f"already processed ({row['status']}); nothing to do")
        return {"status": row["status"]}
    s = app.settings
    root = app.storage.ensure(pid)
    incoming = app.storage.ensure(pid, "incoming")
    tmp_file = incoming / f"{upload_id}.assembled"

    # quota check before spending I/O
    if s.project_quota_bytes:
        used = app.storage.usage(pid)
        if used + row["size"] > s.project_quota_bytes:
            msg = (f"project storage quota exceeded ({used / 1e9:.2f} GB used + {row['size'] / 1e9:.2f} GB new > "
                   f"{s.project_quota_bytes / 1e9:.2f} GB). Raise TERRASCOPE_PROJECT_QUOTA_BYTES or free space.")
            _set_status(app, upload_id, "failed", msg)
            raise JobFailure(msg, "quota_exceeded")
    try:
        info = app.uploads.assemble(upload_id, tmp_file)
    except UploadError as e:
        _set_status(app, upload_id, "failed", str(e))
        raise JobFailure(str(e), e.code) from e
    ctx.log(f"assembled {row['filename']} ({info['size']} bytes) sha256={info['sha256'][:12]}")

    dup = app.db.one("SELECT id, filename FROM images WHERE project_id=? AND sha256=?", (pid, info["sha256"]))
    if dup:
        tmp_file.unlink(missing_ok=True)
        app.uploads.discard(upload_id)
        _set_status(app, upload_id, "duplicate", f"identical to already uploaded file {dup['filename']}", duplicate_of=dup["id"])
        ctx.log(f"duplicate of {dup['filename']}", "warning")
        return {"status": "duplicate", "duplicate_of": dup["id"]}

    try:
        meta = exif.read_image_metadata(tmp_file, s.max_image_megapixels * 1_000_000)
    except ValueError as e:
        rej = app.storage.ensure(pid, "rejected")
        os.replace(tmp_file, rej / f"{upload_id}_{row['filename']}")
        app.uploads.discard(upload_id)
        _set_status(app, upload_id, "rejected", str(e))
        ctx.log(f"rejected: {e} (file kept in rejected/)", "error")
        return {"status": "rejected", "error": str(e)}

    image_id = new_id()
    rel = f"images/{image_id}_{row['filename']}"
    dest = app.storage.path(pid, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_file, dest)
    thumb_rel = f"thumbs/{image_id}.jpg"
    try:
        make_thumbnail(dest, app.storage.path(pid, thumb_rel))
    except Exception as e:  # thumbnail problems must not lose the image
        ctx.log(f"thumbnail failed: {e}", "warning")
        thumb_rel = None
    geo = exif.georef_assessment(meta)
    inserted = app.db.insert("images", {
        "id": image_id, "project_id": pid, "batch_id": row["batch_id"], "filename": row["filename"], "sha256": info["sha256"],
        "size": info["size"], "rel_path": rel, "thumb_path": thumb_rel, "width": meta["width"], "height": meta["height"],
        "lat": meta["lat"], "lon": meta["lon"], "alt_msl": meta["alt_msl"], "rel_alt": meta["rel_alt"],
        "captured_at": meta["captured_at"], "make": meta["make"], "model": meta["model"],
        "georef_status": geo["status"], "issues_json": dumps(geo["issues"]), "meta_json": dumps(meta), "created_at": now(),
    }, or_ignore=True)
    if not inserted:     # lost a race with an identical upload in another worker
        dest.unlink(missing_ok=True)
        app.uploads.discard(upload_id)
        _set_status(app, upload_id, "duplicate", "identical file was ingested concurrently")
        return {"status": "duplicate"}
    app.uploads.discard(upload_id)
    _set_status(app, upload_id, "complete", None, image_id=image_id)
    app.queue.enqueue("analyze_flights", pid, {"project_id": pid}, dedupe_key=f"analyze:{pid}", delay=2.0)
    return {"status": "complete", "image_id": image_id, "georef": geo["status"]}
