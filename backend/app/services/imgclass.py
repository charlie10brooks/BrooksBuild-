"""Per-photo land-cover classification of the ORIGINAL drone images.

This classifies each uploaded photo on its own, at a reduced working size (default 1600 px on the long side) so large batches stay tractable.
It is deliberately separate from the orthomosaic classification editor:

* the masks are in each photo's own pixel grid. Photos are not orthorectified, so they are NOT georeferenced, and nothing here feeds the
  orthomosaic mask or any elevation product;
* the same replaceable models are used (heuristic / learned / ONNX). The heuristic is colour rules, not a trained network, and it assumes
  roughly nadir photos: sky, horizon haze or strongly oblique views can be misread (blue sky looks like water to colour rules).

Mask file: PNG, BGR channels = (confidence 0..100, 0, class id), i.e. R = class id, matching the orthomosaic mask-tile layout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..context import AppContext
from ..core import classes as C
from ..db import dumps, loads
from ..jobs import JobContext, JobFailure
from . import classify as K

DIR = "imgclass"


def mask_path(app: AppContext, pid: str, iid: str) -> Path:
    return app.storage.path(pid, f"{DIR}/{iid}.png")


def load_rgb(path: Path, max_side: int):
    """RGB uint8 array reduced so its long side is <= max_side; returns (array, scale = working/original)."""
    with Image.open(path) as im:
        w0, h0 = im.size
        if im.format == "JPEG":
            im.draft("RGB", (max_side, max_side))
        im = im.convert("RGB")
        s = min(1.0, max_side / max(im.size))
        if s < 1.0:
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        return np.asarray(im), im.width / w0


def predict_tiled(model, rgb: np.ndarray):
    """Run `model` over the image, tiled with context overlap when the image is bigger than the model's preferred tile."""
    h, w = rgb.shape[:2]
    t, ctx = int(getattr(model, "tile_px", 1024)), int(getattr(model, "context_px", 0))
    if max(h, w) <= t + 2 * ctx:
        cls, conf = model.predict(rgb)
        return cls.astype(np.uint8), conf.astype(np.float32)
    cls_out = np.zeros((h, w), np.uint8)
    conf_out = np.zeros((h, w), np.float32)
    for y in range(0, h, t):
        for x in range(0, w, t):
            y0, x0, y1, x1 = max(0, y - ctx), max(0, x - ctx), min(h, y + t + ctx), min(w, x + t + ctx)
            c, f = model.predict(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            ys, xs = y - y0, x - x0
            hh, ww = min(t, h - y), min(t, w - x)
            cls_out[y:y + hh, x:x + ww] = c[ys:ys + hh, xs:xs + ww]
            conf_out[y:y + hh, x:x + ww] = f[ys:ys + hh, xs:xs + ww]
    return cls_out, conf_out


def classify_one(app: AppContext, pid: str, row, model, model_id: str, max_side: int, job_id: Optional[str]) -> dict:
    rgb, scale = load_rgb(app.storage.path(pid, row["rel_path"]), max_side)
    cls, conf = predict_tiled(model, rgb)
    h, w = cls.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[..., 0] = np.clip(np.round(conf * 100), 0, 100).astype(np.uint8)
    out[..., 2] = cls
    dst = mask_path(app, pid, row["id"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    tmp.replace(dst)
    hist = np.bincount(cls.ravel(), minlength=256)
    counts = {str(i): int(n) for i, n in enumerate(hist) if n}
    known = cls > 0
    rec = {"image_id": row["id"], "project_id": pid, "model": model_id, "width": w, "height": h, "scale": float(scale),
           "counts_json": dumps(counts), "unclassified": float(1.0 - known.mean()),
           "mean_conf": float(conf[known].mean()) if known.any() else None, "job_id": job_id, "created_at": time.time()}
    with app.db.tx() as c:
        c.execute("INSERT OR REPLACE INTO image_classification(image_id,project_id,model,width,height,scale,counts_json,unclassified,mean_conf,job_id,created_at) "
                  "VALUES(:image_id,:project_id,:model,:width,:height,:scale,:counts_json,:unclassified,:mean_conf,:job_id,:created_at)", rec)
    return rec


def target_images(app: AppContext, pid: str, scope: str, batch_id: Optional[str], ids: Optional[List[str]], overwrite: bool) -> list:
    where, args = ["i.project_id=?"], [pid]
    if scope == "batch":
        where.append("i.batch_id=?"); args.append(batch_id)
    elif scope == "ids":
        if not ids:
            return []
        where.append(f"i.id IN ({','.join('?' * len(ids))})"); args += list(ids)
    if not overwrite:
        where.append("i.id NOT IN (SELECT image_id FROM image_classification)")
    return app.db.query(f"SELECT i.* FROM images i WHERE {' AND '.join(where)} ORDER BY i.captured_at, i.filename", args)


def handle_classify_images(app: AppContext, ctx: JobContext) -> dict:
    p = ctx.params
    pid = p["project_id"]
    model_id = p.get("model", "heuristic")
    model = K.load_model(app, pid, model_id)
    ctx.log(f"model: {model.name} - {model.description}")
    rows = target_images(app, pid, p.get("scope", "all"), p.get("batch_id"), p.get("ids"), bool(p.get("overwrite", False)))
    if not rows:
        return {"classified": 0, "failed": 0, "note": "nothing to classify"}
    max_side = int(p.get("max_side", 1600))
    done = failed = 0
    for n, row in enumerate(rows):
        ctx.check_cancel()
        try:
            classify_one(app, pid, row, model, model_id, max_side, ctx.id)
            done += 1
        except Exception as e:                           # one bad photo must not sink the batch
            failed += 1
            ctx.log(f"{row['filename']}: {e}", "warning")
        ctx.progress((n + 1) / len(rows), f"{n + 1}/{len(rows)} photos")
    if done == 0:
        raise JobFailure(f"none of the {len(rows)} photos could be classified (see log)", "all_failed")
    ctx.log(f"classified {done} photo(s), {failed} failed")
    return {"classified": done, "failed": failed, "model": model_id, "max_side": max_side}


def summary(app: AppContext, pid: str) -> dict:
    total = app.db.one("SELECT COUNT(*) n FROM images WHERE project_id=?", (pid,))["n"]
    rows = app.db.query("SELECT counts_json, model FROM image_classification WHERE project_id=?", (pid,))
    agg: Dict[str, int] = {}
    models: Dict[str, int] = {}
    for r in rows:
        models[r["model"]] = models.get(r["model"], 0) + 1
        for k, v in loads(r["counts_json"], {}).items():
            agg[k] = agg.get(k, 0) + v
    px = sum(agg.values())
    return {"images": total, "classified": len(rows), "by_model": models,
            "class_fractions": {k: v / px for k, v in sorted(agg.items(), key=lambda kv: int(kv[0]))} if px else {},
            "note": "Per-photo labels in each photo's own pixels. Not georeferenced, not elevation evidence."}


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["classify_images"] = handle_classify_images
