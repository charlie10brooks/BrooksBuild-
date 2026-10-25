"""Manual editing of the point-cloud DISPLAY SAMPLE (reclassify points, mark noise), with persistent undo/redo.

What this edits, exactly: the ~1.5 M point sample that the 3D viewer shows (``terrain/viewer_cloud*.npz``), not the full photogrammetry
cloud and not any elevation raster. A manual edit therefore does NOT change the DSM/DTM/confidence products; re-running terrain
reconstruction starts from the full cloud again. Edits can be exported as an edited-sample LAS.

Storage (``terrain/cloud_edits/``): ``state.npy`` (uint8 per sample point, 255 = unedited), ``ops/<k>.npz`` (idx, before, after) and
``meta.json`` (fingerprint of the sample file, number of ops, head). Undo/redo move ``head``; a new edit drops the redo tail.
If the sample file is rebuilt (different fingerprint) old edits are NOT applied, because the point order would no longer match.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..context import AppContext

NONE = 255
# ASPRS classes the viewer knows; 0 "never classified" is not offered as an edit target.
EDIT_CLASSES = {1: "unclassified", 2: "ground", 3: "low vegetation", 4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "noise"}
_LOCK = threading.Lock()


class CloudEditError(ValueError):
    def __init__(self, msg: str, code: str = "bad_edit", status: int = 400):
        super().__init__(msg)
        self.code, self.status = code, status


def sample_file(app: AppContext, pid: str) -> Tuple[Optional[Path], Optional[str]]:
    w = app.storage.path(pid, "terrain")
    for n, kind in (("viewer_cloud.npz", "classified"), ("viewer_cloud_raw.npz", "raw")):
        if (w / n).exists():
            return w / n, kind
    return None, None


def _fingerprint(f: Path) -> str:
    st = f.stat()
    return f"{f.name}:{st.st_size}:{int(st.st_mtime)}"


def _dir(app: AppContext, pid: str) -> Path:
    return app.storage.path(pid, "terrain/cloud_edits")


def _count(f: Path) -> int:
    with np.load(f, allow_pickle=False) as z:
        return int(len(z["x"]))


def _load(app: AppContext, pid: str):
    """(file, kind, dir, meta) with meta reset when the sample was rebuilt since the edits were made."""
    f, kind = sample_file(app, pid)
    if f is None:
        raise CloudEditError("There is no point-cloud display sample to edit yet. Build one first.", "no_cloud", 409)
    d = _dir(app, pid)
    fp = _fingerprint(f)
    mp = d / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else None
    stale = bool(meta and meta.get("fingerprint") != fp)
    if meta is None or stale:
        meta = {"fingerprint": fp, "n": _count(f), "ops": 0, "head": 0, "stale_dropped": bool(stale)}
    return f, kind, d, meta, stale


def _write_meta(d: Path, meta: dict) -> None:
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, d / "meta.json")


def _state(d: Path, n: int) -> np.ndarray:
    p = d / "state.npy"
    if p.exists():
        a = np.load(p)
        if len(a) == n:
            return a
    return np.full(n, NONE, np.uint8)


def _save_state(d: Path, a: np.ndarray) -> None:
    tmp = d / "state.tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, d / "state.npy")


def _reset_dir(d: Path) -> None:
    if d.exists():
        for p in (d / "ops").glob("*.npz") if (d / "ops").exists() else []:
            p.unlink()
        for n in ("state.npy", "meta.json"):
            (d / n).unlink(missing_ok=True)


def info(app: AppContext, pid: str) -> dict:
    with _LOCK:
        try:
            f, kind, d, meta, stale = _load(app, pid)
        except CloudEditError as e:
            return {"available": False, "reason": str(e)}
        st = _state(d, meta["n"]) if not stale and (d / "state.npy").exists() else None
        return {"available": True, "kind": kind, "points": meta["n"], "head": meta["head"], "ops": meta["ops"],
                "can_undo": meta["head"] > 0, "can_redo": meta["head"] < meta["ops"],
                "edited_points": int((st != NONE).sum()) if st is not None else 0, "stale_dropped": bool(stale),
                "classes": {str(k): v for k, v in EDIT_CLASSES.items()}}


def apply_overrides(app: AppContext, pid: str, cls: np.ndarray) -> np.ndarray:
    """Return `cls` with the current edits applied (no-op when there are none or they belong to an older sample)."""
    with _LOCK:
        try:
            f, _, d, meta, stale = _load(app, pid)
        except CloudEditError:
            return cls
        if stale or meta["head"] == 0 or not (d / "state.npy").exists():
            return cls
        st = _state(d, len(cls))
        return np.where(st != NONE, st, cls).astype(np.uint8)


def edit(app: AppContext, pid: str, idx: np.ndarray, new_cls: int) -> dict:
    if new_cls not in EDIT_CLASSES:
        raise CloudEditError(f"class {new_cls} cannot be assigned; use one of {sorted(EDIT_CLASSES)}", "bad_class")
    with _LOCK:
        f, _, d, meta, stale = _load(app, pid)
        n = meta["n"]
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        if idx.size == 0:
            raise CloudEditError("no points selected", "empty_selection")
        if idx[0] < 0 or idx[-1] >= n:
            raise CloudEditError("selection refers to points outside the display sample; reload the viewer", "bad_index")
        if stale:
            _reset_dir(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ops").mkdir(exist_ok=True)
        with np.load(f, allow_pickle=False) as z:
            base = z["cls"]
            st = _state(d, n)
            cur = np.where(st != NONE, st, base)[idx]
        change = cur != new_cls
        if not change.any():
            return {**_info_from(meta, st), "changed": 0}
        idx, before = idx[change], cur[change]
        for k in range(meta["head"], meta["ops"]):                    # drop the redo tail
            (d / "ops" / f"{k}.npz").unlink(missing_ok=True)
        np.savez_compressed(d / "ops" / f"{meta['head']}.npz", idx=idx.astype(np.uint32), before=before.astype(np.uint8),
                            after=np.full(len(idx), new_cls, np.uint8))
        st[idx] = new_cls
        meta["head"] += 1
        meta["ops"] = meta["head"]
        _save_state(d, st)
        _write_meta(d, meta)
        return {**_info_from(meta, st), "changed": int(len(idx))}


def _info_from(meta: dict, st: np.ndarray) -> dict:
    return {"head": meta["head"], "ops": meta["ops"], "can_undo": meta["head"] > 0, "can_redo": meta["head"] < meta["ops"],
            "edited_points": int((st != NONE).sum())}


def _step(app: AppContext, pid: str, direction: int) -> dict:
    with _LOCK:
        f, _, d, meta, stale = _load(app, pid)
        if stale or not (d / "meta.json").exists():
            raise CloudEditError("nothing to undo", "nothing", 409)
        st = _state(d, meta["n"])
        if direction < 0:
            if meta["head"] == 0:
                raise CloudEditError("nothing to undo", "nothing", 409)
            k = meta["head"] - 1
            z = np.load(d / "ops" / f"{k}.npz")
            idx, val = z["idx"].astype(np.int64), z["before"]
            with np.load(f, allow_pickle=False) as c:
                base = c["cls"][idx]
            st[idx] = np.where(val == base, NONE, val)               # back to "unedited" when it equals the original
            meta["head"] = k
        else:
            if meta["head"] >= meta["ops"]:
                raise CloudEditError("nothing to redo", "nothing", 409)
            k = meta["head"]
            z = np.load(d / "ops" / f"{k}.npz")
            st[z["idx"].astype(np.int64)] = z["after"]
            meta["head"] = k + 1
        _save_state(d, st)
        _write_meta(d, meta)
        return {**_info_from(meta, st), "idx": z["idx"].tolist() if len(z["idx"]) <= 200000 else None,
                "value": None, "direction": direction}


def undo(app, pid):
    return _step(app, pid, -1)


def redo(app, pid):
    return _step(app, pid, +1)


def reset(app: AppContext, pid: str) -> dict:
    with _LOCK:
        _reset_dir(_dir(app, pid))
    return {"head": 0, "ops": 0, "can_undo": False, "can_redo": False, "edited_points": 0}


def write_edited_las(app: AppContext, pid: str, dst: Path) -> dict:
    """Edited display sample as LAS (with the edited classes). It is a SAMPLE, so the point count is far below the full cloud."""
    from . import las as L
    f, kind = sample_file(app, pid)
    if f is None:
        raise CloudEditError("no display sample", "no_cloud", 409)
    with np.load(f, allow_pickle=False) as z:
        origin, epsg, total = z["origin"], int(z["epsg"]) or None, int(z["total"])
        cls = apply_overrides(app, pid, z["cls"])
        rgb = (z["rgb"].astype(np.uint16) * 257) if "rgb" in z.files else None
        x, y, zz = (z[k].astype(np.float64) + origin[i] for i, k in enumerate(("x", "y", "z")))
        n = len(x)
    L.write_las(dst, x, y, zz, cls, rgb, epsg, offset=(float(np.floor(origin[0])), float(np.floor(origin[1])), float(np.floor(origin[2]))))
    return {"points": n, "total_points": total, "epsg": epsg, "kind": kind}


def edited_surface(app: AppContext, pid: str, maxdim: int = 400, exclude=(7,), fill_cells: int = 6) -> dict:
    """PREVIEW surface built from the display sample with the excluded classes (default: points you marked as noise / removed) left out.

    Per grid cell the highest remaining point is used (so it is a top surface, like a DSM, minus what you removed). Cells that had
    points only because of removed features, or small gaps next to remaining points (within ``fill_cells`` cells), are filled by linear
    interpolation: those cells are ESTIMATED, and are reported as ``filled_fraction``. Larger gaps stay empty.
    This is a visual planning aid built from a thinned sample, not a survey surface, and it does not touch the real DSM/DTM."""
    from scipy import ndimage as ndi
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    f, kind = sample_file(app, pid)
    if f is None:
        raise CloudEditError("There is no point-cloud display sample yet. Build one first.", "no_cloud", 409)
    with np.load(f, allow_pickle=False) as z:
        x, y, zz = z["x"].astype(np.float64), z["y"].astype(np.float64), z["z"].astype(np.float64)
        origin = [float(v) for v in z["origin"]]
        cls = apply_overrides(app, pid, z["cls"])
    keep = ~np.isin(cls, list(exclude))
    n_keep = int(keep.sum())
    if n_keep < 50:
        raise CloudEditError("Too few points remain after removal to build a surface.", "too_few_points", 409)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    ext = max(xmax - xmin, ymax - ymin, 1e-6)
    area = max((xmax - xmin) * (ymax - ymin), 1e-6)
    res = max(ext / max(32, min(maxdim, 1200)), float(np.sqrt(area / len(x))) * 1.5)      # based on ALL points so the grid does not change when you remove some
    w, h = int(np.ceil((xmax - xmin) / res)) + 1, int(np.ceil((ymax - ymin) / res)) + 1
    col = np.minimum(((x - xmin) / res).astype(np.int64), w - 1)
    row = np.minimum(((ymax - y) / res).astype(np.int64), h - 1)
    top = np.full((h, w), -np.inf)
    np.maximum.at(top, (row[keep], col[keep]), zz[keep])
    valid = np.isfinite(top)
    dist = ndi.distance_transform_edt(~valid)
    want = (~valid) & (dist <= fill_cells)
    out = np.where(valid, top, np.nan)
    filled = 0
    if want.any():
        rr, cc = np.nonzero(valid)
        pts = np.column_stack([cc, rr]).astype(float)
        vals = top[valid]
        wr, wc = np.nonzero(want)
        q = np.column_stack([wc, wr]).astype(float)
        try:
            v = LinearNDInterpolator(pts, vals)(q)
        except Exception:                                             # degenerate geometry (collinear points)
            v = np.full(len(q), np.nan)
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = NearestNDInterpolator(pts, vals)(q[bad])
        out[wr, wc] = v
        filled = int(len(q))
    nvalid = int(np.isfinite(out).sum())
    grid = {"x0": xmin + origin[0] - res / 2, "y0": ymax + origin[1] + res / 2, "res": res, "width": w, "height": h, "stride": 1, "native_res": res,
            "kind": "estimated", "label": "EDITED surface (preview from sample points; removed features filled by interpolation)",
            "filled_cells": filled, "filled_fraction": (filled / nvalid) if nvalid else 0.0, "removed_points": int((~keep).sum()),
            "points_used": n_keep, "preview": True}
    return {"data": (out + origin[2]).astype("<f4"), "grid": grid}
