"""Land-cover labels on the PHOTO-TEXTURED 3D MODEL (one class id per mesh vertex), editable from the Classify tab.

* Labels live on the 3D model only (``terrain/mesh/labels.npy``). Editing them does NOT change the orthomosaic classification, the point-cloud
  classes or any elevation product. They exist so a viewer can see what is what on the stitched model.
* "Seed" copies the orthomosaic classification onto the model by looking up each vertex's map position in the classification mask. It needs a
  mesh that was confirmed to overlap the project's georeferenced orthomosaic. Vertices that sit on the sides of trees or buildings take the
  class of the ground position they are above; that is a projection of a 2D map, not a 3D classification, which is why editing is offered.
* Every change is recorded (indices, old values, new values) so it can be undone and redone; labels are discarded if the model is rebuilt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..context import AppContext
from ..core.tiling import Window
from ..db import loads
from . import classify as K
from . import meshprep as M
from . import ortho as O
from . import products as P


class MeshClassError(Exception):
    def __init__(self, message: str, code: str = "error", status: int = 400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def _dir(app: AppContext, pid: str) -> Path:
    return app.storage.path(pid, M.DIR)


def _meta_path(app, pid) -> Path:
    return _dir(app, pid) / "labels_meta.json"


def _mesh_meta(app, pid) -> dict:
    p = _dir(app, pid) / "mesh.json"
    if not p.exists() or not (_dir(app, pid) / "mesh.bin").exists():
        raise MeshClassError("There is no prepared photo-textured model yet.", "no_mesh", 409)
    return json.loads(p.read_text())


def _load_meta(app, pid, mm: dict) -> Optional[dict]:
    p = _meta_path(app, pid)
    if not p.exists() or not (_dir(app, pid) / "labels.npy").exists():
        return None
    m = json.loads(p.read_text())
    if m.get("mesh_created_at") != mm["created_at"]:      # the model was rebuilt: stale labels would not line up
        return None
    return m


def _save_meta(app, pid, m: dict) -> None:
    _meta_path(app, pid).write_text(json.dumps(m))


def _labels(app, pid) -> np.ndarray:
    return np.load(_dir(app, pid) / "labels.npy")


def _counts(lab: np.ndarray) -> Dict[str, int]:
    u, c = np.unique(lab, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, c)}


def _classes(app, pid) -> list:
    from ..core import classes as C
    proj = app.db.one("SELECT classes_json FROM projects WHERE id=?", (pid,))
    return loads(proj["classes_json"], None) or [c.to_dict() for c in C.DEFAULT_CLASSES]


def info(app: AppContext, pid: str) -> dict:
    try:
        mm = _mesh_meta(app, pid)
    except MeshClassError as e:
        return {"available": False, "reason": e.message}
    m = _load_meta(app, pid, mm)
    ortho = P.get(app, pid, "orthophoto")
    can_seed = bool(mm["aligned"] and ortho and ortho["exists"] and ortho["meta"].get("georeferenced") and ortho["meta"].get("grid"))
    out = {"available": True, "has_labels": m is not None, "vertices": mm["vertices"], "can_seed": can_seed, "classes": _classes(app, pid),
           "seed_reason": None if can_seed else ("The model is not confirmed to line up with the map, so it cannot be labelled from the orthomosaic."
                                                  if not mm["aligned"] else "There is no georeferenced orthomosaic classification to copy from."),
           "note": "Labels on the 3D model only. They do not change the orthomosaic classification or any elevation product."}
    if m:
        lab = _labels(app, pid)
        out.update({"version": m["version"], "source": m["source"], "can_undo": m["cursor"] > 0, "can_redo": m["cursor"] < len(m["ops"]),
                    "counts": _counts(lab)})
    return out


def _ensure(app, pid, mm: dict) -> dict:
    m = _load_meta(app, pid, mm)
    if m is None:
        d = _dir(app, pid)
        for old in (d / "ops").glob("*.npz") if (d / "ops").exists() else []:
            old.unlink()
        np.save(d / "labels.npy", np.zeros(mm["vertices"], np.uint8))
        m = {"mesh_created_at": mm["created_at"], "version": 0, "source": "blank (nothing labelled yet)", "ops": [], "cursor": 0}
        _save_meta(app, pid, m)
    return m


def _record(app, pid, m: dict, idx: np.ndarray, old: np.ndarray, new: np.ndarray, label: str) -> None:
    d = _dir(app, pid) / "ops"
    d.mkdir(exist_ok=True)
    for k in range(m["cursor"], len(m["ops"])):          # a new edit discards the redo branch
        f = d / f"{k}.npz"
        if f.exists():
            f.unlink()
    m["ops"] = m["ops"][: m["cursor"]]
    np.savez_compressed(d / f"{m['cursor']}.npz", idx=idx.astype(np.uint32), old=old, new=new)
    m["ops"].append(label)
    m["cursor"] += 1
    m["version"] += 1


def edit(app: AppContext, pid: str, idx: np.ndarray, cls: int) -> dict:
    mm = _mesh_meta(app, pid)
    ids = {0} | {int(c["id"]) for c in _classes(app, pid)}
    if cls not in ids:
        raise MeshClassError(f"class {cls} does not exist in this project", "bad_class")
    if idx.size == 0:
        raise MeshClassError("nothing selected", "empty_selection")
    if int(idx.max()) >= mm["vertices"]:
        raise MeshClassError("selection refers to vertices that do not exist", "bad_selection")
    idx = np.unique(idx)
    m = _ensure(app, pid, mm)
    lab = _labels(app, pid)
    old = lab[idx].copy()
    lab[idx] = cls
    _record(app, pid, m, idx, old, np.full(len(idx), cls, np.uint8), f"set {len(idx)} vertices to class {cls}")
    m["source"] = "edited by hand" if m["source"].startswith("blank") else m["source"] + " + edited by hand"
    np.save(_dir(app, pid) / "labels.npy", lab)
    _save_meta(app, pid, m)
    return info(app, pid)


def _step(app, pid, direction: int) -> dict:
    mm = _mesh_meta(app, pid)
    m = _load_meta(app, pid, mm)
    if not m:
        raise MeshClassError("nothing to undo", "nothing")
    if direction < 0:
        if m["cursor"] == 0:
            raise MeshClassError("nothing to undo", "nothing")
        k = m["cursor"] - 1
    else:
        if m["cursor"] >= len(m["ops"]):
            raise MeshClassError("nothing to redo", "nothing")
        k = m["cursor"]
    z = np.load(_dir(app, pid) / "ops" / f"{k}.npz")
    lab = _labels(app, pid)
    lab[z["idx"]] = z["old"] if direction < 0 else z["new"]
    m["cursor"] += direction
    m["version"] += 1
    np.save(_dir(app, pid) / "labels.npy", lab)
    _save_meta(app, pid, m)
    return info(app, pid)


def undo(app, pid):
    return _step(app, pid, -1)


def redo(app, pid):
    return _step(app, pid, +1)


def clear(app: AppContext, pid: str) -> dict:
    mm = _mesh_meta(app, pid)
    m = _ensure(app, pid, mm)
    lab = _labels(app, pid)
    nz = np.nonzero(lab)[0]
    if nz.size:
        _record(app, pid, m, nz, lab[nz].copy(), np.zeros(len(nz), np.uint8), "clear all labels")
        lab[:] = 0
        m["source"] = "blank (cleared)"
        np.save(_dir(app, pid) / "labels.npy", lab)
        _save_meta(app, pid, m)
    return info(app, pid)


def vertex_world_xy(app: AppContext, pid: str, mm: dict) -> np.ndarray:
    n = mm["vertices"]
    pos = np.fromfile(_dir(app, pid) / "mesh.bin", np.float32, count=n * 3).reshape(n, 3).astype(np.float64)
    return pos[:, :2] + np.array(mm["origin"][:2])


def seed(app: AppContext, pid: str, block_rows: int = 512) -> dict:
    mm = _mesh_meta(app, pid)
    i = info(app, pid)
    if not i["can_seed"]:
        raise MeshClassError(i["seed_reason"], "cannot_seed", 409)
    og = P.get(app, pid, "orthophoto")["meta"]["grid"]
    xy = vertex_world_xy(app, pid, mm)
    mask, _ = O.class_rasters(app, pid)
    oc = np.floor((xy[:, 0] - og["x0"]) / og["res"]).astype(np.int64)
    orr = np.floor((og["y0"] - xy[:, 1]) / og["res"]).astype(np.int64)
    inside = (oc >= 0) & (oc < og["width"]) & (orr >= 0) & (orr < og["height"])
    new = np.zeros(mm["vertices"], np.uint8)
    for r0 in range(0, og["height"], block_rows):
        sel = np.nonzero(inside & (orr >= r0) & (orr < r0 + block_rows))[0]
        if sel.size == 0:
            continue
        win = Window(int(oc[sel].min()), int(orr[sel].min()), int(oc[sel].max()) + 1, int(orr[sel].max()) + 1)
        sub = mask.read(win) & K.CLASS_MASK
        new[sel] = sub[orr[sel] - win.y0, oc[sel] - win.x0]
    m = _ensure(app, pid, mm)
    lab = _labels(app, pid)
    ch = np.nonzero(lab != new)[0]
    if ch.size:
        _record(app, pid, m, ch, lab[ch].copy(), new[ch].copy(), "copy the orthomosaic classification onto the model")
        np.save(_dir(app, pid) / "labels.npy", new)
    m["source"] = "copied from the orthomosaic classification"
    _save_meta(app, pid, m)
    out = info(app, pid)
    out["seeded_inside"] = int(inside.sum())
    out["seeded_outside"] = int((~inside).sum())
    return out


def labels_bytes(app: AppContext, pid: str) -> bytes:
    mm = _mesh_meta(app, pid)
    m = _load_meta(app, pid, mm)
    if not m:
        return np.zeros(mm["vertices"], np.uint8).tobytes()
    return _labels(app, pid).astype(np.uint8).tobytes()
