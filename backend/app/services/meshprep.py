"""Prepares OpenDroneMap's PHOTO-TEXTURED 3D model (``odm_texturing/*.obj``) for the browser viewer.

What this is: ODM projects the original photos onto a triangle mesh built from the photogrammetry point cloud, so the result can be orbited
like a Polycam capture (building walls, tree sides, overhangs). It is an OBSERVED model: what the cameras saw, including vegetation.

What this is not: a survey surface, bare earth, or something the point-removal tools edit. It is only produced when ODM builds a 3D model,
which the "draft" preset (``--fast-orthophoto``) skips.

UNVERIFIED against real ODM output: the OBJ/MTL layout and the georeferencing offset convention below follow ODM's documentation as I know it.
The platform checks the mesh against the project's DSM extent and reports whether they line up instead of assuming they do.

Output (``terrain/mesh/``): ``mesh.json`` (metadata), ``mesh.bin`` (float32 positions, float32 uvs, uint32 indices) and ``tex_<i>.jpg``.
Coordinates are stored relative to ``origin`` so float32 precision is fine for UTM-sized values.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..context import AppContext
from ..jobs import JobContext, JobFailure
from . import geotiff as gt
from . import products as P

DIR = "terrain/mesh"
MAX_TRIANGLES = 2_500_000
MAX_TEX_SIDE = 4096
Image.MAX_IMAGE_PIXELS = None            # ODM textures can be large; they are downscaled below

CANDIDATES = ("odm/odm_texturing/odm_textured_model_geo.obj", "odm/odm_texturing/odm_textured_model.obj")


def source_obj(app: AppContext, pid: str) -> Optional[Path]:
    for c in CANDIDATES:
        p = app.storage.path(pid, c)
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _int(tok: str, n: int) -> int:
    i = int(tok)
    return i - 1 if i > 0 else n + i          # OBJ is 1-based; negative indices count back from the end


def parse_obj(path: Path) -> dict:
    v: List[Tuple[float, float, float]] = []
    vt: List[Tuple[float, float]] = []
    mats: List[str] = ["__none__"]
    faces: Dict[str, List[Tuple[int, int]]] = {}
    cur = "__none__"
    mtllib: Optional[str] = None
    ntri = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                a = line.split()
                v.append((float(a[1]), float(a[2]), float(a[3])))
            elif line.startswith("vt "):
                a = line.split()
                vt.append((float(a[1]), float(a[2])))
            elif line.startswith("usemtl"):
                cur = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else "__none__"
            elif line.startswith("mtllib") and mtllib is None:
                mtllib = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                toks = line.split()[1:]
                if len(toks) < 3:
                    continue
                vs = []
                for t in toks:
                    a = t.split("/")
                    vi = _int(a[0], len(v))
                    ti = _int(a[1], len(vt)) if len(a) > 1 and a[1] else -1
                    vs.append((vi, ti))
                lst = faces.setdefault(cur, [])
                for k in range(1, len(vs) - 1):               # fan-triangulate polygons
                    lst += [vs[0], vs[k], vs[k + 1]]
                    ntri += 1
                if ntri > MAX_TRIANGLES:
                    raise JobFailure(f"the textured model has more than {MAX_TRIANGLES:,} triangles; it is too large for the browser viewer. "
                                     "Re-run ODM with a smaller --mesh-size.", "mesh_too_large")
    if not v or not faces:
        raise JobFailure("the textured model file has no geometry", "mesh_empty")
    return {"v": np.asarray(v, np.float64), "vt": np.asarray(vt, np.float64).reshape(-1, 2), "faces": faces, "mtllib": mtllib, "triangles": ntri}


def parse_mtl(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    cur = None
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith("newmtl"):
            cur = s.split(None, 1)[1].strip()
            out[cur] = {"map": None, "kd": (0.7, 0.7, 0.7)}
        elif cur and s.startswith("map_Kd"):
            out[cur]["map"] = s.split(None, 1)[1].strip()
        elif cur and s.startswith("Kd"):
            try:
                a = s.split()
                out[cur]["kd"] = (float(a[1]), float(a[2]), float(a[3]))
            except (ValueError, IndexError):
                pass
    return out


def _bounds_of(path: Path) -> Optional[Tuple[float, float, float, float]]:
    try:
        gi = gt.read_georef(path)
    except Exception:
        return None
    g = gi.grid
    if not g:
        return None
    return (g.x0, g.y0 - g.height * g.res, g.x0 + g.width * g.res, g.y0)


def _overlaps(mesh_xy: Tuple[float, float, float, float], ref: Tuple[float, float, float, float]) -> bool:
    cx, cy = (mesh_xy[0] + mesh_xy[2]) / 2, (mesh_xy[1] + mesh_xy[3]) / 2
    ew, eh = (ref[2] - ref[0]), (ref[3] - ref[1])
    return ref[0] - 0.25 * ew <= cx <= ref[2] + 0.25 * ew and ref[1] - 0.25 * eh <= cy <= ref[3] + 0.25 * eh


def _coords_offset(app: AppContext, pid: str) -> Optional[Tuple[float, float]]:
    p = app.storage.path(pid, "odm/odm_georeferencing/coords.txt")
    if not p.exists():
        return None
    try:
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        a = lines[1].split()
        return float(a[0]), float(a[1])
    except (IndexError, ValueError):
        return None


def build(app: AppContext, pid: str, log=lambda *_: None, progress=lambda *_: None) -> dict:
    src = source_obj(app, pid)
    if not src:
        raise JobFailure("OpenDroneMap produced no textured 3D model for this project. The 'draft' preset skips it (--fast-orthophoto); run "
                         "the 'balanced' or 'high' preset.", "no_textured_model")
    progress(0.05, "reading the textured model")
    m = parse_obj(src)
    v, vt = m["v"], m["vt"]
    nvt = len(vt)
    # one output vertex per distinct (position, uv) pair
    keys, groups, ranges = [], [], []
    mtl_path = (src.parent / m["mtllib"]) if m["mtllib"] else src.with_suffix(".mtl")
    mats = parse_mtl(mtl_path)
    cursor = 0
    for name, lst in m["faces"].items():
        a = np.asarray(lst, np.int64)
        keys.append(a[:, 0] * (nvt + 1) + (a[:, 1] + 1))
        n = len(a) // 3
        groups.append(name)
        ranges.append((cursor, n))
        cursor += n
    allk = np.concatenate(keys)
    uk, inv = np.unique(allk, return_inverse=True)
    vi = uk // (nvt + 1)
    ti = uk % (nvt + 1) - 1
    pos = v[vi]
    uv = np.zeros((len(uk), 2), np.float32)
    has = ti >= 0
    uv[has] = vt[ti[has]]
    progress(0.35, "aligning with the project")
    bb = (float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max()))
    ref = None
    for name in ("odm_dsm", "orthophoto"):
        pr = P.get(app, pid, name)
        if pr and pr["exists"]:
            ref = _bounds_of(pr["path"])
            if ref:
                break
    aligned, note, offset = False, "No georeferenced DSM was available to check the model against, so its placement is unverified.", (0.0, 0.0)
    if ref:
        if _overlaps(bb, ref):
            aligned, note = True, "The model's extent overlaps the project's DSM extent."
        else:
            off = _coords_offset(app, pid)
            if off and _overlaps((bb[0] + off[0], bb[1] + off[1], bb[2] + off[0], bb[3] + off[1]), ref):
                aligned, offset = True, off
                note = f"The model was in offset coordinates; ODM's coords.txt offset ({off[0]:.2f}, {off[1]:.2f}) was applied and it then overlaps the DSM."
            else:
                note = ("The model does NOT overlap the project's DSM extent (it may be in local coordinates), so it can be viewed but not "
                        "placed against the map.")
    pos[:, 0] += offset[0]
    pos[:, 1] += offset[1]
    origin = np.array([(pos[:, 0].min() + pos[:, 0].max()) / 2, (pos[:, 1].min() + pos[:, 1].max()) / 2, float(pos[:, 2].min())])
    loc = (pos - origin).astype(np.float32)
    idx = inv.astype(np.uint32)
    out_dir = app.storage.path(pid, DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("tex_*.jpg"):
        old.unlink()
    progress(0.5, "preparing textures")
    tex_files: Dict[str, Optional[str]] = {}
    gl = []
    for k, (name, (start, n)) in enumerate(zip(groups, ranges)):
        info = mats.get(name, {"map": None, "kd": (0.7, 0.7, 0.7)})
        tf = None
        if info["map"]:
            tp = (mtl_path.parent / info["map"])
            if tp.exists():
                if str(tp) not in tex_files:
                    with Image.open(tp) as im:
                        im = im.convert("RGB")
                        s = max(im.size)
                        if s > MAX_TEX_SIDE:
                            im = im.resize((max(1, im.width * MAX_TEX_SIDE // s), max(1, im.height * MAX_TEX_SIDE // s)), Image.LANCZOS)
                        fn = f"tex_{len(tex_files)}.jpg"
                        im.save(out_dir / fn, "JPEG", quality=88)
                        tex_files[str(tp)] = fn
                tf = tex_files[str(tp)]
        gl.append({"material": name, "texture": tf, "color": list(info["kd"]), "start": start, "count": n})
    progress(0.8, "writing the viewer files")
    with open(out_dir / "mesh.bin", "wb") as f:
        f.write(loc.tobytes())
        f.write(uv.tobytes())
        f.write(idx.tobytes())
    meta = {
        "version": 1, "vertices": int(len(loc)), "triangles": int(len(idx) // 3), "origin": [float(x) for x in origin],
        "bbox_local": [loc.min(axis=0).tolist(), loc.max(axis=0).tolist()], "groups": gl,
        "aligned": aligned, "alignment_note": note, "source": str(src.relative_to(app.storage.project_root(pid))),
        "textures": len(tex_files), "created_at": time.time(),
        "label": "OBSERVED photo-textured model (what the cameras saw, including vegetation). Not bare earth, not a survey surface.",
    }
    (out_dir / "mesh.json").write_text(json.dumps(meta))
    return meta


def info(app: AppContext, pid: str) -> dict:
    p = app.storage.path(pid, f"{DIR}/mesh.json")
    src = source_obj(app, pid)
    if p.exists() and (p.parent / "mesh.bin").exists():
        return {"available": True, "source_available": bool(src), **json.loads(p.read_text())}
    return {"available": False, "source_available": bool(src),
            "reason": ("The textured model from OpenDroneMap is ready to prepare." if src else
                       "No textured 3D model yet. It comes from an OpenDroneMap run with the 'balanced' or 'high' preset "
                       "(the 'draft' preset skips it).")}


def handle_mesh_prep(app: AppContext, ctx: JobContext) -> dict:
    pid = ctx.params["project_id"]
    meta = build(app, pid, ctx.log, ctx.progress)
    ctx.progress(1.0, "done")
    ctx.log(f"textured model: {meta['triangles']:,} triangles, {meta['textures']} texture(s); {meta['alignment_note']}")
    return {k: meta[k] for k in ("vertices", "triangles", "textures", "aligned", "alignment_note")}


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["mesh_prep"] = handle_mesh_prep
