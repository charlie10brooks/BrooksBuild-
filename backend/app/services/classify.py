"""Automatic land-cover classification.

Models are replaceable: anything implementing ``SegmentationModel`` plugs into the same tiled inference.

Built-in models
* ``heuristic``  - colour/texture rules. NOT a trained model. It only predicts vegetation (trees/brush/grass), bare soil,
                   rock, road and water, and leaves everything it is unsure about *unclassified*. It cannot recognise
                   buildings, species, or what lies beneath canopy.
* ``learned``    - multinomial logistic regression trained from *this project's* manual corrections. Its accuracy is
                   measured on held-out tiles and reported.
* ``onnx``       - any user-supplied ONNX semantic-segmentation network (path in TERRASCOPE_SEGMENTATION_MODEL plus a JSON
                   config mapping network classes to project classes). No pretrained network ships with this repository
                   because the sandbox that built it had no network access.

Predictions are written with a provenance tag (``TAG_AUTO``). Manual edits carry ``TAG_MANUAL`` and are never overwritten
unless the user explicitly asks to replace everything.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from ..context import AppContext
from ..core import classes as C
from ..core import regions
from ..core.rasterstore import TiledRaster
from ..core.tiling import Window, iter_tiles
from ..jobs import JobContext, JobFailure
from . import ortho as O
from . import products as P

CLASS_MASK = 0x3F
TAG_MANUAL = 0x40
TAG_AUTO = 0x80
OVERWRITE_MODES = ("unclassified", "auto", "all")


class SegmentationModel(Protocol):
    name: str
    description: str
    context_px: int          # overlap the tiled runner should provide around each tile
    tile_px: int             # preferred tile size

    def predict(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """uint8 (H,W,3) -> (class ids uint8 (H,W), confidence float32 0..1 (H,W)). Class 0 = not classified."""


# ------------------------------------------------------------------------------------------ features
def pixel_features(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] * (2 * np.pi / 180.0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    exg = 2 * g - r - b
    vari = (g - r) / (g + r - b + 1e-3)

    def std(k):
        m = cv2.blur(gray, (k, k))
        return np.sqrt(np.maximum(cv2.blur(gray * gray, (k, k)) - m * m, 0))
    return np.dstack([r, g, b, np.cos(h), np.sin(h), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0, lab[..., 0], lab[..., 1], lab[..., 2],
                      exg, np.clip(vari, -1, 1), std(5), std(15), cv2.blur(gray, (15, 15)), cv2.blur(exg, (15, 15))]).astype(np.float32)


FEATURE_NAMES = ["r", "g", "b", "hue_cos", "hue_sin", "sat", "val", "L", "a", "b_lab", "exg", "vari", "tex5", "tex15", "gray_mean15", "exg_mean15"]


# ------------------------------------------------------------------------------------------ heuristic
@dataclass
class HeuristicParams:
    exg_veg: float = 0.05
    tree_gray: float = 0.30
    tree_texture: float = 0.05
    grass_gray: float = 0.42
    grass_texture: float = 0.04
    water_blue_margin: float = 0.04
    soil_sat: float = 0.18
    soil_rb: float = 0.07
    soil_max_rg: float = 0.2         # soil is brown; strongly red surfaces (roofs, equipment) are not soil
    gray_sat: float = 0.12
    max_confidence: float = 0.8      # a rule set never claims more than this


class HeuristicModel:
    name = "heuristic"
    description = ("Colour/texture rules (not a trained model). Predicts vegetation, bare soil, rock, road and water only; "
                   "leaves uncertain pixels unclassified; cannot detect buildings or see under canopy.")
    context_px = 16
    tile_px = 1024
    confidence_note = ("Rule margin: how far a pixel is from the rule thresholds. It is NOT a calibrated probability; "
                       "wrong predictions can still score high.")

    def __init__(self, params: Optional[HeuristicParams] = None):
        self.p = params or HeuristicParams()

    def predict(self, rgb: np.ndarray):
        p = self.p
        f = rgb.astype(np.float32) / 255.0
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        mx, mn = f.max(axis=2), f.min(axis=2)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        m = cv2.blur(gray, (9, 9))
        tex = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - m * m, 0))
        exg = 2 * g - r - b
        out = np.zeros(gray.shape, np.uint8)
        marg = np.zeros(gray.shape, np.float32)

        def sc(x, scale):          # normalised distance from a decision boundary, 0..1
            return np.clip(np.abs(x) / scale, 0, 1)

        veg = exg > p.exg_veg
        tree = veg & ((gray < p.tree_gray) | ((tex > p.tree_texture) & (gray < p.grass_gray)))
        grass = veg & ~tree & (gray >= p.grass_gray) & (tex < p.grass_texture)
        brush = veg & ~tree & ~grass
        water = ~veg & (b > r + p.water_blue_margin) & (b >= g - 0.02) & (tex < 0.03)
        soil = ~veg & ~water & (r >= g) & (g >= b) & (sat > p.soil_sat) & ((r - b) > p.soil_rb) & ((r - g) < p.soil_max_rg)
        grey = ~veg & ~water & ~soil & (sat < p.gray_sat)
        rock = grey & (gray > 0.45) & (tex >= 0.02)
        road = grey & ~rock & (gray > 0.2) & (gray <= 0.55) & (tex < 0.03)
        for cid, mk in ((C.TREES, tree), (C.BRUSH, brush), (C.GRASS, grass), (C.WATER, water), (C.SOIL, soil), (C.ROCK, rock), (C.ROAD, road)):
            out[mk] = cid
        marg[veg] = sc(exg[veg] - p.exg_veg, 0.15)
        marg[tree] = np.minimum(marg[tree], sc(gray[tree] - p.tree_gray, 0.1) * 0.5 + 0.5)
        marg[grass] = np.minimum(marg[grass], sc(gray[grass] - p.grass_gray, 0.1) * 0.5 + 0.5)
        marg[water] = sc(b[water] - r[water] - p.water_blue_margin, 0.1)
        marg[soil] = np.minimum(sc(sat[soil] - p.soil_sat, 0.15), sc((r - b)[soil] - p.soil_rb, 0.1))
        marg[rock | road] = sc(p.gray_sat - sat[rock | road], 0.08)
        conf = np.where(out > 0, 0.25 + 0.65 * marg, 0.0).astype(np.float32)
        return out, np.minimum(conf, p.max_confidence)


# ------------------------------------------------------------------------------------------ learned
class LearnedModel:
    name = "learned"
    description = "Logistic regression trained from this project's manual corrections; accuracy is measured on held-out tiles."
    confidence_note = "Softmax probability of the model; roughly calibrated only for scenes similar to the training areas."
    context_px = 16
    tile_px = 1024

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)           # plain arrays only: no pickle, safe to load
        self.classes = z["classes"].astype(np.int64)
        self.mean, self.std = z["mean"], z["std"]
        self.coef, self.intercept = z["coef"], z["intercept"]
        self.report = json.loads(str(z["report"]))

    def predict(self, rgb: np.ndarray):
        X = (pixel_features(rgb) - self.mean) / self.std
        z = X @ self.coef.T + self.intercept
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        k = p.argmax(axis=-1)
        return self.classes[k].astype(np.uint8), np.minimum(p.max(axis=-1), 0.99).astype(np.float32)


# ------------------------------------------------------------------------------------------ ONNX
class OnnxSegModel:
    """User-supplied ONNX semantic segmentation network.

    ``<model>.onnx`` needs a sibling ``<model>.json``::
        {"name": "my-model", "input_size": 512 | null, "mean": [..3], "std": [..3], "scale": 255,
         "class_map": {"0": 4, "1": 1, ...}}        # network class index -> project class id (unlisted -> unclassified)
    Input NCHW float32 RGB; output (1,K,H,W) scores/logits (softmax is applied here) or (1,H,W) integer labels.
    """
    description = "User-supplied ONNX semantic segmentation model."
    confidence_note = "Softmax probability from the network; calibration depends on the model."

    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            from .las import MissingDependency
            raise MissingDependency("onnxruntime is not installed: pip install onnxruntime") from e
        path = Path(path)
        cfg_path = path.with_suffix(".json")
        if not cfg_path.exists():
            raise JobFailure(f"missing model config {cfg_path.name} next to {path.name}; see docs/CLASSIFICATION.md", "bad_model")
        cfg = json.loads(cfg_path.read_text())
        self.name = cfg.get("name", path.stem)
        self.cfg = cfg
        self.class_map = {int(k): int(v) for k, v in cfg["class_map"].items()}
        self.mean = np.array(cfg.get("mean", [0, 0, 0]), np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg.get("std", [1, 1, 1]), np.float32).reshape(1, 1, 3)
        self.scale = float(cfg.get("scale", 255))
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        shp = self.inp.shape
        fixed = shp[2] if isinstance(shp[2], int) and isinstance(shp[3], int) and shp[2] == shp[3] else None
        self.tile_px = int(cfg.get("input_size") or fixed or 512)
        self.context_px = int(cfg.get("context_px", 32))
        self.fixed = fixed

    def predict(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        S = self.fixed
        pad = None
        if S:
            if h > S or w > S:
                raise JobFailure(f"model needs {S}x{S} input but got a {w}x{h} tile", "bad_model")
            pad = np.zeros((S, S, 3), np.uint8)
            pad[:h, :w] = rgb
            src = pad
        else:
            src = rgb
        x = ((src.astype(np.float32) / self.scale) - self.mean) / self.std
        out = self.sess.run(None, {self.inp.name: x.transpose(2, 0, 1)[None]})[0]
        if out.ndim == 4:
            e = np.exp(out[0] - out[0].max(axis=0, keepdims=True))
            prob = e / e.sum(axis=0, keepdims=True)
            lab, conf = prob.argmax(axis=0), prob.max(axis=0)
        else:
            lab, conf = out[0].astype(np.int64), np.ones(out.shape[1:], np.float32) * 0.5
        lut = np.zeros(max(max(self.class_map) + 1, int(lab.max()) + 1), np.uint8)
        for k, v in self.class_map.items():
            lut[k] = v
        cls = lut[lab][:h, :w]
        return cls, conf[:h, :w].astype(np.float32)


def available_models(app: AppContext, pid: str) -> List[dict]:
    learned = app.storage.path(pid, "classification/models/learned.npz")
    out = [{"id": "heuristic", "name": HeuristicModel.name, "kind": "heuristic", "available": True, "description": HeuristicModel.description,
            "confidence_note": HeuristicModel.confidence_note},
           {"id": "learned", "name": "learned (from your corrections)", "kind": "learned", "available": learned.exists(),
            "description": LearnedModel.description, "confidence_note": LearnedModel.confidence_note,
            "report": json.loads(str(np.load(learned, allow_pickle=False)["report"])) if learned.exists() else None}]
    mp = app.settings.model_path
    out.append({"id": "onnx", "name": Path(mp).stem if mp else "ONNX segmentation model", "kind": "onnx", "available": bool(mp and Path(mp).exists()),
                "description": OnnxSegModel.description, "confidence_note": OnnxSegModel.confidence_note,
                "note": None if mp else "Not configured. Set TERRASCOPE_SEGMENTATION_MODEL to an .onnx file (see docs/CLASSIFICATION.md)."})
    return out


def load_model(app: AppContext, pid: str, model_id: str) -> SegmentationModel:
    if model_id == "heuristic":
        return HeuristicModel()
    if model_id == "learned":
        p = app.storage.path(pid, "classification/models/learned.npz")
        if not p.exists():
            raise JobFailure("no learned model yet: correct some areas manually (at least two classes), then train from corrections", "no_model")
        return LearnedModel(p)
    if model_id == "onnx":
        mp = app.settings.model_path
        if not mp or not Path(mp).exists():
            raise JobFailure("no ONNX model configured. Set TERRASCOPE_SEGMENTATION_MODEL to a model file, or use the heuristic model "
                             "or manual classification.", "no_model")
        return OnnxSegModel(Path(mp))
    raise JobFailure(f"unknown model {model_id!r}", "no_model")


# ------------------------------------------------------------------------------------------ region helpers
def region_filter(region: Optional[dict]):
    """Return fn(window)->bool mask, or None for 'everywhere'. Coordinates are ortho pixel coordinates."""
    if not region:
        return None, None
    t = region.get("type")
    if t == "rect":
        x0, y0, x1, y1 = region["rect"]
        bbox = Window(int(np.floor(min(x0, x1))), int(np.floor(min(y0, y1))), int(np.ceil(max(x0, x1))), int(np.ceil(max(y0, y1))))
        return bbox, lambda w: regions.rect_mask(x0, y0, x1, y1, w)
    if t in ("polygon", "lasso"):
        pts = [tuple(p) for p in region["points"]]
        a = np.asarray(pts)
        bbox = Window(int(np.floor(a[:, 0].min())), int(np.floor(a[:, 1].min())), int(np.ceil(a[:, 0].max())) + 1, int(np.ceil(a[:, 1].max())) + 1)
        return bbox, lambda w: regions.polygon_mask(pts, w)
    raise JobFailure(f"unknown region type {t!r}", "bad_request")


def run_classification(app: AppContext, pid: str, model: SegmentationModel, region: Optional[dict], overwrite: str,
                       ctx: Optional[JobContext] = None, tile: Optional[int] = None, overlap: Optional[int] = None) -> dict:
    if overwrite not in OVERWRITE_MODES:
        raise JobFailure(f"overwrite must be one of {OVERWRITE_MODES}", "bad_request")
    prod = P.require(app, pid, "orthophoto")
    mask, conf = O.class_rasters(app, pid)
    rd = O.ProjectOrthoReader(prod["path"])
    bbox, rfn = region_filter(region)
    tile = tile or getattr(model, "tile_px", 1024)
    overlap = overlap if overlap is not None else getattr(model, "context_px", 32)
    specs = list(iter_tiles(rd.width, rd.height, tile, overlap))
    if bbox:
        specs = [s for s in specs if s.keep.intersect(bbox) is not None]
    counts = np.zeros(256, np.int64)
    n_written = 0
    t0 = time.time()
    try:
        for i, s in enumerate(specs):
            if ctx:
                ctx.check_cancel()
                ctx.progress(0.05 + 0.93 * i / max(len(specs), 1), f"tile {i + 1}/{len(specs)}")
            rgb, valid = rd.read_rgb_valid(s.window)
            cls, cf = model.predict(rgb)
            ky0, kx0 = s.keep.y0 - s.window.y0, s.keep.x0 - s.window.x0
            ksl = (slice(ky0, ky0 + s.keep.height), slice(kx0, kx0 + s.keep.width))
            cls, cf, valid = cls[ksl], cf[ksl], valid[ksl]
            sel = valid.copy()
            if rfn:
                sel &= rfn(s.keep)
            if not sel.any():
                continue
            with mask.edit("auto-classify", record_history=False):
                cur = mask.read(s.keep)
                manual = (cur & TAG_MANUAL) != 0
                cur_cls = cur & CLASS_MASK
                if overwrite == "unclassified":
                    target = sel & ~manual & (cur_cls == 0)
                elif overwrite == "auto":
                    target = sel & ~manual
                else:
                    target = sel
                if target.any():
                    mask.write(s.keep, (cls | TAG_AUTO).astype(np.uint8), target)
                    n_written += int(target.sum())
                    counts += np.bincount(cls[target], minlength=256)
            if target.any():
                with conf.edit("auto-confidence", record_history=False):
                    conf.write(s.keep, np.clip(cf * 100, 0, 100).astype(np.uint8), target)
    finally:
        rd.close()
    by_class = {int(k): int(v) for k, v in enumerate(counts) if v}
    return {"model": getattr(model, "name", "?"), "pixels_written": n_written, "tiles": len(specs), "by_class": by_class,
            "overwrite": overwrite, "seconds": round(time.time() - t0, 1),
            "notice": "Automatic classification is a visual aid. It does not measure ground elevation and cannot see beneath canopy."}


# ------------------------------------------------------------------------------------------ snapshots
def _snap_root(app, pid) -> Path:
    return app.storage.ensure(pid, "classification/snapshots")


def take_snapshot(app: AppContext, pid: str, label: str) -> dict:
    mask, conf = O.class_rasters(app, pid)
    sid = f"{int(time.time())}_{label[:40].replace('/', '_').replace(' ', '_')}"
    dest = _snap_root(app, pid) / sid
    dest.mkdir(parents=True)
    n = 0
    with mask.edit("snapshot"):                              # holds the writer lock so the copy is consistent; no writes => history untouched
        for name, store in (("mask", mask), ("confidence", conf)):
            (dest / name).mkdir()
            for (ty, tx) in store.iter_existing_tiles():
                (dest / name / str(ty)).mkdir(exist_ok=True)
                shutil.copy2(store._path(ty, tx), dest / name / str(ty) / f"{tx}.npz")
                n += 1
    (dest / "info.json").write_text(json.dumps({"id": sid, "label": label, "created_at": time.time(), "tiles": n}))
    return {"id": sid, "label": label, "tiles": n}


def list_snapshots(app: AppContext, pid: str) -> List[dict]:
    out = []
    for d in sorted(_snap_root(app, pid).iterdir(), reverse=True):
        if (d / "info.json").exists():
            out.append(json.loads((d / "info.json").read_text()))
    return out


def restore_snapshot(app: AppContext, pid: str, sid: str) -> dict:
    src = _snap_root(app, pid) / sid
    if "/" in sid or ".." in sid or not (src / "info.json").exists():
        raise JobFailure("snapshot not found", "not_found")
    mask, conf = O.class_rasters(app, pid)
    for name, store in (("mask", mask), ("confidence", conf)):
        snap_tiles = set()
        base = src / name
        for d in base.iterdir() if base.exists() else []:
            for f in d.glob("*.npz"):
                snap_tiles.add((int(d.name), int(f.stem)))
        live = set(store.iter_existing_tiles())
        with store.edit("restore snapshot", record_history=False):
            for (ty, tx) in snap_tiles | live:
                h, w = store._tile_shape(ty, tx)
                win = Window(tx * store.tile, ty * store.tile, tx * store.tile + w, ty * store.tile + h)
                if (ty, tx) in snap_tiles:
                    with np.load(base / str(ty) / f"{tx}.npz") as z:
                        store.write(win, z["a"])
                else:
                    store.write(win, store.fill)
    return {"restored": sid}


# ------------------------------------------------------------------------------------------ training
def train_from_corrections(app: AppContext, pid: str, ctx: Optional[JobContext] = None, per_tile: int = 600,
                           per_class_cap: int = 60000, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    prod = P.require(app, pid, "orthophoto")
    mask, _ = O.class_rasters(app, pid)
    rd = O.ProjectOrthoReader(prod["path"])
    rng = np.random.default_rng(seed)
    X, y, tid = [], [], []
    keys = list(mask.iter_existing_tiles())
    try:
        for n, (ty, tx) in enumerate(keys):
            if ctx:
                ctx.check_cancel()
                ctx.progress(0.05 + 0.6 * n / max(len(keys), 1), "collecting labelled pixels")
            t = mask.read_tile(ty, tx)
            lab = ((t & TAG_MANUAL) != 0) & ((t & CLASS_MASK) != 0)
            rr, cc = np.nonzero(lab)
            if len(rr) == 0:
                continue
            if len(rr) > per_tile:
                sel = rng.choice(len(rr), per_tile, replace=False)
                rr, cc = rr[sel], cc[sel]
            pad = 16
            win = Window(tx * mask.tile - pad, ty * mask.tile - pad, tx * mask.tile + t.shape[1] + pad, ty * mask.tile + t.shape[0] + pad)
            w = win.clip(rd.width, rd.height)
            rgb = rd.read_rgb(w)
            feats = pixel_features(rgb)
            oy, ox = ty * mask.tile - w.y0, tx * mask.tile - w.x0
            X.append(feats[rr + oy, cc + ox])
            y.append((t[rr, cc] & CLASS_MASK).astype(np.int64))
            tid.append(np.full(len(rr), (ty * 100003 + tx)))
    finally:
        rd.close()
    if not X:
        raise JobFailure("no manual corrections found. Paint or fill some areas with at least two different classes first.", "no_training_data")
    X, y, tid = np.concatenate(X), np.concatenate(y), np.concatenate(tid)
    classes = np.unique(y)
    if len(classes) < 2:
        raise JobFailure("manual corrections only cover one class; label at least two different classes.", "no_training_data")
    if len(y) < 200:
        raise JobFailure(f"only {len(y)} labelled pixels; label more areas (need at least 200).", "no_training_data")
    keep = np.zeros(len(y), bool)
    for c in classes:
        idx = np.nonzero(y == c)[0]
        if len(idx) > per_class_cap:
            idx = rng.choice(idx, per_class_cap, replace=False)
        keep[idx] = True
    X, y, tid = X[keep], y[keep], tid[keep]
    ids = np.unique(tid)
    test_tiles = set(ids[::4].tolist()) if len(ids) >= 4 else set()          # hold out every 4th tile (spatial split)
    test = np.isin(tid, list(test_tiles)) if test_tiles else np.zeros(len(y), bool)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(Z[~test], y[~test])
    holdout = None
    if test.any() and len(np.unique(y[test])) >= 1:
        pred = clf.predict(Z[test])
        holdout = {"accuracy": float((pred == y[test]).mean()), "pixels": int(test.sum()), "tiles": len(test_tiles)}
    clf.fit(Z, y)                                                              # final fit on everything
    report = {"trained_at": time.time(), "samples": int(len(y)), "classes": {int(c): int((y == c).sum()) for c in classes},
              "holdout": holdout, "features": FEATURE_NAMES,
              "notice": "Trained only on this project's manual corrections; accuracy is measured on held-out tiles of the same orthomosaic "
                        "and will not transfer to other sites, lighting, or seasons."}
    d = app.storage.ensure(pid, "classification/models")
    tmp = d / "learned.tmp.npz"
    np.savez(tmp, classes=clf.classes_.astype(np.int64), mean=mean, std=std, coef=clf.coef_, intercept=clf.intercept_,
             report=np.array(json.dumps(report)))
    tmp.replace(d / "learned.npz")
    return report


# ------------------------------------------------------------------------------------------ job handlers
def handle_classify(app: AppContext, ctx: JobContext) -> dict:
    p = ctx.params
    pid = p["project_id"]
    model = load_model(app, pid, p.get("model", "heuristic"))
    ctx.log(f"model: {model.name} - {model.description}")
    snap = take_snapshot(app, pid, f"before auto-classify {model.name}")
    ctx.log(f"snapshot {snap['id']} saved ({snap['tiles']} tiles); undo history is cleared by auto-classification, use the snapshot to roll back")
    res = run_classification(app, pid, model, p.get("region"), p.get("overwrite", "auto"), ctx)
    res["snapshot"] = snap["id"]
    ctx.log(f"wrote {res['pixels_written']:,} pixels in {res['tiles']} tiles")
    return res


def handle_train(app: AppContext, ctx: JobContext) -> dict:
    rep = train_from_corrections(app, ctx.params["project_id"], ctx)
    ctx.log(f"trained on {rep['samples']} pixels; holdout={rep['holdout']}")
    return rep


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["classify"] = handle_classify
HANDLERS["train_classifier"] = handle_train
