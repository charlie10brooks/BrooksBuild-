"""Orthomosaic preparation: normalised tiled GeoTIFF, multiresolution tile pyramid, classification rasters.

Everything streams tile by tile (O(tile) memory), so multi-gigapixel orthomosaics are fine. The original orthomosaic is
never modified; classification lives in separate tiled rasters (see ``core.rasterstore``).
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tifffile

from ..context import AppContext
from ..core.rasterstore import TiledRaster
from ..core.tiling import Window
from ..jobs import JobContext, JobFailure
from . import geotiff as gt
from . import products as P

TILE = 256
MASK_TILE = 256


class OrthoPyramid:
    """Reads tiles from ``ortho/pyramid``. Level 0 is native resolution; level z is downsampled by 2**z."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "pyramid.json").read_text())

    @property
    def ext(self) -> str:
        return self.info["ext"]

    def tile_path(self, z: int, x: int, y: int) -> Optional[Path]:
        if not (0 <= z < self.info["levels"]):
            return None
        p = self.root / str(z) / f"{y}_{x}.{self.ext}"
        return p if p.exists() else None


def _tile_encode(arr: np.ndarray, ext: str) -> bytes:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode("." + ext, bgr, [cv2.IMWRITE_JPEG_QUALITY, 88] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("tile encoding failed")
    return buf.tobytes()


def _tile_decode(path: Path) -> np.ndarray:
    a = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"cannot decode {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA if a.shape[2] == 4 else cv2.COLOR_BGR2RGB)


def _to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    return np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)


def normalise_ortho(src: Path, dst: Path, ctx: Optional[JobContext] = None, tile: int = TILE) -> gt.GeoInfo:
    """Stream-convert any supported RGB(A) GeoTIFF into a tiled, deflate-compressed uint8 GeoTIFF with the same georeferencing."""
    info = gt.read_georef(src)
    if info.bands < 3:
        raise JobFailure(f"orthomosaic must have at least 3 (RGB) bands; {src.name} has {info.bands}", "bad_ortho")
    bands = min(info.bands, 4)
    rd = gt.TiffWindowReader(src)

    def gen():
        n = 0
        total = math.ceil(info.height / tile) * math.ceil(info.width / tile)
        for y in range(0, info.height, tile):
            for x in range(0, info.width, tile):
                w = Window(x, y, min(x + tile, info.width), min(y + tile, info.height))
                a = _to_u8(rd.read(w))[..., :bands]
                full = np.zeros((tile, tile, bands), np.uint8)
                full[:a.shape[0], :a.shape[1]] = a
                n += 1
                if ctx and n % 32 == 0:
                    ctx.check_cancel()
                    ctx.progress(0.05 + 0.25 * n / total, "normalising orthomosaic")
                yield full

    tmp = dst.with_suffix(".tmp.tif")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tifffile.TiffWriter(str(tmp), bigtiff=info.width * info.height * bands > 3.5e9) as tw:
            tw.write(gen(), shape=(info.height, info.width, bands), dtype=np.uint8, tile=(tile, tile),
                     photometric="rgb", compression="deflate",
                     extrasamples=("unassalpha",) if bands == 4 else None,
                     extratags=gt._extratags(info.grid, info.epsg, info.nodata, info.horizontal_units or "metre",
                                             info.vertical_epsg, info.vertical_units or "metre"))
    finally:
        rd.close()
    tmp.replace(dst)
    return gt.read_georef(dst)


def build_pyramid(src: Path, root: Path, ctx: Optional[JobContext] = None, tile: int = TILE) -> dict:
    info = gt.read_georef(src)
    has_alpha = info.bands == 4
    ext = "png" if has_alpha else "jpg"
    levels = 1
    while max(info.width, info.height) / (2 ** (levels - 1)) > tile:
        levels += 1
    (root).mkdir(parents=True, exist_ok=True)
    rd = gt.TiffWindowReader(src)
    nx0, ny0 = math.ceil(info.width / tile), math.ceil(info.height / tile)
    try:
        # level 0
        for ty in range(ny0):
            for tx in range(nx0):
                w = Window(tx * tile, ty * tile, min((tx + 1) * tile, info.width), min((ty + 1) * tile, info.height))
                a = rd.read(w)
                if a.ndim == 2:
                    a = np.repeat(a[..., None], 3, axis=2)
                d = root / "0"
                d.mkdir(exist_ok=True)
                (d / f"{ty}_{tx}.{ext}").write_bytes(_tile_encode(_to_u8(a), ext))
            if ctx:
                ctx.check_cancel()
                ctx.progress(0.35 + 0.25 * (ty + 1) / ny0, "building tile pyramid (level 0)")
    finally:
        rd.close()
    # upper levels from the level below (4 children -> 1 tile)
    nx, ny = nx0, ny0
    for z in range(1, levels):
        nnx, nny = math.ceil(nx / 2), math.ceil(ny / 2)
        d = root / str(z)
        d.mkdir(exist_ok=True)
        ch = 4 if has_alpha else 3
        for ty in range(nny):
            for tx in range(nnx):
                canvas = np.zeros((tile * 2, tile * 2, ch), np.uint8)
                for dy in (0, 1):
                    for dx in (0, 1):
                        cp = root / str(z - 1) / f"{ty * 2 + dy}_{tx * 2 + dx}.{ext}"
                        if cp.exists():
                            c = _tile_decode(cp)
                            canvas[dy * tile:dy * tile + c.shape[0], dx * tile:dx * tile + c.shape[1]] = c
                (d / f"{ty}_{tx}.{ext}").write_bytes(_tile_encode(cv2.resize(canvas, (tile, tile), interpolation=cv2.INTER_AREA), ext))
        nx, ny = nnx, nny
        if ctx:
            ctx.check_cancel()
            ctx.progress(0.6 + 0.3 * z / max(levels - 1, 1), f"building tile pyramid (level {z})")
    meta = {"tile": tile, "levels": levels, "ext": ext, "width": info.width, "height": info.height, "alpha": has_alpha}
    (root / "pyramid.json").write_text(json.dumps(meta))
    return meta


class ProjectOrthoReader:
    """Window reader for flood-fill / similarity tools (the normalised ortho is tiled, so windows read only what they need)."""

    def __init__(self, path: Path):
        self._r = gt.TiffWindowReader(path)
        self.width, self.height = self._r.width, self._r.height

    def read_rgb(self, window: Window) -> np.ndarray:
        return self._r.read_rgb(window)

    def read_rgb_valid(self, window: Window):
        """(rgb uint8, valid bool). ``valid`` is False where the alpha band marks no-data (outside the flown area)."""
        a = self._r.read(window)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        valid = (a[..., 3] > 0) if a.shape[2] == 4 else np.ones(a.shape[:2], bool)
        return _to_u8(a[..., :3]), valid

    def close(self):
        self._r.close()


# ----------------------------------------------------------------------------------- raster registry
_rasters: Dict[Tuple[str, str], TiledRaster] = {}
_rlock = threading.Lock()


def class_rasters(app: AppContext, pid: str) -> Tuple[TiledRaster, TiledRaster]:
    """Process-wide cached (mask, confidence) stores for a project."""
    root = app.storage.path(pid, "classification")
    with _rlock:
        out = []
        for name in ("mask", "confidence"):
            key = (str(root), name)
            if key not in _rasters:
                _rasters[key] = TiledRaster(root / name, cache_tiles=512)
            out.append(_rasters[key])
        return out[0], out[1]


def forget_rasters(app: AppContext, pid: str) -> None:
    root = str(app.storage.path(pid, "classification"))
    with _rlock:
        for k in [k for k in _rasters if k[0] == root]:
            _rasters.pop(k).close()


def create_class_rasters(app: AppContext, pid: str, width: int, height: int) -> None:
    root = app.storage.ensure(pid, "classification")
    for name in ("mask", "confidence"):
        TiledRaster.create(root / name, width, height, tile=MASK_TILE, dtype="uint8", fill=0, exist_ok=True).close()


# ----------------------------------------------------------------------------------- job
def prepare_ortho(app: AppContext, ctx: JobContext) -> dict:
    pid = ctx.params["project_id"]
    src = app.storage.path(pid, ctx.params["rel_path"])
    if not src.exists():
        raise JobFailure(f"orthomosaic file not found: {ctx.params['rel_path']}", "missing_file")
    try:
        info = gt.read_georef(src)
    except Exception as e:
        raise JobFailure(f"cannot read orthomosaic as GeoTIFF: {e}", "bad_ortho")
    ctx.log(f"source {src.name}: {info.width}x{info.height}, {info.bands} bands, {info.dtype}, epsg={info.epsg}, "
            f"georeferenced={info.georeferenced}, tiled={info.tile}")
    proj = app.db.one("SELECT epsg, horizontal_units FROM projects WHERE id=?", (pid,))
    warnings = []
    if proj["epsg"] and info.epsg and proj["epsg"] != info.epsg:
        raise JobFailure(f"orthomosaic CRS EPSG:{info.epsg} differs from the project CRS EPSG:{proj['epsg']}. Reproject the "
                         "orthomosaic (e.g. gdalwarp -t_srs EPSG:%d) or change the project CRS before importing." % proj["epsg"], "crs_mismatch")
    if not info.georeferenced:
        warnings.append("The orthomosaic carries no georeferencing. Classification works in pixel space, but areas, coordinates, "
                        "terrain products and exports cannot be tied to the ground until it is georeferenced.")
    elif info.horizontal_units == "degree":
        warnings.append("The orthomosaic is in a geographic CRS (degrees). Pixel areas are not computed; reproject to a projected CRS.")
    dst = app.storage.path(pid, P.LAYOUT["orthophoto"][0])
    ctx.progress(0.03, "normalising")
    if src.resolve() != dst.resolve():
        norm = normalise_ortho(src, dst, ctx)
    else:
        norm = info
    ctx.progress(0.32, "building tile pyramid")
    pyr = build_pyramid(dst, app.storage.ensure(pid, "ortho/pyramid"), ctx)
    create_class_rasters(app, pid, norm.width, norm.height)
    forget_rasters(app, pid)
    meta = {**norm.to_dict(), "pyramid": pyr, "source_name": src.name, "warnings": warnings}
    P.register(app, pid, "orthophoto", P.LAYOUT["orthophoto"][0], "raster", meta, ctx.id)
    upd = {}
    if norm.epsg and not proj["epsg"]:
        upd["epsg"] = norm.epsg
        ctx.log(f"project CRS set to EPSG:{norm.epsg} from the orthomosaic")
    if norm.horizontal_units in ("metre", "foot", "us_survey_foot") and not proj["horizontal_units"]:
        upd["horizontal_units"] = norm.horizontal_units
    if upd:
        app.db.update("projects", {"id": pid}, upd)
    for w in warnings:
        ctx.log(w, "warning")
    ctx.progress(1.0, "ready")
    return {"width": norm.width, "height": norm.height, "levels": pyr["levels"], "georeferenced": norm.georeferenced, "warnings": warnings}


from ..workers.handlers import HANDLERS  # noqa: E402  (registration)
HANDLERS["prepare_ortho"] = prepare_ortho
