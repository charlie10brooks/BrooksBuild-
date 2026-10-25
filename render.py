"""Server-side rendering of elevation/evidence rasters for the viewers: colour ramps, hillshade, window sampling.

A render is a *picture of a layer for a map-coordinate window*; clients overlay it in their own viewport, so the 2D and 3D
viewers, the split-screen comparison and the measurement tools all share one coordinate system (the project CRS).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..core.dtm import GridSpec
from . import geotiff as gt

KIND = {"dsm": "observed", "odm_dsm": "observed", "dtm": "estimated", "odm_dtm": "estimated", "ndsm": "derived",
        "confidence": "evidence", "evidence": "evidence", "source": "evidence"}
KIND_TEXT = {"observed": "OBSERVED surface (canopy and structure tops as seen by the photogrammetry)",
             "estimated": "ESTIMATED bare-earth surface (not directly observed under vegetation)",
             "derived": "DERIVED: observed surface minus estimated ground",
             "evidence": "EVIDENCE layer describing how well the ground estimate is supported"}

_RAMPS = {
    "terrain": [(0.0, (43, 131, 186)), (0.25, (171, 221, 164)), (0.5, (255, 255, 191)), (0.75, (253, 174, 97)), (1.0, (215, 25, 28))],
    "hypsometric": [(0.0, (38, 70, 83)), (0.2, (42, 157, 143)), (0.4, (138, 177, 125)), (0.6, (233, 196, 106)), (0.8, (244, 162, 97)), (1.0, (231, 111, 81))],
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
    "gray": [(0.0, (20, 20, 20)), (1.0, (240, 240, 240))],
    "heat": [(0.0, (255, 255, 204)), (0.5, (253, 141, 60)), (1.0, (128, 0, 38))],
}
CONF_COLORS = {0: (90, 90, 100, 150), 1: (214, 48, 39, 190), 2: (253, 174, 97, 190), 3: (49, 163, 84, 190)}
EVID_COLORS = {0: (0, 0, 0, 0), 1: (49, 163, 84, 200), 2: (253, 174, 97, 200), 3: (66, 146, 198, 200)}


def ramp_lut(name: str) -> np.ndarray:
    if name not in _RAMPS:
        raise ValueError(f"unknown ramp {name!r}; choose from {sorted(_RAMPS)}")
    stops = _RAMPS[name]
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], float)
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, xs, cols[:, i]) for i in range(3)], axis=1).astype(np.uint8)


def hillshade(dem: np.ndarray, res: float, zexag: float = 1.0, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """0..1 shade (NaN where the DEM is NaN). Gaps are filled by nearest neighbour only for the gradient, then re-masked."""
    nan = np.isnan(dem)
    z = dem.astype(np.float64)
    if nan.any() and not nan.all():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    gy, gx = np.gradient(z * zexag, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    shade[nan] = np.nan
    return shade.astype(np.float32)


class RasterCache:
    """Small LRU of whole-raster arrays keyed by (path, mtime). DEM-class rasters are bounded by TerrainParams.max_cells."""

    def __init__(self, size: int = 4):
        self.size = size
        self._d: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> Tuple[np.ndarray, GridSpec, Optional[float]]:
        st = Path(path).stat()
        key = (str(path), st.st_mtime_ns)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        info = gt.read_georef(path)
        if info.grid is None:
            raise ValueError(f"{Path(path).name} has no georeferencing and cannot be placed on the map")
        arr = gt.read_full(path)
        item = (arr, info.grid, info.nodata)
        with self._lock:
            self._d[key] = item
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return item


class ShadeCache:
    def __init__(self, size: int = 3):
        self.size = size
        self._d: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path, dem: np.ndarray, res: float, zexag: float) -> np.ndarray:
        key = (str(path), Path(path).stat().st_mtime_ns, round(zexag, 2))
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        h = hillshade(dem, res, zexag)
        with self._lock:
            self._d[key] = h
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return h


RASTERS = RasterCache()
SHADES = ShadeCache()


def window_indices(grid: GridSpec, bbox: Tuple[float, float, float, float], w: int, h: int):
    """Source (row, col) index arrays for every output pixel centre, plus an in-bounds mask. bbox=(minx,miny,maxx,maxy)."""
    x0, y0, x1, y1 = bbox
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y1 - (np.arange(h) + 0.5) * (y1 - y0) / h
    cols = np.floor((xs - grid.x0) / grid.res).astype(np.int64)
    rows = np.floor((grid.y0 - ys) / grid.res).astype(np.int64)
    inside = ((rows[:, None] >= 0) & (rows[:, None] < grid.height)) & ((cols[None, :] >= 0) & (cols[None, :] < grid.width))
    return np.clip(rows, 0, grid.height - 1)[:, None], np.clip(cols, 0, grid.width - 1)[None, :], inside


def render_layer(path: Path, layer: str, bbox, w: int, h: int, ramp: str = "terrain", vmin: Optional[float] = None,
                 vmax: Optional[float] = None, shade: bool = False, zexag: float = 1.0, opacity: float = 1.0) -> np.ndarray:
    """RGBA uint8 (h, w, 4). Nodata and out-of-extent pixels are transparent."""
    arr, grid, nodata = RASTERS.get(path)
    r, c, inside = window_indices(grid, bbox, w, h)
    v = arr[r, c]
    out = np.zeros((h, w, 4), np.uint8)
    if layer == "confidence":
        for k, col in CONF_COLORS.items():
            out[(v == k) & inside] = col
    elif layer == "evidence":
        for k, col in EVID_COLORS.items():
            out[(v == k) & inside] = col
    elif layer == "source":
        rng = np.random.default_rng(5)
        pal = rng.integers(60, 230, (256, 3)).astype(np.uint8)
        pal[0] = 0
        out[..., :3] = pal[v.astype(np.uint8)]
        out[..., 3] = np.where((v > 0) & inside, 200, 0)
    else:
        vf = v.astype(np.float32)
        if nodata is not None and not np.isnan(nodata):
            vf[vf == nodata] = np.nan
        valid = np.isfinite(vf) & inside
        if vmin is None or vmax is None:
            src = arr[np.isfinite(arr)] if arr.size < 4_000_000 else arr[::max(1, arr.shape[0] // 1000), ::max(1, arr.shape[1] // 1000)]
            src = src[np.isfinite(src)]
            lo, hi = (np.percentile(src, [1, 99]) if src.size else (0.0, 1.0))
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        t = np.clip((vf - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        t = np.where(valid, t, 0)
        rgb = ramp_lut(ramp)[(t * 255).astype(np.uint8)].astype(np.float32)
        if shade:
            sh = SHADES.get(path, np.where(np.isfinite(arr), arr, np.nan), grid.res, zexag)[r, c]
            f = 0.45 + 0.55 * np.nan_to_num(sh, nan=0.7)
            rgb = rgb * f[..., None]
        out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(valid, 255, 0)
    if opacity < 1.0:
        out[..., 3] = (out[..., 3] * opacity).astype(np.uint8)
    return out


def png_rgba(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return buf.tobytes()


def raster_stats(path: Path) -> Dict[str, Optional[float]]:
    arr, grid, nodata = RASTERS.get(path)
    a = arr.astype(np.float64)
    if nodata is not None and not np.isnan(nodata):
        a[a == nodata] = np.nan
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {"min": None, "max": None, "mean": None, "p2": None, "p98": None, "valid_fraction": 0.0}
    p2, p98 = np.percentile(v, [2, 98])
    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "p2": float(p2), "p98": float(p98),
            "valid_fraction": float(v.size / a.size)}


def sample_point(path: Path, x: float, y: float, interpolate: bool = True) -> Optional[float]:
    """Value at a map coordinate; None if outside the raster or nodata. Categorical rasters use the containing cell."""
    from ..core.planning import bilinear_sample
    arr, grid, nodata = RASTERS.get(path)
    r, c = grid.xy_to_rc(float(x), float(y))
    r, c = int(np.floor(r)), int(np.floor(c))
    if not (0 <= r < grid.height and 0 <= c < grid.width):
        return None
    cell = arr[r, c]
    if arr.dtype.kind != "f":
        return float(cell)
    if nodata is not None and not np.isnan(nodata) and cell == nodata:
        return None
    if interpolate:
        v = float(bilinear_sample(arr, grid, np.array([x]), np.array([y]))[0])
        if not np.isnan(v):
            return v
    return None if np.isnan(cell) else float(cell)
