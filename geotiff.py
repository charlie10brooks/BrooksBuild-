"""GeoTIFF read/write on top of ``tifffile`` (no GDAL required).

Why not rasterio here: exports must be verifiable in minimal environments, and what we need is small:
north-up rasters with an EPSG code, nodata and units, written tiled + compressed and read back by window.
Rasterio/GDAL remain the production reader for foreign files (see ``open_reader``).

Georeferencing rules
--------------------
* A product is written *georeferenced* only when an EPSG code is supplied. Otherwise it is written as a plain
  TIFF **and** the caller must document the (lack of) CRS in the sidecar metadata; ``write_geotiff`` returns
  ``georeferenced=False`` so exporters can surface that rather than hide it.
* Vertical datum/units are recorded in GeoKeys when given and always repeated in the sidecar JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import tifffile

from ..core.dtm import GridSpec
from ..core.tiling import Window

TAG_PIXEL_SCALE, TAG_TIEPOINT, TAG_GEOKEYS, TAG_GEODOUBLE, TAG_GEOASCII, TAG_NODATA = 33550, 33922, 34735, 34736, 34737, 42113

# linear unit codes (EPSG)
UNIT_CODES = {"metre": 9001, "foot": 9002, "us_survey_foot": 9003}
UNIT_NAMES = {v: k for k, v in UNIT_CODES.items()}
UNIT_TO_METRES = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200.0 / 3937.0}


try:  # float predictor (better DEM compression) needs imagecodecs; fall back to plain deflate without it
    import imagecodecs  # noqa: F401
    HAVE_IMAGECODECS = True
except ImportError:
    HAVE_IMAGECODECS = False


class GeoTiffError(ValueError):
    pass


@dataclass
class GeoInfo:
    width: int
    height: int
    bands: int
    dtype: str
    grid: Optional[GridSpec]            # None when the file carries no georeferencing
    epsg: Optional[int]                 # horizontal CRS (projected or geographic)
    vertical_epsg: Optional[int]
    horizontal_units: Optional[str]     # metre | foot | us_survey_foot | degree
    vertical_units: Optional[str]
    nodata: Optional[float]
    georeferenced: bool
    tile: Optional[Tuple[int, int]]     # (tile_h, tile_w) if tiled

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["grid"] = self.grid.to_dict() if self.grid else None
        return d


# ------------------------------------------------------------------------------ writing
def _geokeys(epsg: Optional[int], horizontal_units: str, vertical_epsg: Optional[int], vertical_units: str):
    if epsg is None:
        return None
    geographic = epsg in (4326, 4269, 4258, 4283, 6318) or 4000 <= epsg < 4900
    keys = [(1024, 0, 1, 2 if geographic else 1), (1025, 0, 1, 1)]          # model type, PixelIsArea
    if geographic:
        keys.append((2048, 0, 1, epsg))
    else:
        keys.append((3072, 0, 1, epsg))
        keys.append((3076, 0, 1, UNIT_CODES.get(horizontal_units, 9001)))
    if vertical_epsg:
        keys.append((4096, 0, 1, vertical_epsg))
    if vertical_units in UNIT_CODES:
        keys.append((4099, 0, 1, UNIT_CODES[vertical_units]))
    flat = [1, 1, 0, len(keys)] + [v for k in keys for v in k]
    return tuple(flat)


def _extratags(grid: Optional[GridSpec], epsg: Optional[int], nodata: Optional[float], horizontal_units: str,
               vertical_epsg: Optional[int], vertical_units: str):
    tags = []
    if grid is not None and epsg is not None:
        tags.append((TAG_PIXEL_SCALE, "d", 3, (grid.res, grid.res, 0.0), True))
        tags.append((TAG_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, grid.x0, grid.y0, 0.0), True))
        gk = _geokeys(epsg, horizontal_units, vertical_epsg, vertical_units)
        tags.append((TAG_GEOKEYS, "H", len(gk), gk, True))
    if nodata is not None:
        txt = "nan" if (isinstance(nodata, float) and math.isnan(nodata)) else repr(float(nodata)).rstrip("0").rstrip(".")
        tags.append((TAG_NODATA, "s", 0, txt, True))
    return tags


def write_geotiff(path: Path, data: np.ndarray, grid: Optional[GridSpec], epsg: Optional[int],
                  nodata: Optional[float] = None, horizontal_units: str = "metre",
                  vertical_epsg: Optional[int] = None, vertical_units: str = "metre",
                  tile: int = 256, compression: str = "deflate", description: str = "") -> dict:
    """Write an (H,W) or (H,W,B) array. Returns a small report incl. ``georeferenced``."""
    data = np.asarray(data)
    if data.ndim not in (2, 3):
        raise GeoTiffError("data must be (H,W) or (H,W,B)")
    if grid is not None and (grid.height, grid.width) != data.shape[:2]:
        raise GeoTiffError(f"grid {grid.shape} does not match array {data.shape[:2]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    big = data.nbytes > 3.5e9
    photometric = "rgb" if (data.ndim == 3 and data.shape[2] in (3, 4) and data.dtype == np.uint8) else "minisblack"
    kw = dict(photometric=photometric, compression=compression, bigtiff=big, description=description or None,
              extratags=_extratags(grid, epsg, nodata, horizontal_units, vertical_epsg, vertical_units))
    if data.dtype.kind == "f" and compression == "deflate" and HAVE_IMAGECODECS:
        kw["predictor"] = 3
    if data.shape[0] > tile and data.shape[1] > tile:
        kw["tile"] = (tile, tile)
    if photometric == "rgb" and data.shape[2] == 4:
        kw["extrasamples"] = ("unassalpha",)
    tifffile.imwrite(str(tmp), data, **kw)
    tmp.replace(path)
    return {"path": str(path), "georeferenced": bool(grid is not None and epsg is not None), "epsg": epsg,
            "width": int(data.shape[1]), "height": int(data.shape[0]), "dtype": data.dtype.name}


def write_geotiff_tiles(path: Path, width: int, height: int, dtype, tile_fn: Callable[[Window], np.ndarray],
                        grid: Optional[GridSpec], epsg: Optional[int], nodata: Optional[float] = None,
                        horizontal_units: str = "metre", vertical_epsg: Optional[int] = None,
                        vertical_units: str = "metre", tile: int = 256, compression: str = "deflate",
                        description: str = "") -> dict:
    """Stream a very large single-band raster to disk tile by tile; memory use is O(tile)."""
    dtype = np.dtype(dtype)

    def gen() -> Iterator[np.ndarray]:
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                w = Window(x, y, min(x + tile, width), min(y + tile, height))
                a = np.asarray(tile_fn(w), dtype=dtype)
                if a.shape != (w.height, w.width):
                    raise GeoTiffError(f"tile_fn returned {a.shape} for window {w}")
                pad = np.zeros((tile, tile), dtype=dtype)
                if nodata is not None:
                    pad[:] = nodata
                pad[:a.shape[0], :a.shape[1]] = a
                yield pad

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kw = dict(shape=(height, width), dtype=dtype, tile=(tile, tile), photometric="minisblack",
              compression=compression, bigtiff=height * width * dtype.itemsize > 3.5e9,
              description=description or None,
              extratags=_extratags(grid, epsg, nodata, horizontal_units, vertical_epsg, vertical_units))
    if dtype.kind == "f" and compression == "deflate" and HAVE_IMAGECODECS:
        kw["predictor"] = 3
    tifffile.imwrite(str(tmp), gen(), **kw)
    tmp.replace(path)
    return {"path": str(path), "georeferenced": bool(grid is not None and epsg is not None), "epsg": epsg,
            "width": width, "height": height, "dtype": dtype.name}


def write_sidecar(path: Path, info: dict) -> Path:
    """``<file>.meta.json`` next to a raster: CRS, units, nodata and processing notes. Always written."""
    p = Path(str(path) + ".meta.json")
    p.write_text(json.dumps(info, indent=2, sort_keys=True, default=str))
    return p


# ------------------------------------------------------------------------------ reading
def _decode_geokeys(page) -> dict:
    t = page.tags.get(TAG_GEOKEYS)
    if t is None:
        return {}
    v = list(t.value)
    n = v[3]
    out = {}
    for i in range(n):
        key, loc, count, val = v[4 + 4 * i: 8 + 4 * i]
        if loc == 0:
            out[key] = val
    return out


def read_georef(path: Path) -> GeoInfo:
    with tifffile.TiffFile(str(path)) as tf:
        pg = tf.pages[0]
        h, w = pg.shape[0], pg.shape[1]
        bands = pg.shape[2] if len(pg.shape) == 3 else 1
        tags = pg.tags
        gk = _decode_geokeys(pg)
        epsg = gk.get(3072) or gk.get(2048)
        grid = None
        if TAG_PIXEL_SCALE in tags and TAG_TIEPOINT in tags:
            sx, sy = tags[TAG_PIXEL_SCALE].value[:2]
            tp = tags[TAG_TIEPOINT].value
            x0 = tp[3] - tp[0] * sx
            y0 = tp[4] + tp[1] * sy
            if abs(sx - sy) > 1e-9 * max(abs(sx), 1e-12):
                raise GeoTiffError(f"non-square pixels ({sx} x {sy}) are not supported")
            grid = GridSpec(float(x0), float(y0), float(sx), w, h)
        nd = None
        if TAG_NODATA in tags:
            s = str(tags[TAG_NODATA].value).strip("\x00 ")
            try:
                nd = float(s)
            except ValueError:
                nd = None
        hu = UNIT_NAMES.get(gk.get(3076)) if gk.get(3076) else ("degree" if gk.get(2048) else None)
        vu = UNIT_NAMES.get(gk.get(4099))
        return GeoInfo(w, h, bands, np.dtype(pg.dtype).name, grid, int(epsg) if epsg else None,
                       int(gk[4096]) if gk.get(4096) else None, hu, vu, nd,
                       bool(grid is not None and epsg), (pg.tilelength, pg.tilewidth) if pg.is_tiled else None)


class TiffWindowReader:
    """Windowed reads of an uncompressed/compressed striped or tiled, chunky TIFF without loading it whole.

    Only the segments overlapping the window are read and decoded. Not thread-safe; open one per thread.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tf = tifffile.TiffFile(str(path))
        self._pg = self._tf.pages[0]
        pg = self._pg
        if pg.planarconfig != 1 and len(pg.shape) == 3:
            raise GeoTiffError("planar-separate TIFFs are not supported for windowed reads")
        self.height, self.width = pg.shape[0], pg.shape[1]
        self.bands = pg.shape[2] if len(pg.shape) == 3 else 1
        self.dtype = np.dtype(pg.dtype)
        self.info = read_georef(path)

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def read(self, window: Window) -> np.ndarray:
        """Returns (H,W) for single band, else (H,W,B)."""
        pg, fh = self._pg, self._tf.filehandle
        win = window.clip(self.width, self.height)
        if win is None or win != window:
            raise GeoTiffError(f"window {window} is outside the {self.width}x{self.height} raster")
        out = np.zeros((window.height, window.width, self.bands), dtype=self.dtype)
        if pg.is_tiled:
            th, tw = pg.tilelength, pg.tilewidth
            tiles_across = math.ceil(self.width / tw)
            idxs = [ty * tiles_across + tx
                    for ty in range(window.y0 // th, (window.y1 - 1) // th + 1)
                    for tx in range(window.x0 // tw, (window.x1 - 1) // tw + 1)]
        else:
            rps = pg.rowsperstrip
            idxs = list(range(window.y0 // rps, (window.y1 - 1) // rps + 1))
        for i in idxs:
            fh.seek(pg.dataoffsets[i])
            raw = fh.read(pg.databytecounts[i])
            seg, (_, _, y, x, _), _ = pg.decode(raw, i)
            seg = seg.reshape(seg.shape[-3:]) if seg.ndim > 3 else seg
            if seg.ndim == 2:
                seg = seg[..., None]
            sh, sw = seg.shape[0], seg.shape[1]
            y0, x0 = max(y, window.y0), max(x, window.x0)
            y1, x1 = min(y + sh, window.y1, self.height), min(x + sw, window.x1, self.width)
            if y1 > y0 and x1 > x0:
                out[y0 - window.y0:y1 - window.y0, x0 - window.x0:x1 - window.x0] = seg[y0 - y:y1 - y, x0 - x:x1 - x]
        return out[..., 0] if self.bands == 1 else out

    def read_rgb(self, window: Window) -> np.ndarray:
        a = self.read(window)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        a = a[..., :3]
        if a.dtype != np.uint8:
            a = np.clip(a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 1.0) * 255.0, 0, 255).astype(np.uint8)
        return a


def read_full(path: Path) -> np.ndarray:
    """Whole-array read; only for rasters known to be small (DEM grids, previews)."""
    return tifffile.imread(str(path))
