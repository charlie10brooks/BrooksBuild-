"""Slope, aspect, depression filling and D8 drainage. NumPy/SciPy only.

All functions treat NaN as nodata and never invent values for it. Hydrology results derived from an
*estimated* surface inherit its uncertainty: depressions and channels under dense canopy may be
interpolation artefacts, so callers should mask by terrain confidence before presenting them.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .dtm import GridSpec

# D8 neighbour offsets (drow, dcol) and distances in cell units
_OFFS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIST = [1.0, 2 ** 0.5, 1.0, 2 ** 0.5, 1.0, 2 ** 0.5, 1.0, 2 ** 0.5]


def slope_aspect(dem: np.ndarray, res: float) -> Tuple[np.ndarray, np.ndarray]:
    """Horn (1981) slope in degrees and downslope aspect (compass azimuth, 0 = N, clockwise).

    Cells with a NaN in their 3x3 neighbourhood are NaN. Flat cells have NaN aspect.
    """
    z = dem.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / (8 * res)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / (8 * res)
    bad = ndimage.binary_dilation(np.isnan(z), structure=np.ones((3, 3), bool))
    zf = np.where(np.isnan(z), 0.0, z)
    dzdx = ndimage.correlate(zf, kx, mode="nearest")      # per metre east
    dzdrow = ndimage.correlate(zf, ky, mode="nearest")    # per metre southward (rows increase south)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdrow)))
    aspect = np.degrees(np.arctan2(-dzdx, dzdrow)) % 360.0   # downslope = (-dz/dx east, +dz/drow = -dz/dnorth)
    flat = np.hypot(dzdx, dzdrow) < 1e-9
    slope[bad] = np.nan
    aspect[bad | flat] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)


def fill_depressions(dem: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Priority-Flood + epsilon (Barnes et al. 2014). NaN cells and the raster edge act as outlets."""
    H, W = dem.shape
    z = dem.astype(np.float64).ravel().tolist()
    nod = np.isnan(dem).ravel().tolist()
    closed = [False] * (H * W)
    heap: List[Tuple[float, int]] = []
    isnan2d = np.isnan(dem)
    near_nodata = ndimage.binary_dilation(isnan2d, structure=np.ones((3, 3), bool)) & ~isnan2d
    edge = np.zeros_like(isnan2d)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    for i in np.flatnonzero(((near_nodata | edge) & ~isnan2d).ravel()).tolist():
        closed[i] = True
        heap.append((z[i], i))
    heapq.heapify(heap)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        zc, c = pop(heap)
        r, col = divmod(c, W)
        for dr, dc in _OFFS:
            rr, cc = r + dr, col + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            n = rr * W + cc
            if closed[n] or nod[n]:
                continue
            closed[n] = True
            if z[n] <= zc:
                z[n] = zc + epsilon
            push(heap, (z[n], n))
    out = np.array(z, dtype=np.float64).reshape(H, W)
    out[isnan2d] = np.nan
    return out.astype(np.float32)


def flow_direction_d8(filled: np.ndarray) -> np.ndarray:
    """Index 0..7 into ``_OFFS`` of the steepest-descent neighbour, or -1 for outlets/nodata."""
    H, W = filled.shape
    z = filled.astype(np.float64)
    pad = np.pad(z, 1, constant_values=np.nan)
    best = np.full((H, W), -np.inf)
    d = np.full((H, W), -1, dtype=np.int8)
    for k, ((dr, dc), dist) in enumerate(zip(_OFFS, _DIST)):
        nb = pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        slope = (z - nb) / dist
        slope = np.where(np.isnan(nb), 1e-9, slope)           # nodata / raster edge = weak outlet
        better = (slope > best) & (slope > 0) & ~np.isnan(z)
        best = np.where(better, slope, best)
        d = np.where(better, k, d).astype(np.int8)
    return d


def _downstream_index(d: np.ndarray) -> np.ndarray:
    H, W = d.shape
    rr, cc = np.mgrid[0:H, 0:W]
    dr = np.array([o[0] for o in _OFFS] + [0])
    dc = np.array([o[1] for o in _OFFS] + [0])
    k = np.where(d < 0, 8, d)
    r2, c2 = rr + dr[k], cc + dc[k]
    ok = (d >= 0) & (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
    return np.where(ok, r2 * W + c2, -1).ravel()


def flow_accumulation(filled: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Upstream cell count (including the cell itself) for every valid cell."""
    ds = _downstream_index(d).tolist()
    valid = ~np.isnan(filled).ravel()
    order = np.argsort(-np.where(valid, filled.ravel(), -np.inf), kind="stable")
    acc = np.where(valid, 1.0, 0.0).tolist()
    for i in order.tolist():
        if not valid[i]:
            break
        j = ds[i]
        if j >= 0:
            acc[j] += acc[i]
    out = np.array(acc, dtype=np.float32).reshape(filled.shape)
    out[~valid.reshape(filled.shape)] = np.nan
    return out


def drainage_segments(acc: np.ndarray, d: np.ndarray, grid: GridSpec, min_area_m2: float,
                      max_features: int = 50_000) -> List[dict]:
    """Channel segments (cell centre -> downstream cell centre) above an upstream-area threshold."""
    min_cells = max(1.0, min_area_m2 / grid.cell_area)
    ds = _downstream_index(d)
    flat_acc = acc.ravel()
    sel = np.flatnonzero((flat_acc >= min_cells) & (ds >= 0))
    if sel.size > max_features:
        sel = sel[np.argsort(-flat_acc[sel])[:max_features]]
    H, W = acc.shape
    out = []
    for i in sel.tolist():
        r0, c0 = divmod(i, W)
        r1, c1 = divmod(int(ds[i]), W)
        x0, y0 = grid.cell_centers_xy(r0, c0)
        x1, y1 = grid.cell_centers_xy(r1, c1)
        out.append({"coords": [[float(x0), float(y0)], [float(x1), float(y1)]],
                    "upstream_area_m2": float(flat_acc[i] * grid.cell_area)})
    return out


def depression_report(dem: np.ndarray, filled: np.ndarray, res: float, min_depth_m: float = 0.15) -> dict:
    depth = np.where(np.isnan(dem), 0.0, filled.astype(np.float64) - dem)
    mask = depth >= min_depth_m
    labels, n = ndimage.label(mask)
    return {"mask": mask, "count": int(n), "area_m2": float(mask.sum() * res * res),
            "max_depth_m": float(depth.max()) if depth.size else 0.0,
            "volume_m3": float(depth[mask].sum() * res * res)}


def analyse_drainage(dem: np.ndarray, grid: GridSpec, min_channel_area_m2: float = 500.0) -> dict:
    filled = fill_depressions(dem)
    d = flow_direction_d8(filled)
    acc = flow_accumulation(filled, d)
    return {"filled": filled, "direction": d, "accumulation": acc,
            "segments": drainage_segments(acc, d, grid, min_channel_area_m2),
            "depressions": depression_report(dem, filled, grid.res)}
