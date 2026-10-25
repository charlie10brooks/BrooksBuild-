"""Bare-earth terrain estimation from classified ground points.

Principles
----------
* **Measured is never overwritten.** Cells that contain ground returns keep their measured value; every
  other cell is *estimated*. A separate flag raster records which is which.
* **No flattening.** Interpolators either preserve local gradients (``local_plane``, ``tin``, ``tps``) or are
  limited to a search radius (``idw``, ``nearest``). Cells with no evidence within ``max_radius_m`` stay
  NaN: there is no global fill.
* **Confidence is evidence-based and heuristic.** It reflects distance to, and density of, real ground
  observations (optionally inflated on steep slopes). It is *not* a statistical confidence interval; use
  ``cross_validate`` to get measured error against distance for a given dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import cKDTree

NODATA = np.float32(np.nan)

# estimate-source flag values
FLAG_NONE, FLAG_MEASURED, FLAG_INTERPOLATED = 0, 1, 2
# confidence categories
CONF_UNKNOWN, CONF_LOW, CONF_MODERATE, CONF_HIGH = 0, 1, 2, 3
CONF_NAMES = {CONF_UNKNOWN: "unknown", CONF_LOW: "low", CONF_MODERATE: "moderate", CONF_HIGH: "high"}


class InsufficientGroundError(ValueError):
    pass


@dataclass(frozen=True)
class GridSpec:
    """North-up raster grid. (x0, y0) is the *top-left corner*; rows run downward."""

    x0: float
    y0: float
    res: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, minx: float, miny: float, maxx: float, maxy: float, res: float) -> "GridSpec":
        if res <= 0:
            raise ValueError("res must be positive")
        w = int(np.ceil((maxx - minx) / res))
        h = int(np.ceil((maxy - miny) / res))
        if w <= 0 or h <= 0:
            raise ValueError("empty bounds")
        return cls(float(minx), float(miny + h * res), float(res), w, h)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def cell_area(self) -> float:
        return self.res * self.res

    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    def xy_to_rc(self, x, y):
        """Continuous (row, col) in 'pixel-edge' space: cell (i,j) spans [i,i+1) x [j,j+1)."""
        return (self.y0 - np.asarray(y)) / self.res, (np.asarray(x) - self.x0) / self.res

    def rc_to_xy(self, r, c):
        return self.x0 + np.asarray(c) * self.res, self.y0 - np.asarray(r) * self.res

    def cell_centers_xy(self, rows, cols):
        return self.x0 + (np.asarray(cols) + 0.5) * self.res, self.y0 - (np.asarray(rows) + 0.5) * self.res

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "res": self.res, "width": self.width, "height": self.height}


# ------------------------------------------------------------------------------ gridding
def bin_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, grid: GridSpec, stat: str = "median"
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce points to one value per cell. Returns (z_grid float32 with NaN, count int32)."""
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
    r, c = grid.xy_to_rc(x, y)
    r, c = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
    ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width) & np.isfinite(z)
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    count = np.zeros(grid.shape, dtype=np.int32)
    if not ok.any():
        return out, count
    idx = r[ok] * grid.width + c[ok]
    zz = z[ok]
    order = np.lexsort((zz, idx))          # by cell, then ascending z
    idx_s, z_s = idx[order], zz[order]
    first = np.r_[True, idx_s[1:] != idx_s[:-1]]
    starts = np.flatnonzero(first)
    cnt = np.diff(np.r_[starts, idx_s.size])
    cells = idx_s[starts]
    if stat == "min":
        val = z_s[starts]
    elif stat == "max":
        val = z_s[starts + cnt - 1]
    elif stat == "median":
        val = 0.5 * (z_s[starts + (cnt - 1) // 2] + z_s[starts + cnt // 2])
    elif stat == "mean":
        val = np.add.reduceat(z_s, starts) / cnt
    else:
        raise ValueError(f"unknown stat {stat!r}")
    out.flat[cells] = val
    count.flat[cells] = cnt
    return out, count


# ------------------------------------------------------------------------- interpolation
@dataclass
class InterpParams:
    method: str = "local_plane"      # local_plane | idw | tin | tps | nearest
    k: int = 16                      # neighbours for local_plane / idw
    power: float = 2.0               # idw exponent
    max_radius_m: float = 30.0       # never estimate farther than this from an observation
    smoothing: float = 0.0           # tps smoothing
    tps_neighbors: int = 48
    extrap_clamp_m: float = 0.5      # local_plane may overshoot neighbour range by this + 25 %
    chunk: int = 200_000


def _query_xy(rc: np.ndarray, res: float) -> np.ndarray:
    return rc.astype(np.float64) * res


def interpolate_points(src_rc: np.ndarray, src_z: np.ndarray, tgt_rc: np.ndarray, res: float,
                       p: InterpParams) -> np.ndarray:
    """Estimate z at target cell positions from source cell positions (both (n,2) row/col arrays)."""
    if src_rc.shape[0] < 3:
        raise InsufficientGroundError("need at least 3 ground cells to interpolate")
    sxy, txy = _query_xy(src_rc, res), _query_xy(tgt_rc, res)
    out = np.full(len(txy), np.nan, dtype=np.float64)
    tree = cKDTree(sxy)
    if p.method == "nearest":
        d, i = tree.query(txy, distance_upper_bound=p.max_radius_m)
        ok = np.isfinite(d)
        out[ok] = src_z[i[ok]]
        return out
    if p.method == "tin":
        f = LinearNDInterpolator(sxy, src_z, fill_value=np.nan)
        out = f(txy)
        d, _ = tree.query(txy, distance_upper_bound=p.max_radius_m)
        out[~np.isfinite(d)] = np.nan
        return out
    if p.method == "tps":
        n = min(p.tps_neighbors, len(sxy))
        f = RBFInterpolator(sxy, src_z, neighbors=n, kernel="thin_plate_spline", smoothing=p.smoothing)
        d, _ = tree.query(txy, distance_upper_bound=p.max_radius_m)
        ok = np.isfinite(d)
        for s in range(0, int(ok.sum()), p.chunk):
            sel = np.flatnonzero(ok)[s:s + p.chunk]
            out[sel] = f(txy[sel])
        return out
    if p.method not in ("idw", "local_plane"):
        raise ValueError(f"unknown interpolation method {p.method!r}")

    k = min(p.k, len(sxy))
    for s in range(0, len(txy), p.chunk):
        q = txy[s:s + p.chunk]
        d, i = tree.query(q, k=k, distance_upper_bound=p.max_radius_m)
        if k == 1:
            d, i = d[:, None], i[:, None]
        valid = np.isfinite(d)
        zn = np.where(valid, src_z[np.minimum(i, len(src_z) - 1)], np.nan)
        w = np.where(valid, 1.0 / (np.where(valid, d, 1.0) ** p.power + 1e-9), 0.0)
        wsum = w.sum(axis=1)
        has = wsum > 0
        zw = np.where(valid, zn, 0.0)
        mean = np.full(len(q), np.nan)
        mean[has] = (w * zw).sum(axis=1)[has] / wsum[has]
        if p.method == "idw":
            out[s:s + p.chunk] = mean
            continue
        # weighted local plane z = a + b*dx + c*dy (preserves slope; IDW alone flattens it)
        nvalid = valid.sum(axis=1)
        dxy = np.where(valid[:, :, None], sxy[np.minimum(i, len(sxy) - 1)] - q[:, None, :], 0.0)
        A = np.stack([np.ones_like(d), dxy[:, :, 0], dxy[:, :, 1]], axis=2)         # (m,k,3)
        Aw = A * w[:, :, None]
        ata = np.einsum("mki,mkj->mij", Aw, A)
        atz = np.einsum("mki,mk->mi", Aw, zw)
        det = np.linalg.det(ata)
        good = (nvalid >= 4) & (np.abs(det) > 1e-12 * np.maximum(1.0, np.abs(ata).max(axis=(1, 2)) ** 3))
        est = mean.copy()
        if good.any():
            sol = np.linalg.solve(ata[good], atz[good][:, :, None])[:, :, 0]
            est[good] = sol[:, 0]
            lo = np.where(valid, zn, np.inf).min(axis=1)[good]
            hi = np.where(valid, zn, -np.inf).max(axis=1)[good]
            clamp = p.extrap_clamp_m + 0.25 * (hi - lo)
            est[good] = np.clip(est[good], lo - clamp, hi + clamp)
        out[s:s + p.chunk] = est
    return out


@dataclass
class DTMResult:
    dtm: np.ndarray           # float32, NaN where there is no estimate
    flag: np.ndarray          # uint8: 0 none, 1 measured, 2 interpolated
    count: np.ndarray         # ground points per cell
    dist_m: np.ndarray        # float32 distance to nearest measured cell (m)
    measured: np.ndarray      # bool
    params: InterpParams
    stat: str


def build_dtm(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, grid: GridSpec,
              params: Optional[InterpParams] = None, stat: str = "median", min_points: int = 3) -> DTMResult:
    """Grid ground points, keep them as measured cells, estimate the rest within ``max_radius_m``."""
    params = params or InterpParams()
    measured_z, count = bin_points(gx, gy, gz, grid, stat)
    measured = ~np.isnan(measured_z)
    if int(measured.sum()) < min_points:
        raise InsufficientGroundError(
            f"only {int(measured.sum())} ground cells; need >= {min_points}. "
            "Dense canopy may have blocked ground returns: check the ground classification, "
            "or add LiDAR / survey data.")
    dist_m = (ndimage.distance_transform_edt(~measured) * grid.res).astype(np.float32)
    dtm = measured_z.copy()
    flag = np.where(measured, FLAG_MEASURED, FLAG_NONE).astype(np.uint8)
    # only cells that are within reach of an observation are worth estimating
    target = (~measured) & (dist_m <= params.max_radius_m)
    if target.any():
        src_rc = np.argwhere(measured)
        tgt_rc = np.argwhere(target)
        est = interpolate_points(src_rc, measured_z[measured].astype(np.float64), tgt_rc, grid.res, params)
        est = est.astype(np.float32)
        dtm[target] = est
        ok = np.isfinite(est)
        tr, tc = tgt_rc[ok, 0], tgt_rc[ok, 1]
        flag[tr, tc] = FLAG_INTERPOLATED
    return DTMResult(dtm, flag, count, dist_m, measured, params, stat)


# ---------------------------------------------------------------------------- confidence
@dataclass
class ConfidenceParams:
    high_dist_m: float = 2.0
    moderate_dist_m: float = 6.0
    low_dist_m: float = 15.0
    density_radius_m: float = 4.0
    min_density_high: float = 0.10     # fraction of cells within radius that are measured
    slope_penalty: float = 2.0         # effective distance *= 1 + slope_penalty * tan(slope)


def slope_tan(dem: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(np.float64), res)
    return np.hypot(gx, gy)


def terrain_confidence(dtm: np.ndarray, measured: np.ndarray, res: float,
                       p: Optional[ConfidenceParams] = None) -> np.ndarray:
    """Per-cell evidence class: high / moderate / low / unknown (uint8, see CONF_*)."""
    p = p or ConfidenceParams()
    dist = ndimage.distance_transform_edt(~measured) * res
    s = slope_tan(np.where(np.isnan(dtm), np.nanmean(dtm) if np.isfinite(dtm).any() else 0.0, dtm), res)
    d_eff = dist * (1.0 + p.slope_penalty * s)
    size = max(1, int(round(2 * p.density_radius_m / res)) | 1)
    density = ndimage.uniform_filter(measured.astype(np.float32), size=size, mode="constant")
    conf = np.zeros(dtm.shape, dtype=np.uint8)
    valid = ~np.isnan(dtm)
    conf[valid & (d_eff <= p.low_dist_m)] = CONF_LOW
    conf[valid & (d_eff <= p.moderate_dist_m)] = CONF_MODERATE
    conf[valid & (d_eff <= p.high_dist_m) & (density >= p.min_density_high)] = CONF_HIGH
    return conf


def confidence_summary(conf: np.ndarray, res: float, region: Optional[np.ndarray] = None) -> dict:
    sel = conf if region is None else conf[region]
    total = max(1, sel.size)
    out = {}
    for k, name in CONF_NAMES.items():
        n = int((sel == k).sum())
        out[name] = {"cells": n, "area_m2": n * res * res, "fraction": n / total}
    weak = out["low"]["fraction"] + out["unknown"]["fraction"]
    out["weak_fraction"] = weak
    out["advice"] = (
        "Ground evidence is weak across much of this area (dense canopy likely blocked ground returns). "
        "Supplemental ground-based survey, LiDAR, or other elevation data is recommended before relying on this surface."
        if weak > 0.25 else "")
    return out


# ------------------------------------------------------------------------------- sources
@dataclass
class ElevationSource:
    id: int                      # 1..255
    name: str
    kind: str                    # photogrammetry_ground | lidar | external_dem | survey | other
    priority: int                # higher wins
    data: np.ndarray             # float32 on the project grid, NaN = no data
    measured: Optional[np.ndarray] = None
    z_offset: float = 0.0        # vertical datum correction added to this source
    meta: dict = field(default_factory=dict)


@dataclass
class MergeResult:
    dtm: np.ndarray
    source_id: np.ndarray        # uint8, 0 = none
    measured: np.ndarray


def merge_sources(sources: Sequence[ElevationSource]) -> MergeResult:
    """Per-cell pick of the highest-priority valid source (ties -> lower id). Preserves provenance."""
    if not sources:
        raise ValueError("no sources")
    shape = sources[0].data.shape
    ordered = sorted(sources, key=lambda s: (s.priority, -s.id))   # ascending priority; later wins
    dtm = np.full(shape, np.nan, dtype=np.float32)
    sid = np.zeros(shape, dtype=np.uint8)
    meas = np.zeros(shape, dtype=bool)
    for s in ordered:
        if s.data.shape != shape:
            raise ValueError(f"source {s.name!r} grid mismatch")
        valid = ~np.isnan(s.data)
        dtm[valid] = s.data[valid] + np.float32(s.z_offset)
        sid[valid] = s.id
        meas[valid] = True if s.measured is None else s.measured[valid]
    return MergeResult(dtm, sid, meas)


def estimate_vertical_offset(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None,
                             min_cells: int = 50) -> dict:
    """Robust median(a - b) over cells where both exist. Use to reconcile vertical datums
    (e.g. drone ellipsoid heights vs. NAVD88 LiDAR). The user should confirm before applying."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    if mask is not None:
        valid &= mask
    n = int(valid.sum())
    if n < min_cells:
        raise ValueError(f"only {n} overlapping cells; need >= {min_cells} to estimate an offset")
    diff = (a[valid] - b[valid]).astype(np.float64)
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) * 1.4826
    inl = np.abs(diff - med) <= 3 * max(mad, 1e-6)
    med2 = float(np.median(diff[inl]))
    return {"offset_m": med2, "robust_sigma_m": mad, "cells": n, "inlier_fraction": float(inl.mean()),
            "warning": "Sources disagree by more than a constant offset (tilt, scale or real change); "
                       "a single vertical offset may not be appropriate." if mad > 0.5 else ""}


# ------------------------------------------------------------------------------ validation
def cross_validate(measured_z: np.ndarray, grid: GridSpec, params: Optional[InterpParams] = None,
                   holdout: str = "block", fraction: float = 0.15, block_m: float = 20.0, seed: int = 0,
                   bin_edges_m: Sequence[float] = (0, 1, 2, 4, 8, 15, 30, 60)) -> dict:
    """Hold out measured cells, re-estimate them, report error vs. distance to nearest remaining observation.

    ``block`` removes whole spatial blocks, mimicking canopy gaps, and gives realistic gap-filling error.
    ``random`` removes scattered cells and is optimistic. Error is measured against *observed ground*,
    so it cannot capture systematic bias where ground was never observed (e.g. under dense canopy).
    """
    params = params or InterpParams()
    rng = np.random.default_rng(seed)
    meas = ~np.isnan(measured_z)
    rc = np.argwhere(meas)
    if len(rc) < 20:
        raise InsufficientGroundError("not enough measured cells for cross-validation")
    if holdout == "random":
        hold = rng.random(len(rc)) < fraction
    elif holdout == "block":
        b = max(1, int(round(block_m / grid.res)))
        keys = (rc[:, 0] // b) * 100003 + (rc[:, 1] // b)
        uniq = np.unique(keys)
        chosen = rng.random(len(uniq)) < fraction
        if not chosen.any():
            chosen[rng.integers(len(uniq))] = True
        hold = np.isin(keys, uniq[chosen])
    else:
        raise ValueError("holdout must be 'block' or 'random'")
    if hold.all() or not hold.any():
        raise InsufficientGroundError("degenerate hold-out split; adjust fraction/block size")
    train, test = rc[~hold], rc[hold]
    z_train = measured_z[train[:, 0], train[:, 1]].astype(np.float64)
    z_true = measured_z[test[:, 0], test[:, 1]].astype(np.float64)
    pred = interpolate_points(train, z_train, test, grid.res, params)
    dist, _ = cKDTree(train.astype(np.float64) * grid.res).query(test.astype(np.float64) * grid.res)
    err = pred - z_true
    rows = []
    edges = list(bin_edges_m)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (dist >= lo) & (dist < hi)
        pe = err[sel & np.isfinite(err)]
        rows.append({"dist_from_m": lo, "dist_to_m": hi, "n": int(pe.size),
                     "rmse_m": float(np.sqrt(np.mean(pe ** 2))) if pe.size else None,
                     "bias_m": float(pe.mean()) if pe.size else None,
                     "mae_m": float(np.abs(pe).mean()) if pe.size else None,
                     "p90_abs_m": float(np.percentile(np.abs(pe), 90)) if pe.size else None})
    fin = np.isfinite(err)
    return {"holdout": holdout, "train_cells": int(len(train)), "test_cells": int(len(test)),
            "unpredicted_cells": int((~fin).sum()),
            "overall_rmse_m": float(np.sqrt(np.mean(err[fin] ** 2))) if fin.any() else None,
            "by_distance": rows,
            "note": "Error is measured against observed ground only; it does not capture bias where ground was never observed."}
