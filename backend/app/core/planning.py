"""Preliminary site-planning calculations on an estimated terrain surface.

Everything returned from here is a **preliminary planning estimate**. It is only as good as the
underlying surface and its confidence layer, and it is not suitable for final foundation design,
permitting, or construction staking without independent survey validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import regions
from .dtm import CONF_HIGH, CONF_LOW, CONF_MODERATE, CONF_NAMES, CONF_UNKNOWN, GridSpec
from .tiling import Window

DISCLAIMER = ("Preliminary planning estimate only. Not suitable for final foundation design, permitting or "
              "construction staking without validation by a licensed surveyor.")

Point = Tuple[float, float]


def polygon_area(points: Sequence[Point]) -> float:
    """Shoelace area in squared map units. Requires a *projected* CRS (metres/feet), not lon/lat."""
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_window(points: Sequence[Point], grid: GridSpec) -> Optional[Window]:
    r, c = grid.xy_to_rc(np.array([p[0] for p in points]), np.array([p[1] for p in points]))
    w = Window(int(np.floor(c.min())), int(np.floor(r.min())), int(np.ceil(c.max())) + 1, int(np.ceil(r.max())) + 1)
    return w.clip(grid.width, grid.height)


def polygon_cell_mask(points: Sequence[Point], grid: GridSpec) -> Tuple[Optional[Window], Optional[np.ndarray]]:
    """Cells whose centre lies inside the polygon, as (window, mask-in-window)."""
    win = polygon_window(points, grid)
    if win is None:
        return None, None
    pts = [(float(grid.xy_to_rc(x, y)[1]), float(grid.xy_to_rc(x, y)[0])) for x, y in points]   # (col,row)
    return win, regions.polygon_mask(pts, win)


def bilinear_sample(arr: np.ndarray, grid: GridSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample at map coordinates. NaN if outside or any contributing cell is nodata (no invention)."""
    r, c = grid.xy_to_rc(np.asarray(x, float), np.asarray(y, float))
    r, c = r - 0.5, c - 0.5                      # cell-centre index space
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0
    out = np.full(np.shape(r), np.nan)
    ok = (r0 >= 0) & (r0 + 1 < grid.height) & (c0 >= 0) & (c0 + 1 < grid.width)
    if ok.any():
        a = arr.astype(np.float64)
        v = ((1 - fr) * (1 - fc) * a[r0[ok], c0[ok]] + (1 - fr) * fc * a[r0[ok], c0[ok] + 1]
             + fr * (1 - fc) * a[r0[ok] + 1, c0[ok]] + fr * fc * a[r0[ok] + 1, c0[ok] + 1])
        out[ok] = v
    return out


def nearest_sample(arr: np.ndarray, grid: GridSpec, x, y, fill=0):
    r, c = grid.xy_to_rc(np.asarray(x, float), np.asarray(y, float))
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    out = np.full(np.shape(r), fill, dtype=arr.dtype)
    out[ok] = arr[r[ok], c[ok]]
    return out


def elevation_profile(dem: np.ndarray, grid: GridSpec, line: Sequence[Point], spacing: Optional[float] = None,
                      conf: Optional[np.ndarray] = None, dsm: Optional[np.ndarray] = None) -> dict:
    """Profile along a polyline. Samples with no valid surface are NaN, never interpolated across."""
    pts = np.asarray(line, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("profile needs at least 2 points")
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = float(cum[-1])
    if total <= 0:
        raise ValueError("zero-length profile")
    step = spacing or grid.res
    if step <= 0:
        raise ValueError("spacing must be positive")
    if total / step > 200_000:
        raise ValueError("profile would have too many samples; increase spacing")
    s = np.unique(np.r_[np.arange(0.0, total, step), total])
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    z = bilinear_sample(dem, grid, x, y)
    out = {"distance_m": s, "x": x, "y": y, "ground_z": z, "total_length": total}
    if conf is not None:
        out["confidence"] = nearest_sample(conf, grid, x, y, CONF_UNKNOWN)
    if dsm is not None:
        sz = bilinear_sample(dsm, grid, x, y)
        out["surface_z"] = sz
        out["canopy_height"] = sz - z
    valid = ~np.isnan(z)
    out["valid_fraction"] = float(valid.mean())
    if valid.sum() >= 2:
        out["min_z"], out["max_z"] = float(np.nanmin(z)), float(np.nanmax(z))
        ds = np.diff(s[valid]); dz = np.diff(z[valid])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.abs(dz) / np.where(ds > 0, ds, np.nan)
        out["max_grade_percent"] = float(np.nanmax(g) * 100) if np.isfinite(g).any() else None
    return out


def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    xm, ym = x.mean(), y.mean()
    A = np.c_[np.ones_like(x), x - xm, y - ym]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])       # z at centroid, dz/dx, dz/dy


def footprint_report(dem: np.ndarray, grid: GridSpec, polygon: Sequence[Point], conf: Optional[np.ndarray] = None,
                     dsm: Optional[np.ndarray] = None, steep_deg: float = 15.0) -> dict:
    """Elevation, slope, relief and evidence summary under a proposed footprint."""
    win, inside = polygon_cell_mask(polygon, grid)
    area = polygon_area(polygon)
    rep: Dict[str, object] = {"area_map_units2": area, "disclaimer": DISCLAIMER, "warnings": []}
    if win is None or inside is None or not inside.any():
        rep["status"] = "outside_surface"
        rep["warnings"].append("Footprint does not overlap the terrain surface.")
        return rep
    sub = dem[win.slices()]
    valid = inside & ~np.isnan(sub)
    n_in, n_valid = int(inside.sum()), int(valid.sum())
    rep["cells_in_footprint"], rep["cells_with_surface"] = n_in, n_valid
    rep["coverage_fraction"] = n_valid / n_in
    if n_valid < 3:
        rep["status"] = "insufficient_surface"
        rep["warnings"].append("Fewer than 3 terrain cells under the footprint; no estimate is possible.")
        return rep
    rr, cc = np.nonzero(valid)
    x, y = grid.cell_centers_xy(rr + win.y0, cc + win.x0)
    z = sub[valid].astype(np.float64)
    z0, dzdx, dzdy = _fit_plane(x, y, z)
    resid = z - (z0 + dzdx * (x - x.mean()) + dzdy * (y - y.mean()))
    slope = np.hypot(dzdx, dzdy)
    rep.update({
        "status": "ok",
        "min_z": float(z.min()), "max_z": float(z.max()), "mean_z": float(z.mean()),
        "relief": float(z.max() - z.min()),
        "p5_z": float(np.percentile(z, 5)), "p95_z": float(np.percentile(z, 95)),
        "best_fit_plane": {"z_at_centroid": z0, "dz_dx": dzdx, "dz_dy": dzdy},
        "mean_slope_percent": float(slope * 100), "mean_slope_degrees": float(np.degrees(np.arctan(slope))),
        "downslope_azimuth_deg": float(np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0),
        "plane_residual_rms": float(np.sqrt(np.mean(resid ** 2))),
    })
    from .terrain_analysis import slope_aspect
    sl, _ = slope_aspect(dem[max(0, win.y0 - 1):win.y1 + 1, max(0, win.x0 - 1):win.x1 + 1], grid.res)
    pad_r, pad_c = win.y0 - max(0, win.y0 - 1), win.x0 - max(0, win.x0 - 1)
    sl = sl[pad_r:pad_r + win.height, pad_c:pad_c + win.width]
    sv = sl[valid]
    if np.isfinite(sv).any():
        rep["max_cell_slope_degrees"] = float(np.nanmax(sv))
        rep["steep_fraction"] = float(np.nanmean(sv > steep_deg))
    if conf is not None:
        cs = conf[win.slices()][inside]
        total = max(1, cs.size)
        rep["confidence"] = {CONF_NAMES[k]: float((cs == k).sum() / total) for k in CONF_NAMES}
        weak = rep["confidence"]["low"] + rep["confidence"]["unknown"]
        if weak > 0.25:
            rep["warnings"].append(
                f"{weak:.0%} of the footprint rests on low-confidence or unknown ground. Supplemental "
                "ground survey, LiDAR or other elevation data is recommended before relying on these numbers.")
    if rep["coverage_fraction"] < 0.9:                                  # type: ignore[operator]
        rep["warnings"].append("Part of the footprint has no terrain estimate; statistics cover only the covered part.")
    if dsm is not None:
        ds = dsm[win.slices()]
        both = valid & ~np.isnan(ds)
        if both.any():
            rep["mean_canopy_height"] = float((ds[both] - sub[both]).mean())
    # corner elevations (bilinear; NaN if unsupported)
    pts = np.asarray(polygon, float)
    rep["vertex_elevations"] = [{"x": float(a), "y": float(b), "z": (None if np.isnan(v) else float(v))}
                                for (a, b), v in zip(pts, bilinear_sample(dem, grid, pts[:, 0], pts[:, 1]))]
    return rep


@dataclass
class ProposedSurface:
    """Plane z = z_ref + dzdx*(x-x_ref) + dzdy*(y-y_ref). ``flat(z)`` is a level pad."""
    z_ref: float
    x_ref: float = 0.0
    y_ref: float = 0.0
    dzdx: float = 0.0
    dzdy: float = 0.0

    @classmethod
    def flat(cls, z: float) -> "ProposedSurface":
        return cls(z_ref=z)

    @classmethod
    def from_three_points(cls, p1, p2, p3) -> "ProposedSurface":
        a, b, c = (np.asarray(p, float) for p in (p1, p2, p3))
        n = np.cross(b - a, c - a)
        if abs(n[2]) < 1e-12:
            raise ValueError("points are collinear; cannot define a plane")
        return cls(z_ref=float(a[2]), x_ref=float(a[0]), y_ref=float(a[1]), dzdx=float(-n[0] / n[2]), dzdy=float(-n[1] / n[2]))

    def z_at(self, x, y):
        return self.z_ref + self.dzdx * (np.asarray(x) - self.x_ref) + self.dzdy * (np.asarray(y) - self.y_ref)


def cut_fill(dem: np.ndarray, grid: GridSpec, polygon: Sequence[Point], surface: ProposedSurface,
             conf: Optional[np.ndarray] = None, min_coverage: float = 0.95, max_weak_fraction: float = 0.25,
             error_rmse_m: Optional[float] = None) -> dict:
    """Preliminary earthwork volumes between the estimated ground and a proposed surface.

    Refuses (status ``refused``) when the existing-ground evidence under the polygon is not
    defensible: too many cells without an estimate, or too much low-confidence / unknown ground.
    Volumes are in cubic map units (m^3 for metre grids); fill = proposed above existing.
    """
    out: Dict[str, object] = {"disclaimer": DISCLAIMER}
    if conf is None:
        out.update(status="refused", reason="No terrain-confidence layer is available, so the existing-ground "
                                            "surface cannot be judged defensible. Run terrain reconstruction first.")
        return out
    win, inside = polygon_cell_mask(polygon, grid)
    if win is None or inside is None or not inside.any():
        out.update(status="refused", reason="Polygon does not overlap the terrain surface.")
        return out
    sub, cs = dem[win.slices()], conf[win.slices()]
    n_in = int(inside.sum())
    have = inside & ~np.isnan(sub)
    coverage = float(have.sum() / n_in)
    weak = float((inside & ((cs <= CONF_LOW) | np.isnan(sub))).sum() / n_in)   # low + unknown + missing
    out.update(coverage_fraction=coverage, weak_fraction=weak)
    if coverage < min_coverage:
        out.update(status="refused", reason=f"Only {coverage:.0%} of the polygon has an existing-ground estimate "
                                            f"(need >= {min_coverage:.0%}). Add ground survey or LiDAR data.")
        return out
    if weak > max_weak_fraction:
        out.update(status="refused", reason=f"{weak:.0%} of the polygon rests on low-confidence or unknown ground "
                                            f"(limit {max_weak_fraction:.0%}). Supplemental ground survey, LiDAR or other "
                                            "elevation data is needed for a defensible estimate.")
        return out
    rr, cc = np.nonzero(have)
    x, y = grid.cell_centers_xy(rr + win.y0, cc + win.x0)
    diff = surface.z_at(x, y) - sub[have].astype(np.float64)        # + = fill, - = cut
    a = grid.cell_area
    fill, cut = float(diff[diff > 0].sum() * a), float(-diff[diff < 0].sum() * a)
    by_conf = {}
    ccs = cs[have]
    for k in (CONF_HIGH, CONF_MODERATE, CONF_LOW):
        sel = ccs == k
        by_conf[CONF_NAMES[k]] = {"cells": int(sel.sum()),
                                  "cut": float(-diff[sel & (diff < 0)].sum() * a),
                                  "fill": float(diff[sel & (diff > 0)].sum() * a)}
    out.update(status="ok", cut=cut, fill=fill, net=fill - cut, area=float(have.sum() * a),
               mean_depth_of_change=float(diff.mean()), by_confidence=by_conf,
               proposed={"z_ref": surface.z_ref, "dzdx": surface.dzdx, "dzdy": surface.dzdy})
    if error_rmse_m is not None:
        unc = float(have.sum() * a * error_rmse_m)
        out["volume_uncertainty_worst_case"] = unc
        out["uncertainty_note"] = ("Worst-case systematic bias of one RMSE across the whole polygon, from "
                                   "cross-validation at the relevant distance; real error may be larger "
                                   "where ground was never observed.")
    return out
