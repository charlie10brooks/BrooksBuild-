"""Selection geometry for the classification editor.

All coordinates are *global raster pixel coordinates* in the browser convention: pixel
(col, row) covers [col, col+1) x [row, row+1), so its centre is at (col+.5, row+.5).
Functions take the ``window`` they should rasterise into and return window-shaped bool masks.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .tiling import Window

Point = Tuple[float, float]
_SHIFT = 4
_SCALE = 1 << _SHIFT


def _to_local_fixed(points: Sequence[Point], window: Window) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("non-finite coordinate")
    p = p - 0.5 - np.array([window.x0, window.y0])   # OpenCV integer coords are pixel centres
    return np.round(p * _SCALE).astype(np.int32)


def polygon_mask(points: Sequence[Point], window: Window) -> np.ndarray:
    """Even-odd polygon fill (also used for lasso). Needs >= 3 vertices.

    A pixel is selected iff its *centre* lies inside the polygon, the same rule as
    ``rect_mask``. (``cv2.fillPoly`` also paints the outline, which inflates every polygon by
    about half a pixel, so it is deliberately not used here.) Self-intersecting lassos are
    handled by the even-odd rule.
    """
    if len(points) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("non-finite coordinate")
    H, W = window.shape
    out = np.zeros((H, W), dtype=bool)
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    keep = y0 != y1
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    if x0.size == 0 or H == 0 or W == 0:
        return out
    # rows whose centre (r + .5) lies in the half-open span [ylo, yhi) of each edge
    r0 = np.maximum(np.ceil(np.minimum(y0, y1) - 0.5).astype(np.int64), window.y0)
    r1 = np.minimum(np.ceil(np.maximum(y0, y1) - 0.5).astype(np.int64), window.y1)
    n = np.maximum(r1 - r0, 0)
    total = int(n.sum())
    if total == 0:
        return out
    edge = np.repeat(np.arange(n.size), n)
    rows = r0[edge] + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))
    xs = x0[edge] + (rows + 0.5 - y0[edge]) * (x1[edge] - x0[edge]) / (y1[edge] - y0[edge])
    order = np.lexsort((xs, rows))
    rows, xs = rows[order], xs[order]
    if rows.size % 2:                      # numerical edge case; a closed polygon is always even
        rows, xs = rows[:-1], xs[:-1]
    ra, xa, xb = rows[0::2], xs[0::2], xs[1::2]
    c0 = np.clip(np.ceil(xa - 0.5).astype(np.int64) - window.x0, 0, W)
    c1 = np.clip(np.ceil(xb - 0.5).astype(np.int64) - window.x0, 0, W)
    ok = c1 > c0
    diff = np.zeros((H, W + 1), dtype=np.int32)
    np.add.at(diff, (ra[ok] - window.y0, c0[ok]), 1)
    np.add.at(diff, (ra[ok] - window.y0, c1[ok]), -1)
    return np.cumsum(diff[:, :W], axis=1) > 0


def rect_mask(x0: float, y0: float, x1: float, y1: float, window: Window) -> np.ndarray:
    """Pixels whose centre lies inside the rectangle."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    cols = np.arange(window.x0, window.x1) + 0.5
    rows = np.arange(window.y0, window.y1) + 0.5
    return ((rows[:, None] >= ya) & (rows[:, None] < yb) &
            (cols[None, :] >= xa) & (cols[None, :] < xb))


def brush_mask(points: Sequence[Point], radius: float, window: Window) -> np.ndarray:
    """Round-capped stroke through ``points`` with the given pixel radius."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    img = np.zeros(window.shape, dtype=np.uint8)
    pts = _to_local_fixed(points, window)
    r = int(round(radius * _SCALE))
    for i, p in enumerate(pts):
        cv2.circle(img, (int(p[0]), int(p[1])), r, 1, thickness=-1, lineType=cv2.LINE_8, shift=_SHIFT)
        if i:
            q = pts[i - 1]
            cv2.line(img, (int(q[0]), int(q[1])), (int(p[0]), int(p[1])), 1,
                     thickness=max(1, int(round(2 * radius))), lineType=cv2.LINE_8, shift=_SHIFT)
    return img.astype(bool)


# ----------------------------------------------------------------------------- colour
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H,W,3) -> float32 CIE Lab with L in 0..100."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def color_distance(lab: np.ndarray, seed_lab: np.ndarray) -> np.ndarray:
    """CIE76 delta-E between every pixel and one reference colour."""
    d = lab - seed_lab.reshape(1, 1, 3)
    return np.sqrt((d * d).sum(axis=-1))


def similar_mask(rgb: np.ndarray, seed_rc: Tuple[int, int], tolerance: float,
                 restrict: Optional[np.ndarray] = None, sample_radius: int = 1) -> np.ndarray:
    """Select-by-colour: every pixel in the window within ``tolerance`` delta-E of the seed.

    The reference colour is the mean Lab of a (2r+1)^2 patch around the seed so a single noisy
    pixel does not dominate. Not connected-component limited.
    """
    lab = rgb_to_lab(rgb)
    r, c = seed_rc
    h, w = lab.shape[:2]
    if not (0 <= r < h and 0 <= c < w):
        raise ValueError("seed outside window")
    patch = lab[max(0, r - sample_radius): r + sample_radius + 1, max(0, c - sample_radius): c + sample_radius + 1]
    ref = patch.reshape(-1, 3).mean(axis=0)
    sel = color_distance(lab, ref) <= float(tolerance)
    if restrict is not None:
        sel &= restrict
    sel[r, c] = True
    return sel


def flood_select(rgb: np.ndarray, seed_rc: Tuple[int, int], tolerance: float, connectivity: int = 4,
                 restrict: Optional[np.ndarray] = None, sample_radius: int = 1) -> Tuple[np.ndarray, bool]:
    """Flood fill selection: the connected region of similar colour that contains the seed.

    Returns (mask, touches_window_edge). If the region reaches the edge of the analysis window
    it may continue beyond it; callers should tell the user the fill was truncated.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    cand = similar_mask(rgb, seed_rc, tolerance, restrict, sample_radius)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, _ = ndimage.label(cand, structure=structure)
    region = labels == labels[seed_rc]
    edge = bool(region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any())
    return region, edge


def clean_mask(sel: np.ndarray, min_area: int = 0, fill_holes: bool = False, smooth_radius: int = 0) -> np.ndarray:
    """Tidy a selection: morphological smoothing, small-blob removal, hole filling."""
    out = sel.copy()
    if smooth_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * smooth_radius + 1,) * 2)
        u = out.astype(np.uint8)
        u = cv2.morphologyEx(cv2.morphologyEx(u, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        out = u.astype(bool)
    if min_area > 0:
        labels, n = ndimage.label(out)
        if n:
            sizes = ndimage.sum(out, labels, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:] = sizes >= min_area
            out = keep[labels]
    if fill_holes:
        out = ndimage.binary_fill_holes(out)
    return out
