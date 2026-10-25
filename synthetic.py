"""Synthetic development/test dataset: **not real survey data** and always labelled as such.

A known bare-earth surface (hills, a gully, a gentle regional slope) is covered with tree crowns, brush patches and
open grass/soil. A photogrammetry-like point cloud is sampled from it: where canopy is present the sensor sees the
canopy top, and only a small fraction of samples reach the ground through gaps. Because the true terrain is known
we can measure how well the ground filter + DTM pipeline recovers it, including under canopy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy import ndimage

from ..core import classes as C
from ..core.dtm import GridSpec

SYNTHETIC_NOTICE = "SYNTHETIC SAMPLE DATA - generated for development and testing; not a real site."


@dataclass
class SyntheticSite:
    grid: GridSpec
    terrain: np.ndarray          # true bare earth (m)
    canopy_h: np.ndarray         # vegetation height above ground (m)
    dsm: np.ndarray              # terrain + canopy_h
    labels: np.ndarray           # uint8 class ids (core.classes)
    rgb: np.ndarray              # uint8 (H,W,3) orthophoto-like image
    epsg: int = 32616
    notice: str = SYNTHETIC_NOTICE

    def points(self, density: float = 4.0, canopy_gap: float = 0.04, seed: int = 1,
               ground_sigma: float = 0.05, canopy_sigma: float = 0.15) -> Dict[str, np.ndarray]:
        """Sample a cloud. Under canopy only ``canopy_gap`` of the returns reach the ground."""
        rng = np.random.default_rng(seed)
        g = self.grid
        n = int(g.width * g.height * g.res ** 2 * density)
        x = g.x0 + rng.random(n) * g.width * g.res
        y = g.y0 - rng.random(n) * g.height * g.res
        r, c = np.clip(g.xy_to_rc(x, y)[0].astype(int), 0, g.height - 1), np.clip(g.xy_to_rc(x, y)[1].astype(int), 0, g.width - 1)
        h = self.canopy_h[r, c]
        t = self.terrain[r, c]
        covered = h > 0.5
        reaches_ground = ~covered | (rng.random(n) < canopy_gap)
        z = np.where(reaches_ground, t + rng.normal(0, ground_sigma, n), t + h + rng.normal(0, canopy_sigma, n))
        return {"x": x, "y": y, "z": z, "is_ground_truth": reaches_ground}


def make_site(width_m: float = 120.0, height_m: float = 90.0, res: float = 0.5, seed: int = 7,
              origin=(571000.0, 4402000.0), tree_cover: float = 0.55, epsg: int = 32616) -> SyntheticSite:
    rng = np.random.default_rng(seed)
    w, h = int(width_m / res), int(height_m / res)
    grid = GridSpec(origin[0], origin[1] + h * res, res, w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = xx * res, (h - yy) * res
    terrain = 250.0 + 0.04 * X + 0.02 * Y                                       # regional slope
    for _ in range(6):                                                           # hills
        cx, cy, s, a = rng.uniform(0, width_m), rng.uniform(0, height_m), rng.uniform(12, 30), rng.uniform(-3, 6)
        terrain += a * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s)))
    gully = np.abs(Y - (0.45 * height_m + 8 * np.sin(X / 18.0)))                # a meandering drainage gully
    terrain -= 2.2 * np.exp(-(gully ** 2) / (2 * 4.0 ** 2))
    # vegetation
    canopy = np.zeros((h, w))
    labels = np.full((h, w), C.GRASS, dtype=np.uint8)
    soil_patch = ndimage.gaussian_filter(rng.random((h, w)), 6) > 0.54
    labels[soil_patch] = C.SOIL
    area, placed, target = w * h, 0.0, tree_cover * w * h
    while placed < target:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rad = rng.uniform(3.0, 6.0) / res
        ht = rng.uniform(10, 24)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (rad * rad)
        crown = ht * np.clip(1 - d2, 0, None) ** 0.5
        canopy = np.maximum(canopy, crown)
        placed += np.pi * rad * rad * 0.6
    tree_mask = canopy > 1.0
    labels[tree_mask] = C.TREES
    brush = (ndimage.gaussian_filter(rng.random((h, w)), 3) > 0.56) & ~tree_mask
    canopy[brush] = np.maximum(canopy[brush], rng.uniform(0.8, 1.8, brush.sum()))
    labels[brush] = C.BRUSH
    # a small shed (structure) on high ground
    sy, sx = int(h * 0.2), int(w * 0.75)
    sh = (slice(sy, sy + int(4 / res)), slice(sx, sx + int(6 / res)))
    canopy[sh], labels[sh] = 3.0, C.BUILDING
    # imagery
    base = {C.TREES: (28, 70, 32), C.BRUSH: (88, 130, 52), C.GRASS: (150, 160, 80), C.SOIL: (134, 104, 84),
            C.BUILDING: (190, 60, 50)}
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for k, col in base.items():
        rgb[labels == k] = col
    tex = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 0.8)[..., None] * 14
    shade = (ndimage.gaussian_filter(canopy, 1.5)[..., None] * 0.0)
    rgb = np.clip(rgb + tex + shade + rng.normal(0, 3, rgb.shape), 0, 255).astype(np.uint8)
    return SyntheticSite(grid, terrain.astype(np.float32), canopy.astype(np.float32),
                         (terrain + canopy).astype(np.float32), labels, rgb, epsg)
