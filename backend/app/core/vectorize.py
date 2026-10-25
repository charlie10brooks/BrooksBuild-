"""Raster mask -> GeoJSON polygons (with holes)."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .dtm import GridSpec


def mask_to_polygons(mask: np.ndarray, grid: GridSpec, min_cells: int = 4, simplify_cells: float = 0.0) -> List[dict]:
    """GeoJSON Polygon geometries (map coordinates) for connected True regions.

    The mask is traced at 2x resolution so outlines sit within ~0.25 cell of the true cell edges.
    """
    up = np.kron(mask.astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    contours, hier = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]

    def ring(cnt) -> List[List[float]]:
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if simplify_cells > 0:
            pts = cv2.approxPolyDP(cnt, simplify_cells * 2, True).reshape(-1, 2).astype(np.float64)
        cols, rows = (pts[:, 0] + 0.5) / 2.0, (pts[:, 1] + 0.5) / 2.0
        x, y = grid.rc_to_xy(rows, cols)
        coords = [[float(a), float(b)] for a, b in zip(x, y)]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords

    polys = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:          # a hole; handled with its parent
            continue
        if cv2.contourArea(cnt) / 4.0 < min_cells:
            continue
        rings = [ring(cnt)]
        child = hier[i][2]
        while child != -1:
            if len(contours[child]) >= 3:
                rings.append(ring(contours[child]))
            child = hier[child][0]
        if len(rings[0]) >= 4:
            polys.append({"type": "Polygon", "coordinates": rings})
    return polys
