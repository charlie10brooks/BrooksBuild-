"""Flight grouping, camera footprints and overlap estimation from EXIF-level metadata.

Footprints are *approximate*: they assume a nadir camera, flat terrain and use the 35 mm-equivalent
focal length and relative altitude when present. They are for planning/QA, not for processing; ODM
computes the real camera poses. When relative altitude is missing no footprint is invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6_378_137.0
DIAG_35MM_MM = 43.2666


def suggest_utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def lonlat_to_local_m(lon, lat, lon0: float, lat0: float):
    """Equirectangular projection around (lon0, lat0); fine for sites a few km across."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    x = np.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def local_m_to_lonlat(x, y, lon0: float, lat0: float):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return lon0 + np.degrees(x / (R_EARTH * math.cos(math.radians(lat0)))), lat0 + np.degrees(y / R_EARTH)


def sensor_size_35eq_mm(aspect: float) -> Tuple[float, float]:
    """Width/height (mm) of a virtual sensor with a 43.27 mm diagonal and the given aspect (w/h)."""
    h = DIAG_35MM_MM / math.sqrt(1 + aspect ** 2)
    return h * aspect, h


def ground_sample_distance_cm(rel_alt_m: float, focal35_mm: float, width_px: int, aspect: float) -> float:
    sw, _ = sensor_size_35eq_mm(aspect)
    return rel_alt_m * (sw / focal35_mm) / width_px * 100.0


def footprint_local(rel_alt: float, focal35: float, aspect: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Ground rectangle corners (east, north) in metres around the nadir point. ``yaw`` is the heading
    of the image 'up' direction, clockwise from north."""
    sw, sh = sensor_size_35eq_mm(aspect)
    gw, gh = rel_alt * sw / focal35, rel_alt * sh / focal35
    u = np.array([-gw / 2, gw / 2, gw / 2, -gw / 2])
    v = np.array([-gh / 2, -gh / 2, gh / 2, gh / 2])
    psi = math.radians(yaw_deg)
    east = u * math.cos(psi) + v * math.sin(psi)
    north = -u * math.sin(psi) + v * math.cos(psi)
    return np.stack([east, north], axis=1)


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of convex polygons (counter-clockwise)."""
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out_list = out, []
        if len(inp) == 0:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def inter(p, q):
            d1, d2 = q - p, b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den if den else 0.0
            return p + t * d1

        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out_list.append(inter(p, q))
                out_list.append(q)
            elif inside(p):
                out_list.append(inter(p, q))
        out = np.array(out_list) if out_list else np.zeros((0, 2))
    return out


def _area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    return poly if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) >= 0 else poly[::-1]


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area / area of ``a`` for two convex quadrilaterals."""
    inter = _clip(_ccw(a), _ccw(b))
    aa = _area(a)
    return _area(inter) / aa if aa > 0 else 0.0


@dataclass
class ImageGeo:
    id: str
    lon: Optional[float]
    lat: Optional[float]
    captured_at: Optional[str] = None
    rel_alt: Optional[float] = None
    focal35: Optional[float] = None
    yaw: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pitch: Optional[float] = None


def _ts(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def group_flights(images: Sequence[ImageGeo], gap_minutes: float = 15.0) -> Dict[str, int]:
    """Assign flight-group numbers (1..n) by capture-time gaps. Images without a timestamp go to group 0
    ('unassigned') rather than being merged into a flight on a guess."""
    stamped = sorted(((_ts(i.captured_at), i.id) for i in images if _ts(i.captured_at) is not None))
    groups: Dict[str, int] = {i.id: 0 for i in images}
    g, last = 0, None
    for t, iid in stamped:
        if last is None or (t - last) > gap_minutes * 60.0:
            g += 1
        groups[iid] = g
        last = t
    return groups


@dataclass
class OverlapResult:
    footprint_available: bool
    forward: Optional[float] = None      # best overlap with a temporal neighbour
    side: Optional[float] = None         # best overlap with a non-adjacent image
    best: Optional[float] = None
    neighbours: int = 0
    poor: bool = False
    reason: str = ""


def analyse_overlap(images: Sequence[ImageGeo], poor_threshold: float = 0.6,
                    max_neighbours: int = 16) -> Tuple[Dict[str, OverlapResult], dict]:
    """Per-image overlap estimate and a project summary. Unknown stays unknown."""
    res: Dict[str, OverlapResult] = {}
    placed = [i for i in images if i.lon is not None and i.lat is not None]
    fp_ok = [i for i in placed if i.rel_alt and i.focal35 and i.width and i.height
             and (i.pitch is None or abs(i.pitch + 90) <= 20)]
    for i in images:
        res[i.id] = OverlapResult(False, reason="no GPS position" if i not in placed else
                                  ("camera not near nadir" if (i.pitch is not None and abs(i.pitch + 90) > 20)
                                   else "no relative altitude / focal length for footprint"))
    summary = {"images": len(images), "with_gps": len(placed), "with_footprint": len(fp_ok), "poor_overlap": 0,
               "mean_best_overlap": None, "note": ("Overlap is estimated from EXIF assuming a nadir camera and flat "
                                                  "ground; ODM computes true camera poses.")}
    if len(fp_ok) < 2:
        return res, summary
    lon0, lat0 = float(np.mean([i.lon for i in fp_ok])), float(np.mean([i.lat for i in fp_ok]))
    xs, ys = lonlat_to_local_m([i.lon for i in fp_ok], [i.lat for i in fp_ok], lon0, lat0)
    polys = [footprint_local(i.rel_alt, i.focal35, i.width / i.height, i.yaw or 0.0) + np.array([x, y])
             for i, x, y in zip(fp_ok, xs, ys)]
    radius = max(float(np.hypot(*(p.max(axis=0) - p.min(axis=0)))) for p in polys)
    tree = cKDTree(np.c_[xs, ys])
    order = {i.id: k for k, i in enumerate(sorted(fp_ok, key=lambda im: (_ts(im.captured_at) or 0.0, im.id)))}
    bests = []
    for k, im in enumerate(fp_ok):
        _, idx = tree.query([xs[k], ys[k]], k=min(max_neighbours + 1, len(fp_ok)), distance_upper_bound=radius)
        fwd, side, n = 0.0, 0.0, 0
        for j in np.atleast_1d(idx):
            if j >= len(fp_ok) or j == k:
                continue
            f = overlap_fraction(polys[k], polys[j])
            if f <= 0:
                continue
            n += 1
            temporal = abs(order[fp_ok[j].id] - order[im.id]) == 1
            if temporal:
                fwd = max(fwd, f)
            else:
                side = max(side, f)
        best = max(fwd, side)
        poor = best < poor_threshold
        res[im.id] = OverlapResult(True, fwd or None, side or None, best, n, poor,
                                   f"best neighbour overlap {best:.0%} < {poor_threshold:.0%}" if poor else "")
        bests.append(best)
    summary["poor_overlap"] = int(sum(1 for r in res.values() if r.poor))
    summary["mean_best_overlap"] = float(np.mean(bests))
    return res, summary


def footprint_geojson(im: ImageGeo, lon0: float, lat0: float) -> Optional[dict]:
    if im.lon is None or im.lat is None or not (im.rel_alt and im.focal35 and im.width and im.height):
        return None
    x, y = lonlat_to_local_m(im.lon, im.lat, lon0, lat0)
    ring = footprint_local(im.rel_alt, im.focal35, im.width / im.height, im.yaw or 0.0) + np.array([float(x), float(y)])
    lon, lat = local_m_to_lonlat(ring[:, 0], ring[:, 1], lon0, lat0)
    coords = [[float(a), float(b)] for a, b in zip(lon, lat)]
    coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}
