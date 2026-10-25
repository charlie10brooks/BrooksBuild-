"""CAD export: a dependency-free ASCII DXF (R12 / AC1009) writer and a marching-squares contour generator.

Everything drawn from the terrain is the ESTIMATED bare-earth model. Contours are therefore split onto layers by evidence quality, so a
CAD user can see at a glance which stretches rest on measured / well-supported ground and which are extrapolation. Nothing here is
survey-grade; the drawing carries a visible "PRELIMINARY PLANNING ESTIMATE" note.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Pt = Tuple[float, float]

# ----------------------------------------------------------------------------- marching squares
# edge ids around a cell with corners a(top-left) b(top-right) c(bottom-right) d(bottom-left):  0 top(a-b) 1 right(b-c) 2 bottom(d-c) 3 left(a-d)
_CASES: Dict[int, List[Tuple[int, int]]] = {
    0: [], 15: [], 1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(0, 1)], 5: [(3, 0), (2, 1)], 6: [(0, 2)], 7: [(3, 0)],
    8: [(3, 0)], 9: [(0, 2)], 10: [(3, 2), (0, 1)], 11: [(0, 1)], 12: [(3, 1)], 13: [(2, 1)], 14: [(3, 2)],
}


def contour_lines(z: np.ndarray, level: float) -> List[List[Tuple[float, float]]]:
    """Iso-lines of `z` at `level` in (row, col) fractional pixel-centre coordinates. Cells touching NaN are skipped."""
    h, w = z.shape
    if h < 2 or w < 2:
        return []
    a, b, c, d = z[:-1, :-1], z[:-1, 1:], z[1:, 1:], z[1:, :-1]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    idx = (a > level) * 8 + (b > level) * 4 + (c > level) * 2 + (d > level) * 1
    mixed = ok & (idx > 0) & (idx < 15)
    rows, cols = np.nonzero(mixed)
    if rows.size == 0:
        return []

    def crossing(edge: int, i: int, j: int):
        # returns (key, (row, col)); keys are shared between neighbouring cells so segments chain exactly
        if edge == 0:
            za, zb, key, r0, c0, r1, c1 = z[i, j], z[i, j + 1], ("h", i, j), i, j, i, j + 1
        elif edge == 2:
            za, zb, key, r0, c0, r1, c1 = z[i + 1, j], z[i + 1, j + 1], ("h", i + 1, j), i + 1, j, i + 1, j + 1
        elif edge == 3:
            za, zb, key, r0, c0, r1, c1 = z[i, j], z[i + 1, j], ("v", i, j), i, j, i + 1, j
        else:
            za, zb, key, r0, c0, r1, c1 = z[i, j + 1], z[i + 1, j + 1], ("v", i, j + 1), i, j + 1, i + 1, j + 1
        t = (level - za) / (zb - za) if zb != za else 0.5
        return key, (r0 + (r1 - r0) * float(t), c0 + (c1 - c0) * float(t))

    pos: Dict[tuple, Tuple[float, float]] = {}
    adj: Dict[tuple, List[tuple]] = {}
    for i, j in zip(rows.tolist(), cols.tolist()):
        for e0, e1 in _CASES[int(idx[i, j])]:
            k0, p0 = crossing(e0, i, j)
            k1, p1 = crossing(e1, i, j)
            pos[k0], pos[k1] = p0, p1
            adj.setdefault(k0, []).append(k1)
            adj.setdefault(k1, []).append(k0)
    used = set()
    out: List[List[Tuple[float, float]]] = []

    def is_used(p, q):
        return (p, q) in used or (q, p) in used

    for k in [k for k, v in adj.items() if len(v) == 1] + list(adj):      # open ends first, then closed loops
        if all(is_used(k, n) for n in adj[k]):
            continue
        path, cur = [k], k
        while True:
            nxt = next((n for n in adj[cur] if not is_used(cur, n)), None)
            if nxt is None:
                break
            used.add((cur, nxt))
            path.append(nxt)
            cur = nxt
            if cur == k:
                break
        if len(path) >= 2:
            out.append([pos[p] for p in path])
    return out


def nice_interval(relief: float, target_lines: int = 20) -> float:
    """A round contour interval giving roughly `target_lines` lines across the relief."""
    raw = max(relief, 1e-6) / target_lines
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def terrain_contours(dem: np.ndarray, x0: float, y0: float, res: float, interval: float, conf: Optional[np.ndarray] = None,
                     smooth_cells: float = 1.5, max_cells: int = 4_000_000) -> dict:
    """Contours of an ESTIMATED DTM. Returns {"lines": [(level, is_major, weak, [(x, y, z)...])], "interval", "stride", "smoothed_cells"}.

    The DTM is lightly Gaussian-smoothed first (raw 0.5 m cells give jittery lines) and very large rasters are decimated so the DXF stays
    usable; both are reported. `weak` marks runs whose cells have low/unknown terrain confidence."""
    import cv2

    stride = max(1, int(math.ceil(math.sqrt(dem.size / max_cells))))
    z = dem[::stride, ::stride].astype(np.float64)
    cf = conf[::stride, ::stride] if conf is not None else None
    valid = np.isfinite(z)
    if not valid.any():
        return {"lines": [], "interval": interval, "stride": stride, "smoothed_cells": 0.0}
    sm = 0.0
    if smooth_cells > 0 and min(z.shape) >= 5:
        wgt = cv2.GaussianBlur(valid.astype(np.float64), (0, 0), smooth_cells)
        zs = cv2.GaussianBlur(np.where(valid, z, 0.0), (0, 0), smooth_cells) / np.maximum(wgt, 1e-9)
        z = np.where(valid & (wgt > 0.3), zs, np.nan)
        sm = smooth_cells * stride
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    lo = math.ceil(zmin / interval) * interval
    lines = []
    n = 0
    level = lo
    while level <= zmax and n < 2000:
        major = abs(round(level / (interval * 5)) * (interval * 5) - level) < 1e-9
        for ln in contour_lines(z, level):
            pts = [(x0 + (c + 0.5) * res * stride, y0 - (r + 0.5) * res * stride, level) for r, c in ln]
            weak_flags = []
            for r, c in ln:
                if cf is None:
                    weak_flags.append(False)
                else:
                    v = cf[min(cf.shape[0] - 1, int(round(r))), min(cf.shape[1] - 1, int(round(c)))]
                    weak_flags.append(int(v) in (0, 1))                       # unknown / low
            run: List[tuple] = [pts[0]]
            cur = weak_flags[0]
            for p, wk in zip(pts[1:], weak_flags[1:]):
                if wk == cur:
                    run.append(p)
                else:
                    run.append(p)
                    lines.append((level, major, cur, run))
                    run, cur = [p], wk
            if len(run) >= 2:
                lines.append((level, major, cur, run))
        level = round(level + interval, 9)
        n += 1
    return {"lines": lines, "interval": interval, "stride": stride, "smoothed_cells": sm, "zmin": zmin, "zmax": zmax}


# ----------------------------------------------------------------------------- DXF writer
class Dxf:
    """Minimal ASCII DXF (AC1009). Entities: POLYLINE (2D/3D), POINT, TEXT."""

    def __init__(self, units: str = "metre"):
        self.layers: Dict[str, int] = {}
        self.ents: List[str] = []
        self.units = units
        self.ext = [math.inf, math.inf, -math.inf, -math.inf]

    def layer(self, name: str, color: int) -> str:
        self.layers[name] = color
        return name

    @staticmethod
    def _g(code: int, val) -> str:
        return f"{code}\n{val}\n"

    def _f(self, v: float) -> str:
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e15 else str(v)

    def _bb(self, x: float, y: float):
        self.ext = [min(self.ext[0], x), min(self.ext[1], y), max(self.ext[2], x), max(self.ext[3], y)]

    def polyline(self, layer: str, pts: Sequence[Sequence[float]], closed: bool = False, elevated: bool = False):
        if len(pts) < 2:
            return
        g = self._g
        flag = (8 if elevated else 0) | (1 if closed else 0)
        s = g(0, "POLYLINE") + g(8, layer) + g(66, 1) + g(10, 0) + g(20, 0) + g(30, 0) + g(70, flag)
        for p in pts:
            z = p[2] if (elevated and len(p) > 2) else 0.0
            s += g(0, "VERTEX") + g(8, layer) + g(10, self._f(p[0])) + g(20, self._f(p[1])) + g(30, self._f(z)) + (g(70, 32) if elevated else "")
            self._bb(p[0], p[1])
        self.ents.append(s + g(0, "SEQEND") + g(8, layer))

    def point(self, layer: str, x: float, y: float, z: float = 0.0):
        g = self._g
        self.ents.append(g(0, "POINT") + g(8, layer) + g(10, self._f(x)) + g(20, self._f(y)) + g(30, self._f(z)))
        self._bb(x, y)

    def text(self, layer: str, x: float, y: float, height: float, s: str, z: float = 0.0):
        g = self._g
        s = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s.replace("\r", " ").replace("\n", " "))[:250]
        self.ents.append(g(0, "TEXT") + g(8, layer) + g(10, self._f(x)) + g(20, self._f(y)) + g(30, self._f(z)) + g(40, self._f(height)) + g(1, s))
        self._bb(x, y)

    def dumps(self) -> str:
        g = self._g
        insunits = {"metre": 6, "foot": 2, "us_survey_foot": 21}.get(self.units, 0)
        ext = self.ext if self.ext[0] != math.inf else [0, 0, 1, 1]
        out = g(0, "SECTION") + g(2, "HEADER") + g(9, "$ACADVER") + g(1, "AC1009") + g(9, "$INSUNITS") + g(70, insunits)
        out += g(9, "$EXTMIN") + g(10, self._f(ext[0])) + g(20, self._f(ext[1])) + g(30, 0) + g(9, "$EXTMAX") + g(10, self._f(ext[2])) + g(20, self._f(ext[3])) + g(30, 0)
        out += g(0, "ENDSEC") + g(0, "SECTION") + g(2, "TABLES") + g(0, "TABLE") + g(2, "LTYPE") + g(70, 1) + g(0, "LTYPE") + g(2, "CONTINUOUS") + g(70, 0) + g(3, "Solid line") + g(72, 65) + g(73, 0) + g(40, 0) + g(0, "ENDTAB")
        out += g(0, "TABLE") + g(2, "LAYER") + g(70, len(self.layers))
        for name, col in self.layers.items():
            out += g(0, "LAYER") + g(2, name) + g(70, 0) + g(62, col) + g(6, "CONTINUOUS")
        out += g(0, "ENDTAB") + g(0, "ENDSEC") + g(0, "SECTION") + g(2, "ENTITIES") + "".join(self.ents) + g(0, "ENDSEC") + g(0, "EOF")
        return out

    def write(self, path) -> None:
        with open(path, "w", encoding="ascii", newline="\r\n") as f:
            f.write(self.dumps())
