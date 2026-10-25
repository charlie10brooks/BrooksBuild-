"""Ground classification of dense point clouds.

Engines
-------
``pdal_smrf`` (default) and ``pdal_pmf``
    Established filters executed by PDAL (``filters.smrf`` / ``filters.pmf``) in a subprocess. If PDAL is not
    installed the job **fails with install instructions**; there is never a silent substitution.
``builtin_pmf``
    A streaming progressive-morphological filter (Zhang et al. 2003) in numpy/scipy. It exists so development
    mode and CI can exercise the whole terrain pipeline without PDAL. It must be selected explicitly.
    Known limit of PMF-style filters: terrain features wider than ``builtin_max_window_m`` *and* taller than
    ``builtin_max_distance`` are mistaken for objects and shaved, while tree crowns wider than the window are
    kept as ground. Tune both to the site; prefer SMRF (slope-adaptive) for production.

No engine can see through dense canopy: all of them only label which *observed* points look like ground.
Where no ground was observed, downstream code must report low confidence / unknown, never fill silently.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from ..core.dtm import GridSpec, bin_points
from . import las as L

ENGINES = ("pdal_smrf", "pdal_pmf", "builtin_pmf")
PDAL_INSTALL_HINT = ("PDAL was not found on PATH. Install it with `conda install -c conda-forge pdal` or "
                     "`apt install pdal`, or run the worker from the provided Docker image (which bundles PDAL). "
                     "To work without PDAL in development choose ground engine 'builtin_pmf' explicitly.")


class GroundFilterError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


@dataclass
class GroundFilterParams:
    engine: str = "pdal_smrf"
    outlier_removal: bool = True
    outlier_mean_k: int = 8
    outlier_multiplier: float = 3.0
    # SMRF (PDAL names). window/threshold in metres; slope is rise/run.
    smrf_cell: float = 1.0
    smrf_slope: float = 0.15
    smrf_window: float = 18.0
    smrf_threshold: float = 0.5
    smrf_scalar: float = 1.25
    # PMF
    pmf_cell: float = 1.0
    pmf_max_window: float = 33.0
    pmf_slope: float = 0.5
    pmf_initial_distance: float = 0.15
    pmf_max_distance: float = 2.5
    # builtin
    builtin_cell: float = 1.0
    builtin_max_window_m: float = 15.0
    builtin_slope: float = 0.3
    builtin_initial_distance: float = 0.15
    builtin_max_distance: float = 3.0
    builtin_ground_tolerance: float = 0.25
    builtin_noise_depth: float = 3.0

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        for name in ("smrf_cell", "smrf_window", "pmf_cell", "builtin_cell", "builtin_max_window_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------ PDAL
def build_pdal_pipeline(src: str, dst: str, p: GroundFilterParams) -> List[dict]:
    """The exact pipeline handed to PDAL (also stored with the job for reproducibility)."""
    p.validate()
    if p.engine == "builtin_pmf":
        raise ValueError("builtin_pmf does not use PDAL")
    stages: List[dict] = [{"type": "readers.las", "filename": src},
                          {"type": "filters.assign", "assignment": "Classification[:]=1"}]
    ignore = None
    if p.outlier_removal:
        stages.append({"type": "filters.outlier", "method": "statistical", "mean_k": p.outlier_mean_k,
                       "multiplier": p.outlier_multiplier})
        ignore = "Classification[7:7]"
    if p.engine == "pdal_smrf":
        st = {"type": "filters.smrf", "cell": p.smrf_cell, "slope": p.smrf_slope, "window": p.smrf_window,
              "threshold": p.smrf_threshold, "scalar": p.smrf_scalar}
    else:
        st = {"type": "filters.pmf", "cell_size": p.pmf_cell, "max_window_size": p.pmf_max_window,
              "slope": p.pmf_slope, "initial_distance": p.pmf_initial_distance, "max_distance": p.pmf_max_distance}
    if ignore:
        st["ignore"] = ignore
    stages.append(st)
    stages.append({"type": "writers.las", "filename": dst, "compression": "false", "forward": "all"})
    return stages


def run_pdal_pipeline(pipeline: List[dict], workdir: Path, pdal_bin: str = "pdal", timeout_s: Optional[float] = None,
                      log: Callable[[str], None] = lambda s: None,
                      cancelled: Callable[[], bool] = lambda: False) -> None:
    if shutil.which(pdal_bin) is None:
        raise L.MissingDependency(PDAL_INSTALL_HINT)
    workdir.mkdir(parents=True, exist_ok=True)
    pj = workdir / "pdal_pipeline.json"
    pj.write_text(json.dumps({"pipeline": pipeline}, indent=2))
    proc = subprocess.Popen([pdal_bin, "pipeline", str(pj)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    timer = None
    if timeout_s:
        timer = threading.Timer(timeout_s, proc.kill)
        timer.start()
    tail: List[str] = []
    try:
        for line in proc.stdout:                     # type: ignore[union-attr]
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            log(line)
            if cancelled():
                proc.kill()
                raise Cancelled("cancelled by user")
        rc = proc.wait()
    finally:
        if timer:
            timer.cancel()
    if rc != 0:
        raise GroundFilterError(f"PDAL exited with code {rc}: " + " | ".join(tail[-8:])[:1200])


# --------------------------------------------------------------------------- built-in PMF
def _fill_nearest(a: np.ndarray) -> np.ndarray:
    bad = np.isnan(a)
    if not bad.any():
        return a
    if bad.all():
        raise GroundFilterError("point cloud has no points inside the grid")
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def pmf_surface(zmin: np.ndarray, cell: float, p: GroundFilterParams) -> np.ndarray:
    """Progressive morphological filter on a minimum-elevation grid -> filtered 'ground surface' grid."""
    z = _fill_nearest(zmin.astype(np.float64))
    cur = z.copy()
    prev_w, w = 1, 3
    max_w = max(3, int(round(p.builtin_max_window_m / cell)) | 1)
    while True:
        opened = ndimage.grey_opening(cur, size=(w, w))
        dh = min(p.builtin_slope * (w - prev_w) * cell + p.builtin_initial_distance, p.builtin_max_distance)
        # cells that stick out of the opened surface by more than the allowed step are non-ground: lower them
        cur = np.where(cur - opened > dh, opened, cur)
        if w >= max_w:
            break
        prev_w, w = w, min(2 * w - 1, max_w)
    return cur


def _grid_for(bounds, cell: float) -> GridSpec:
    minx, miny, maxx, maxy = bounds
    return GridSpec.from_bounds(minx, miny, maxx + cell, maxy + cell, cell)


def classify_builtin(src: Path, dst: Path, p: GroundFilterParams, epsg: Optional[int] = None,
                     progress: Callable[[float, str], None] = lambda f, m: None,
                     cancelled: Callable[[], bool] = lambda: False) -> Dict[str, int]:
    """Two streaming passes over ``src``; writes a LAS with ground = 2, other = 1, outliers = 7."""
    hdr = L.read_header(src)
    grid = _grid_for(hdr.bounds_xy, p.builtin_cell)
    zmin = np.full(grid.shape, np.nan, dtype=np.float32)
    for ch in L.iter_points(src):
        if cancelled():
            raise Cancelled("cancelled by user")
        m, _ = bin_points(ch["x"], ch["y"], ch["z"], grid, "min")
        zmin = np.fmin(zmin, m)
    progress(0.35, "built minimum-elevation grid")
    surf = pmf_surface(zmin, grid.res, p)
    progress(0.5, "filtered surface computed")
    counts = {"ground": 0, "non_ground": 0, "noise": 0, "total": 0}
    done = 0
    with L.LasWriter(dst, epsg or hdr.epsg, scale=hdr.scale[0], offset=hdr.offset) as w:
        for ch in L.iter_points(src):
            if cancelled():
                raise Cancelled("cancelled by user")
            r, c = grid.xy_to_rc(ch["x"], ch["y"])
            r = np.clip(r.astype(int), 0, grid.height - 1)
            c = np.clip(c.astype(int), 0, grid.width - 1)
            h = ch["z"] - surf[r, c]
            cls = np.where(h <= p.builtin_ground_tolerance, 2, 1).astype(np.uint8)
            cls[h < -p.builtin_noise_depth] = 7                                       # far below the surface
            w.write(ch["x"], ch["y"], ch["z"], cls, ch.get("rgb"), ch.get("intensity"))
            counts["ground"] += int((cls == 2).sum())
            counts["non_ground"] += int((cls == 1).sum())
            counts["noise"] += int((cls == 7).sum())
            done += len(cls)
            progress(0.5 + 0.5 * done / max(hdr.count, 1), "classifying points")
    counts["total"] = counts["ground"] + counts["non_ground"] + counts["noise"]
    return counts


def classify_ground(src: Path, dst: Path, p: GroundFilterParams, workdir: Path, epsg: Optional[int] = None,
                    pdal_bin: str = "pdal", log: Callable[[str], None] = lambda s: None,
                    progress: Callable[[float, str], None] = lambda f, m: None,
                    cancelled: Callable[[], bool] = lambda: False, timeout_s: Optional[float] = None) -> Dict[str, int]:
    """Classify ``src`` (LAS/LAZ) into ``dst`` (LAS). Returns class counts."""
    p.validate()
    if p.engine == "builtin_pmf":
        las_src = L.ensure_las(src, workdir, pdal_bin)
        return classify_builtin(las_src, dst, p, epsg, progress, cancelled)
    run_pdal_pipeline(build_pdal_pipeline(str(src), str(dst), p), workdir, pdal_bin, timeout_s, log, cancelled)
    counts = {"ground": 0, "non_ground": 0, "noise": 0, "total": 0}
    for ch in L.iter_points(dst):
        cl = ch["classification"]
        counts["ground"] += int((cl == 2).sum())
        counts["noise"] += int((cl == 7).sum())
        counts["total"] += len(cl)
    counts["non_ground"] = counts["total"] - counts["ground"] - counts["noise"]
    if counts["total"] == 0:
        raise GroundFilterError("PDAL produced an empty point cloud")
    return counts


def iter_ground_points(path: Path, ground_class: int = 2) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    for ch in L.iter_points(path):
        m = ch["classification"] == ground_class
        if m.any():
            yield ch["x"][m], ch["y"][m], ch["z"][m]


# ------------------------------------------------------------- vegetation / building labelling
def label_by_height_above_ground(src: Path, dst: Path, dtm: np.ndarray, grid: GridSpec, epsg: Optional[int],
                                 building_mask: Optional[np.ndarray] = None,
                                 low: float = 0.5, high: float = 2.0) -> Dict[str, int]:
    """Re-write a ground-classified cloud with 3/4/5 (low/medium/high vegetation) from height above the DTM.

    Heights are only computed where the DTM has an estimate; elsewhere the label stays 1 (unclassified) so no
    guess is made. Points on ``building_mask`` cells that are not ground become 6.
    """
    counts: Dict[str, int] = {}
    hdr = L.read_header(src)
    with L.LasWriter(dst, epsg or hdr.epsg, scale=hdr.scale[0], offset=hdr.offset) as w:
        for ch in L.iter_points(src):
            r, c = grid.xy_to_rc(ch["x"], ch["y"])
            r, c = np.floor(r).astype(int), np.floor(c).astype(int)
            inside = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
            rr, cc = np.clip(r, 0, grid.height - 1), np.clip(c, 0, grid.width - 1)
            hag = np.where(inside, ch["z"] - dtm[rr, cc], np.nan)
            cls = ch["classification"].copy()
            ng = (cls != 2) & (cls != 7) & np.isfinite(hag)
            cls[ng & (hag < low)] = 3
            cls[ng & (hag >= low) & (hag < high)] = 4
            cls[ng & (hag >= high)] = 5
            if building_mask is not None:
                cls[ng & inside & building_mask[rr, cc]] = 6
            for k, v in zip(*np.unique(cls, return_counts=True)):
                counts[int(k)] = counts.get(int(k), 0) + int(v)
            w.write(ch["x"], ch["y"], ch["z"], cls, ch.get("rgb"), ch.get("intensity"))
    return {str(k): v for k, v in sorted(counts.items())}
