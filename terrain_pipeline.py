"""Bare-earth terrain reconstruction from point clouds and supplemental elevation sources.

Evidence flags in ``evidence.tif``
    0 none      no estimate (unknown)
    1 measured  cell contains classified ground points (photogrammetry, LiDAR or survey)
    2 interpolated  estimated from nearby measured cells within the search radius
    3 external  taken from an imported terrain model (not observed by this project's data)

The visual land-cover mask is *never* an input to elevation values. The only use of land cover is a safety check
(``veg_mask``): photogrammetric "ground" cells under mapped vegetation whose surface is within
``canopy_clearance_m`` of the DSM are suspect (the filter probably labelled canopy as ground) and are demoted from
measured to estimated, which lowers their confidence.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core import dtm as D
from ..core.dtm import ConfidenceParams, ElevationSource, GridSpec, InterpParams
from ..core.tiling import Window
from . import geotiff as gt
from . import groundfilter as G
from . import las as L

EV_NONE, EV_MEASURED, EV_INTERP, EV_EXTERNAL = 0, 1, 2, 3
SOURCE_PHOTO = 1
TRUST_TO_CONF = {"high": D.CONF_HIGH, "moderate": D.CONF_MODERATE, "low": D.CONF_LOW}


class TerrainError(RuntimeError):
    pass


@dataclass
class SourceSpec:
    id: int                      # 2..255 (1 is reserved for this project's photogrammetry)
    name: str
    kind: str                    # lidar | survey | external_dem
    path: Path
    priority: int = 20
    z_offset: float = 0.0        # vertical datum correction (m), applied only if the user confirmed it
    trust: str = "moderate"      # external_dem only: confidence granted to cells taken from this model
    meta: dict = field(default_factory=dict)


@dataclass
class TerrainParams:
    resolution_m: float = 0.5
    max_cells: int = 60_000_000
    ground: G.GroundFilterParams = field(default_factory=G.GroundFilterParams)
    interp: InterpParams = field(default_factory=InterpParams)
    confidence: ConfidenceParams = field(default_factory=ConfidenceParams)
    stat: str = "median"
    photo_priority: int = 10
    canopy_clearance_m: float = 1.0
    max_ground_points: int = 150_000_000
    cross_validate: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class TerrainResult:
    grid: GridSpec
    paths: Dict[str, str]
    summary: dict


def choose_grid(bounds: Tuple[float, float, float, float], res: float, max_cells: int) -> Tuple[GridSpec, float]:
    """Grid covering ``bounds``; coarsens ``res`` (reported) if it would exceed ``max_cells``."""
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise TerrainError("point cloud has zero extent")
    r = res
    while (w / r) * (h / r) > max_cells:
        r *= 1.5
    return GridSpec.from_bounds(minx, miny, maxx, maxy, r), r


# ----------------------------------------------------------------------------- source loading
def load_survey_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = [], [], []
    seen_data = False
    with open(path, newline="") as f:
        rd = csv.reader(f)
        for i, row in enumerate(rd):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                vals = [float(v) for v in row[-3:]] if len(row) >= 3 else None
            except ValueError:
                if not seen_data:
                    seen_data = True
                    continue                       # the first non-comment row may be a header
                raise TerrainError(f"{Path(path).name}: line {i + 1} is not numeric")
            seen_data = True
            if vals is None:
                raise TerrainError(f"{Path(path).name}: line {i + 1} needs x,y,z columns")
            xs.append(vals[0]); ys.append(vals[1]); zs.append(vals[2])
    if not xs:
        raise TerrainError(f"{Path(path).name}: no survey points found")
    return np.array(xs), np.array(ys), np.array(zs)


def resample_raster_to_grid(path: Path, grid: GridSpec, epsg: Optional[int]) -> np.ndarray:
    """Resample a same-CRS GeoTIFF DEM onto ``grid`` (float32, NaN outside/nodata). Refuses CRS mismatches."""
    info = gt.read_georef(path)
    if not info.georeferenced or info.grid is None:
        raise TerrainError(f"{Path(path).name} has no georeferencing; cannot use it as an elevation source")
    if epsg is not None and info.epsg != epsg:
        raise TerrainError(f"{Path(path).name} is in EPSG:{info.epsg} but the project is EPSG:{epsg}. Reproject it first "
                           f"(e.g. `gdalwarp -t_srs EPSG:{epsg} in.tif out.tif`); silent reprojection is not done.")
    sg = info.grid
    # source window covering the target bounds (with a margin)
    tx0, ty0, tx1, ty1 = grid.bounds()
    c0 = int(np.floor((tx0 - sg.x0) / sg.res)) - 2
    c1 = int(np.ceil((tx1 - sg.x0) / sg.res)) + 2
    r0 = int(np.floor((sg.y0 - ty1) / sg.res)) - 2
    r1 = int(np.ceil((sg.y0 - ty0) / sg.res)) + 2
    win = Window(max(c0, 0), max(r0, 0), min(c1, sg.width), min(r1, sg.height))
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    if win.is_empty():
        return out
    with gt.TiffWindowReader(path) as rd:
        a = rd.read(win).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if info.nodata is not None:
        a[(a == info.nodata) | (np.isnan(info.nodata) & np.isnan(a))] = np.nan
    if sg.res * 2 <= grid.res:                                  # much finer than target: area-average first
        f = grid.res / sg.res
        a = cv2.resize(np.nan_to_num(a, nan=-9999.0), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
        a[a < -9000] = np.nan
        step = grid.res
        sx0, sy0 = sg.x0 + win.x0 * sg.res, sg.y0 - win.y0 * sg.res
    else:
        step = sg.res
        sx0, sy0 = sg.x0 + win.x0 * sg.res, sg.y0 - win.y0 * sg.res
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    x, y = grid.cell_centers_xy(rows, cols)
    mx = ((x - sx0) / step - 0.5).astype(np.float32)
    my = ((sy0 - y) / step - 0.5).astype(np.float32)
    ok = np.isfinite(a)
    filled = np.where(ok, a, 0.0).astype(np.float32)
    val = cv2.remap(filled, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wgt = cv2.remap(ok.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    good = wgt > 0.999                                          # only fully-supported cells; no edge smearing
    out[good] = val[good] / wgt[good]
    return out


def _bin_points_chunks(chunks, grid: GridSpec, stat: str, max_points: int):
    xs, ys, zs = [], [], []
    n = 0
    for x, y, z in chunks:
        n += len(x)
        if n > max_points:
            raise TerrainError(f"more than {max_points:,} ground points; raise TerrainParams.max_ground_points or "
                               "thin the cloud")
        xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return np.full(grid.shape, np.nan, np.float32), np.zeros(grid.shape, np.int32)
    return D.bin_points(np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), grid, stat)


# ------------------------------------------------------------------------------- main pipeline
def run_terrain(cloud: Path, workdir: Path, epsg: Optional[int], params: Optional[TerrainParams] = None,
                sources: Sequence[SourceSpec] = (), veg_mask_fn: Optional[Callable[[GridSpec], np.ndarray]] = None,
                building_mask_fn: Optional[Callable[[GridSpec], np.ndarray]] = None,
                odm_dsm: Optional[Path] = None, vertical_units: str = "metre", horizontal_units: str = "metre",
                vertical_epsg: Optional[int] = None, pdal_bin: str = "pdal",
                log: Callable[[str], None] = lambda s: None,
                progress: Callable[[float, str], None] = lambda f, m: None,
                cancelled: Callable[[], bool] = lambda: False) -> TerrainResult:
    params = params or TerrainParams()
    t0 = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    if not epsg:
        warnings.append("Project CRS is not set: rasters are written WITHOUT georeferencing and LAS files without a "
                        "CRS. Coordinates are in the local frame of the input cloud.")

    # 1. grid
    las_in = L.ensure_las(Path(cloud), workdir, pdal_bin)      # LAZ is converted by PDAL (or fails with install hints)
    hdr = L.read_header(las_in)
    grid, res = choose_grid(hdr.bounds_xy, params.resolution_m, params.max_cells)
    if res != params.resolution_m:
        warnings.append(f"Requested {params.resolution_m} m resolution would exceed {params.max_cells:,} cells; "
                        f"using {res:.2f} m.")
    progress(0.02, "grid defined")

    # 2. ground classification
    classified = workdir / "classified_ground.las"
    counts = G.classify_ground(las_in, classified, params.ground, workdir, epsg or hdr.epsg, pdal_bin, log,
                               lambda f, m: progress(0.05 + 0.35 * f, m), cancelled)
    log(f"ground classification: {counts}")
    if counts["ground"] < 10:
        raise TerrainError(f"only {counts['ground']} ground points were found in {counts['total']:,} points. "
                           "The area may be fully canopy-covered. Add LiDAR, survey points or an external DEM.")

    # 3. DSM (all non-noise points, max per cell) and photogrammetry ground evidence
    dsm = np.full(grid.shape, np.nan, np.float32)
    for ch in L.iter_points(classified):
        if cancelled():
            raise G.Cancelled("cancelled by user")
        keep = ch["classification"] != 7
        m, _ = D.bin_points(ch["x"][keep], ch["y"][keep], ch["z"][keep], grid, "max")
        dsm = np.fmax(dsm, m)
    dsm_source = "points (max z per cell, noise removed)"
    if odm_dsm is not None and Path(odm_dsm).exists():
        try:
            ext = resample_raster_to_grid(Path(odm_dsm), grid, epsg)
            dsm = np.where(np.isfinite(ext), ext, dsm).astype(np.float32)
            dsm_source = f"ODM DSM {Path(odm_dsm).name}, gaps filled from points"
        except TerrainError as e:
            warnings.append(f"ODM DSM not used: {e}")
    progress(0.45, "DSM built")

    photo_z, photo_n = _bin_points_chunks(G.iter_ground_points(classified), grid, params.stat, params.max_ground_points)
    srcs: List[ElevationSource] = [ElevationSource(SOURCE_PHOTO, "Photogrammetry ground points", "photogrammetry_ground",
                                                   params.photo_priority, photo_z, ~np.isnan(photo_z))]
    external_trust: Dict[int, str] = {}
    src_report = [{"id": SOURCE_PHOTO, "name": "Photogrammetry ground points", "kind": "photogrammetry_ground",
                   "priority": params.photo_priority, "z_offset_m": 0.0,
                   "cells": int((~np.isnan(photo_z)).sum()), "points": int(photo_n.sum())}]

    # 4. supplemental sources
    for s in sources:
        if not 2 <= s.id <= 255:
            raise TerrainError("supplemental source ids must be 2..255")
        if cancelled():
            raise G.Cancelled("cancelled by user")
        if s.kind == "lidar":
            p = L.ensure_las(Path(s.path), workdir, pdal_bin)
            h = L.read_header(p)
            if h.epsg and epsg and h.epsg != epsg:
                raise TerrainError(f"LiDAR source {s.name!r} is EPSG:{h.epsg}; project is EPSG:{epsg}. Reproject it first "
                                   "(pdal translate --filters.reprojection).")
            has_ground = any((ch["classification"] == 2).any() for ch in L.iter_points(p))
            if not has_ground:
                out = workdir / f"source_{s.id}_classified.las"
                G.classify_ground(p, out, params.ground, workdir / f"src{s.id}", epsg or h.epsg, pdal_bin, log,
                                  lambda f, m: None, cancelled)
                p = out
                warnings.append(f"LiDAR source {s.name!r} had no ground classification; it was classified with "
                                f"{params.ground.engine}.")
            z, n = _bin_points_chunks(G.iter_ground_points(p), grid, params.stat, params.max_ground_points)
            srcs.append(ElevationSource(s.id, s.name, "lidar", s.priority, z, ~np.isnan(z), s.z_offset, dict(s.meta)))
        elif s.kind == "survey":
            x, y, z0 = load_survey_csv(Path(s.path))
            z, n = D.bin_points(x, y, z0, grid, params.stat)
            srcs.append(ElevationSource(s.id, s.name, "survey", s.priority, z, ~np.isnan(z), s.z_offset, dict(s.meta)))
        elif s.kind == "external_dem":
            z = resample_raster_to_grid(Path(s.path), grid, epsg)
            n = np.isfinite(z).astype(np.int32)
            srcs.append(ElevationSource(s.id, s.name, "external_dem", s.priority, z, np.zeros(grid.shape, bool),
                                        s.z_offset, dict(s.meta)))
            external_trust[s.id] = s.trust
        else:
            raise TerrainError(f"unknown source kind {s.kind!r}")
        src_report.append({"id": s.id, "name": s.name, "kind": s.kind, "priority": s.priority, "z_offset_m": s.z_offset,
                           "cells": int(np.isfinite(z).sum()), "points": int(np.sum(n)), **({"trust": s.trust} if s.kind == "external_dem" else {})})
        if int(np.isfinite(z).sum()) == 0:
            warnings.append(f"Source {s.name!r} does not overlap the project area and contributed nothing.")
    progress(0.55, "sources gridded")

    # 5. merge with provenance
    merged = D.merge_sources(srcs)
    measured = merged.measured & ~np.isin(merged.source_id, list(external_trust)) if external_trust else merged.measured.copy()
    base = merged.dtm.copy()

    if veg_mask_fn is None:
        warnings.append("No land-cover mask was supplied, so photogrammetric 'ground' under vegetation could not be checked "
                        "for canopy contamination. Point geometry alone cannot tell canopy from ground in a single-surface "
                        "photogrammetry cloud; classify vegetation first and re-run, and treat dense-canopy areas with caution.")
    # 6. safety check: 'ground' under mapped canopy that is not below the canopy surface
    demoted = 0
    if veg_mask_fn is not None:
        veg = veg_mask_fn(grid)
        suspect = (merged.source_id == SOURCE_PHOTO) & measured & veg & ((dsm - base) < params.canopy_clearance_m)
        demoted = int(suspect.sum())
        if demoted:
            measured &= ~suspect
            base[suspect] = np.nan
            warnings.append(f"{demoted:,} photogrammetric ground cells lie under mapped vegetation but within "
                            f"{params.canopy_clearance_m} m of the canopy surface; they were treated as canopy, not ground.")

    # 7. estimate missing cells from measured ones (bounded radius, never a global fill)
    have_model = np.isfinite(base) & ~measured
    mx, my = grid.cell_centers_xy(*np.nonzero(measured))
    if measured.sum() < 3:
        raise TerrainError("fewer than 3 measured ground cells remain; cannot estimate terrain. Supplemental ground "
                           "survey, LiDAR, or an external DEM is required.")
    res_d = D.build_dtm(mx, my, base[measured].astype(np.float64), grid, params.interp, "median")
    dtm = res_d.dtm.copy()
    dtm[have_model] = base[have_model]                  # model cells (external DEM) outrank interpolation
    evidence = np.where(measured, EV_MEASURED, EV_NONE).astype(np.uint8)
    evidence[(res_d.flag == D.FLAG_INTERPOLATED) & ~have_model] = EV_INTERP
    evidence[have_model] = EV_EXTERNAL
    dtm[measured] = base[measured]
    source = merged.source_id.copy()
    source[np.isnan(dtm)] = 0
    progress(0.75, "DTM estimated")

    # 8. confidence
    conf = D.terrain_confidence(dtm, measured, grid.res, params.confidence)
    for sid, trust in external_trust.items():
        cells = (source == sid) & (evidence == EV_EXTERNAL)
        # The value in these cells came from the external model itself, not from this project's observations, so its declared
        # trust sets the confidence (proximity to measured ground elsewhere does not corroborate the model's values).
        conf[cells] = TRUST_TO_CONF.get(trust, D.CONF_MODERATE)
    conf[np.isnan(dtm)] = D.CONF_UNKNOWN
    csum = D.confidence_summary(conf, grid.res)
    ndsm = (dsm - dtm).astype(np.float32)

    # 9. outputs
    out: Dict[str, str] = {}
    kw = dict(epsg=epsg, horizontal_units=horizontal_units, vertical_epsg=vertical_epsg, vertical_units=vertical_units)

    def put(name: str, arr, nodata, desc: str):
        p = workdir / f"{name}.tif"
        rep = gt.write_geotiff(p, arr, grid, nodata=nodata, description=desc, **kw)
        gt.write_sidecar(p, {"product": name, "description": desc, "epsg": epsg, "horizontal_units": horizontal_units,
                             "vertical_units": vertical_units, "vertical_epsg": vertical_epsg,
                             "vertical_datum": "as supplied by input data; not verified" if not vertical_epsg else f"EPSG:{vertical_epsg}",
                             "nodata": "NaN" if nodata is not None and np.isnan(nodata) else nodata,
                             "georeferenced": rep["georeferenced"], "grid": grid.to_dict(),
                             "estimate_notice": "PRELIMINARY ESTIMATE. Not suitable for final design, permitting or staking "
                             "without survey validation.", "settings": params.to_dict()})
        out[name] = str(p)

    nan = float("nan")
    put("dsm", dsm, nan, f"Observed surface (canopy/structure tops). Source: {dsm_source}")
    put("dtm", dtm, nan, "ESTIMATED bare-earth surface; see evidence.tif for measured vs interpolated")
    put("ndsm", ndsm, nan, "DSM minus DTM: height of objects above estimated ground")
    put("confidence", conf, None, "Terrain confidence: 0 unknown, 1 low, 2 moderate, 3 high")
    put("evidence", evidence, None, "Elevation evidence: 0 none, 1 measured ground, 2 interpolated, 3 external model")
    put("source", source, None, "Elevation source id per cell (see processing summary)")
    progress(0.85, "rasters written")

    labelled = workdir / "classified_cloud.las"
    bmask = building_mask_fn(grid) if building_mask_fn else None
    lab = G.label_by_height_above_ground(classified, labelled, dtm, grid, epsg or hdr.epsg, bmask)
    out["classified_cloud"] = str(labelled)

    # 10. validation + summary
    cv = None
    if params.cross_validate and int((~np.isnan(photo_z)).sum()) >= 200:
        try:
            cv = D.cross_validate(photo_z, grid, params.interp)
        except D.InsufficientGroundError as e:
            warnings.append(f"cross-validation skipped: {e}")
    if csum["advice"]:
        warnings.append(csum["advice"])
    n_cells = grid.width * grid.height
    summary = {
        "grid": grid.to_dict(), "epsg": epsg, "horizontal_units": horizontal_units, "vertical_units": vertical_units,
        "vertical_epsg": vertical_epsg, "resolution_m": grid.res, "dsm_source": dsm_source,
        "ground_engine": params.ground.engine, "ground_filter": counts,
        "ground_fraction_of_points": counts["ground"] / max(counts["total"], 1),
        "cells": n_cells,
        "coverage": {"measured": float(measured.sum()) / n_cells,
                     "interpolated": float((evidence == EV_INTERP).sum()) / n_cells,
                     "external": float((evidence == EV_EXTERNAL).sum()) / n_cells,
                     "none": float((evidence == EV_NONE).sum()) / n_cells},
        "confidence": csum, "point_labels": lab, "sources": src_report,
        "demoted_canopy_cells": demoted, "cross_validation": cv, "warnings": warnings,
        "elevation_range_m": [float(np.nanmin(dtm)), float(np.nanmax(dtm))] if np.isfinite(dtm).any() else None,
        "seconds": round(time.time() - t0, 1), "settings": params.to_dict(),
        "notice": "Estimated terrain. Not measured where evidence is 'interpolated' or 'external'. Preliminary planning "
                  "estimate only; validate with a ground survey before design, permitting or staking.",
    }
    progress(1.0, "done")
    return TerrainResult(grid, out, summary)
