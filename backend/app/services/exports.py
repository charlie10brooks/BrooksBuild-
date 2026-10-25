"""The ``export`` job: assembles deliverables into ``exports/<export_id>/`` with a manifest.

Rules enforced here:
* Every raster/vector file carries its CRS and units (GeoTIFF tags, GeoJSON ``crs`` member, ``*.meta.json`` sidecars and the
  bundle-level ``metadata.json``). A file that cannot be georeferenced is exported anyway but says so in the manifest.
* Observed surfaces (DSM) and estimated surfaces (DTM) are never mixed up: file names and metadata carry the label.
* Visual land-cover classification is exported as land cover only; it never alters elevation products.
* An item that cannot be produced is recorded under ``skipped`` with the reason. It is never silently replaced or faked.
* Everything carries the "preliminary planning estimate" disclaimer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..context import AppContext
from ..core import classes as C
from ..core import crs as CRS
from ..core import dtm as D
from ..core import planning as PL
from ..core.dtm import GridSpec
from ..core.tiling import Window
from ..core.vectorize import mask_to_polygons
from ..db import loads, now
from ..jobs import JobContext, JobFailure
from . import cad as CAD
from . import classify as K
from . import geotiff as gt
from . import ortho as O
from . import products as P
from . import render as R

ITEMS = ("originals", "orthophoto", "classification", "class_stats", "class_polygons", "dsm", "dtm", "confidence", "evidence",
         "pointcloud", "classified_cloud", "edited_cloud", "image_classes", "features", "profiles", "footprints", "cad_dxf", "report")
DEFAULT_ITEMS = tuple(i for i in ITEMS if i != "originals")      # originals can be many GB: opt in
DISCLAIMER = PL.DISCLAIMER
VISUAL_NOTE = ("Land-cover classification is a VISUAL interpretation of the orthomosaic. It is not elevation evidence and was not "
               "used as proof of ground height.")
MAX_POLY_PIXELS = 16_000_000
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(s: str, default: str = "item") -> str:
    s = _SAFE.sub("_", s or "").strip("._")
    return s[:80] or default


def exports_root(app: AppContext, pid: str) -> Path:
    return app.storage.ensure(pid, "exports")


def export_dir(app: AppContext, pid: str, eid: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", eid):
        raise ValueError("bad export id")
    return exports_root(app, pid) / eid


def list_exports(app: AppContext, pid: str) -> List[dict]:
    out = []
    root = exports_root(app, pid)
    for d in sorted(root.iterdir(), reverse=True) if root.exists() else []:
        m = d / "manifest.json"
        if m.exists():
            try:
                j = json.loads(m.read_text())
            except ValueError:
                continue
            out.append({"id": d.name, "created_at": j.get("created_at"), "label": j.get("label"), "files": len(j.get("files", [])),
                        "bytes": sum(f["bytes"] for f in j.get("files", [])), "skipped": j.get("skipped", [])})
    return out


class Exporter:
    def __init__(self, app: AppContext, ctx: JobContext, pid: str, eid: str, items: List[str], options: dict):
        self.app, self.ctx, self.pid, self.eid = app, ctx, pid, eid
        self.items, self.options = items, options
        self.dir = export_dir(app, pid, eid)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.proj = app.db.one("SELECT * FROM projects WHERE id=?", (pid,))
        self.classes = loads(self.proj["classes_json"], None) or [c.to_dict() for c in C.DEFAULT_CLASSES]
        self.files: List[dict] = []
        self.skipped: List[dict] = []
        self.epsg = self.proj["epsg"]

    # ---------------------------------------------------------------- bookkeeping
    def add(self, item: str, path: Path, description: str, crs: Optional[int] = ..., extra: Optional[dict] = None) -> None:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        self.files.append({"item": item, "name": path.relative_to(self.dir).as_posix(), "bytes": path.stat().st_size, "sha256": h.hexdigest(),
                           "description": description, "epsg": self.epsg if crs is ... else crs, **(extra or {})})

    def skip(self, item: str, reason: str) -> None:
        self.skipped.append({"item": item, "reason": reason})
        self.ctx.log(f"export item {item!r} skipped: {reason}", "warning")

    def product(self, name: str):
        p = P.get(self.app, self.pid, name)
        return p if p and p["exists"] else None

    def copy_raster(self, item: str, prod: str, fname: str, description: str) -> None:
        p = self.product(prod)
        if not p:
            return self.skip(item, f"product {prod!r} has not been produced for this project")
        dst = self.dir / fname
        shutil.copy2(p["path"], dst)
        side = Path(str(p["path"]) + ".meta.json")
        info = {}
        if side.exists():
            shutil.copy2(side, Path(str(dst) + ".meta.json"))
        try:
            gi = gt.read_georef(dst)
            info = {"georeferenced": gi.georeferenced, "grid": gi.grid.to_dict() if gi.grid else None, "units": gi.horizontal_units}
            if not gi.georeferenced:
                self.ctx.log(f"{fname} carries no georeferencing", "warning")
            crs = gi.epsg
        except Exception as e:                                                     # noqa: BLE001
            crs, info = self.epsg, {"note": f"georeference not readable: {e}"}
        self.add(item, dst, description, crs, info)

    # ---------------------------------------------------------------- items
    def originals(self) -> None:
        rows = self.app.db.query("SELECT * FROM images WHERE project_id=? ORDER BY filename, id", (self.pid,))
        if not rows:
            return self.skip("originals", "the project has no images")
        zp = self.dir / "original_images.zip"
        used: Dict[str, int] = {}
        man = io.StringIO()
        keys = [k for k in rows[0].keys() if k not in ("rel_path", "thumb_path", "project_id")]
        w = csv.writer(man)
        w.writerow(["archive_name"] + keys)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:         # JPEG/TIFF are already compressed
            for i, r in enumerate(rows):
                if self.ctx.cancelled():
                    from ..jobs import JobCancelled
                    raise JobCancelled()
                src = self.app.storage.path(self.pid, r["rel_path"])
                name = safe_name(r["filename"], "image")
                if name in used:
                    used[name] += 1
                    stem, dot, ext = name.rpartition(".")
                    name = f"{stem or name}_{used[name]}{dot}{ext}" if dot else f"{name}_{used[name]}"
                else:
                    used[name] = 0
                if not src.exists():
                    self.ctx.log(f"original {r['filename']} missing on disk", "warning")
                    continue
                z.write(src, f"images/{name}")
                w.writerow([f"images/{name}"] + [r[k] for k in keys])
                if i % 50 == 0:
                    self.ctx.progress(self.base + 0.05 * (i / len(rows)), f"zipping originals {i}/{len(rows)}")
            z.writestr("images_manifest.csv", man.getvalue())
        self.add("originals", zp, f"{len(rows)} original image files, byte-identical to the uploads, with a metadata CSV", None)

    def orthophoto(self) -> None:
        self.copy_raster("orthophoto", "orthophoto", "orthophoto.tif", "Orthomosaic (RGB). Visual imagery, not elevation.")

    def dsm(self) -> None:
        self.copy_raster("dsm", "dsm", "dsm_OBSERVED_surface.tif",
                         "OBSERVED surface: tops of canopy and structures as seen by photogrammetry. Not bare earth.")

    def dtm(self) -> None:
        self.copy_raster("dtm", "dtm", "dtm_ESTIMATED_bare_earth.tif",
                         "ESTIMATED bare-earth surface (DTM). Under canopy it is partly interpolated; see the evidence and confidence rasters.")

    def confidence(self) -> None:
        self.copy_raster("confidence", "confidence", "terrain_confidence.tif", "Terrain confidence: 0 unknown, 1 low, 2 moderate, 3 high.")

    def evidence(self) -> None:
        self.copy_raster("evidence", "evidence", "terrain_evidence.tif",
                         "Evidence behind each DTM cell: 0 none, 1 measured ground, 2 interpolated, 3 external model.")

    def pointcloud(self) -> None:
        p = self.product("pointcloud")
        if not p:
            return self.skip("pointcloud", "no point cloud has been produced")
        if self.options.get("laz"):
            las = self.laz(p["path"], "pointcloud")
            if las:
                self.add("pointcloud", las, "Photogrammetry point cloud (LAZ). Unclassified surface points.", self.epsg)
            return
        dst = self.dir / ("pointcloud" + p["path"].suffix.lower())
        shutil.copy2(p["path"], dst)
        self.add("pointcloud", dst, "Photogrammetry point cloud. Unclassified surface points.", self.epsg)

    def edited_cloud(self) -> None:
        from . import cloudedit as CE
        ci = CE.info(self.app, self.pid)
        if not ci.get("available"):
            return self.skip("edited_cloud", ci.get("reason", "no point-cloud display sample"))
        dst = self.dir / "pointcloud_display_sample_EDITED.las"
        r = CE.write_edited_las(self.app, self.pid, dst)
        self.add("edited_cloud", dst, f"Display SAMPLE of the point cloud ({r['points']:,} of {r['total_points']:,} points) with your manual class edits "
                 f"({ci['edited_points']:,} points edited). Not the full cloud; does not change the DTM.", r["epsg"] or self.epsg,
                 {"sample": True, "points": r["points"], "total_points": r["total_points"], "edited_points": ci["edited_points"]})

    def image_classes(self) -> None:
        import csv
        rows = self.app.db.query("SELECT i.filename, i.width iw, i.height ih, c.* FROM image_classification c JOIN images i ON i.id=c.image_id "
                                 "WHERE c.project_id=? ORDER BY i.filename", (self.pid,))
        if not rows:
            return self.skip("image_classes", "no drone photos have been classified")
        names = {int(c["id"]): c["name"] for c in self.classes}
        ids = sorted(names)
        dst = self.dir / "photo_classification_stats.csv"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "model", "work_width_px", "work_height_px", "unclassified_fraction", "mean_confidence"] + [f"frac_{names[i]}" for i in ids])
            for r in rows:
                counts = loads(r["counts_json"], {})
                px = max(1, sum(counts.values()))
                w.writerow([r["filename"], r["model"], r["width"], r["height"], f"{r['unclassified']:.4f}",
                            "" if r["mean_conf"] is None else f"{r['mean_conf']:.3f}"] + [f"{counts.get(str(i), 0) / px:.4f}" for i in ids])
        self.add("image_classes", dst, "Per-photo land-cover fractions (each photo's own pixels; not georeferenced, not elevation evidence).", None)

    def classified_cloud(self) -> None:
        p = self.product("classified_cloud")
        if not p:
            return self.skip("classified_cloud", "ground filtering has not been run (no classified cloud)")
        if self.options.get("laz"):
            las = self.laz(p["path"], "classified_cloud")
            if las:
                self.add("classified_cloud", las, "Point cloud with ground classification (ASPRS class 2 = ground).", self.epsg)
            return
        dst = self.dir / "pointcloud_ground_classified.las"
        shutil.copy2(p["path"], dst)
        self.add("classified_cloud", dst, "Point cloud with ground classification (ASPRS class 2 = ground, 1 = unclassified).", self.epsg)

    def laz(self, src: Path, item: str) -> Optional[Path]:
        pdal = self.app.settings.pdal_bin
        if shutil.which(pdal) is None:
            self.skip(item, "LAZ output requires PDAL, which is not installed; export without 'laz' to get the uncompressed LAS")
            return None
        dst = self.dir / (("pointcloud_ground_classified" if item == "classified_cloud" else "pointcloud") + ".laz")
        r = subprocess.run([pdal, "translate", str(src), str(dst), "--writers.las.compression=true"], capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            self.skip(item, f"pdal translate failed: {r.stderr.strip()[:300]}")
            return None
        return dst

    def _class_inputs(self):
        prod = self.product("orthophoto")
        if not prod:
            self.skip("classification", "no orthomosaic, so no classification raster exists")
            return None
        mask, _ = O.class_rasters(self.app, self.pid)
        g = prod["meta"].get("grid")
        return prod, mask, (GridSpec(**g) if g else None)

    def classification(self) -> None:
        got = self._class_inputs()
        if not got:
            return
        prod, mask, grid = got
        hist = mask.histogram()
        if not any((v & K.CLASS_MASK) for v in hist):
            self.skip("classification", "nothing has been classified yet (all pixels are Unknown)")
            return
        units = self.proj["horizontal_units"]
        desc = "VISUAL land-cover class ids (0 = unknown). " + VISUAL_NOTE
        out = self.dir / "classification.tif"
        gt.write_geotiff_tiles(out, mask.width, mask.height, np.uint8, lambda w: mask.read(w) & K.CLASS_MASK, grid, self.epsg if grid else None,
                               nodata=None, horizontal_units=units or "metre", description=desc)
        gt.write_sidecar(out, {"description": desc, "epsg": self.epsg, "units": units, "georeferenced": grid is not None,
                               "classes": self.classes, "disclaimer": DISCLAIMER})
        self.add("classification", out, desc, self.epsg if grid else None, {"georeferenced": grid is not None})
        prv = self.dir / "classification_provenance.tif"
        gt.write_geotiff_tiles(prv, mask.width, mask.height, np.uint8,
                               lambda w: np.where(mask.read(w) & K.TAG_MANUAL, 1, np.where(mask.read(w) & K.TAG_AUTO, 2, 0)).astype(np.uint8),
                               grid, self.epsg if grid else None, nodata=None, horizontal_units=units or "metre",
                               description="Provenance: 0 none, 1 manually set, 2 automatic (model) classification.")
        self.add("classification", prv, "Per-pixel provenance of the class: 0 none, 1 manual, 2 automatic", self.epsg if grid else None)
        tb = self.dir / "classification_classes.csv"
        with open(tb, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "key", "name", "color_hex", "vegetation_obscures_ground"])
            for c in self.classes:
                w.writerow([c["id"], c.get("key", ""), c["name"], c["color"], int(bool(c.get("vegetation")))])
        self.add("classification", tb, "Class table (id, name, colour)", None)

    def class_stats(self) -> None:
        got = self._class_inputs()
        if not got:
            return
        prod, mask, grid = got
        unit = None
        if grid and self.proj["horizontal_units"] != "degree":
            f = {"metre": 1.0, "foot": 0.3048, "us_survey_foot": 1200 / 3937}.get(self.proj["horizontal_units"], 1.0)
            unit = (grid.res * f) ** 2
        h = mask.histogram()
        total = mask.width * mask.height
        p = self.dir / "class_statistics.csv"
        with open(p, "w", newline="") as f:
            f.write("# " + VISUAL_NOTE + "\n# " + DISCLAIMER + "\n")
            w = csv.writer(f)
            w.writerow(["class_id", "name", "pixels", "percent", "area_m2", "area_acres", "manual_pixels", "automatic_pixels", "unreviewed_pixels"])
            for c in self.classes:
                n = {"none": 0, "manual": 0, "automatic": 0}
                for v, cnt in h.items():
                    if v & K.CLASS_MASK == c["id"]:
                        n["manual" if v & K.TAG_MANUAL else ("automatic" if v & K.TAG_AUTO else "none")] += cnt
                px = sum(n.values())
                w.writerow([c["id"], c["name"], px, round(100.0 * px / total, 4) if total else 0,
                            round(px * unit, 3) if unit else "", round(px * unit / 4046.8564224, 5) if unit else "",
                            n["manual"], n["automatic"], n["none"]])
        self.add("class_stats", p, "Pixel and area statistics per class" + ("" if unit else " (areas unavailable: no projected georeferencing)"), None)

    def class_polygons(self) -> None:
        got = self._class_inputs()
        if not got:
            return
        prod, mask, grid = got
        if grid is None:
            return self.skip("class_polygons", "orthomosaic is not georeferenced; polygons would have no coordinates")
        step = 1
        while (mask.width // step) * (mask.height // step) > MAX_POLY_PIXELS:
            step *= 2
        # decimate by sampling tile by tile (never loads the whole mask)
        T = mask.tile
        if step > T:
            return self.skip("class_polygons", "raster too large to vectorize")
        h2, w2 = mask.height // step, mask.width // step
        small = np.zeros((h2, w2), np.uint8)
        for ty in range(0, mask.height, T):
            for tx in range(0, mask.width, T):
                a = mask.read(Window(tx, ty, min(tx + T, mask.width), min(ty + T, mask.height)))[::step, ::step] & K.CLASS_MASK
                r0, c0 = ty // step, tx // step
                sub = small[r0:r0 + a.shape[0], c0:c0 + a.shape[1]]
                sub[:] = a[:sub.shape[0], :sub.shape[1]]
        g2 = GridSpec(grid.x0, grid.y0, grid.res * step, w2, h2)
        feats = []
        for c in self.classes:
            if c["id"] == C.UNKNOWN:
                continue
            for geom in mask_to_polygons(small == c["id"], g2, min_cells=4, simplify_cells=0.5):
                feats.append({"type": "Feature", "geometry": geom, "properties": {"class_id": c["id"], "class": c["name"],
                                                                                   "vegetation": bool(c.get("vegetation")),
                                                                                   "kind": "visual_land_cover"}})
        gj = self.geojson(feats, {"note": VISUAL_NOTE + (f" Vectorized at 1/{step} of native resolution." if step > 1 else ""),
                                  "disclaimer": DISCLAIMER})
        p = self.dir / "land_cover_polygons.geojson"
        p.write_text(json.dumps(gj))
        self.add("class_polygons", p, "Land-cover polygons (visual interpretation; not elevation evidence)", self.epsg,
                 {"features": len(feats), "decimation": step})

    # ---------------------------------------------------------------- vector helpers
    def geojson(self, feats: List[dict], props: Optional[dict] = None, epsg: Optional[int] = ...) -> dict:
        e = self.epsg if epsg is ... else epsg
        gj = {"type": "FeatureCollection", "features": feats, **(props or {})}
        if e:
            gj["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{e}"}}
        gj["horizontal_units"] = self.proj["horizontal_units"]
        return gj

    def features(self) -> None:
        feats = []
        if self.proj["boundary_geojson"]:
            feats.append({"type": "Feature", "geometry": self.boundary_in_project_crs(loads(self.proj["boundary_geojson"])),
                          "properties": {"kind": "project_boundary", "name": "Project boundary"}})
        for r in self.app.db.query("SELECT * FROM features WHERE project_id=? ORDER BY created_at", (self.pid,)):
            feats.append({"type": "Feature", "geometry": loads(r["geometry_json"]),
                          "properties": {"kind": r["kind"], "name": r["name"], **loads(r["props_json"], {}), "disclaimer": "preliminary planning estimate"}})
        if not feats:
            return self.skip("features", "no boundary, footprints, profiles or measurements have been drawn")
        p = self.dir / f"features_epsg{self.epsg or 'unknown'}.geojson"
        feat_proj = feats                       # every feature (boundary included) is now in the project CRS
        p.write_text(json.dumps(self.geojson(feats, {"disclaimer": DISCLAIMER})))
        self.add("features", p, "Project boundary and drawn features in the PROJECT CRS" + ("" if self.epsg else " (CRS unknown!)"), self.epsg)
        if self.epsg and self.epsg != 4326 and feat_proj:
            try:
                t = [{**f, "geometry": CRS.transform_geometry(f["geometry"], self.epsg, 4326)} for f in feat_proj]
                p2 = self.dir / "features_wgs84.geojson"
                gj = {"type": "FeatureCollection", "features": t, "disclaimer": DISCLAIMER}
                p2.write_text(json.dumps(gj))
                self.add("features", p2, "Drawn features in WGS84 lon/lat (RFC 7946)", 4326)
            except CRS.UnsupportedCrs as e:
                self.ctx.log(f"WGS84 copy of features not written: {e}", "warning")

    def boundary_in_project_crs(self, g: dict) -> dict:
        """The API accepts boundaries as lon/lat or already in the project CRS. Exports always state one CRS, so lon/lat is converted."""
        def pts(c):
            if c and isinstance(c[0], (int, float)):
                yield c
            else:
                for i in c:
                    yield from pts(i)
        allp = list(pts(g["coordinates"]))
        looks_lonlat = bool(allp) and all(abs(p[0]) <= 180 and abs(p[1]) <= 90 for p in allp)
        if looks_lonlat and self.epsg and self.epsg != 4326:
            try:
                return CRS.transform_geometry(g, 4326, self.epsg)
            except CRS.UnsupportedCrs as e:
                self.ctx.log(f"boundary left in lon/lat: {e}", "warning")
                self.skip("boundary_crs", f"boundary is in lon/lat and could not be converted to EPSG:{self.epsg}: {e}")
        return g

    def _dem(self):
        d = self.product("dtm")
        if not d:
            return None
        arr, grid, _ = R.RASTERS.get(d["path"])
        c, s = self.product("confidence"), self.product("dsm")
        return arr, grid, (R.RASTERS.get(c["path"])[0] if c else None), (R.RASTERS.get(s["path"])[0] if s else None)

    def profiles(self) -> None:
        rows = self.app.db.query("SELECT * FROM features WHERE project_id=? AND kind='profile' ORDER BY created_at", (self.pid,))
        if not rows:
            return self.skip("profiles", "no profile lines have been drawn")
        dem = self._dem()
        if not dem:
            return self.skip("profiles", "no DTM yet; run the terrain job first")
        arr, grid, conf, dsm = dem
        for i, r in enumerate(rows, 1):
            g = loads(r["geometry_json"])
            if g.get("type") != "LineString":
                continue
            prof = PL.elevation_profile(arr, grid, [tuple(p[:2]) for p in g["coordinates"]], loads(r["props_json"], {}).get("spacing"), conf, dsm)
            cols = {"distance": np.round(prof["distance_m"], 4), "x": np.round(prof["x"], 4), "y": np.round(prof["y"], 4),
                    "ground_z_ESTIMATED": np.round(prof["ground_z"], 4)}
            if "surface_z" in prof:
                cols["surface_z_OBSERVED"] = np.round(prof["surface_z"], 4)
                cols["canopy_height"] = np.round(prof["canopy_height"], 4)
            if "confidence" in prof:
                cols["terrain_confidence"] = [D.CONF_NAMES[int(c)] for c in prof["confidence"]]
            p = self.dir / f"profile_{i:02d}_{safe_name(r['name'] or 'line')}.csv"
            with open(p, "w", newline="") as f:
                f.write("# " + DISCLAIMER + "\n# ground_z_ESTIMATED is an estimate of bare earth; surface_z_OBSERVED is the observed surface. "
                        f"Horizontal CRS EPSG:{self.epsg}; distances in {self.proj['horizontal_units']}; elevations in {self.proj['vertical_units']}. Empty = no data.\n")
                w = csv.writer(f)
                w.writerow(list(cols))
                for row in zip(*cols.values()):
                    w.writerow(["" if (isinstance(v, (float, np.floating)) and not np.isfinite(v)) else v for v in row])
            self.add("profiles", p, f"Elevation profile '{r['name'] or i}'", self.epsg)

    def footprint_reports(self) -> List[dict]:
        rows = self.app.db.query("SELECT * FROM features WHERE project_id=? AND kind='footprint' ORDER BY created_at", (self.pid,))
        dem = self._dem()
        out = []
        if not rows or not dem:
            return out
        arr, grid, conf, dsm = dem
        for r in rows:
            g = loads(r["geometry_json"])
            if g.get("type") != "Polygon":
                continue
            ring = [tuple(p[:2]) for p in g["coordinates"][0]]
            rep = PL.footprint_report(arr, grid, ring, conf, dsm, loads(r["props_json"], {}).get("steep_deg", 15.0))
            out.append({"name": r["name"] or r["id"][:8], "report": rep})
        return out

    def footprints(self) -> None:
        if not self.app.db.one("SELECT 1 FROM features WHERE project_id=? AND kind='footprint'", (self.pid,)):
            return self.skip("footprints", "no footprints have been drawn")
        if not self._dem():
            return self.skip("footprints", "no DTM yet; run the terrain job first")
        reps = self.footprint_reports()
        p = self.dir / "footprint_reports.json"
        p.write_text(json.dumps({"disclaimer": DISCLAIMER, "epsg": self.epsg, "units": {"horizontal": self.proj["horizontal_units"],
                                                                                         "vertical": self.proj["vertical_units"]},
                                 "footprints": reps}, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else
                                 (o.tolist() if hasattr(o, "tolist") else str(o))))
        self.add("footprints", p, "Elevation, slope and evidence summary under each drawn footprint (preliminary planning estimate)", self.epsg)

    def cad_dxf(self) -> None:
        """DXF for CAD packages: boundary, footprints, profile lines, control points and ESTIMATED-ground contours, one layer each.
        Contours are split onto layers by terrain confidence so extrapolated stretches are visibly different from supported ones."""
        dx = CAD.Dxf(self.proj["horizontal_units"])
        L = {n: dx.layer(n, c) for n, c in (("TS_BOUNDARY", 30), ("TS_FOOTPRINTS", 5), ("TS_PROFILES", 6), ("TS_MEASUREMENTS", 4), ("TS_NOTES", 7),
                                            ("TS_CONTROL_GCP", 1), ("TS_CONTROL_CHECK", 3), ("TS_CONTOUR_EST_MINOR", 94), ("TS_CONTOUR_EST_MAJOR", 52),
                                            ("TS_CONTOUR_LOWCONF_MINOR", 12), ("TS_CONTOUR_LOWCONF_MAJOR", 10), ("TS_CONTOUR_LABELS", 7))}
        drawn = 0
        counts: dict = {}

        def rings(g):
            if g.get("type") == "Polygon":
                return g["coordinates"]
            if g.get("type") == "MultiPolygon":
                return [r for poly in g["coordinates"] for r in poly]
            return []

        if self.proj["boundary_geojson"]:
            for ring in rings(self.boundary_in_project_crs(loads(self.proj["boundary_geojson"]))):
                dx.polyline(L["TS_BOUNDARY"], ring[:-1] if len(ring) > 3 and ring[0] == ring[-1] else ring, closed=True)
                drawn += 1
            counts["boundary"] = 1
        labels: List[tuple] = []
        for r in self.app.db.query("SELECT * FROM features WHERE project_id=? ORDER BY created_at", (self.pid,)):
            g, kind = loads(r["geometry_json"]), r["kind"]
            layer = L["TS_FOOTPRINTS"] if kind == "footprint" else L["TS_PROFILES"] if kind == "profile" else L["TS_MEASUREMENTS"] if kind == "measurement" else L["TS_NOTES"]
            if g.get("type") in ("Polygon", "MultiPolygon"):
                for ring in rings(g):
                    dx.polyline(layer, ring[:-1] if len(ring) > 3 and ring[0] == ring[-1] else ring, closed=True)
                    drawn += 1
                    if r["name"]:
                        labels.append((layer, ring[0][0], ring[0][1], r["name"]))
            elif g.get("type") == "LineString":
                dx.polyline(layer, g["coordinates"])
                drawn += 1
                if r["name"]:
                    labels.append((layer, g["coordinates"][0][0], g["coordinates"][0][1], r["name"]))
            elif g.get("type") == "Point":
                dx.point(layer, g["coordinates"][0], g["coordinates"][1])
                drawn += 1
                if r["name"]:
                    labels.append((layer, g["coordinates"][0], g["coordinates"][1], r["name"]))
            counts[kind] = counts.get(kind, 0) + 1
        for r in self.app.db.query("SELECT * FROM control_points WHERE project_id=? ORDER BY created_at", (self.pid,)):
            layer = L["TS_CONTROL_GCP"] if r["kind"] == "gcp" else L["TS_CONTROL_CHECK"]
            dx.point(layer, r["x"], r["y"], r["z"] or 0.0)
            labels.append((layer, r["x"], r["y"], f"{r['label']} ({r['kind']})" + (f" z={r['z']:g}" if r["z"] is not None else "")))
            drawn += 1
            counts["control_points"] = counts.get("control_points", 0) + 1
        info: dict = {"entities": 0, "layers": sorted(L)}
        dem = self._dem()
        if dem:
            arr, grid, conf, _ = dem
            finite = arr[np.isfinite(arr)]
            if finite.size:
                interval = float(self.options.get("contour_interval") or CAD.nice_interval(float(finite.max() - finite.min())))
                res = CAD.terrain_contours(arr, grid.x0, grid.y0, grid.res, interval, conf)
                for level, major, weak, pts in res["lines"]:
                    name = ("TS_CONTOUR_LOWCONF_" if weak else "TS_CONTOUR_EST_") + ("MAJOR" if major else "MINOR")
                    dx.polyline(L[name], pts, elevated=True)
                    drawn += 1
                    if major and not weak and len(pts) > 6 and len(labels) < 4000:
                        m = pts[len(pts) // 2]
                        labels.append((L["TS_CONTOUR_LABELS"], m[0], m[1], f"{level:g}"))
                info.update({"contour_interval": interval, "contour_vertical_units": self.proj["vertical_units"], "contour_lines": len(res["lines"]),
                             "contours_smoothed_over_m": round(res["smoothed_cells"] * grid.res if res["smoothed_cells"] else 0.0, 3),
                             "contour_decimation": res["stride"]})
                counts["contours"] = len(res["lines"])
        if not drawn:
            return self.skip("cad_dxf", "nothing to draw: no boundary, drawn features, control points or terrain model yet")
        ex = dx.ext
        th = max(0.5, max(ex[2] - ex[0], ex[3] - ex[1]) / 160)
        for layer, x, y, t in labels:
            dx.text(layer, x + th * 0.3, y + th * 0.3, th, t)
        note = ["PRELIMINARY PLANNING ESTIMATE - NOT A SURVEY. Do not use for foundation design, permitting or staking without validation by a licensed surveyor.",
                f"Source: BrooksBuild. Horizontal CRS EPSG:{self.epsg or 'UNKNOWN'}, units {self.proj['horizontal_units']}; elevations in {self.proj['vertical_units']}.",
                "Contours come from the ESTIMATED bare-earth model (smoothed). TS_CONTOUR_LOWCONF_* = low/unknown confidence (largely extrapolated); do not rely on them."]
        for i, t in enumerate(note):
            dx.text(L["TS_NOTES"], ex[0], ex[1] - th * (2 + 1.6 * i), th, t)
        p = self.dir / f"site_plan_epsg{self.epsg or 'unknown'}_PRELIMINARY.dxf"
        dx.write(p)
        info["entities"] = len(dx.ents)
        self.add("cad_dxf", p, "CAD drawing (DXF R12): boundary, footprints, profiles, control points and ESTIMATED-ground contours on separate layers. "
                 "Preliminary planning estimate." + ("" if self.epsg else " CRS unknown!"), self.epsg, {"cad": info, "drawn": counts})

    # ---------------------------------------------------------------- report & metadata
    def report(self) -> None:
        from .report_pdf import build_report
        p = self.dir / "site_report.pdf"
        build_report(self, p)
        self.add("report", p, "PDF summary report (preliminary planning estimate)", None)

    def metadata(self) -> dict:
        prods = {}
        for r in P.list_products(self.app, self.pid):
            prods[r["name"]] = {"kind": r["kind"], "exists": r["exists"], "bytes": r["size"], "created_at": r["created_at"],
                                "meta": {k: v for k, v in r["meta"].items() if k not in ("pyramid", "raw")}}
        summary = (self.product("terrain_summary") or {}).get("meta")
        return {"generator": "BrooksBuild", "created_at": now(), "export_id": self.eid, "disclaimer": DISCLAIMER, "visual_classification_note": VISUAL_NOTE,
                "surface_definitions": {"land_cover": "visual classification of the orthomosaic (not elevation evidence)",
                                        "observed_surface": "DSM / photogrammetry surface points: canopy and structure tops",
                                        "classified_ground_points": "point-cloud returns accepted as ground by the ground filter",
                                        "estimated_bare_earth": "DTM: measured where ground returns exist, interpolated elsewhere",
                                        "insufficient_evidence": "terrain-confidence class 0 / evidence 0: no defensible elevation estimate"},
                "project": {"id": self.pid, "name": self.proj["name"], "address": self.proj["address"], "synthetic_sample": bool(self.proj["is_sample"]),
                            "horizontal_epsg": self.epsg, "vertical_epsg": self.proj["vertical_epsg"], "horizontal_units": self.proj["horizontal_units"],
                            "vertical_units": self.proj["vertical_units"], "unit_system": self.proj["unit_system"],
                            "settings": loads(self.proj["settings_json"], {})},
                "classes": self.classes, "products": prods, "terrain_run": summary,
                "files": self.files, "skipped": self.skipped}


def readme_text(meta: dict) -> str:
    p = meta["project"]
    return "\n".join([
        "BrooksBuild export", "=" * 17, "", meta["disclaimer"], "",
        f"Project: {p['name']}   (horizontal EPSG:{p['horizontal_epsg']}, {p['horizontal_units']}; vertical {p['vertical_units']})",
        "SYNTHETIC SAMPLE DATA - not a real site." if p["synthetic_sample"] else "", "",
        "Read this before using the elevation files:",
        " * *_OBSERVED_* = what the photogrammetry saw (tops of trees, roofs). It is NOT the ground.",
        " * *_ESTIMATED_* = bare-earth estimate. Under dense canopy it is interpolated from sparse returns.",
        "   Check terrain_confidence.tif / terrain_evidence.tif before trusting any cell. Obtain supplemental survey data",
        "   (LiDAR, total-station or GNSS ground shots) where confidence is low or unknown.",
        " * Land-cover classification is visual and is never evidence of ground elevation.", "",
        "Files: see manifest.json (sha256 for every file) and metadata.json (CRS, units, settings, provenance).", ""])


def run_export(app: AppContext, ctx: JobContext) -> dict:
    pid = ctx.params["project_id"]
    req = ctx.params.get("items") or list(DEFAULT_ITEMS)
    if req == "all":
        req = list(ITEMS)
    bad = [i for i in req if i not in ITEMS]
    if bad:
        raise JobFailure(f"unknown export item(s): {', '.join(bad)}. Choose from {', '.join(ITEMS)}", "bad_request")
    items = [i for i in ITEMS if i in req]
    eid = ctx.id[:16]
    ex = Exporter(app, ctx, pid, eid, items, {"laz": bool(ctx.params.get("laz")), "contour_interval": ctx.params.get("contour_interval")})
    tot = len(items)
    for n, item in enumerate(items):
        if ctx.cancelled():
            from ..jobs import JobCancelled
            raise JobCancelled()
        ex.base = 0.02 + 0.85 * n / tot
        ctx.progress(ex.base, f"exporting {item}")
        try:
            getattr(ex, item)()
        except JobFailure:
            raise
        except Exception as e:                                                       # noqa: BLE001
            if e.__class__.__name__ == "JobCancelled":
                raise
            ctx.log(f"export of {item} failed: {type(e).__name__}: {e}", "error")
            ex.skip(item, f"failed: {type(e).__name__}: {e}")
    if not ex.files:
        raise JobFailure("nothing could be exported: " + "; ".join(f"{s['item']}: {s['reason']}" for s in ex.skipped), "nothing_to_export")
    ctx.progress(0.9, "writing metadata and bundle")
    meta = ex.metadata()
    (ex.dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (ex.dir / "README.txt").write_text(readme_text(meta))
    ex.add("metadata", ex.dir / "metadata.json", "CRS, units, settings, product provenance", None)
    ex.add("metadata", ex.dir / "README.txt", "How to read the deliverables", None)
    bundle = ex.dir / f"terrascope_{safe_name(ex.proj['name'], 'project')}_{eid[:8]}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in ex.files:
            if f["item"] == "originals":                                  # kept as its own archive: can be tens of GB
                continue
            z.write(ex.dir / f["name"], f["name"])
            sidecar = ex.dir / (f["name"] + ".meta.json")
            if sidecar.exists():
                z.write(sidecar, f["name"] + ".meta.json")
    ex.add("bundle", bundle, "Everything except the original-image archive, in one zip", None)
    manifest = {"export_id": eid, "created_at": now(), "label": ctx.params.get("label"), "project_id": pid, "requested": items,
                "files": ex.files, "skipped": ex.skipped, "disclaimer": DISCLAIMER}
    (ex.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    ctx.progress(1.0, "done")
    return {"export_id": eid, "files": len(ex.files), "skipped": ex.skipped, "bytes": sum(f["bytes"] for f in ex.files)}


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["export"] = run_export
