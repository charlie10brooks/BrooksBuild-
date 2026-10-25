"""OpenDroneMap orchestration (Docker).  ODM does the photogrammetry; this module prepares inputs, runs it, watches it,
and registers its outputs. It never fabricates outputs: if Docker or the ODM image is unavailable the job FAILS with install
instructions, and a job only succeeds if ODM exited 0 *and* the expected products exist.

Integration notes (verify against your ODM version):
* Invocation:  docker run --rm -v <project dir>:/datasets/code <image> --project-path /datasets code <options>
* Before running, ``<image> --help`` is parsed and every option we pass is checked against it, so a renamed/removed option
  is reported up-front instead of failing deep into a multi-hour run.
* ODM resumes from the last completed stage when the project directory is intact, so "retry" after a failure/cancel continues
  rather than restarting. There is no true pause: cancel stops the container and retry resumes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..context import AppContext
from ..db import dumps, loads
from ..jobs import JobCancelled, JobContext, JobFailure
from . import geotiff as gt
from . import las as L
from . import products as P

DOCKER_HINT = ("Install Docker Engine and make sure the daemon is running (https://docs.docker.com/engine/install/), then "
               "`docker pull opendronemap/odm`. Alternatively run the provided docker-compose stack, which mounts the Docker socket.")

PRESETS: Dict[str, Dict[str, object]] = {
    "draft": {"fast-orthophoto": True, "pc-quality": "low", "feature-quality": "low", "orthophoto-resolution": 10.0, "dem-resolution": 20.0},
    "balanced": {"pc-quality": "medium", "feature-quality": "medium", "orthophoto-resolution": 5.0, "dem-resolution": 10.0},
    "high": {"pc-quality": "high", "feature-quality": "high", "orthophoto-resolution": 2.5, "dem-resolution": 5.0},
}
# every ODM option the platform may pass. Values are validated by type; anything else is refused.
ALLOWED = {"fast-orthophoto": bool, "pc-quality": ("ultra", "high", "medium", "low", "lowest"), "feature-quality": ("ultra", "high", "medium", "low", "lowest"),
           "orthophoto-resolution": float, "dem-resolution": float, "min-num-features": int, "max-concurrency": int,
           "auto-boundary": bool, "use-3dmesh": bool, "skip-report": bool}
ALWAYS = ["dsm", "dtm", "pc-las"]          # we need DSM, ODM's own DTM for reference, and an uncompressed LAS if available
STAGES = ["dataset", "split", "merge", "opensfm", "openmvs", "odm_filterpoints", "odm_meshing", "mvs_texturing", "odm_georeferencing",
          "odm_dem", "odm_orthophoto", "odm_report", "odm_postprocess"]


def build_options(preset: str, overrides: Optional[dict], has_gcp: bool) -> List[str]:
    if preset not in PRESETS:
        raise JobFailure(f"unknown preset {preset!r}; choose one of {sorted(PRESETS)}", "bad_request")
    opts = dict(PRESETS[preset])
    for k, v in (overrides or {}).items():
        k = k.lstrip("-")
        if k not in ALLOWED:
            raise JobFailure(f"ODM option --{k} is not allowed through the platform (allowed: {sorted(ALLOWED)})", "bad_request")
        t = ALLOWED[k]
        if isinstance(t, tuple):
            if v not in t:
                raise JobFailure(f"--{k} must be one of {t}", "bad_request")
        elif t is bool:
            v = bool(v)
        else:
            try:
                v = t(v)
            except (TypeError, ValueError):
                raise JobFailure(f"--{k} must be a {t.__name__}", "bad_request")
            if v <= 0:
                raise JobFailure(f"--{k} must be positive", "bad_request")
        opts[k] = v
    out: List[str] = [f"--{a}" for a in ALWAYS]
    for k, v in opts.items():
        if v is True:
            out.append(f"--{k}")
        elif v is False:
            continue
        else:
            out += [f"--{k}", str(v)]
    if has_gcp:
        out += ["--gcp", "/datasets/code/gcp_list.txt"]
    return out


def option_names(args: List[str]) -> List[str]:
    return [a for a in args if a.startswith("--")]


def check_docker(app: AppContext) -> str:
    docker = shutil.which(app.settings.docker_bin)
    from .las import MissingDependency
    if not docker:
        raise MissingDependency(f"Docker was not found on PATH ({app.settings.docker_bin!r}). " + DOCKER_HINT)
    try:
        r = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=20)
    except Exception as e:
        raise MissingDependency(f"Docker could not be queried ({e}). " + DOCKER_HINT)
    if r.returncode != 0:
        raise MissingDependency("Docker is installed but its daemon is not reachable: " + (r.stderr.strip()[:300] or "no details") + ". " + DOCKER_HINT)
    return docker


def verify_options(app: AppContext, docker: str, args: List[str]) -> None:
    r = subprocess.run([docker, "run", "--rm", app.settings.odm_image, "--help"], capture_output=True, text=True, timeout=300)
    text = r.stdout + r.stderr
    if r.returncode != 0 and "--project-path" not in text:
        from .las import MissingDependency
        raise MissingDependency(f"Could not run the ODM image {app.settings.odm_image!r}: {text.strip()[:300]}. "
                                f"Pull it with `docker pull {app.settings.odm_image}`.")
    missing = [o for o in sorted(set(option_names(args) + ["--project-path"])) if o not in text]
    if missing:
        raise JobFailure(f"The installed ODM does not recognise: {', '.join(missing)}. The option set was written for a different ODM "
                         "version; adjust services/odm.py (ALWAYS / PRESETS) or pin the image version.", "odm_option_mismatch")


def prepare_inputs(app: AppContext, pid: str, exclude: List[str], ctx: JobContext) -> dict:
    work = app.storage.ensure(pid, "odm")
    imgs = work / "images"
    imgs.mkdir(exist_ok=True)
    rows = app.db.query("SELECT id, filename, rel_path, lat, lon FROM images WHERE project_id=?", (pid,))
    n = linked = skipped = 0
    want = set()
    for r in rows:
        if r["id"] in exclude:
            skipped += 1
            continue
        src = app.storage.path(pid, r["rel_path"])
        name = f"{r['id'][:8]}_{r['filename']}"
        want.add(name)
        dst = imgs / name
        if not dst.exists():
            try:
                os.link(src, dst)                 # hardlink: no copy of potentially thousands of large images
            except OSError:
                shutil.copy2(src, dst)            # different filesystem
        n += 1
        linked += 1
    for f in imgs.iterdir():                      # remove inputs that were excluded/deleted since a previous run
        if f.name not in want:
            f.unlink()
    with_gps = sum(1 for r in rows if r["lat"] is not None and r["id"] not in exclude)
    return {"images": n, "with_gps": with_gps, "excluded": skipped}


def _progress(line: str, state: dict) -> Optional[float]:
    m = re.search(r"Running (\w+) stage", line)
    if m and m.group(1) in STAGES:
        state["stage"] = m.group(1)
        return (STAGES.index(m.group(1)) + 0.1) / len(STAGES)
    return None


def _find(base: Path, *cands: str) -> Optional[Path]:
    for c in cands:
        p = base / c
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _first_number(d: dict, *keys: str):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def parse_report(work: Path) -> dict:
    """Pull quality statistics from ODM's ``odm_report/stats.json`` when present.

    The key names below follow ODM 3.x as best known and are UNVERIFIED against a live run in this repository. Every field
    is optional: anything not found is reported as missing, never guessed.
    """
    sp = work / "odm_report" / "stats.json"
    if not sp.exists():
        return {"available": False, "note": "odm_report/stats.json not found; quality statistics unavailable"}
    try:
        s = json.loads(sp.read_text())
    except ValueError:
        return {"available": False, "note": "stats.json unreadable"}
    rs = s.get("reconstruction_statistics") or {}
    ps = s.get("odm_processing_statistics") or s.get("processing_statistics") or {}
    rec = _first_number(rs, "reconstructed_shots_count")
    tot = _first_number(rs, "initial_shots_count")
    return {"available": True,
            "images_reconstructed": int(rec) if rec is not None else None,
            "images_total": int(tot) if tot is not None else None,
            "reprojection_error_px": _first_number(rs, "reprojection_error_pixels", "average_error", "reprojection_error_normalized"),
            "gsd_cm": _first_number(ps, "average_gsd"), "area_m2": _first_number(ps, "area"),
            "raw": {k: s[k] for k in ("reconstruction_statistics", "odm_processing_statistics", "gps_errors", "gcp_errors") if k in s}}


def quality_checks(stats: dict, expected_images: int, georef: bool, dsm_ok: bool, cloud_ok: bool, ortho_ok: bool) -> List[dict]:
    chk = []

    def add(name, status, msg):
        chk.append({"name": name, "status": status, "message": msg})
    add("orthomosaic produced", "pass" if ortho_ok else "fail", "odm_orthophoto.tif present" if ortho_ok else "orthomosaic missing")
    add("point cloud produced", "pass" if cloud_ok else "fail", "georeferenced point cloud present" if cloud_ok else "point cloud missing")
    add("DSM produced", "pass" if dsm_ok else "fail", "dsm.tif present" if dsm_ok else "DSM missing")
    add("georeferenced outputs", "pass" if georef else "warn",
        "orthomosaic carries a CRS" if georef else "outputs carry no CRS; supply GPS-tagged images or GCPs")
    rec, tot = stats.get("images_reconstructed"), stats.get("images_total") or expected_images
    if rec and tot:
        f = rec / tot
        add("images reconstructed", "pass" if f >= 0.9 else "warn", f"{rec}/{tot} images ({f:.0%}) were used in the reconstruction"
            + ("" if f >= 0.9 else "; check overlap and blur for the rest"))
    else:
        add("images reconstructed", "unknown", "ODM did not report how many images were reconstructed")
    err = stats.get("reprojection_error_px")
    if isinstance(err, (int, float)):
        add("reprojection error", "pass" if err <= 1.5 else "warn", f"average reprojection error {err:.2f} px" + ("" if err <= 1.5 else " (high)"))
    return chk


def run_odm(app: AppContext, ctx: JobContext) -> dict:
    p = ctx.params
    pid = p["project_id"]
    proj = app.db.one("SELECT epsg FROM projects WHERE id=?", (pid,))
    n_img = app.db.one("SELECT COUNT(*) n, SUM(lat IS NOT NULL) g FROM images WHERE project_id=?", (pid,))
    if n_img["n"] < 3:
        raise JobFailure(f"photogrammetry needs at least 3 overlapping images (this project has {n_img['n']}).", "insufficient_images")
    gcp = P.get(app, pid, "gcp_list")
    if (n_img["g"] or 0) == 0 and not (gcp and gcp["exists"]) and not p.get("allow_ungeoreferenced"):
        raise JobFailure("None of the images carry GPS positions and no GCP list was provided, so the outputs could not be georeferenced. "
                         "Upload a gcp_list.txt, or re-submit with allow_ungeoreferenced=true to produce a local, non-georeferenced model.",
                         "no_georeference")
    opts = build_options(p.get("preset", "balanced"), p.get("options"), bool(gcp and gcp["exists"]))
    docker = check_docker(app)
    ctx.progress(0.01, "verifying ODM options")
    verify_options(app, docker, opts)
    ctx.progress(0.02, "preparing inputs")
    info = prepare_inputs(app, pid, p.get("exclude_images", []), ctx)
    work = app.storage.path(pid, "odm")
    if gcp and gcp["exists"]:
        shutil.copy2(gcp["path"], work / "gcp_list.txt")
    ctx.log(f"ODM inputs: {info}; options: {' '.join(opts)}")
    name = f"terrascope-odm-{ctx.id[:12]}"
    cmd = [docker, "run", "--rm", "--name", name, "-v", f"{work}:/datasets/code", app.settings.odm_image,
           "--project-path", "/datasets", *opts, *app.settings.odm_extra_args.split(), "code"]
    ctx.log("$ " + " ".join(cmd))
    state = {"stage": None}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    stop = threading.Event()

    def watcher():
        while not stop.wait(2.0):
            if ctx.cancelled():
                subprocess.run([docker, "stop", "-t", "20", name], capture_output=True)
                proc.terminate()
                return
    threading.Thread(target=watcher, daemon=True).start()
    tail: List[str] = []
    try:
        for line in proc.stdout:                               # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            tail = (tail + [line])[-30:]
            ctx.log(line[:1000])
            f = _progress(line, state)
            if f is not None:
                ctx.progress(0.03 + 0.9 * f, f"ODM stage: {state['stage']}")
        rc = proc.wait()
    finally:
        stop.set()
        if proc.poll() is None:
            proc.kill()
    if ctx.cancelled():
        raise JobCancelled()
    if rc != 0:
        raise JobFailure(f"ODM exited with code {rc} (stage: {state['stage']}). Last output: " + " | ".join(tail[-5:])[:900] +
                         " Retry to resume from the last completed stage.", "odm_failed", retryable=False)

    # -------- outputs
    ctx.progress(0.94, "registering outputs")
    ortho = _find(work, "odm_orthophoto/odm_orthophoto.tif")
    cloud = _find(work, "odm_georeferencing/odm_georeferenced_model.las", "odm_georeferencing/odm_georeferenced_model.laz")
    dsm = _find(work, "odm_dem/dsm.tif")
    dtm = _find(work, "odm_dem/dtm.tif")
    ok_ortho, ok_cloud, ok_dsm = ortho is not None, cloud is not None, dsm is not None
    georef = False
    if ortho:
        try:
            gi = gt.read_georef(ortho)
            georef = gi.georeferenced
            if gi.epsg and proj["epsg"] and gi.epsg != proj["epsg"]:
                raise JobFailure(f"ODM produced EPSG:{gi.epsg} outputs but the project CRS is EPSG:{proj['epsg']}. Set the project CRS to "
                                 f"EPSG:{gi.epsg} or pass a gcp_list in the desired CRS.", "crs_mismatch")
            if gi.epsg and not proj["epsg"]:
                app.db.update("projects", {"id": pid}, {"epsg": gi.epsg})
                ctx.log(f"project CRS set to EPSG:{gi.epsg} from ODM output")
        except gt.GeoTiffError as e:
            ctx.log(f"cannot read ODM orthophoto georeferencing: {e}", "warning")
    stats = parse_report(work)
    checks = quality_checks(stats, info["images"], georef, ok_dsm, ok_cloud, ok_ortho)
    missing = [c["message"] for c in checks if c["status"] == "fail"]
    if missing:
        raise JobFailure("ODM finished but outputs are missing: " + "; ".join(missing) + ". It is NOT reported as successful.", "odm_outputs_missing")
    rel = lambda q: str(q.relative_to(app.storage.project_root(pid)))
    P.register(app, pid, "pointcloud", rel(cloud), "pointcloud", {"source": "ODM", "compressed": cloud.suffix == ".laz"}, ctx.id)
    P.register(app, pid, "odm_dsm", rel(dsm), "raster", {"source": "ODM", "note": "Observed surface (canopy/structure tops)."}, ctx.id)
    if dtm:
        P.register(app, pid, "odm_dtm", rel(dtm), "raster", {"source": "ODM", "note": "ODM's own SMRF ground estimate; reference only. "
                                                              "The platform's DTM comes from the terrain pipeline."}, ctx.id)
    app.queue.enqueue("prepare_ortho", pid, {"project_id": pid, "rel_path": rel(ortho)})
    chain = ctx.params.get("terrain_chain")
    if chain:                                    # "Build 3D terrain model": continue straight into terrain reconstruction
        app.queue.enqueue("terrain", pid, {"project_id": pid, "use_classification": True, "params": dict(chain)})
        ctx.log(f"terrain reconstruction queued automatically (ground filter: {(chain.get('ground') or {}).get('engine')})")
    P.register(app, pid, "odm_quality", "odm/odm_report/stats.json" if (work / "odm_report/stats.json").exists() else "odm", "report",
               {"checks": checks, "stats": {k: v for k, v in stats.items() if k != "raw"}}, ctx.id)
    ctx.log("quality checks: " + json.dumps(checks))
    ctx.progress(1.0, "done")
    return {"checks": checks, "stats": {k: v for k, v in stats.items() if k != "raw"}, "inputs": info,
            "note": "Orthomosaic is being prepared for the editor by a follow-up job."}


from ..workers.handlers import HANDLERS  # noqa: E402
HANDLERS["odm"] = run_odm
