# BrooksBuild-# TerraScope

Drone-based terrain reconstruction and vegetation removal for **residential construction planning on wooded lots**.

Upload drone photos (any number), get an orthomosaic, point cloud and surface model, classify what is on the ground, and
estimate the bare earth underneath. Measure, profile and draw footprints on it, and export the results to GIS and CAD.

> **Everything this tool produces about the ground under trees is an estimate, and it says so.**
> A camera sees treetops, not the soil beneath them. TerraScope keeps five things strictly apart and labels them everywhere:
> **land cover** (a visual interpretation of the imagery), the **observed surface** (DSM), **classified ground points**,
> the **estimated bare earth** (DTM) and **insufficient-evidence** areas (the confidence layer).
> *Painting a pixel as "soil" never creates elevation evidence.* All elevations, slopes, volumes and drainage results are a
> **preliminary planning estimate**, not for foundation design, permitting or staking without a licensed surveyor's validation.

Inspired by the workflow of tools like AGTEK Reveal Maps / Clean; this is an independent implementation with its own code and design.

---

## What has and has not been verified

Read this first. It is the honest status of the build.

| Area | Status |
| --- | --- |
| Backend logic: uploads, ingest, flights, classification, terrain, planning, exports, job queue | **Tested.** 223 automated tests pass (`python3 -m unittest discover -s tests -t .` in `backend/`). |
| Web UI (React/TS): dashboard, upload, classification editor, Terrain 2D, Planning, 3D, Exports | **Driven in real Chromium** against the synthetic sample project (screenshots, real mouse/keyboard input, server-side pixel counts checked). No automated browser tests are committed. |
| Photogrammetry (OpenDroneMap) | **Never run.** The sandbox had the Docker CLI but no daemon. The orchestration is tested against a *stub* `docker`; real ODM output has not been seen. ODM options are verified against `--help` before each run, so a renamed option fails fast with a message. |
| PDAL ground filters (SMRF / PMF), LAZ | **Never run.** PDAL was not installed. The pipelines are built and unit-tested up to the subprocess call. The built-in progressive-morphological filter (`builtin_pmf`) *was* exercised end-to-end and is the only engine that has produced real output. |
| Docker image / docker-compose | **Never built.** Written from knowledge of the tools; expect small fixes on first build. |
| PostgreSQL/PostGIS | **Not used.** Persistence is SQLite (WAL). `deploy/postgres_schema.sql` is an unexecuted translation of the schema. |
| Celery / Redis | **Not used.** The job queue is a persistent table in the database with its own worker process (heartbeats, stale-job recovery, retries, cancellation). |
| S3 storage | **Not implemented.** There is a `Storage` interface and a local implementation; `S3Storage` is a documented placeholder. |
| Pretrained land-cover model | **None ships.** Auto-classification uses a colour/texture heuristic, a per-project "learned" model trained from *your* corrections, or an ONNX model you supply. |
| Real drone data | **Never tested.** Everything was exercised on a synthetic wooded lot (`POST /api/dev/sample-project`, marked SYNTHETIC in the UI). Accuracy on real sites is unknown. |
| Coordinate systems | WGS84 / UTM (any zone) and Web Mercator are supported in-process. **NAD83, State Plane and other datums are refused**, never guessed. Use WGS84/UTM or provide an EPSG the app understands. |

### Where this deviates from the requested stack, and why

| Requested | Built | Reason |
| --- | --- | --- |
| FastAPI | **Starlette + Pydantic v2** | FastAPI could not be installed in the offline build sandbox. Routes, validation and JSON responses are equivalent; there is no auto-generated `/docs`. |
| PostgreSQL + PostGIS | **SQLite (WAL)** | No server available. Schema is portable; see `deploy/postgres_schema.sql`. |
| Celery + Redis | **DB-backed queue + worker process** | Avoids two more services; semantics documented in `docs/ARCHITECTURE.md`. |
| Vite | **esbuild** | Vite could not be installed offline. `npm run build` bundles `src/main.tsx`. |
| Tailwind CLI | **Tailwind v4 via its JS `compile()` API** (in `frontend/build.mjs`) | Same reason. Output is ordinary Tailwind CSS. |
| MapLibre GL | **Hand-written Canvas 2D map + WebGL2 3D viewer** | MapLibre could not be installed offline, and the data is project-CRS rasters, not web-mercator tiles. |
| Rasterio / GDAL | **tifffile + NumPy** (GeoTIFF read/write with GeoKeys, in `services/geotiff.py`) | Neither could be installed. GDAL remains a good choice for production GeoTIFF handling and reprojection. |
| `@types/react` | An **ambient shim** (`frontend/src/react-shim.d.ts`) | Not installable offline. Delete the file after `npm install` pulls the real types. |

---

## Quick start (no Docker, development)

Requirements: Python 3.11+ (developed on 3.13), Node 20+.

```bash
git clone <this repo> && cd terrascope
pip install -r backend/requirements.txt
scripts/dev.sh              # builds the UI, starts the API (:8765) and one worker, DEV_MODE on
```

Open <http://localhost:8765>, click **Add synthetic sample project**, and explore: the sample has an orthomosaic, a point cloud and
all downstream products, so every tab works without Docker, PDAL or real photos. Run **Terrain 2D → Rebuild terrain** (engine:
*built-in*) to regenerate the terrain products.

Manual equivalent:

```bash
cd frontend && npm install && node build.mjs          # or: node build.mjs --watch
cd ../backend
export TERRASCOPE_DATA_DIR=$PWD/../data TERRASCOPE_DEV_MODE=1
python3 -m app.workers.runner &                        # worker
python3 -m uvicorn app.api.main:app_factory --factory --port 8765
```

### Tests

```bash
cd backend && python3 -m unittest discover -s tests -t .       # backend (223 tests, ~1 min)
cd frontend && node build.mjs --check                           # TypeScript strict type-check + bundle
```

## Quick start (Docker Compose) — untested

```bash
cp .env.example .env     # set TERRASCOPE_DATA_DIR to an absolute host path
docker compose up --build
```

Serves the UI and API on <http://localhost:8000> with one worker. The data directory is mounted at the **same path** inside the
containers because the worker launches OpenDroneMap as a sibling container through the host Docker socket (`docker run -v <path>:/datasets/code`),
and the daemon resolves that path on the host. Mounting the socket is root-equivalent on the host: do not expose this to untrusted users.

---

## Using it

1. **Dashboard / Upload.** Create a project, optionally enter its EPSG code (otherwise it is suggested from image GPS). Drag in photos
   or whole folders. Uploads are chunked (8 MB), resumable after a browser close, retried, checksummed and de-duplicated; EXIF is parsed;
   thumbnails are generated; files missing GPS are flagged, never discarded. There is **no limit on image count**, only byte-based
   safeguards (free disk, per-file, optional project quota).
   You can also import an existing orthomosaic, LiDAR / DEM, survey points, and a GCP / check-point list (CSV).
   **Classify the photos:** once several photos are ingested, the Upload view offers *Classify drone photos* (all, or one batch): per-photo land-cover
   masks with a thumbnail gallery, overlay viewer and CSV export. These are per-photo labels, not georeferenced, and not elevation evidence (see `docs/CLASSIFICATION.md`).
2. **Photogrammetry** (Dashboard → OpenDroneMap panel; needs Docker). Presets draft/balanced/high; excluded images; refuses ungeoreferenced sets unless you allow it.
3. **Classify.** Paint the orthomosaic into land-cover classes (10 defaults, colours and names editable). Tools: sample, brush, polygon, rectangle,
   lasso, flood fill with tolerance, select-similar (both preview before applying), reclassify, erase-to-unknown. Undo/redo, autosave, side-by-side
   view, legend with live statistics, overlay opacity. **Auto-classify** can seed the mask; automatic labels carry a confidence and a provenance
   tag, never overwrite your manual edits unless you say so, and snapshots are taken first. See `docs/CLASSIFICATION.md`.
4. **Terrain 2D.** Run terrain reconstruction (ground filter → DSM, DTM, confidence, evidence). Inspect any point (observed vs estimated elevation,
   evidence, confidence, source), measure, profile, swipe-compare layers, show steep slopes and drainage. See `docs/TERRAIN.md`.
5. **Terrain 3D.** WebGL terrain meshes (DSM and DTM separately or together, with transparency), draped imagery / classes / confidence,
   point cloud by RGB / class / height, vertical exaggeration, split-screen *observed | estimated*, 3D point-to-point measurement.
   **Edit points:** lasso/box-select points and reclassify them (ground, vegetation, building, noise…) with saved undo/redo. Edits apply to the 3D display
   sample only; they do not change the DSM/DTM or the full cloud (see `docs/TERRAIN.md`). Export them as an edited-sample LAS.
6. **Planning.** Draw a boundary and building footprints; get elevation, slope and confidence under each footprint; propose a flat or sloped grading
   plane and compute cut / fill **only where the evidence supports it** (otherwise it refuses and says why).
7. **Export.** Original images, orthomosaic / classification / DSM / DTM / confidence / evidence GeoTIFFs, LAS (LAZ with PDAL), GeoJSON (project CRS and WGS84),
   profile CSVs, class statistics CSV, **DXF** for CAD, a PDF report, and `metadata.json` with CRS, units, settings and a SHA-256 per file.

### Keyboard

`Alt+1…7` switch tabs · `?` help · **Classify:** `G` pan, `S` sample, `B` brush, `P` polygon, `R` rectangle, `L` lasso, `F` fill, `M` select-similar,
`E` erase-to-unknown, `1…9` class, `[` `]` brush size, `H` hide overlay, `\` peek at imagery, `Enter` apply/close, `Esc` cancel, `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo ·
**Terrain / Planning:** `H` pan, `I` inspect, `D` distance, `A` area, `P` profile, `F` footprint, `B` boundary, `Enter` finish, `Backspace` remove last point, `Esc` cancel ·
**3D:** drag = orbit, right-drag or Shift-drag = pan, wheel = zoom.

---

## Dependencies

| Needed for | Dependency | If missing |
| --- | --- | --- |
| Everything | Python 3.11+, packages in `backend/requirements.txt` | — |
| Web UI build | Node 20+, `npm install` (react, esbuild, tailwindcss, typescript) | Prebuilt `frontend/dist/` can be served as-is |
| Photogrammetry | Docker daemon + `opendronemap/odm` image (several GB) | ODM job fails with install instructions; nothing is fabricated |
| SMRF / PMF ground filter, LAZ | [PDAL](https://pdal.io) on `PATH` (or `TERRASCOPE_PDAL_BIN`) | Job fails with instructions; choose `builtin_pmf` explicitly if you accept a simpler filter |
| ONNX segmentation | `onnxruntime` + `TERRASCOPE_SEGMENTATION_MODEL` + JSON class map | Option is shown as unavailable with the reason |
| 3D view | A browser with WebGL2 | 2D views keep working |

`GET /api/system` reports which of these are present.

## Configuration

Environment variables (prefix `TERRASCOPE_`): `DATA_DIR`, `MAX_FILE_BYTES` (8 GiB), `PROJECT_QUOTA_BYTES` (0 = none), `MIN_FREE_BYTES` (2 GiB),
`MAX_IMAGE_MEGAPIXELS` (400), `MAX_CHUNK_BYTES` (64 MiB), `WORKER_CONCURRENCY` (2), `HEAVY_CONCURRENCY` (1), `STALE_JOB_SECONDS` (90),
`MAX_ATTEMPTS` (3), `PDAL_BIN`, `DOCKER_BIN`, `ODM_IMAGE`, `ODM_ARGS`, `SEGMENTATION_MODEL`, `DEV_MODE`. See `backend/app/config.py`.

## Deployment notes

* **Single host.** Run the API and one or more workers against the same `DATA_DIR` (they coordinate through SQLite WAL and file locks). Put TLS and
  authentication in front (a reverse proxy): **TerraScope has no user accounts or authentication.** Anyone who can reach the API can read and
  delete every project. Do not expose it to the internet as is.
* Project isolation is by directory and by checked project id on every route; there is no multi-tenant access control beyond that.
* Heavy jobs (photogrammetry, terrain, classification) are limited to `HEAVY_CONCURRENCY` at once. ODM is memory-hungry: size the host from ODM's own guidance for your image count and quality preset.
* Back up `DATA_DIR` (it holds the database, originals and all products). Free disk space below `MIN_FREE_BYTES` makes uploads fail with a clear error rather than filling the disk.
* Restart safety: a worker killed mid-job stops heart-beating; after `STALE_JOB_SECONDS` its job is re-queued (or failed once attempts are used up). Uploads resume from the chunks already on disk.
* Linux/macOS only (the tiled mask store uses `fcntl` locks).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "Cannot reach the TerraScope server" | API not running, or you opened the UI from another origin. The UI calls `/api` on the page's own origin (override with `window.TERRASCOPE_API`). |
| Jobs stay *queued* | No worker is running (`python3 -m app.workers.runner`), or the queue is paused (left sidebar → Queue → Resume). |
| ODM job fails immediately: "Docker not available / daemon not running" | Install and start Docker; `docker pull opendronemap/odm`. `GET /api/system` shows what was detected. |
| ODM job fails: "unsupported option" | Your ODM version renamed an option. The message names it; adjust the preset or `TERRASCOPE_ODM_ARGS`. |
| Terrain job fails: "PDAL not found" | Install PDAL, or choose *Built-in progressive morphological filter* (simpler; less proven on steep or dense terrain). |
| Terrain job: "No point cloud yet" | The terrain job needs the photogrammetry point cloud: run OpenDroneMap first. (Imported LiDAR/DEM files become *elevation sources*, not the base cloud.) |
| Upload says "no space" | Free disk is below `MIN_FREE_BYTES`. Free space or lower the threshold. |
| Project CRS "cannot be transformed" / export refuses | The EPSG is not WGS84/UTM/Web Mercator. Reproject the data or set a supported EPSG. |
| 3D view says WebGL2 unavailable | Enable hardware acceleration, or use a browser with WebGL2. |
| Boundary does not draw | The project CRS is unknown or unsupported in the browser; set the EPSG on the project. |
| Cut/fill says *refused* | Intentional: too little measured ground under the polygon, or too much low-confidence area. The message states the reason; survey the area. |
| DSM looks full of holes | The DSM is built from observed points and gaps are **not** filled. The synthetic sample's DSM is sparse (about half the cells); a real ODM run should be denser, but that has not been seen. |

## Repository layout

```
backend/app/api         HTTP routes (Starlette)            backend/app/core      pure algorithms (CRS, DTM, planning, raster store, editing…)
backend/app/services    jobs: ingest, flights, ODM, ortho,  backend/app/workers   worker runner + handler registry
                        classify, terrain, exports, CAD…    backend/tests         223 tests
frontend/src            React + TS UI, canvas map, WebGL    deploy/               Postgres schema (unverified)
docs/                   architecture, classification, terrain   scripts/dev.sh    local dev launcher
```

More: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/CLASSIFICATION.md`](docs/CLASSIFICATION.md) · [`docs/TERRAIN.md`](docs/TERRAIN.md)
