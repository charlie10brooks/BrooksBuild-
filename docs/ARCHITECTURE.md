# Architecture

## Shape of the system

```
browser (React + TS, canvas 2D map, WebGL2 3D)
   │  JSON / binary over HTTP  (same origin, /api)
   ▼
API process (Starlette + uvicorn)            worker process(es)  python -m app.workers.runner
   routes in app/api/*  ── enqueue ──►  SQLite (WAL)  ◄── claim / heartbeat / log ──  handlers in app/services/*
   │                                  jobs, job_logs, projects, images, products, features, …
   └──────────────── files ────────────  DATA_DIR/projects/<project_id>/…  ────────────────┘
                                         external tools: Docker → OpenDroneMap, PDAL (subprocess)
```

* **No request does heavy work.** Uploads only write chunks; ingest (checksum, EXIF, thumbnail), flight analysis, ODM, orthomosaic
  pyramid building, classification, terrain, point-cloud sampling and exports are all jobs.
* **API and workers share nothing but the database and the data directory**, so they can be restarted independently.
* Dependencies run *down*: `api → services → core`. `core/` is pure NumPy/SciPy algorithms with no I/O framework (CRS maths, flight geometry,
  tiled raster store, edit operations, DTM interpolation, terrain analysis, planning, vectorisation).

## Job system (`app/jobs.py`)

A persistent queue in the `jobs` table.

* States: `queued → running → succeeded | failed | cancelled` (`cancelling` while a cooperative cancel is pending).
* A job is `succeeded` **only** if its handler returned normally; any exception stores the reason and `failed`.
* Workers heartbeat every `HEARTBEAT_SECONDS`. A job whose worker stopped for `STALE_JOB_SECONDS` is re-queued (or failed once `MAX_ATTEMPTS` is used up), so a killed worker never leaves a job "running" forever and never reports it done.
* Heavy kinds (`odm`, `terrain`, `classify`, `classify_images`, `train_classifier`, `export`…) share `HEAVY_CONCURRENCY`; light kinds (`ingest_image`) use the rest of `WORKER_CONCURRENCY`.
* Dedupe keys prevent duplicates (e.g. one terrain job per project at a time; 409 otherwise). Retry and cancel are API calls; the queue can be paused.
* Every job has structured logs (`job_logs`), readable from the UI.

Job kinds: `ingest_image`, `analyze_flights`, `import_asset`, `odm`, `prepare_ortho`, `classify`, `train_classifier`, `terrain`, `viewer_cloud`, `export`.

## Data layout

```
DATA_DIR/terrascope.sqlite3            database (WAL)
DATA_DIR/uploads/…                     chunked uploads in flight (resumable across restarts)
DATA_DIR/projects/<id>/                one directory per project; every path goes through Storage.path(), which rejects traversal
    images/ thumbs/ rejected/          originals (byte-identical), thumbnails, files that failed validation (kept, not deleted)
    incoming/ source/                  imported assets (orthomosaic, LiDAR, DEM, survey points)
    odm/                               ODM working directory
    ortho/pyramid/                     256 px tile pyramid for the orthomosaic
    classification/                    tiled mask store (+ models/, snapshots/)
    terrain/                           intermediate terrain files
    products/                          orthophoto, dsm, dtm, ndsm, confidence, evidence, source (.tif), pointcloud(.las), classified cloud
    exports/<export_id>/               files, manifest.json, metadata.json, README.txt, bundle .zip
```

## Storage

`Storage` is a small interface (`project_root`, `path`, `usage`, `delete_project`) with `LocalStorage`. Native tools (PDAL, ODM, tifffile) need real paths, so
workers always work on local files. `S3Storage` is a placeholder and **not implemented**.

## Uploads (`core/uploadstore.py`, `services/ingest.py`)

create upload → `PUT` chunks in any order, retried freely (8 MB from the browser, 3 in parallel) → `complete` → an `ingest_image` job assembles the file,
verifies SHA-256 (and the client's, if sent), de-duplicates on `(project, sha256)`, validates it is an image and parses EXIF/GPS, writes a thumbnail, and
records the outcome in `upload_files`. A browser that reconnects asks for the chunk indices already received (`client_key` + `purpose`). Rejected files are
moved to `rejected/` with a reason. There is no cap on image count. Guards: free disk (`MIN_FREE_BYTES`), per-file size, optional project quota, image megapixels.

## Coordinates (`core/crs.py`, `frontend/src/util.ts`)

Products live in the **project CRS** (EPSG on the project). In-process support: WGS84 geographic, Web Mercator, and WGS84 / UTM zones (Krüger series; checked against
the meridian-arc integral in tests). The client mirrors the transform so boundaries and GPS can be drawn without a round trip. Anything else (NAD83, State Plane, …) is
**refused with a message**, not approximated. Horizontal and vertical units are stored separately (metre / international foot / US survey foot) and shown per `unit_system`.
Boundaries may arrive as lon/lat or project coordinates; exports always convert to one stated CRS.

## Classification mask (`core/rasterstore.py`, `core/editing.py`)

Tiled, sparse, persistent, with undo/redo patches and an incremental class histogram; edits are serialised by a file lock and journaled for crash recovery. See `CLASSIFICATION.md`.

## HTTP API

All JSON except tiles/images/binary feeds. Errors are `{"error": {"code", "message", "details"}}`. Main groups (see `app/api/*.py` for the full list):

| Group | Routes |
| --- | --- |
| System | `GET /health`, `/system` (dependencies), `/storage`, `/queue` (+ `pause`, `resume`), `POST /dev/sample-project` (dev mode only) |
| Projects | `/projects`, `/projects/{id}` (GET/PATCH/DELETE with name confirmation), `/batches`, `/flights`, `/footprints`, `/images…`, `/control-points` |
| Uploads | `/uploads` (create/list/summary), `/uploads/{id}/chunks/{n}`, `/uploads/{id}/complete`, abort |
| Jobs | `/jobs`, `/projects/{id}/jobs`, `/jobs/{id}` (+ `logs`, `cancel`, `retry`) |
| Photogrammetry | `GET/POST /projects/{id}/photogrammetry` |
| Imagery & classes | `/ortho`, `/ortho/tiles/{z}/{x}/{y}`, `/classification` (+ `tiles`, `overview`, `edit`, `undo`, `redo`, `select-preview`, `sample`, `stats`, `auto`, `train`, `snapshots…`) |
| Photo classification | `/image-classification` (GET state, POST start, DELETE clear), `/image-classification/items`, `/images/{id}/classification` (+ `overlay.png`) |
| Terrain | `/terrain` (+ `run`, `render/{layer}`, `heightmap/{layer}`, `stats/{layer}`, `inspect`, `sample`), `/elevation-sources`, `/products`, `/pointcloud` (+ `data`, `build`, `edit`, `edit/undo`, `edit/redo`, `edit/reset`) |
| Planning | `/planning/profile`, `/footprint`, `/cut-fill`, `/steep`, `/drainage`, `/features` |
| Exports | `/exports/catalog`, `/exports` (create/list), `/exports/{id}` (GET/DELETE), `/exports/{id}/files/{name}` (manifest-whitelisted) |

There is **no authentication** (see README → Deployment).

## Security and safeguards

* Project isolation: project ids are validated (32 hex); every file path is resolved under the project directory; export downloads only serve names listed in that export's manifest.
* Upload safety: content is validated as an image (not trusted by extension), decompression-bomb limit via `MAX_IMAGE_MEGAPIXELS`, filenames sanitised, size/disk guards.
* Subprocesses (ODM, PDAL) are started with argument lists (no shell), monitored, and terminated on cancel.
* Destructive actions need confirmation (project delete requires the project name).
* Resource limits: heavy-job concurrency, decimated point-cloud feed to the browser, heightmap and render size caps, drainage analysis refuses very large rasters.

## Frontend (`frontend/src`)

React 19 + TypeScript (strict), bundled by esbuild, Tailwind v4 compiled in `build.mjs`. State: a tiny external store (`store.ts`) + per-view hooks.
`MapCanvas` is a pan/zoom canvas in **map coordinates**; imagery comes from the tile pyramid (`tiles.ts`), the mask from separately-served mask tiles
colourised in the browser, and elevation layers from server-rendered PNG windows. `gl3d.ts` is a small WebGL2 engine (orbit camera, heightfield meshes,
point clouds, draped lines, CPU ray-picking). Layout: left sidebar (projects, batches, layers, jobs), workspace, right sidebar (tool / terrain / settings), status bar, header toolbar.
