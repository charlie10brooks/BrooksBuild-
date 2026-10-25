-- PostgreSQL 15+ / PostGIS 3 translation of backend/app/db.py (SQLite).
--
-- !! UNVERIFIED !!  No PostgreSQL server was available while this was written, so this file has never been executed and the
-- application does not use it: TerraScope runs on SQLite (WAL) today. It documents the intended migration target.
-- A migration also needs a Database adapter (the code uses sqlite3 placeholders, BEGIN IMMEDIATE, INSERT OR IGNORE and
-- last_insert_rowid-style behaviour), and the job claim query should become `... FOR UPDATE SKIP LOCKED`.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,                       -- 32 hex chars (uuid4().hex)
    name TEXT NOT NULL, address TEXT,
    lat DOUBLE PRECISION, lon DOUBLE PRECISION,
    epsg INTEGER, vertical_epsg INTEGER,
    horizontal_units TEXT NOT NULL DEFAULT 'metre', vertical_units TEXT NOT NULL DEFAULT 'metre',
    unit_system TEXT NOT NULL DEFAULT 'metric' CHECK (unit_system IN ('metric','imperial')),
    boundary_geojson JSONB,                    -- lon/lat OR project CRS (the API accepts both); could become geometry(MultiPolygon, <epsg>) per project
    classes_json JSONB, settings_json JSONB NOT NULL DEFAULT '{}',
    is_sample BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE batches (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label TEXT, created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE images (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    batch_id TEXT, filename TEXT NOT NULL, sha256 TEXT NOT NULL, size BIGINT NOT NULL,
    rel_path TEXT NOT NULL, thumb_path TEXT, width INTEGER, height INTEGER,
    lat DOUBLE PRECISION, lon DOUBLE PRECISION, alt_msl DOUBLE PRECISION, rel_alt DOUBLE PRECISION,
    captured_at TEXT, make TEXT, model TEXT,
    flight_group INTEGER NOT NULL DEFAULT 0, georef_status TEXT NOT NULL DEFAULT 'missing',
    issues_json JSONB NOT NULL DEFAULT '[]', meta_json JSONB NOT NULL DEFAULT '{}', overlap_json JSONB,
    created_at DOUBLE PRECISION NOT NULL,
    geom geometry(Point, 4326) GENERATED ALWAYS AS
        (CASE WHEN lat IS NOT NULL AND lon IS NOT NULL THEN ST_SetSRID(ST_MakePoint(lon, lat), 4326) END) STORED,
    UNIQUE (project_id, sha256)                -- checksum de-duplication
);
CREATE INDEX idx_images_project ON images(project_id, flight_group);
CREATE INDEX idx_images_geom ON images USING GIST (geom);

CREATE TABLE upload_files (
    upload_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    batch_id TEXT, filename TEXT NOT NULL, size BIGINT NOT NULL, status TEXT NOT NULL, error TEXT,
    image_id TEXT, duplicate_of TEXT, client_key TEXT, purpose TEXT NOT NULL DEFAULT 'image',
    created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX idx_upfiles_project ON upload_files(project_id, status);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id) ON DELETE CASCADE, kind TEXT NOT NULL,
    status TEXT NOT NULL, params_json JSONB NOT NULL DEFAULT '{}', priority INTEGER NOT NULL DEFAULT 0,
    progress DOUBLE PRECISION NOT NULL DEFAULT 0, message TEXT, result_json JSONB, error TEXT, error_code TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, worker_id TEXT, heartbeat_at DOUBLE PRECISION,
    run_after DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL, started_at DOUBLE PRECISION, finished_at DOUBLE PRECISION
);
CREATE INDEX idx_jobs_queue ON jobs(status, priority, created_at);
CREATE INDEX idx_jobs_project ON jobs(project_id, created_at);

CREATE TABLE job_logs (
    id BIGSERIAL PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts DOUBLE PRECISION NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL
);
CREATE INDEX idx_logs_job ON job_logs(job_id, id);

CREATE TABLE products (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, name TEXT NOT NULL,
    rel_path TEXT NOT NULL, kind TEXT NOT NULL, meta_json JSONB NOT NULL DEFAULT '{}', job_id TEXT,
    created_at DOUBLE PRECISION NOT NULL, PRIMARY KEY (project_id, name)
);

CREATE TABLE control_points (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('gcp','check')), label TEXT NOT NULL,
    x DOUBLE PRECISION NOT NULL, y DOUBLE PRECISION NOT NULL, z DOUBLE PRECISION,   -- project CRS
    residual_json JSONB, created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE elevation_sources (
    id INTEGER NOT NULL, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL, kind TEXT NOT NULL, rel_path TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 20,
    z_offset DOUBLE PRECISION NOT NULL DEFAULT 0, trust TEXT NOT NULL DEFAULT 'moderate' CHECK (trust IN ('high','moderate','low')),
    enabled BOOLEAN NOT NULL DEFAULT TRUE, meta_json JSONB NOT NULL DEFAULT '{}', created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (project_id, id)
);

CREATE TABLE features (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('boundary','footprint','profile','measurement','note')), name TEXT,
    geometry_json JSONB NOT NULL,              -- GeoJSON in the PROJECT CRS. To use PostGIS operators add a geometry column
                                               -- with SRID = projects.epsg (per-project SRID needs a typmod-less `geometry` column + CHECK).
    props_json JSONB NOT NULL DEFAULT '{}',
    created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX idx_features_project ON features(project_id, kind);

INSERT INTO meta(key, value) VALUES ('schema_version', '1');
