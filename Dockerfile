# Verified by GitHub Actions (.github/workflows/build.yml): the image builds, the backend tests pass inside it, and the app answers
# on /api/health and serves the web UI. The built-in ODM start-up check runs at build time; a real photo run is the 'odm-real' CI job.
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN node build.mjs

# OpenDroneMap (AGPL-3.0, https://github.com/OpenDroneMap/ODM) is taken from its official image, not copied into this repository.
# It is self-contained (its own pixi/conda environment + SuperBuild), so it is copied to the same path (/code) and run natively.
ARG ODM_IMAGE=opendronemap/odm:latest
FROM ${ODM_IMAGE} AS odm

FROM python:3.12-slim-trixie
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# docker CLI: the worker launches OpenDroneMap as a sibling container via the host socket.
# PDAL (ground filtering SMRF/PMF + LAZ) is installed when the distribution packages it; if not, the image still builds and the app
# reports "PDAL not found" (the explicit built-in filter still works). Debian bookworm has no pdal package; trixie does.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io libgl1 libglib2.0-0 \
    && (apt-get install -y --no-install-recommends pdal || echo "WARNING: pdal package not available; building without PDAL") \
    && rm -rf /var/lib/apt/lists/* \
    && (which pdal && pdal --version | head -1 || echo "PDAL: not installed")
COPY --from=odm /code /code
# Fails the build if the copied ODM cannot start on this base image (missing libraries, wrong paths).
RUN cd /code && ./scripts/docker-entrypoint.sh python3 scripts/smoke.py && ./scripts/docker-entrypoint.sh --help | head -3
ENV TERRASCOPE_ODM_HOME=/code
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
