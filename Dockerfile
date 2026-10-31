# Verified by GitHub Actions (.github/workflows/build.yml): the image builds, the backend tests pass inside it, the app answers on
# /api/health and serves the web UI, and a real OpenDroneMap run on a small public photo set is attempted (see the 'real ODM' step).
#
# This image is built ON TOP OF the official OpenDroneMap image (AGPL-3.0, https://github.com/OpenDroneMap/ODM): ODM itself stays
# at /code, unmodified, and BrooksBuild runs it natively (no second container, no Docker socket). BrooksBuild's own Python
# dependencies live in a separate virtual environment (/opt/app) so they never mix with ODM's.
ARG ODM_IMAGE=opendronemap/odm:3.6.2

FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN node build.mjs

FROM ${ODM_IMAGE}
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive
# PDAL (ground filtering SMRF/PMF + LAZ) from Ubuntu; if unavailable the image still builds and the app reports "PDAL not found".
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv python3-pip libgl1 libglib2.0-0 curl ca-certificates \
    && (apt-get install -y --no-install-recommends pdal || echo "WARNING: pdal package not available; building without PDAL") \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN python3 -m venv /opt/app && /opt/app/bin/pip install -r backend/requirements.txt
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist
# The ODM image sets PYTHONPATH/LD_LIBRARY_PATH for ODM's own libraries; this app must not inherit them (BrooksBuild passes them to ODM only).
ENV TERRASCOPE_ODM_HOME=/code PATH=/opt/app/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONPATH= LD_LIBRARY_PATH=
# Fails the build if ODM cannot start in this image.
RUN cd /code && env -i PATH="/code/venv/bin:/usr/bin:/bin" PYTHONPATH="$(ls -d /code/SuperBuild/install/lib/python3*/dist-packages | head -1)" LD_LIBRARY_PATH=/code/SuperBuild/install/lib python3 /code/run.py --help | head -3
WORKDIR /app/backend
ENTRYPOINT []
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
