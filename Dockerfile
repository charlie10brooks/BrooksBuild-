# BUILD NOT VERIFIED: the sandbox that wrote this could not reach Docker Hub (base images are blocked), so no image was built here.
# Treat it as a starting point and expect to fix small things on first build. See README "What has and has not been verified".
FROM node:20-bookworm-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN node build.mjs

FROM python:3.12-slim-trixie
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# docker CLI: the worker launches OpenDroneMap as a sibling container via the host socket.
# PDAL (ground filtering SMRF/PMF + LAZ) is installed when the distribution packages it; if not, the image still builds and the app
# reports "PDAL not found" (the explicit built-in filter still works). Debian bookworm has no pdal package; trixie does.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io libgl1 libglib2.0-0 \
    && (apt-get install -y --no-install-recommends pdal || echo "WARNING: pdal package not available; building without PDAL") \
    && rm -rf /var/lib/apt/lists/* \
    && (which pdal && pdal --version | head -1 || echo "PDAL: not installed")
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend/ backend/
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
