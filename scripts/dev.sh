#!/usr/bin/env bash
# Local development without Docker: builds the UI, then runs the API (serving the UI) and one worker.
# Usage: scripts/dev.sh [data_dir]     -> http://localhost:8765
set -euo pipefail
cd "$(dirname "$0")/.."
export TERRASCOPE_DATA_DIR="${1:-$PWD/data}"
export TERRASCOPE_DEV_MODE=1
(cd frontend && npm install --no-audit --no-fund >/dev/null && node build.mjs)
cd backend
trap 'kill 0' EXIT
python3 -m app.workers.runner &
python3 -m uvicorn app.api.main:app_factory --factory --port "${PORT:-8765}"
