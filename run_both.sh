#!/usr/bin/env bash
set -euo pipefail
# Run frontend dev server and backend (uvicorn) together from repo root.
# Usage: ./run_both.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_CMD="uvicorn main:app --reload --host 0.0.0.0 --port 8000"

cd "$ROOT_DIR"

echo "Starting frontend (Vite) in $FRONTEND_DIR ..."
(cd "$FRONTEND_DIR" && npm run dev) &
FRONT_PID=$!

echo "Starting backend (uvicorn) ..."
$BACKEND_CMD &
BACK_PID=$!

cleanup() {
  echo "Stopping children..."
  kill "$FRONT_PID" "$BACK_PID" 2>/dev/null || true
  wait || true
}

trap cleanup EXIT INT TERM

wait
