#!/usr/bin/env bash
# Starts the backend (8004) and a production build of the frontend (5176) against a throwaway
# database, runs the Playwright journey, then stops both servers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MONGO_DB="${E2E_MONGO_DB:-nutrisnap_e2e}"
PY="${PYTHON:-python3}"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  "$PY" -c "import pymongo,os; pymongo.MongoClient(os.environ.get('MONGO_URI','mongodb://127.0.0.1:27017')).drop_database('$MONGO_DB')" || true
}
trap cleanup EXIT

wait_for() {
  for _ in $(seq 1 90); do
    curl -fsS "$1" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "timed out waiting for $1" >&2
  return 1
}

cd "$ROOT/backend"
"$PY" -m scripts.seed
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8004 --log-level warning &
PIDS+=($!)

cd "$ROOT/frontend"
npm run build --silent >/dev/null
# run the vite binary directly (not via npx) so $! is the server process itself
./node_modules/.bin/vite preview --host 127.0.0.1 --port 5176 --strictPort >/dev/null 2>&1 &
PIDS+=($!)

wait_for http://127.0.0.1:8004/api/health
wait_for http://127.0.0.1:5176/

cd "$ROOT"
"$PY" -m pytest -q e2e/test_e2e.py "$@"
