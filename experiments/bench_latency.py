"""Measure /api/recognize latency in-process (FastAPI TestClient, real model, throwaway DB).

    cd backend && python ../experiments/bench_latency.py
Writes experiments/results/<date>_latency/latency.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["MONGO_DB"] = f"bench_{uuid.uuid4().hex[:8]}"

import numpy as np  # noqa: E402
import pymongo  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

N = 20
IMAGES = [ROOT / "backend/tests/fixtures/pizza.jpg", ROOT / "backend/tests/fixtures/dosa.jpg"]


def main() -> None:
    out = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requests_per_setting": N,
           "cpu_threads": get_settings().onnx_threads, "note": "wall-clock per request incl. upload parsing, decode, "
           "inference, Mongo writes; shared 2-CPU build machine", "results": {}}
    with TestClient(app) as c:
        tok = c.post("/api/auth/register", json={"name": "b", "email": "b@example.com", "password": "secret123"}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        for multi in (False, True):
            times, server = [], []
            for i in range(N + 2):
                data = IMAGES[i % 2].read_bytes()
                t = time.perf_counter()
                r = c.post("/api/recognize", params={"multi": multi}, files={"image": ("x.jpg", data, "image/jpeg")}, headers=h)
                dt = (time.perf_counter() - t) * 1000
                assert r.status_code == 200, r.text
                if i >= 2:  # warm-up
                    times.append(dt)
                    server.append(r.json()["timing_ms"])
            out["results"]["with_multi_item_crops" if multi else "whole_image_only"] = {
                "median_ms": round(float(np.median(times)), 1), "p90_ms": round(float(np.percentile(times, 90)), 1),
                "server_inference_median_ms": round(float(np.median(server)), 1)}
    pymongo.MongoClient(get_settings().mongo_uri).drop_database(os.environ["MONGO_DB"])
    d = ROOT / "experiments" / "results" / f"{datetime.now(timezone.utc):%Y-%m-%d}_latency"
    d.mkdir(parents=True, exist_ok=True)
    (d / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
