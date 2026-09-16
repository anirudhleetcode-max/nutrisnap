"""Compute (and cache) CLIP image embeddings for the evaluation sets.

Embeddings are cached in data/processed/emb_<set>_<hash>.npz; the hash covers the image bytes and
the vision model file size, so a changed dataset or model invalidates the cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from app.ml.clip_model import ClipEncoder
from app.ml.preprocess import decode_image

from .datasets import MAPPING, PROCESSED

BACKEND = Path(__file__).resolve().parents[1]
TESTSET = BACKEND / "ml" / "testset"


def load_set(name: str) -> list[dict]:
    """Rows with keys: image (bytes), dataset, split, source_class, row_id, accept (food ids)."""
    if name == "commons":
        items = json.loads((TESTSET / "manifest.json").read_text(encoding="utf-8"))["items"]
        rows = []
        for it in items:
            p = TESTSET / "images" / it["file"]
            if p.exists():
                rows.append({"image": p.read_bytes(), "dataset": "commons", "split": "ood",
                             "source_class": it["label"], "row_id": it["file"], "accept": it["accept"]})
        return rows
    t = pq.read_table(PROCESSED / f"{name}.parquet")
    rows = t.to_pylist()
    for r in rows:
        r["accept"] = MAPPING[name][r["source_class"]]
    return rows


def embed_rows(enc: ClipEncoder, rows: list[dict], name: str) -> tuple[np.ndarray, float]:
    h = hashlib.sha256()
    for r in rows:
        h.update(hashlib.sha256(r["image"]).digest())
    h.update(str((enc.model_dir / "vision_model_quantized.onnx").stat().st_size).encode())
    path = PROCESSED / f"emb_{name}_{h.hexdigest()[:12]}.npz"
    if path.exists():
        z = np.load(path)
        return z["emb"], float(z["ms"])
    out, times = [], []
    for i, r in enumerate(rows):
        img = decode_image(r["image"])
        t = time.perf_counter()
        out.append(enc.embed_images([img])[0])
        times.append(time.perf_counter() - t)
        if i % 250 == 0:
            print(f"  {name}: {i}/{len(rows)}")
    emb = np.stack(out).astype(np.float32)
    ms = float(np.median(times) * 1000)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, emb=emb, ms=ms)
    return emb, ms


def thumb(data: bytes, px: int = 96) -> Image.Image:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((px, px))
    return im
