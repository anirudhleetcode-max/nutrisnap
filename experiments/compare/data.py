"""Shared data for the model comparison (CNN from scratch, transfer learning, CLIP).

Every model sees exactly the same photos and splits as experiments/run.py:
  train 1,800 / val 900 / test 2,461 (Food-101 30 classes + Indian Food Images 15 classes)
plus the hand-picked Wikimedia Commons photos as the unseen-photo set.

The label space is the 45 training classes (the first food id of each source class in
backend/ml/class_mapping.json). A prediction counts as correct when it is in the photo's accept set,
the same rule run.py uses (e.g. any of the six paneer dishes for indianfoods/paneer).

Pixels for the CNNs are cached once as uint8 256x256 (shortest side -> 256, centre crop), so the
CNN scripts don't decode JPEGs every epoch.
"""
from __future__ import annotations

import hashlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from ml.datasets import PROCESSED  # noqa: E402
from ml.embeddings import load_set  # noqa: E402

DATASETS = ("food101", "indianfoods")
PIX = 256


@dataclass
class Split:
    rows: list[dict]
    y: np.ndarray                 # primary label index in `classes` (-1 if the class isn't one of the 45)
    accept: list[set[int]]        # accepted label indexes (within the 45)
    dataset: np.ndarray
    source_class: list[str]


@dataclass
class Data:
    classes: list[str]            # 45 food ids
    train: Split
    val: Split
    test: Split
    unseen: Split                 # Commons food photos (not-food rows removed)


def _split(rows: list[dict], index: dict[str, int]) -> Split:
    y = np.array([index.get(r["accept"][0], -1) for r in rows])
    acc = [{index[a] for a in r["accept"] if a in index} for r in rows]
    return Split(rows, y, acc, np.array([r["dataset"] for r in rows]),
                 [f"{r['dataset']}/{r['source_class']}" for r in rows])


def load() -> Data:
    rows = [r for name in DATASETS for r in load_set(name)]
    classes = sorted({r["accept"][0] for r in rows if r["split"] == "train"})
    index = {c: i for i, c in enumerate(classes)}
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    commons = [r for r in load_set("commons") if r["source_class"] != "not-food"]
    return Data(classes, _split(by["train"], index), _split(by["val"], index), _split(by["test"], index),
                _split(commons, index))


def to_square(data: bytes, size: int = PIX) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    l, t = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def pixels(split: Split, name: str) -> np.ndarray:
    """(N, 256, 256, 3) uint8, cached in data/processed/pix<size>_<name>_<hash>.npy."""
    h = hashlib.sha256()
    for r in split.rows:
        h.update(hashlib.sha256(r["image"]).digest())
    path = PROCESSED / f"pix{PIX}_{name}_{h.hexdigest()[:12]}.npy"
    if path.exists():
        return np.load(path)
    arr = np.stack([to_square(r["image"]) for r in split.rows])
    np.save(path, arr)
    return arr
