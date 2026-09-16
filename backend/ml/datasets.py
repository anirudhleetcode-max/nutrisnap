"""Download and sample the public evaluation datasets.

    python -m ml.datasets            # prepares data/processed/*.parquet (about 10-15 minutes)

Sources (see data/README.md for licences):
  * Food-101 (Bossard et al., 2014) via https://huggingface.co/datasets/ethz/food101 - 101 classes.
    Its shards are sorted by class, so instead of downloading 5 GB we read the parquet footer,
    use the per-row-group min/max label statistics, and fetch only the row groups that contain the
    classes we map (HTTP range requests).
      train split -> our train (40/class) + val (20/class), disjoint rows
      validation split (the official Food-101 test set) -> our test (50/class)
  * Indian Food Images (https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset),
    15 classes; downloaded whole (690 MB).
      train split -> our train (40/class) + val (20/class); test split -> our test (all 961 images)

Only classes listed in ml/class_mapping.json are kept. Sampling uses a fixed seed. Output rows:
image bytes (re-encoded as JPEG, longest side <= 512 px to keep the cache small), dataset, source
class, split, source row id.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MAPPING = json.loads((Path(__file__).parent / "class_mapping.json").read_text(encoding="utf-8"))
HF = "https://huggingface.co/datasets/"
UA = {"User-Agent": "NutriSnap/1.0 (student project; evaluation)"}
SEED = 42
PER_CLASS = {"train": 40, "val": 20, "test": 50}

FOOD101_SHARDS = {
    "train": [f"ethz/food101/resolve/main/data/train-{i:05d}-of-00008.parquet" for i in range(8)],
    "validation": [f"ethz/food101/resolve/main/data/validation-{i:05d}-of-00003.parquet" for i in range(3)],
}
INDIAN_FILES = {
    "train": ["bharat-raghunathan/indian-foods-dataset/resolve/main/data/train-00000-of-00002-5de00600cb70f0ee.parquet",
              "bharat-raghunathan/indian-foods-dataset/resolve/main/data/train-00001-of-00002-7cca688335b50fc2.parquet"],
    "test": ["bharat-raghunathan/indian-foods-dataset/resolve/main/data/test-00000-of-00001-75236f054535aed8.parquet"],
}


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable remote file over HTTP Range requests, with a small block cache."""

    BLOCK = 4 * 1024 * 1024

    def __init__(self, url: str):
        self.s = requests.Session()
        self.s.headers.update(UA)
        r = self.s.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.url, self.size, self.pos, self.cache = r.url, int(r.headers["Content-Length"]), 0, {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            if len(self.cache) > 32:
                self.cache.pop(next(iter(self.cache)))
            a = i * self.BLOCK
            b = min(self.size, a + self.BLOCK) - 1
            for attempt in range(4):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    r.raise_for_status()
                    self.cache[i] = r.content
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def label_names(pf: pq.ParquetFile) -> list[str]:
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def label_leaf(pf: pq.ParquetFile) -> int:
    """Index of the label column among parquet leaf columns (image is a struct with 2 leaves)."""
    rg = pf.metadata.row_group(0)
    return next(i for i in range(rg.num_columns) if rg.column(i).path_in_schema == "label")


def shrink(data: bytes) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((512, 512))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def rows_from(pf: pq.ParquetFile, wanted: set[int], groups: list[int] | None = None):
    """Yield (row_id, label, image_bytes) for rows whose label is wanted."""
    groups = range(pf.num_row_groups) if groups is None else groups
    base = [0]
    for g in range(pf.num_row_groups):
        base.append(base[-1] + pf.metadata.row_group(g).num_rows)
    for g in groups:
        t = pf.read_row_group(g, columns=["image", "label"])
        for k, (img, lab) in enumerate(zip(t.column("image").to_pylist(), t.column("label").to_pylist())):
            if lab in wanted:
                yield base[g] + k, lab, img["bytes"]


def sample_food101(split: str, quota: dict[int, int], rng: random.Random) -> dict[int, list]:
    """Collect up to quota[label] rows per label, reading only row groups that can contain them."""
    got: dict[int, list] = defaultdict(list)
    shards = FOOD101_SHARDS[split][:]
    rng.shuffle(shards)
    for shard in shards:
        need = {l for l, q in quota.items() if len(got[l]) < q}
        if not need:
            break
        pf = pq.ParquetFile(HttpRangeFile(HF + shard))
        li = label_leaf(pf)
        groups = []
        for g in range(pf.num_row_groups):
            st = pf.metadata.row_group(g).column(li).statistics
            if st is None or any(st.min <= l <= st.max for l in need):
                groups.append(g)
        rng.shuffle(groups)
        print(f"  {shard.split('/')[-1]}: {len(groups)}/{pf.num_row_groups} row groups may hold needed classes")
        for g in groups:
            need = {l for l, q in quota.items() if len(got[l]) < q}
            if not need:
                break
            st = pf.metadata.row_group(g).column(li).statistics
            if st is not None and not any(st.min <= l <= st.max for l in need):
                continue
            for rid, lab, data in rows_from(pf, need, [g]):
                if len(got[lab]) < quota[lab]:
                    got[lab].append((f"{shard.split('/')[-1]}#{rid}", data))
    return got


def local_file(rel: str) -> Path:
    dest = RAW / Path(rel).name
    if not dest.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print("downloading", rel)
        with requests.get(HF + rel, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(rows: list[dict], name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def prepare_food101() -> None:
    rng = random.Random(SEED)
    names = None
    classes = MAPPING["food101"]
    # footer of any shard carries the label names
    pf = pq.ParquetFile(HttpRangeFile(HF + FOOD101_SHARDS["train"][0]))
    names = label_names(pf)
    ids = {names.index(c): c for c in classes}
    print("Food-101 train pool")
    pool = sample_food101("train", {i: PER_CLASS["train"] + PER_CLASS["val"] for i in ids}, rng)
    rows = []
    for lab, items in pool.items():
        rng.shuffle(items)
        for j, (rid, data) in enumerate(items):
            split = "train" if j < PER_CLASS["train"] else "val"
            rows.append({"dataset": "food101", "split": split, "source_class": ids[lab], "row_id": rid, "image": shrink(data)})
    print("Food-101 test (official validation split)")
    test = sample_food101("validation", {i: PER_CLASS["test"] for i in ids}, rng)
    for lab, items in test.items():
        for rid, data in items:
            rows.append({"dataset": "food101", "split": "test", "source_class": ids[lab], "row_id": rid, "image": shrink(data)})
    write(rows, "food101")


def prepare_indian() -> None:
    rng = random.Random(SEED)
    classes = MAPPING["indianfoods"]
    rows = []
    pool: dict[str, list] = defaultdict(list)
    for rel in INDIAN_FILES["train"]:
        pf = pq.ParquetFile(local_file(rel))
        names = label_names(pf)
        for rid, lab, data in rows_from(pf, set(range(len(names)))):
            if names[lab] in classes:
                pool[names[lab]].append((f"{Path(rel).name}#{rid}", data))
    for cls, items in sorted(pool.items()):
        rng.shuffle(items)
        n = PER_CLASS["train"] + PER_CLASS["val"]
        for j, (rid, data) in enumerate(items[:n]):
            rows.append({"dataset": "indianfoods", "split": "train" if j < PER_CLASS["train"] else "val",
                         "source_class": cls, "row_id": rid, "image": shrink(data)})
    for rel in INDIAN_FILES["test"]:
        pf = pq.ParquetFile(local_file(rel))
        names = label_names(pf)
        for rid, lab, data in rows_from(pf, set(range(len(names)))):
            if names[lab] in classes:
                rows.append({"dataset": "indianfoods", "split": "test", "source_class": names[lab],
                             "row_id": f"{Path(rel).name}#{rid}", "image": shrink(data)})
    write(rows, "indianfoods")


def manifest() -> dict:
    out = {}
    for name in ("food101", "indianfoods"):
        p = PROCESSED / f"{name}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=["split", "source_class", "row_id"])
        splits: dict[str, int] = defaultdict(int)
        for s in t.column("split").to_pylist():
            splits[s] += 1
        ids = sorted(t.column("row_id").to_pylist())
        out[name] = {"rows": t.num_rows, "splits": dict(splits),
                     "classes": len(set(t.column("source_class").to_pylist())),
                     "row_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
                     "file_sha256": sha256(p)[:16]}
    (PROCESSED / "manifest.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["indianfoods", "food101"]
    if "indianfoods" in which:
        prepare_indian()
    if "food101" in which:
        prepare_food101()
    print(json.dumps(manifest(), indent=1))
