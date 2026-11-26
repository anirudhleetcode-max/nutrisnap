"""Dataset statistics for docs/DATASET.md (class balance, image sizes, brightness, duplicates).

    cd backend
    python ../experiments/compare/dataset_stats.py

Writes experiments/results/<date>_dataset/dataset_stats.json and class_balance.png.
Image sizes are measured on the cached copies (longest side capped at 512 px by ml/datasets.py);
the original resolution is not kept.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from data import ROOT, load


def main() -> None:
    out = ROOT / "experiments" / "results" / f"{datetime.now(timezone.utc):%Y-%m-%d}_dataset"
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    res: dict = {"classes": len(d.classes)}
    counts = defaultdict(Counter)
    sizes, bright, aspect = defaultdict(list), defaultdict(list), defaultdict(list)
    hashes = defaultdict(set)
    for s in ("train", "val", "test", "unseen"):
        sp = getattr(d, s)
        for r, c in zip(sp.rows, sp.source_class):
            counts[s][c] += 1
            im = Image.open(io.BytesIO(r["image"]))
            w, h = im.size
            ds = r["dataset"]
            sizes[ds].append(max(w, h))
            aspect[ds].append(w / h)
            g = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
            bright[ds].append(g.mean())
            # perceptual-ish hash: 8x8 grey thumbnail above/below its mean
            t = np.asarray(im.convert("L").resize((8, 8)), dtype=np.float32)
            hashes[s].add(hashlib.sha1((t > t.mean()).tobytes()).hexdigest())
    res["per_split"] = {s: {"photos": sum(c.values()), "classes": len(c), "min_per_class": min(c.values()),
                            "max_per_class": max(c.values()), "imbalance_ratio": round(max(c.values()) / min(c.values()), 2)}
                        for s, c in counts.items()}
    res["per_class"] = {c: {s: counts[s].get(c, 0) for s in ("train", "val", "test")} for c in sorted(counts["test"])}
    res["unseen_classes"] = dict(counts["unseen"])
    res["image_stats_by_dataset"] = {ds: {"n": len(v), "longest_side_median": int(np.median(v)),
                                          "longest_side_min": int(np.min(v)),
                                          "aspect_ratio_median": round(float(np.median(aspect[ds])), 3),
                                          "share_not_square": round(float(np.mean(np.abs(np.log(aspect[ds])) > 0.05)), 3),
                                          "mean_brightness_0_255": round(float(np.mean(bright[ds])), 1),
                                          "share_dark_below_60": round(float(np.mean(np.array(bright[ds]) < 60)), 3)}
                                     for ds, v in sizes.items()}
    res["near_duplicate_check"] = {
        "method": "8x8 average-hash; identical hashes across splits are possible duplicates (or very similar plates)",
        "train_test_shared_hashes": len(hashes["train"] & hashes["test"]),
        "train_val_shared_hashes": len(hashes["train"] & hashes["val"]),
        "train_unseen_shared_hashes": len(hashes["train"] & hashes["unseen"])}
    (out / "dataset_stats.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cls = sorted(res["per_class"], key=lambda c: (c.split("/")[0], -res["per_class"][c]["test"]))
    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(cls))
    for s, col in (("train", "#4c78a8"), ("val", "#f58518"), ("test", "#54a24b")):
        v = np.array([res["per_class"][c][s] for c in cls])
        ax.bar(range(len(cls)), v, bottom=bottom, color=col, label=s)
        bottom += v
    ax.set_xticks(range(len(cls)), [c.split("/")[1][:14] for c in cls], rotation=75, fontsize=7)
    ax.axvline(sum(c.startswith("food101") for c in cls) - 0.5, color="#555", ls=":")
    ax.set(ylabel="photos", title="Photos per class (left: Food-101, right: Indian Food Images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=110)
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
