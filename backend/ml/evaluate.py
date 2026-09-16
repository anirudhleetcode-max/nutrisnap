"""Evaluate food recognition on the Wikimedia Commons test set.

    python -m ml.fetch_testset     # once
    python -m ml.evaluate          # writes ml/metrics.json

Configurations compared (same images, same food list of ~190 classes):
  single_template   - one prompt per food: "a photo of {name}, a type of food."
  ensemble          - 7 phrasings per food (plain + descriptive names), embeddings averaged
Also reported: the not-food guard, a simulated personalisation run, and latency.

Scoring: some test classes are broader than our database (e.g. a Commons "Biryani" photo may be
chicken, mutton or veg biryani), so each image carries an `accept` list; a prediction is correct
if it is in that list.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.clip import ClipRecognizer, labels_from_foods, load_image
from app.services.personalize import blend, knn_probs

BASE = Path(__file__).resolve().parents[1]
TESTSET = BASE / "ml" / "testset"
OUT = BASE / "ml" / "metrics.json"
GUARD_THRESHOLD = 0.5


def main() -> None:
    foods = json.loads((BASE / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    ids = [f["id"] for f in foods]
    rec = ClipRecognizer(BASE / "models", labels_from_foods(foods), BASE / "models" / "cache")
    matrices = {
        "single_template": rec.build_label_matrix(ensemble=False),
        "ensemble": rec.label_matrix,
    }
    items = json.loads((TESTSET / "manifest.json").read_text(encoding="utf-8"))["items"]
    items = [it for it in items if (TESTSET / "images" / it["file"]).exists()]

    embs, times = [], []
    for it in items:
        img = load_image((TESTSET / "images" / it["file"]).read_bytes())
        t = time.perf_counter()
        embs.append(rec.embed_images([img])[0])
        times.append(time.perf_counter() - t)
    embs = np.stack(embs)

    food_mask = np.array([it["label"] != "not-food" for it in items])
    food_items = [it for it in items if it["label"] != "not-food"]
    food_embs = embs[food_mask]
    accept = [{ids.index(a) for a in it["accept"]} for it in food_items]

    def score(probs: np.ndarray) -> dict:
        order = np.argsort(-probs, axis=1)
        top1 = [order[i, 0] in accept[i] for i in range(len(accept))]
        top5 = [bool(set(order[i, :5]) & accept[i]) for i in range(len(accept))]
        return {"top1": round(float(np.mean(top1)), 3), "top5": round(float(np.mean(top5)), 3),
                "_top1": top1, "_order": order}

    results = {}
    for name, m in matrices.items():
        results[name] = score(rec.food_probs(food_embs, m))

    # per-class breakdown for the ensemble
    per_class: dict[str, list[bool]] = defaultdict(list)
    for it, ok in zip(food_items, results["ensemble"]["_top1"]):
        per_class[it["label"]].append(ok)
    confusions = []
    for it, ok, order in zip(food_items, results["ensemble"]["_top1"], results["ensemble"]["_order"]):
        if not ok:
            confusions.append({"file": it["file"], "predicted": ids[order[0]]})

    # not-food guard
    scores = np.array([rec.food_mass(e) for e in embs])
    nonfood = ~food_mask
    guard = {
        "threshold": GUARD_THRESHOLD,
        "non_food_images": int(nonfood.sum()),
        "non_food_flagged": round(float(np.mean(scores[nonfood] < GUARD_THRESHOLD)), 3),
        "food_images_wrongly_flagged": round(float(np.mean(scores[food_mask] < GUARD_THRESHOLD)), 3),
    }

    # simulated personalisation (leave-one-out): the "user" has corrected every *other* test
    # photo that zero-shot got wrong. Optimistic by construction - same small dataset, and
    # Commons categories sometimes hold several photos of the same plate.
    zs = rec.food_probs(food_embs)
    wrong = ~np.array(results["ensemble"]["_top1"])
    true_idx = np.array([ids.index(it["accept"][0]) for it in food_items])
    personal = np.zeros_like(zs)
    helped = 0
    for i in range(len(food_items)):
        mem_mask = wrong.copy()
        mem_mask[i] = False
        kp, a = knn_probs(food_embs[i], food_embs[mem_mask], true_idx[mem_mask], len(ids))
        personal[i] = blend(zs[i], kp, a)
        helped += a > 0
    pers = score(personal)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "CLIP ViT-B/32, quantized ONNX (Xenova/clip-vit-base-patch32), CPU",
        "n_classes": len(ids),
        "n_test_images": int(food_mask.sum()),
        "n_test_classes": len(per_class),
        "zero_shot": {k: {"top1": v["top1"], "top5": v["top5"]} for k, v in results.items()},
        "per_class_top1_ensemble": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_class.items())},
        "ensemble_confusions": confusions,
        "not_food_guard": guard,
        "personalisation_simulated": {
            "description": "leave-one-out; memory = other test photos the zero-shot model got wrong, with their true labels",
            "memory_size": int(wrong.sum()),
            "images_where_knn_fired": int(helped),
            "top1": pers["top1"], "top5": pers["top5"],
        },
        "latency_ms_vision_per_image": round(1000 * float(np.median(times)), 1),
    }
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("zero_shot", "not_food_guard", "personalisation_simulated",
                                               "per_class_top1_ensemble", "latency_ms_vision_per_image")}, indent=2))


if __name__ == "__main__":
    main()
