"""FoodRecognizer: everything between a decoded image and the API response.

The settings that came out of experiments (temperature, abstention threshold, not-food threshold,
whether to use the linear probe) are read from artifacts/recognizer.json, which
experiments/run.py writes after fitting them on the validation split. If that file is missing the
recognizer still works, uncalibrated, and says so in its model card.
"""
from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .calibration import apply_temperature
from .clip_model import ClipEncoder, labels_from_foods
from .personalize import blend, knn_probs
from .preprocess import PREPROCESS_VERSION, crops, quality_warnings
from .probe import LinearProbe

log = logging.getLogger(__name__)

DEFAULTS = {
    "version": "uncalibrated",
    "mode": "zero_shot",
    "temperature": 1.0,
    "abstain_below": 0.0,
    "not_food_below": 0.5,
    "hybrid_weight": 0.0,
    "probe_file": None,
    "calibrated": False,
}


def hybrid(zs: np.ndarray, probe_full: np.ndarray, mask: np.ndarray, w: float) -> np.ndarray:
    """Mix the linear probe into zero-shot probabilities *within* the probe's classes only.

    Zero-shot decides how much probability goes to the probe's classes as a group (M); inside that
    group the distribution is (1-w) * zero-shot + w * probe. Classes the probe never saw keep their
    zero-shot probability, so they can still win.
    """
    m = (zs * mask).sum(axis=-1, keepdims=True)
    inside = (1 - w) * zs * mask + w * m * probe_full
    return np.where(mask, inside, zs)


@dataclass
class Analysis:
    embedding: np.ndarray
    zero_shot: np.ndarray              # raw zero-shot probabilities (logit scale 100)
    probs: np.ndarray                  # final probabilities shown to the user
    order: list[int]
    confidence: float
    decision: str                      # confident | uncertain | not_food
    food_score: float
    personal_weight: float = 0.0
    probe_used: bool = False
    also_detected: list[tuple[int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FoodRecognizer:
    def __init__(self, model_dir: Path, foods: list[dict], artifacts_dir: Path, threads: int = 2):
        self.encoder = ClipEncoder(model_dir, labels_from_foods(foods), model_dir / "cache", threads)
        self.ids = self.encoder.ids
        self.artifacts_dir = Path(artifacts_dir)
        cfg_path = self.artifacts_dir / "recognizer.json"
        self.card: dict = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.cfg = {**DEFAULTS, **self.card.get("settings", {})}
        fitted = self.cfg.get("text_tower")
        if fitted and fitted != self.encoder.text_precision:
            log.warning("recognizer.json was fitted with the %s text tower but %s is loaded: temperature and "
                        "thresholds will be off. Run `python -m scripts.download_models`.",
                        fitted, self.encoder.text_precision)
        fitted_vision = self.cfg.get("vision_sha256")
        if fitted_vision and fitted_vision != self.encoder.vision_sha256:
            log.warning("recognizer.json was fitted with vision model %s but %s (%s) is loaded: temperature and "
                        "thresholds may be off. Re-run `python -m scripts.download_models` (same onnxruntime "
                        "version) or re-fit with experiments/run.py.", fitted_vision, self.encoder.vision_file,
                        self.encoder.vision_sha256)
        self.probe: LinearProbe | None = None
        if self.cfg["mode"] == "hybrid" and self.cfg["probe_file"]:
            p = self.artifacts_dir / self.cfg["probe_file"]
            if p.exists():
                self.probe = LinearProbe.load(p)
                self.probe_mask = np.isin(self.ids, self.probe.classes)
            else:
                log.warning("probe file %s missing; falling back to zero-shot", p.name)

    @property
    def labels(self):
        return self.encoder.labels

    def base_probs(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Zero-shot probabilities, optionally mixed with the linear probe, then temperature-scaled."""
        zs = self.encoder.zero_shot(emb)
        p = zs
        used = False
        if self.probe is not None:
            p = hybrid(zs, self.probe.expand(self.probe.predict_proba(emb), self.ids),
                       self.probe_mask, float(self.cfg["hybrid_weight"]))
            used = True
        return zs, apply_temperature(p, float(self.cfg["temperature"])), used

    def analyse(self, img: Image.Image, memory: tuple[np.ndarray, np.ndarray] | None = None,
                multi: bool = True, top_k: int = 5) -> Analysis:
        emb = self.encoder.embed_images([img])[0]
        zs, probs, used = self.base_probs(emb)
        alpha = 0.0
        if memory is not None and len(memory[0]):
            kp, alpha = knn_probs(emb, memory[0], memory[1], len(self.ids))
            probs = blend(probs, kp, alpha)
        order = [int(i) for i in np.argsort(-probs)[:top_k]]
        conf = float(probs[order[0]])
        food_score = float(self.encoder.food_score(emb)[0])
        if food_score < self.cfg["not_food_below"]:
            decision = "not_food"
        elif conf < self.cfg["abstain_below"]:
            decision = "uncertain"
        else:
            decision = "confident"
        a = Analysis(emb, zs, probs, order, conf, decision, food_score, alpha, used,
                     warnings=quality_warnings(img))
        if multi:
            seen: dict[int, float] = {}
            cembs = self.encoder.embed_images(crops(img))
            scores = self.encoder.food_score(cembs)
            for e, s in zip(cembs, scores):
                if s < 0.6:
                    continue
                _, p, _ = self.base_probs(e)
                j = int(p.argmax())
                # report only confident crops that disagree with the whole-image guess
                if j != order[0] and p[j] >= max(0.45, self.cfg["abstain_below"]):
                    seen[j] = max(seen.get(j, 0.0), float(p[j]))
            a.also_detected = sorted(seen.items(), key=lambda kv: -kv[1])[:3]
        return a

    def model_card(self) -> dict:
        return {
            "name": "NutriSnap food recognizer",
            "version": self.cfg["version"],
            "mode": "zero_shot + linear probe (hybrid)" if self.probe else "zero_shot",
            "backbone": {
                "model": "CLIP ViT-B/32 (OpenAI, 2021), pretrained - not trained by this project",
                "weights": f"Xenova/clip-vit-base-patch32 ONNX export; vision {self.encoder.vision_file} "
                           f"(int8 dynamic quantisation), text {self.encoder.text_precision}",
                "source": "https://huggingface.co/Xenova/clip-vit-base-patch32",
                "license": "MIT (OpenAI CLIP)",
                "runtime": f"onnxruntime {ort.__version__} (CPU), python {platform.python_version()}",
                "embedding_dim": 512,
            },
            "preprocessing": PREPROCESS_VERSION,
            "classes": len(self.ids),
            "zero_shot": {"templates_per_class": 7, "logit_scale": 100},
            "probe": self.probe.card if self.probe else None,
            "settings": {k: self.cfg[k] for k in ("temperature", "abstain_below", "not_food_below", "hybrid_weight",
                                                  "calibrated")},
            "metrics": self.card.get("metrics"),
            "evaluated_on": self.card.get("datasets"),
            "trained_at": self.card.get("created_at"),
            "experiment": self.card.get("run_id"),
            "limitations": [
                "Closed set of ~190 foods; anything else is forced onto the nearest one.",
                "Classifies the whole photo; the multi-item hint is a crop heuristic, not detection.",
                "Portion size is not estimated from the image.",
                "Confuses visually similar Indian dishes (see error analysis).",
                "Confidence is calibrated on public web photos; phone photos may differ.",
                "Personalised results (after corrections) are not calibrated.",
            ],
        }
