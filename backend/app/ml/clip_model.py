"""Pretrained CLIP ViT-B/32 (OpenAI, 2021) as ONNX (Xenova/clip-vit-base-patch32 export). Nothing here
is trained by this project.

Only the vision tower runs per request. It is the fp32 export quantised by scripts.download_models with
onnxruntime's dynamic quantisation (int8 weights, per-channel): 2.2x faster on CPU than the 8-bit
export Xenova publishes, and closer to fp32 (embedding cosine 0.973 vs 0.932;
experiments/results/2026-11-26_onnx). The Xenova file is kept as a fallback. The text tower runs once per
(food list, templates, model file) to build the zero-shot label matrix, which is cached as .npy.

The text tower is the full-precision (fp32) export when `text_model.onnx` is present. The 8-bit text
export is only a fallback: its embeddings have 0.74 cosine similarity to fp32 and cost ~22 points of
188-way zero-shot accuracy (experiments/results/2026-11-26_onnx/text_precision_zero_shot.json). Since
the label matrix is cached, fp32 costs nothing per request.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from .calibration import softmax
from .preprocess import clip_tensor

log = logging.getLogger(__name__)

LOGIT_SCALE = 100.0  # CLIP's learned logit scale (exp(4.6052))
MAX_TOKENS = 77
EOS = 49407
VISION_FILES = ("vision_model_int8.onnx", "vision_model_quantized.onnx")  # preference: quantised here, then Xenova's
TEXT_FILES = ("text_model.onnx", "text_model_quantized.onnx")             # preference: fp32, then int8
MODEL_FILES = (VISION_FILES[0], TEXT_FILES[0], "tokenizer.json")          # what scripts.download_models provides

# Baseline: the single prompt from the CLIP paper, with the plain food name.
BASELINE_TEMPLATE = "a photo of {name}, a type of food."

# Ensemble: several phrasings x (plain name, descriptive name), text embeddings averaged
# (CLIP paper, section 3.1.4). Written before any evaluation was run; not tuned on test data.
ENSEMBLE_TEMPLATES = [
    "a photo of {name}, a type of food.",
    "a photo of {desc}.",
    "a close-up photo of {desc}.",
    "a top-down photo of a plate of {name}.",
    "a photo of {name} served in a restaurant.",
    "a homemade {name} on a table.",
    "a photo of {name}, a popular {cuisine} dish.",
]

NON_FOOD_PROMPTS = [
    "a photo of a person.",
    "a selfie of a person's face.",
    "a photo of a room interior.",
    "a photo of a street or a building.",
    "a photo of a landscape.",
    "a photo of a car.",
    "a photo of a dog or a cat.",
    "a screenshot of a phone screen with text.",
    "a photo of a document or a page of text.",
    "a photo of a laptop or a keyboard.",
    "an empty white plate.",
]


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


@dataclass
class Label:
    id: str
    name: str
    desc: str
    cuisine: str


def labels_from_foods(foods: list[dict]) -> list[Label]:
    return [Label(f["id"], f["name"].split(" / ")[0].split(" (")[0], f["clip_name"], f["cuisine"]) for f in foods]


class ClipEncoder:
    def __init__(self, model_dir: Path, labels: list[Label], cache_dir: Path, threads: int = 2):
        self.model_dir = Path(model_dir)
        text_file = next((f for f in TEXT_FILES if (self.model_dir / f).exists()), None)
        vision_file = next((f for f in VISION_FILES if (self.model_dir / f).exists()), None)
        missing = [] if vision_file else [" or ".join(VISION_FILES)]
        missing += [] if (self.model_dir / "tokenizer.json").exists() else ["tokenizer.json"]
        if text_file is None:
            missing.append(" or ".join(TEXT_FILES))
        if missing:
            raise FileNotFoundError(f"missing model files in {self.model_dir}: {', '.join(missing)}")
        self.vision_file = vision_file
        if vision_file != VISION_FILES[0]:
            log.warning("using %s (%s not found); recognition is ~2x slower. Run `python -m scripts.download_models`.",
                        vision_file, VISION_FILES[0])
        self.text_file = text_file
        self.text_precision = "fp32" if text_file == TEXT_FILES[0] else "int8"
        if self.text_precision == "int8":
            log.warning("using the 8-bit text tower (%s not found); zero-shot accuracy will be lower. "
                        "Run `python -m scripts.download_models`.", TEXT_FILES[0])
        self.labels = labels
        self.ids = [lab.id for lab in labels]
        self.cache_dir = Path(cache_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._opts = opts
        self.vision = ort.InferenceSession(str(self.model_dir / vision_file), opts,
                                           providers=["CPUExecutionProvider"])
        self._text: ort.InferenceSession | None = None
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.label_matrix = self.build_label_matrix(ensemble=True)
        self.nonfood_matrix = self._cached("nonfood", NON_FOOD_PROMPTS, lambda: l2n(np.stack(
            [self.encode_text(p) for p in NON_FOOD_PROMPTS])))

    @property
    def vision_sha256(self) -> str:
        """First 16 hex digits of the vision model file's SHA-256 (recorded with the fitted settings)."""
        if not hasattr(self, "_vision_sha"):
            h = hashlib.sha256()
            with open(self.model_dir / self.vision_file, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            self._vision_sha = h.hexdigest()[:16]
        return self._vision_sha

    # ---------- text side (runs once, cached) ----------
    @property
    def text(self) -> ort.InferenceSession:
        if self._text is None:
            self._text = ort.InferenceSession(str(self.model_dir / self.text_file), self._opts,
                                              providers=["CPUExecutionProvider"])
        return self._text

    def encode_text(self, prompt: str) -> np.ndarray:
        # One prompt at a time, no padding: the export pools at the EOS token and takes no
        # attention mask, so padded batches would change the embeddings.
        ids = self.tokenizer.encode(prompt).ids[:MAX_TOKENS]
        if ids[-1] != EOS:
            ids[-1] = EOS
        out = self.text.run(["text_embeds"], {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
        return l2n(out.astype(np.float32))

    def prompts_for(self, lab: Label, ensemble: bool, templates: list[str] | None = None) -> list[str]:
        if not ensemble:
            return [BASELINE_TEMPLATE.format(name=lab.name.lower())]
        cuisine = "Indian" if lab.cuisine == "indian" else "international"
        return [t.format(name=lab.name.lower(), desc=lab.desc, cuisine=cuisine) for t in (templates or ENSEMBLE_TEMPLATES)]

    def _cached(self, kind: str, key_parts: list[str], build) -> np.ndarray:
        onnx_size = (self.model_dir / self.text_file).stat().st_size
        h = hashlib.sha256(json.dumps([kind, key_parts, self.text_file, onnx_size]).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{kind}_{h}.npy"
        if path.exists():
            return np.load(path)
        log.info("building %s text embeddings (%d entries), cached to %s", kind, len(key_parts), path.name)
        mat = build().astype(np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, mat)
        return mat

    def build_label_matrix(self, ensemble: bool, templates: list[str] | None = None) -> np.ndarray:
        prompts = [self.prompts_for(lab, ensemble, templates) for lab in self.labels]
        kind = "labels_ens" if ensemble else "labels_single"

        def build():
            return np.stack([l2n(np.stack([self.encode_text(p) for p in ps]).mean(axis=0)) for ps in prompts])
        return self._cached(kind, [p for ps in prompts for p in ps], build)

    # ---------- image side (per request) ----------
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Batch=1 per image: the dynamically quantised model computes activation ranges per
        batch, so batching would make results depend on the other images in the batch."""
        outs = []
        for im in images:
            x = clip_tensor(im)[None].astype(np.float32)
            outs.append(self.vision.run(["image_embeds"], {"pixel_values": x})[0][0])
        return l2n(np.stack(outs).astype(np.float32))

    def zero_shot(self, emb: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
        m = self.label_matrix if matrix is None else matrix
        return softmax(LOGIT_SCALE * emb @ m.T)

    def food_score(self, emb: np.ndarray) -> np.ndarray:
        """P(food) from a 2-way softmax of the best food logit vs the best non-food logit."""
        emb = np.atleast_2d(emb)
        f = (LOGIT_SCALE * emb @ self.label_matrix.T).max(axis=1)
        n = (LOGIT_SCALE * emb @ self.nonfood_matrix.T).max(axis=1)
        return softmax(np.stack([f, n], axis=1))[:, 0]
