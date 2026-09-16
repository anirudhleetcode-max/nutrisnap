"""Zero-shot food recognition with CLIP ViT-B/32 (quantized ONNX, CPU only).

Pipeline
  image -> CLIP preprocessing -> vision model -> 512-d image embedding (L2-normalised)
  food list x prompt templates -> text model -> averaged, re-normalised label embeddings (cached .npy)
  logits = 100 * cosine(image, label) -> softmax -> ranked candidates

Only the vision model runs per request; the label matrix is computed once and cached on disk,
keyed by a hash of (food names + templates + model file size).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

log = logging.getLogger(__name__)

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224
LOGIT_SCALE = 100.0
MAX_TOKENS = 77

# Baseline: the single prompt from the CLIP paper, with the plain food name.
BASELINE_TEMPLATE = "a photo of {name}, a type of food."

# Ensemble: several phrasings x (plain name, descriptive name). Averaging the text embeddings of
# many phrasings gives a more robust "centroid" for each class (CLIP paper, section 3.1.4).
ENSEMBLE_TEMPLATES = [
    "a photo of {name}, a type of food.",
    "a photo of {desc}.",
    "a close-up photo of {desc}.",
    "a top-down photo of a plate of {name}.",
    "a photo of {name} served in a restaurant.",
    "a homemade {name} on a table.",
    "a photo of {name}, a popular {cuisine} dish.",
]

# "Not food" guard: generic prompts for things people accidentally photograph.
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


def preprocess(img: Image.Image) -> np.ndarray:
    """CLIP preprocessing: shortest side -> 224 (bicubic), center crop, scale, normalise, CHW."""
    img = img.convert("RGB")
    w, h = img.size
    scale = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * scale)), max(SIZE, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF
    return img.convert("RGB")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def crops(img: Image.Image) -> list[Image.Image]:
    """2x2 grid + a centre crop, used for the multi-item hint."""
    w, h = img.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h),
             (w // 5, h // 5, w - w // 5, h - h // 5)]
    return [img.crop(b) for b in boxes]


@dataclass
class Label:
    id: str
    name: str
    desc: str
    cuisine: str


class ClipRecognizer:
    def __init__(self, model_dir: Path, labels: list[Label], cache_dir: Path, threads: int = 2):
        self.model_dir = Path(model_dir)
        self.labels = labels
        self.cache_dir = Path(cache_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._opts = opts
        self.vision = ort.InferenceSession(str(self.model_dir / "vision_model_quantized.onnx"), opts,
                                           providers=["CPUExecutionProvider"])
        self._text: ort.InferenceSession | None = None
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.label_matrix = self.build_label_matrix(ensemble=True)
        self.nonfood_matrix = self._cached("nonfood", NON_FOOD_PROMPTS, lambda: l2n(np.stack(
            [self.encode_text(p) for p in NON_FOOD_PROMPTS])))

    # ---------- text side (runs once, cached) ----------
    @property
    def text(self) -> ort.InferenceSession:
        if self._text is None:
            self._text = ort.InferenceSession(str(self.model_dir / "text_model_quantized.onnx"), self._opts,
                                              providers=["CPUExecutionProvider"])
        return self._text

    def encode_text(self, prompt: str) -> np.ndarray:
        # One prompt at a time, no padding: the model pools at the EOS token, so padding would
        # need an attention mask this export does not take.
        ids = self.tokenizer.encode(prompt).ids[:MAX_TOKENS]
        if ids[-1] != 49407:
            ids[-1] = 49407
        out = self.text.run(["text_embeds"], {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
        return l2n(out.astype(np.float32))

    def prompts_for(self, lab: Label, ensemble: bool, templates: list[str] | None = None) -> list[str]:
        if not ensemble:
            return [BASELINE_TEMPLATE.format(name=lab.name.lower())]
        cuisine = "Indian" if lab.cuisine == "indian" else "international"
        return [t.format(name=lab.name.lower(), desc=lab.desc, cuisine=cuisine) for t in (templates or ENSEMBLE_TEMPLATES)]

    def _cached(self, kind: str, key_parts: list[str], build) -> np.ndarray:
        onnx_size = (self.model_dir / "text_model_quantized.onnx").stat().st_size
        h = hashlib.sha256(json.dumps([kind, key_parts, onnx_size]).encode()).hexdigest()[:16]
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
            rows = []
            for ps in prompts:
                emb = np.stack([self.encode_text(p) for p in ps]).mean(axis=0)
                rows.append(l2n(emb))
            return np.stack(rows)
        return self._cached(kind, [p for ps in prompts for p in ps], build)

    # ---------- image side (per request) ----------
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([preprocess(im) for im in images]).astype(np.float32)
        out = self.vision.run(["image_embeds"], {"pixel_values": batch})[0]
        return l2n(out.astype(np.float32))

    def food_probs(self, emb: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
        m = self.label_matrix if matrix is None else matrix
        return softmax(LOGIT_SCALE * emb @ m.T)

    def food_mass(self, emb: np.ndarray) -> float:
        """Share of probability that lands on *any* food vs the non-food prompts."""
        food_logits = LOGIT_SCALE * emb @ self.label_matrix.T
        non_logits = LOGIT_SCALE * emb @ self.nonfood_matrix.T
        # compare the best food against the best non-food prompt (2-way softmax)
        p = softmax(np.array([food_logits.max(), non_logits.max()]))
        return float(p[0])

    def analyse(self, img: Image.Image, multi: bool = True, top_k: int = 5) -> dict:
        images = [img] + (crops(img) if multi else [])
        embs = self.embed_images(images)
        emb = embs[0]
        probs = self.food_probs(emb)
        order = np.argsort(-probs)[:top_k]
        result = {
            "embedding": emb,
            "probs": probs,
            "top": [(int(i), float(probs[i])) for i in order],
            "food_score": self.food_mass(emb),
            "also_detected": [],
        }
        if multi:
            top1 = int(order[0])
            seen: dict[int, float] = {}
            for e in embs[1:]:
                if self.food_mass(e) < 0.6:
                    continue
                p = self.food_probs(e)
                j = int(p.argmax())
                # only report crops that are confident and disagree with the whole-image guess
                if j != top1 and p[j] >= 0.45:
                    seen[j] = max(seen.get(j, 0.0), float(p[j]))
            result["also_detected"] = sorted(seen.items(), key=lambda kv: -kv[1])[:3]
        return result


def labels_from_foods(foods: list[dict]) -> list[Label]:
    return [Label(f["id"], f["name"].split(" / ")[0].split(" (")[0], f["clip_name"], f["cuisine"]) for f in foods]
