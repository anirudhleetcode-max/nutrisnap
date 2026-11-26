"""Tip-Adapter: a training-free way to adapt CLIP to a task with a few labelled photos.

Zhang et al., "Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification", ECCV 2022.

Zero-shot CLIP scores a photo against *text* embeddings of the class names. Tip-Adapter adds a
second, *visual* term: a key-value cache built from the labelled training photos.

    keys   K  (N x d)  L2-normalised CLIP image embeddings of the N training photos
    values V  (N x C)  one-hot labels of those photos

For a new photo with embedding f (L2-normalised):

    affinity  A = exp(-beta * (1 - f K^T))          (N,)   1 for an identical photo, -> 0 when far
    logits    = 100 * f W^T  +  alpha * A V         (C,)
                 zero-shot      few-shot cache

beta sets how sharply the cache focuses on the nearest photos; alpha sets how much the cache counts
against the text classifier. Both are chosen on the validation split. Nothing is trained, so adding
a photo to the cache is instant, and alpha = 0 gives back plain zero-shot CLIP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import softmax

LOGIT_SCALE = 100.0


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


@dataclass
class TipAdapter:
    text_weights: np.ndarray   # (C, d) zero-shot classifier: one text embedding per class
    keys: np.ndarray           # (N, d)
    values: np.ndarray         # (N, C) one-hot
    alpha: float = 1.0
    beta: float = 5.5

    @classmethod
    def build(cls, text_weights: np.ndarray, train_emb: np.ndarray, train_labels: np.ndarray,
              alpha: float = 1.0, beta: float = 5.5) -> "TipAdapter":
        C = text_weights.shape[0]
        labels = np.asarray(train_labels, dtype=int)
        if labels.size and (labels.min() < 0 or labels.max() >= C):
            raise ValueError("train label outside the class range")
        values = np.zeros((len(labels), C), dtype=np.float32)
        values[np.arange(len(labels)), labels] = 1.0
        return cls(_l2n(text_weights).astype(np.float32), _l2n(train_emb).astype(np.float32), values, alpha, beta)

    def cache_logits(self, emb: np.ndarray) -> np.ndarray:
        aff = np.exp(-self.beta * (1.0 - _l2n(np.atleast_2d(emb)) @ self.keys.T))
        return aff @ self.values

    def logits(self, emb: np.ndarray) -> np.ndarray:
        emb = _l2n(np.atleast_2d(emb))
        return LOGIT_SCALE * emb @ self.text_weights.T + self.alpha * self.cache_logits(emb)

    def predict_proba(self, emb: np.ndarray) -> np.ndarray:
        return softmax(self.logits(emb))


def search_alpha_beta(adapter: TipAdapter, val_emb: np.ndarray, val_labels: np.ndarray,
                      alphas=(0.5, 1, 2, 3, 5, 8, 12, 20), betas=(1, 2, 3.5, 5.5, 7.5, 10)) -> tuple[float, float, list[dict]]:
    """Grid search on val (top-1). Returns the best (alpha, beta) and the whole grid."""
    emb = _l2n(np.atleast_2d(val_emb))
    zs = LOGIT_SCALE * emb @ adapter.text_weights.T
    sims = emb @ adapter.keys.T
    y = np.asarray(val_labels)
    grid, best = [], None
    for b in betas:
        cache = np.exp(-b * (1.0 - sims)) @ adapter.values
        for a in alphas:
            acc = float(((zs + a * cache).argmax(1) == y).mean())
            grid.append({"alpha": a, "beta": b, "val_top1": round(acc, 4)})
            # ties go to the smaller alpha (closer to zero-shot)
            if best is None or acc > best[0] + 1e-12:
                best = (acc, a, b)
    return best[1], best[2], grid
