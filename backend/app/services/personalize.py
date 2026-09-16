"""Few-shot personalisation: a per-user kNN over CLIP image embeddings of past corrections.

When a user logs a photo under a different food than our top guess, we store
(image embedding, chosen food). For a new photo we look up the k most similar stored photos
(cosine similarity, embeddings are unit-length) and turn their labels into a probability
distribution. That distribution is blended with the zero-shot CLIP probabilities:

    p = (1 - a) * p_zero_shot + a * p_knn

The blend weight `a` depends on how close the nearest stored photo is: nothing below
`MIN_SIM` counts, and `a` grows linearly up to `MAX_ALPHA` at `FULL_SIM`. So a near-duplicate of
a photo the user already corrected (same dish, same plate, same kitchen) moves the ranking a lot,
while an unrelated photo leaves the zero-shot result untouched.
"""
from __future__ import annotations

import numpy as np

K = 5
MIN_SIM = 0.72      # CLIP image-image cosine below this is "a different kind of photo"
FULL_SIM = 0.90     # at or above this the neighbour is treated as the same dish
MAX_ALPHA = 0.7
TAU = 0.03          # softmax temperature over neighbour similarities


def knn_probs(query: np.ndarray, mem: np.ndarray, mem_labels: np.ndarray, n_labels: int,
              k: int = K) -> tuple[np.ndarray, float]:
    """Return (probabilities over labels, blend weight alpha). alpha == 0 means 'no opinion'."""
    probs = np.zeros(n_labels, dtype=np.float32)
    if mem.size == 0:
        return probs, 0.0
    sims = mem @ query
    idx = np.argsort(-sims)[:k]
    idx = idx[sims[idx] >= MIN_SIM]
    if idx.size == 0:
        return probs, 0.0
    w = np.exp((sims[idx] - sims[idx].max()) / TAU)
    np.add.at(probs, mem_labels[idx], w)
    probs /= probs.sum()
    best = float(sims[idx[0]])
    alpha = MAX_ALPHA * min(1.0, (best - MIN_SIM) / (FULL_SIM - MIN_SIM))
    return probs, alpha


def blend(zero_shot: np.ndarray, knn: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return zero_shot
    return (1 - alpha) * zero_shot + alpha * knn
