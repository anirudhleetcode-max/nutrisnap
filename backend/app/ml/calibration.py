"""Calibration and abstention utilities (numpy only, used by both the API and the experiments).

A prediction is "correct" when the top-1 class is in the image's accept set, so the negative
log-likelihood uses the probability mass on the accept set.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """Temperature scaling in log-probability space: softmax(log p / T).
    For zero-shot CLIP this is identical to dividing the logits by T."""
    return softmax(np.log(np.clip(probs, 1e-12, None)) / t)


def accept_mass(probs: np.ndarray, accept: list[set[int]]) -> np.ndarray:
    return np.array([probs[i, list(a)].sum() for i, a in enumerate(accept)])


def nll(probs: np.ndarray, accept: list[set[int]]) -> float:
    return float(-np.mean(np.log(np.clip(accept_mass(probs, accept), 1e-12, None))))


def fit_temperature(probs: np.ndarray, accept: list[set[int]], lo: float = 0.05, hi: float = 20.0) -> float:
    """1-D search for the T that minimises validation NLL (log-spaced grid, then refine)."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 60))
    losses = [nll(apply_temperature(probs, t), accept) for t in grid]
    k = int(np.argmin(losses))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    fine = np.linspace(a, b, 40)
    losses = [nll(apply_temperature(probs, t), accept) for t in fine]
    return float(fine[int(np.argmin(losses))])


def top1(probs: np.ndarray, accept: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = np.array([p in a for p, a in zip(pred, accept)])
    return conf, correct


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error (equal-width bins) and the reliability table."""
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if not m.any():
            table.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": 0, "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += m.mean() * abs(c - a)
        table.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                      "confidence": round(c, 3), "accuracy": round(a, 3)})
    return float(ece), table


def brier_top1(conf: np.ndarray, correct: np.ndarray) -> float:
    """Brier score of the top-1 confidence as a probability of being correct."""
    return float(np.mean((conf - correct.astype(float)) ** 2))


def coverage_curve(conf: np.ndarray, correct: np.ndarray, thresholds=None) -> list[dict]:
    thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2) if thresholds is None else thresholds
    out = []
    for t in thresholds:
        m = conf >= t
        out.append({"threshold": float(t), "coverage": round(float(m.mean()), 3),
                    "accuracy": round(float(correct[m].mean()), 3) if m.any() else None})
    return out


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target_accuracy: float) -> float:
    """Lowest threshold whose accepted predictions reach the target accuracy (max coverage)."""
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        m = conf >= t
        if m.sum() >= 20 and correct[m].mean() >= target_accuracy:
            return float(t)
    return 1.0


# ---------------- uncertainty measures ----------------
def entropy(probs: np.ndarray) -> np.ndarray:
    """Predictive entropy in nats (0 = certain; log C = uniform)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def margin(probs: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability (small margin = two classes compete)."""
    s = np.sort(probs, axis=-1)
    return s[..., -1] - s[..., -2]


def aurc(certainty: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better).

    Sort predictions from most to least certain; at each coverage k/N the risk is the error rate of
    the k most certain predictions. AURC averages that risk over all coverages, so it measures how
    well a certainty score *ranks* right answers above wrong ones, independent of any threshold.
    """
    order = np.argsort(-np.asarray(certainty), kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))
