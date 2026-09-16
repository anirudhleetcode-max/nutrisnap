"""Linear probe: multinomial logistic regression on frozen CLIP image embeddings.

Training (offline, scikit-learn) happens in experiments/run.py; the runtime only needs the weight
matrix, so inference is plain numpy: p = softmax(W x + b).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import softmax


class LinearProbe:
    def __init__(self, W: np.ndarray, b: np.ndarray, classes: list[str], card: dict | None = None):
        self.W, self.b, self.classes, self.card = W.astype(np.float32), b.astype(np.float32), classes, card or {}

    def predict_proba(self, emb: np.ndarray) -> np.ndarray:
        return softmax(emb @ self.W.T + self.b)

    def expand(self, probs: np.ndarray, label_ids: list[str]) -> np.ndarray:
        """Place probe probabilities onto the full food list (zeros for classes the probe never saw)."""
        index = {fid: i for i, fid in enumerate(label_ids)}
        out = np.zeros((*probs.shape[:-1], len(label_ids)), dtype=np.float32)
        for j, c in enumerate(self.classes):
            out[..., index[c]] = probs[..., j]
        return out

    def save(self, path: Path) -> None:
        np.savez_compressed(path, W=self.W, b=self.b, classes=np.array(self.classes))
        path.with_suffix(".json").write_text(json.dumps(self.card, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        z = np.load(path)
        card_path = path.with_suffix(".json")
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        return cls(z["W"], z["b"], [str(c) for c in z["classes"]], card)


def train_probe(X: np.ndarray, y: list[str], C: float, seed: int) -> LinearProbe:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(X, y)
    return LinearProbe(clf.coef_, clf.intercept_, [str(c) for c in clf.classes_])
