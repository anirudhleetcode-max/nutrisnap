import numpy as np
import pytest

from app.ml.adapter import TipAdapter, search_alpha_beta
from app.ml.calibration import aurc, entropy, margin


def _unit(rng, n, d=16):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_alpha_zero_is_zero_shot():
    rng = np.random.default_rng(0)
    W, K, f = _unit(rng, 4), _unit(rng, 10), _unit(rng, 3)
    tip = TipAdapter.build(W, K, rng.integers(0, 4, 10), alpha=0.0)
    np.testing.assert_allclose(tip.logits(f), 100 * f @ W.T, rtol=1e-5)


def test_cache_recovers_a_training_photo_the_text_classifier_gets_wrong():
    rng = np.random.default_rng(1)
    W = _unit(rng, 3)
    # a photo that zero-shot puts in class 0, labelled class 2
    photo = W[0] + 0.05 * rng.normal(size=W.shape[1])
    photo /= np.linalg.norm(photo)
    tip = TipAdapter.build(W, photo[None], np.array([2]), alpha=200.0, beta=5.0)
    assert (100 * photo @ W.T).argmax() == 0
    assert tip.predict_proba(photo).argmax() == 2
    np.testing.assert_allclose(tip.predict_proba(photo).sum(), 1.0, rtol=1e-6)


def test_bad_labels_are_rejected():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        TipAdapter.build(_unit(rng, 3), _unit(rng, 2), np.array([0, 3]))


def test_search_prefers_cache_when_it_helps():
    rng = np.random.default_rng(3)
    C, d = 4, 32
    centres = _unit(rng, C, d)
    W = _unit(rng, C, d)  # text weights unrelated to the visual clusters: zero-shot is near chance
    y_tr = np.repeat(np.arange(C), 10)
    y_va = np.repeat(np.arange(C), 5)
    tr = centres[y_tr] + 0.1 * rng.normal(size=(len(y_tr), d))
    va = centres[y_va] + 0.1 * rng.normal(size=(len(y_va), d))
    tip = TipAdapter.build(W, tr, y_tr)
    a, b, grid = search_alpha_beta(tip, va, y_va)
    best = max(g["val_top1"] for g in grid)
    assert best > 0.9 and a > 0.5 and len(grid) == 48


def test_uncertainty_measures():
    p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.6, 0.4, 0.0]])
    np.testing.assert_allclose(entropy(p), [0.0, np.log(3), -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))], atol=1e-6)
    np.testing.assert_allclose(margin(p), [1.0, 0.0, 0.2], atol=1e-9)


def test_aurc_rewards_good_ranking():
    correct = np.array([1, 1, 1, 0, 0])
    good = aurc(np.array([0.9, 0.8, 0.7, 0.2, 0.1]), correct)
    bad = aurc(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), correct)
    assert good < bad
    assert good == pytest.approx(np.mean([0, 0, 0, 1 / 4, 2 / 5]))
