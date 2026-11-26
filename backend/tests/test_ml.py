import numpy as np
import pytest
from PIL import Image

from app.ml.calibration import softmax
from app.ml.personalize import MAX_ALPHA, blend, knn_probs
from app.ml.preprocess import MEAN, STD, clip_tensor as preprocess, crops


def test_preprocess_shape_and_normalisation():
    img = Image.new("RGB", (640, 320), (255, 255, 255))
    x = preprocess(img)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, 0, 0], (1 - MEAN) / STD, rtol=1e-5)


def test_crops_cover_grid_and_centre():
    cs = crops(Image.new("RGB", (400, 300)))
    assert len(cs) == 5 and cs[0].size == (200, 150) and cs[4].size == (240, 180)


def test_softmax_stable():
    p = softmax(np.array([1000.0, 1000.0, 0.0]))
    np.testing.assert_allclose(p, [0.5, 0.5, 0.0], atol=1e-6)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_knn_fires_only_for_similar_photos():
    mem = np.stack([unit([1, 0, 0]), unit([1, 0.05, 0]), unit([0, 1, 0])])
    labels = np.array([2, 2, 0])
    near, a = knn_probs(unit([1, 0.02, 0]), mem, labels, 3)
    assert a == MAX_ALPHA and near.argmax() == 2
    far, a2 = knn_probs(unit([0, 0, 1]), mem, labels, 3)
    assert a2 == 0 and far.sum() == 0
    empty, a3 = knn_probs(unit([1, 0, 0]), np.zeros((0, 3), np.float32), np.array([], int), 3)
    assert a3 == 0


def test_blend_can_flip_the_ranking():
    zs = np.array([0.6, 0.1, 0.3])
    knn = np.array([0.0, 0.0, 1.0])
    assert blend(zs, knn, 0.0).argmax() == 0
    assert blend(zs, knn, MAX_ALPHA).argmax() == 2
    assert np.isclose(blend(zs, knn, 0.5).sum(), 1.0)


# ---------- phase 2: hybrid, calibration, probe ----------
from app.ml.calibration import apply_temperature, fit_temperature, pick_threshold, reliability, top1  # noqa: E402
from app.ml.pipeline import hybrid  # noqa: E402
from app.ml.preprocess import ImageRejected, decode_image, quality_warnings, sniff  # noqa: E402
from app.ml.probe import LinearProbe, train_probe  # noqa: E402


def test_hybrid_keeps_unseen_class_mass():
    zs = np.array([0.5, 0.2, 0.3])          # class 0 is unknown to the probe
    mask = np.array([False, True, True])
    probe = np.array([0.0, 0.9, 0.1])       # probe prefers class 1
    out = hybrid(zs, probe, mask, 1.0)
    assert np.isclose(out[0], 0.5) and np.isclose(out.sum(), 1.0)
    assert np.allclose(out[1:], [0.45, 0.05])
    assert np.allclose(hybrid(zs, probe, mask, 0.0), zs)


def test_temperature_scaling_properties():
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(apply_temperature(p, 1.0), p)
    sharp = apply_temperature(p, 0.5)
    assert sharp[0, 0] > 0.7 and sharp.argmax() == 0      # T<1 sharpens, ranking unchanged
    assert apply_temperature(p, 3.0)[0, 0] < 0.7


def test_fit_temperature_fixes_underconfidence():
    rng = np.random.default_rng(0)
    n, k = 400, 5
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 3.0                         # the model is usually right...
    probs = softmax(logits / 4.0)                          # ...but its probabilities are flattened
    accept = [{int(c)} for c in y]
    t = fit_temperature(probs, accept)
    assert t < 0.5
    before = reliability(*top1(probs, accept))[0]
    after = reliability(*top1(apply_temperature(probs, t), accept))[0]
    assert after < before


def test_pick_threshold_trades_coverage_for_accuracy():
    conf = np.linspace(0, 1, 200)
    correct = conf > 0.4                                   # confident answers are right
    t = pick_threshold(conf, correct, 0.95)
    assert 0.3 <= t <= 0.45 and correct[conf >= t].mean() >= 0.95
    assert correct[conf >= t - 0.01].mean() < 0.95 or t == 0.0   # lowest such threshold
    assert pick_threshold(conf, np.zeros(200, bool), 0.9) == 1.0   # never good enough -> always abstain


def test_probe_trains_and_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    centres = {"idli": unit([1, 0, 0]), "poha": unit([0, 1, 0])}
    X, y = [], []
    for name, c in centres.items():
        for _ in range(30):
            X.append(c + rng.normal(0, 0.05, 3))
            y.append(name)
    probe = train_probe(np.array(X, dtype=np.float32), y, C=1.0, seed=0)
    probe.card = {"synthetic": True, "note": "unit-test data"}
    path = tmp_path / "p.npz"
    probe.save(path)
    loaded = LinearProbe.load(path)
    q = unit([0.9, 0.1, 0])[None]
    assert loaded.classes[int(loaded.predict_proba(q).argmax())] == "idli"
    assert loaded.card["synthetic"] is True
    full = loaded.expand(loaded.predict_proba(q), ["dal", "idli", "poha"])
    assert full.shape == (1, 3) and full[0, 0] == 0


def test_magic_bytes_and_quality_rules():
    assert sniff(b"\xff\xd8\xff\xe0") == "jpeg" and sniff(b"GIF89a") is None
    with pytest.raises(ImageRejected) as e:
        decode_image(b"GIF89a" + b"0" * 100)
    assert e.value.status == 415
    dark = Image.new("RGB", (300, 300), (3, 3, 3))
    assert "too_dark" in quality_warnings(dark)
    flat = Image.new("RGB", (300, 300), (128, 128, 128))
    assert "blurry" in quality_warnings(flat)


def test_encoder_reports_missing_text_tower(tmp_path):
    from app.ml.clip_model import ClipEncoder
    (tmp_path / "vision_model_quantized.onnx").write_bytes(b"x")
    (tmp_path / "tokenizer.json").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="text_model.onnx or text_model_quantized.onnx"):
        ClipEncoder(tmp_path, [], tmp_path / "cache")
