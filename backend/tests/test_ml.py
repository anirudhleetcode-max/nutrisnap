import numpy as np
from PIL import Image

from app.services.clip import MEAN, STD, crops, preprocess, softmax
from app.services.personalize import MAX_ALPHA, blend, knn_probs


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
