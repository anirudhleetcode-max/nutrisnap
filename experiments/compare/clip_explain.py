"""Look inside CLIP: architecture read from the model file, and measurements that explain why
zero-shot classification works (and where it doesn't).

    cd backend
    python ../experiments/compare/clip_explain.py      # needs models/optim/*.onnx from onnx_bench.py

Writes experiments/results/<date>_clip_explain/clip.json and clip_similarity.png. docs/CLIP.md
walks through the numbers.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from data import BACKEND, ROOT, load

from app.ml.calibration import fit_temperature, reliability, softmax, top1  # noqa: E402
from app.ml.clip_model import (BASELINE_TEMPLATE, ENSEMBLE_TEMPLATES, LOGIT_SCALE, ClipEncoder,  # noqa: E402
                               labels_from_foods, l2n)
from ml.embeddings import embed_rows, load_set  # noqa: E402

MD = BACKEND / "models"


def architecture() -> dict:
    """Read shapes from the fp32 ONNX initialisers, so the numbers come from the weights, not memory."""
    import onnx
    from onnx import numpy_helper  # noqa: F401
    info = {}
    for tower, f in (("vision", "vision_model.onnx"), ("text", "text_model.onnx")):
        p = MD / "optim" / f
        if not p.exists():
            continue
        g = onnx.load(str(p)).graph
        shapes = {i.name: list(i.dims) for i in g.initializer}
        n_params = int(sum(np.prod(s) for s in shapes.values()))
        layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in shapes if "layers." in n})
        pick = {n: s for n, s in shapes.items() if any(k in n for k in (
            "patch_embedding", "position_embedding", "class_embedding", "token_embedding", "projection", "logit_scale"))}
        ops = Counter(n.op_type for n in g.node)
        info[tower] = {"parameters": n_params, "transformer_layers": len(layers),
                       "key_tensors": pick, "inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input],
                       "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output],
                       "most_common_ops": dict(ops.most_common(8))}
    return info


def main() -> None:
    out = ROOT / "experiments" / "results" / f"{datetime.now(timezone.utc):%Y-%m-%d}_clip_explain"
    out.mkdir(parents=True, exist_ok=True)
    foods = json.loads((BACKEND / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    enc = ClipEncoder(MD, labels_from_foods(foods), MD / "cache")
    d = load()
    cols = [enc.ids.index(c) for c in d.classes]
    W = enc.label_matrix                  # (188, 512) ensemble text embeddings
    res: dict = {"architecture": architecture()}

    # image embeddings for val (cached by ml.embeddings)
    E_parts = []
    for name in ("food101", "indianfoods"):
        rows = load_set(name)
        e, _ = embed_rows(enc, rows, name)
        E_parts.append(e[np.array([r["split"] == "val" for r in rows])])
    E = np.concatenate(E_parts)
    y = d.val.y

    # 1. the modality gap: matched image-text cosines are small in absolute terms
    S = E @ W[cols].T                     # (900, 45) cosine similarities
    matched = S[np.arange(len(y)), y]
    mask = np.ones_like(S, dtype=bool)
    mask[np.arange(len(y)), y] = False
    img_img = E @ E.T
    same = (y[:, None] == y[None]) & ~np.eye(len(y), dtype=bool)
    diff = y[:, None] != y[None]
    Wn = W[cols]
    tt = Wn @ Wn.T
    res["similarity"] = {
        "image_text_matched_mean": round(float(matched.mean()), 4),
        "image_text_unmatched_mean": round(float(S[mask].mean()), 4),
        "image_text_matched_minus_best_wrong_mean": round(float((matched - np.where(mask, S, -1).max(1)).mean()), 4),
        "image_image_same_class_mean": round(float(img_img[same].mean()), 4),
        "image_image_different_class_mean": round(float(img_img[diff].mean()), 4),
        "text_text_between_classes_mean": round(float(tt[~np.eye(len(cols), dtype=bool)].mean()), 4),
        "mean_image_vs_mean_text_cosine": round(float(l2n(E.mean(0)) @ l2n(Wn.mean(0))), 4),
        "note": "cosines of ~0.3 for the right label vs ~0.2 for wrong ones look small, but multiplied by the "
                "logit scale (100) a gap of 0.05 becomes 5 logits, i.e. e^5 = ~150x more probability",
    }

    # 2. logit scale (inverse temperature) sweep: ranking is unchanged, confidence is not
    sweep = []
    acc = None
    for s in (10, 25, 50, 100, 200):
        P = softmax(s * S)
        conf, corr = top1(P, d.val.accept)
        acc = float(corr.mean())
        sweep.append({"logit_scale": s, "val_top1": round(acc, 4), "mean_confidence": round(float(conf.mean()), 4),
                      "ece": round(reliability(conf, corr)[0], 4)})
    T = fit_temperature(softmax(LOGIT_SCALE * S), d.val.accept)
    res["logit_scale_sweep_val_45"] = {"rows": sweep, "best_temperature_at_scale_100": round(T, 4),
                                       "effective_scale": round(LOGIT_SCALE / T, 1)}

    # 3. prompt sensitivity: each template alone vs the average of all seven
    prompts = {}
    for t in [BASELINE_TEMPLATE, *ENSEMBLE_TEMPLATES[1:]]:
        if t == BASELINE_TEMPLATE:
            M = enc.build_label_matrix(ensemble=False)
        else:
            M = enc.build_label_matrix(ensemble=True, templates=[t])
        P = softmax(LOGIT_SCALE * E @ M[cols].T)
        P188 = softmax(LOGIT_SCALE * E @ M.T)
        acc188 = np.mean([p in {enc.ids.index(a) for a in r["accept"]} for p, r in zip(P188.argmax(1), d.val.rows)])
        prompts[t] = {"val_top1_45": round(float(top1(P, d.val.accept)[1].mean()), 4), "val_top1_188": round(float(acc188), 4)}
    P = softmax(LOGIT_SCALE * E @ W[cols].T)
    P188 = softmax(LOGIT_SCALE * E @ W.T)
    prompts["ENSEMBLE (mean of the 7 normalised text embeddings)"] = {
        "val_top1_45": round(float(top1(P, d.val.accept)[1].mean()), 4),
        "val_top1_188": round(float(np.mean([p in {enc.ids.index(a) for a in r["accept"]} for p, r in zip(P188.argmax(1), d.val.rows)])), 4)}
    res["prompt_sensitivity_val"] = prompts

    # 4. which food names sit closest together in text space (predicts zero-shot confusions)
    name = {f["id"]: f["name"] for f in foods}
    Wall = W @ W.T
    np.fill_diagonal(Wall, -1)
    iu = np.triu_indices(len(W), 1)
    order = np.argsort(-Wall[iu])[:15]
    res["closest_label_pairs_text_space"] = [
        {"a": name[enc.ids[iu[0][k]]], "b": name[enc.ids[iu[1][k]]], "cosine": round(float(Wall[iu][k]), 4)} for k in order]

    # 5. a small image x text matrix for the figure: one val photo per class for 10 classes
    pick_cls = list(range(0, len(cols), max(1, len(cols) // 10)))[:10]
    pick = [int(np.where(y == c)[0][0]) for c in pick_cls]
    Ssmall = E[pick] @ W[[cols[c] for c in pick_cls]].T
    res["figure_classes"] = [name[d.classes[c]] for c in pick_cls]
    res["figure_cosine"] = np.round(Ssmall, 4).tolist()

    (out / "clip.json").write_text(json.dumps(res, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax[0].imshow(Ssmall, cmap="viridis")
    labels = [n.split(" (")[0][:16] for n in res["figure_classes"]]
    ax[0].set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_yticks(range(len(labels)), [f"photo of {lab}" for lab in labels], fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, f"{Ssmall[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if Ssmall[i, j] < Ssmall.max() - 0.05 else "black")
    ax[0].set_title("Cosine(image embedding, text embedding)\nrows: one val photo per class · columns: class prompts", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(matched, bins=40, alpha=0.7, label="photo vs its own label", color="#c8442b")
    ax[1].hist(S[mask], bins=40, alpha=0.5, label="photo vs the other 44 labels", color="#4c78a8", density=False, weights=np.full(mask.sum(), len(y) / mask.sum()))
    ax[1].set(title="Image-text cosine on val (900 photos)", xlabel="cosine similarity", ylabel="photos (unmatched rescaled)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "clip_similarity.png", dpi=110)
    print(json.dumps({k: v for k, v in res.items() if k not in ("figure_cosine",)}, indent=1)[:4000])
    print("results in", out)


if __name__ == "__main__":
    main()
