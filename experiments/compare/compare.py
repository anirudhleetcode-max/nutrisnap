"""Compare conventional CNNs, transfer learning and CLIP on the same photos.

    cd backend
    python ../experiments/compare/train_cnn.py --model scratch            # writes cnn/*.npz first
    python ../experiments/compare/train_cnn.py --model resnet50_frozen
    python ../experiments/compare/train_cnn.py --model resnet50_finetune
    python ../experiments/compare/compare.py

Approaches (all scored on the same 45 classes, same splits, same "accept set" rule):

  1. Traditional CNN       small CNN trained from random init on 1,800 photos
  2. Transfer learning     ImageNet ResNet-50: (a) frozen features + logistic regression,
                           (b) layer4 + head fine-tuned
  3. CLIP zero-shot        no food photos at all: cosine similarity to text embeddings of the class
                           names (single prompt, and the 7-prompt ensemble the app uses)
  4. CLIP + adaptation     (a) linear probe on frozen CLIP image embeddings,
                           (b) Tip-Adapter: zero-shot text classifier + a training-free cache of
                               the labelled photos (alpha, beta chosen on val)

For each: accuracy (top-1/5, macro, per dataset), calibration (temperature fitted on val, ECE, NLL),
uncertainty (AURC for max-probability, entropy and margin; coverage at the val-chosen threshold),
cost (labelled photos, trainable parameters, training time, inference latency), performance on the
unseen Commons photos, and a few-shot learning curve (1-40 labelled photos per class).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

from data import BACKEND, ROOT, load

from app.ml.adapter import TipAdapter, search_alpha_beta  # noqa: E402
from app.ml.calibration import (apply_temperature, aurc, entropy, fit_temperature, margin, nll,  # noqa: E402
                                pick_threshold, reliability, softmax, top1)
from app.ml.clip_model import LOGIT_SCALE, ClipEncoder, labels_from_foods  # noqa: E402
from ml.embeddings import embed_rows  # noqa: E402

NAMES = {
    "cnn_scratch": "CNN from scratch",
    "resnet50_frozen": "ResNet-50 ImageNet, frozen + logistic regression",
    "resnet50_finetune": "ResNet-50 ImageNet, fine-tuned (layer4 + head)",
    "clip_zero_shot_single": "CLIP zero-shot, single prompt",
    "clip_zero_shot_ensemble": "CLIP zero-shot, 7-prompt ensemble (int8 text tower, as shipped)",
    "clip_zero_shot_fp32_text": "CLIP zero-shot, 7-prompt ensemble, fp32 text tower",
    "clip_linear_probe": "CLIP + linear probe",
    "clip_tip_adapter": "CLIP + Tip-Adapter (training-free cache)",
    "clip_tip_adapter_fp32_text": "CLIP + Tip-Adapter, fp32 text tower",
}
FAMILY = {"cnn_scratch": "1 traditional CNN", "resnet50_frozen": "2 transfer learning",
          "resnet50_finetune": "2 transfer learning", "clip_zero_shot_single": "3 CLIP zero-shot",
          "clip_zero_shot_ensemble": "3 CLIP zero-shot", "clip_zero_shot_fp32_text": "3 CLIP zero-shot",
          "clip_linear_probe": "4 CLIP adapted", "clip_tip_adapter_fp32_text": "4 CLIP adapted",
          "clip_tip_adapter": "4 CLIP adapted"}


def fp32_label_matrix(enc: ClipEncoder) -> np.ndarray | None:
    """Same 7-prompt ensemble, embedded with the fp32 text tower (models/optim/text_model.onnx,
    downloaded by onnx_bench.py). The app's label matrix uses the int8 text tower."""
    import hashlib
    import onnxruntime as ort
    from app.ml.clip_model import EOS, MAX_TOKENS, l2n
    path = BACKEND / "models" / "optim" / "text_model.onnx"
    if not path.exists():
        print("no fp32 text tower (run onnx_bench.py first); skipping fp32-text variants")
        return None
    prompts = [enc.prompts_for(lab, True) for lab in enc.labels]
    key = hashlib.sha256(json.dumps([prompts, path.stat().st_size]).encode()).hexdigest()[:16]
    cache = BACKEND / "models" / "cache" / f"labels_ens_fp32text_{key}.npy"
    if cache.exists():
        return np.load(cache)
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def emb(p):
        ids = enc.tokenizer.encode(p).ids[:MAX_TOKENS]
        if ids[-1] != EOS:
            ids[-1] = EOS
        return l2n(s.run(["text_embeds"], {"input_ids": np.array([ids], dtype=np.int64)})[0][0].astype(np.float32))
    W = np.stack([l2n(np.stack([emb(p) for p in ps]).mean(0)) for ps in prompts]).astype(np.float32)
    np.save(cache, W)
    return W


def topk(P: np.ndarray, accept: list[set[int]], k: int) -> np.ndarray:
    order = np.argsort(-P, axis=1)[:, :k]
    return np.array([bool(set(o) & a) for o, a in zip(order, accept)])


def macro(ok: np.ndarray, groups: list[str]) -> float:
    g = np.array(groups)
    return float(np.mean([ok[g == c].mean() for c in np.unique(g)]))


def r4(x):
    return None if x is None else round(float(x), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run = args.run or f"{datetime.now(timezone.utc):%Y-%m-%d}_compare"
    out = ROOT / "experiments" / "results" / run
    cnn_dir = out / "cnn"

    d = load()
    classes = d.classes
    C = len(classes)
    foods = json.loads((BACKEND / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    enc = ClipEncoder(BACKEND / "models", labels_from_foods(foods), BACKEND / "models" / "cache")
    cols = [enc.ids.index(c) for c in classes]
    W_single = enc.build_label_matrix(ensemble=False)
    W_ens = enc.label_matrix
    W_ens32 = fp32_label_matrix(enc)

    # CLIP image embeddings (cached by ml.embeddings; the same cache run.py uses)
    emb, ms = {}, {}
    for name in ("food101", "indianfoods"):
        from ml.embeddings import load_set
        rows = load_set(name)
        e, ms[name] = embed_rows(enc, rows, name)
        for s in ("train", "val", "test"):
            m = np.array([r["split"] == s for r in rows])
            emb.setdefault(s, []).append(e[m])
    emb = {s: np.concatenate(v) for s, v in emb.items()}
    from ml.embeddings import load_set
    com_rows = load_set("commons")
    e_com, ms["commons"] = embed_rows(enc, com_rows, "commons")
    emb["unseen"] = e_com[np.array([r["source_class"] != "not-food" for r in com_rows])]
    for s in ("train", "val", "test", "unseen"):
        assert len(emb[s]) == len(getattr(d, s).rows), s

    # ------------------------------------------------ CLIP models (45-way)
    P, cost = {}, {}

    def zs(W, E):
        return softmax(LOGIT_SCALE * E @ W[cols].T)
    zs_models = [("clip_zero_shot_single", W_single), ("clip_zero_shot_ensemble", W_ens)]
    if W_ens32 is not None:
        zs_models.append(("clip_zero_shot_fp32_text", W_ens32))
    for key, W in zs_models:
        P[key] = {s: zs(W, emb[s]) for s in ("val", "test", "unseen")}
        cost[key] = {"labelled_photos": 0, "trainable_params": 0, "train_seconds": 0.0}

    t = time.time()
    probe_search, best = [], None
    for Cr in (0.1, 0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=Cr, max_iter=3000, random_state=args.seed).fit(emb["train"], d.train.y)
        acc = float(topk(clf.predict_proba(emb["val"]), d.val.accept, 1).mean())
        probe_search.append({"C": Cr, "val_top1": r4(acc)})
        if best is None or acc > best[0]:
            best = (acc, clf, Cr)
    clf = best[1]
    P["clip_linear_probe"] = {s: clf.predict_proba(emb[s]) for s in ("val", "test", "unseen")}
    cost["clip_linear_probe"] = {"labelled_photos": len(d.train.y), "trainable_params": int(clf.coef_.size + C),
                                 "train_seconds": round(time.time() - t, 1), "C": best[2], "C_search": probe_search}

    t = time.time()
    tip = TipAdapter.build(W_ens[cols], emb["train"], d.train.y)
    a, b, tip_grid = search_alpha_beta(tip, emb["val"], d.val.y)
    tip.alpha, tip.beta = a, b
    P["clip_tip_adapter"] = {s: tip.predict_proba(emb[s]) for s in ("val", "test", "unseen")}
    cost["clip_tip_adapter"] = {"labelled_photos": len(d.train.y), "trainable_params": 0,
                                "train_seconds": round(time.time() - t, 1), "alpha": a, "beta": b,
                                "cache_entries": len(d.train.y), "grid": tip_grid}
    if W_ens32 is not None:
        tip32 = TipAdapter.build(W_ens32[cols], emb["train"], d.train.y)
        tip32.alpha, tip32.beta, grid32 = search_alpha_beta(tip32, emb["val"], d.val.y)
        P["clip_tip_adapter_fp32_text"] = {s: tip32.predict_proba(emb[s]) for s in ("val", "test", "unseen")}
        cost["clip_tip_adapter_fp32_text"] = {"labelled_photos": len(d.train.y), "trainable_params": 0, "train_seconds": 0.0,
                                              "alpha": tip32.alpha, "beta": tip32.beta, "cache_entries": len(d.train.y)}

    # ------------------------------------------------ CNN models (from train_cnn.py)
    cnn_meta = {}
    for key, f in (("cnn_scratch", "scratch"), ("resnet50_frozen", "resnet50_frozen"), ("resnet50_finetune", "resnet50_finetune")):
        p = cnn_dir / f"{f}.npz"
        if not p.exists():
            print(f"missing {p}: run train_cnn.py --model {f} first; skipping")
            continue
        meta = json.loads(p.with_suffix(".json").read_text())
        assert meta["classes_list"] == classes, "class order differs"
        z = np.load(p)
        P[key] = {s: z[s] for s in ("val", "test", "unseen")}
        cost[key] = {"labelled_photos": len(d.train.y) + (1_281_167 if "resnet" in key else 0),
                     "labelled_photos_note": "+ ImageNet-1k pretraining" if "resnet" in key else "",
                     "trainable_params": meta["trainable_params"], "total_params": meta["params"],
                     "train_seconds": meta["train_seconds"], "latency_ms_batch1": meta["latency_ms_torch_batch1"],
                     "input_size": meta["input_size"]}
        cnn_meta[key] = {k: v for k, v in meta.items() if k != "classes_list"}
    clip_ms = round(float(np.mean([ms["food101"], ms["indianfoods"]])), 2)
    for k in cost:
        if k.startswith("clip"):
            cost[k].update({"latency_ms_batch1": clip_ms, "input_size": 224, "total_params": 87_849_216,
                            "labelled_photos_note": "+ 400M web image-text pairs (CLIP pretraining)"})

    # ------------------------------------------------ evaluation
    # 49 test photos (all Indian Food Images, mostly kofta and vada pav) are byte-identical copies of a
    # train or val photo in the source dataset. Trained models can memorise them, so every model is
    # also scored on the test split without them.
    import hashlib
    sha = lambda r: hashlib.sha256(r["image"]).hexdigest()  # noqa: E731
    seen_hashes = {sha(r) for r in d.train.rows} | {sha(r) for r in d.val.rows}
    dedup = np.array([sha(r) not in seen_hashes for r in d.test.rows])
    seen = np.array([bool(a) for a in d.unseen.accept])
    results = {}
    for key, Pk in P.items():
        r = {"name": NAMES[key], "family": FAMILY[key]}
        T = fit_temperature(Pk["val"], d.val.accept)
        for s in ("val", "test"):
            sp = getattr(d, s)
            t1, t5 = topk(Pk[s], sp.accept, 1), topk(Pk[s], sp.accept, 5)
            r[s] = {"top1": r4(t1.mean()), "top5": r4(t5.mean()), "macro_top1": r4(macro(t1, sp.source_class)),
                    "food101_top1": r4(t1[sp.dataset == "food101"].mean()),
                    "indian_top1": r4(t1[sp.dataset == "indianfoods"].mean()), "n": len(t1)}
        sp = d.test
        t1d, t5d = topk(Pk["test"], sp.accept, 1)[dedup], topk(Pk["test"], sp.accept, 5)[dedup]
        r["test_dedup"] = {"top1": r4(t1d.mean()), "top5": r4(t5d.mean()),
                           "indian_top1": r4(t1d[sp.dataset[dedup] == "indianfoods"].mean()), "n": int(dedup.sum())}
        te = d.test
        Pc = apply_temperature(Pk["test"], T)
        conf, corr = top1(Pc, te.accept)
        conf_raw, _ = top1(Pk["test"], te.accept)
        conf_va, corr_va = top1(apply_temperature(Pk["val"], T), d.val.accept)
        thr = pick_threshold(conf_va, corr_va, 0.85)
        m = conf >= thr
        r["calibration"] = {"temperature": r4(T), "ece_before": r4(reliability(conf_raw, corr)[0]),
                            "ece_after": r4(reliability(conf, corr)[0]), "nll_before": r4(nll(Pk["test"], te.accept)),
                            "nll_after": r4(nll(Pc, te.accept)), "mean_confidence_after": r4(conf.mean())}
        r["uncertainty"] = {"aurc_max_prob": r4(aurc(conf, corr)), "aurc_neg_entropy": r4(aurc(-entropy(Pc), corr)),
                            "aurc_margin": r4(aurc(margin(Pc), corr)),
                            "aurc_oracle": r4(aurc(corr.astype(float), corr)),
                            "threshold_for_85pct_on_val": thr, "test_coverage": r4(m.mean()),
                            "test_accuracy_when_answering": r4(corr[m].mean()) if m.any() else None,
                            "mean_entropy_correct": r4(entropy(Pc)[corr].mean()),
                            "mean_entropy_wrong": r4(entropy(Pc)[~corr].mean()) if (~corr).any() else None}
        uP = Pk["unseen"][seen]
        ua = [a for a, s in zip(d.unseen.accept, seen) if s]
        uconf, ucorr = top1(apply_temperature(uP, T), ua)
        r["unseen_photos"] = {"n": int(seen.sum()), "top1": r4(topk(uP, ua, 1).mean()), "top5": r4(topk(uP, ua, 5).mean()),
                              "coverage_at_threshold": r4((uconf >= thr).mean()),
                              "accuracy_when_answering": r4(ucorr[uconf >= thr].mean()) if (uconf >= thr).any() else None,
                              "mean_confidence": r4(uconf.mean())}
        r["cost"] = {k: v for k, v in cost[key].items() if k not in ("grid", "C_search")}
        r["_reliability_after"] = reliability(conf, corr)[1]
        r["_risk_coverage"] = [(float(c), float(e)) for c, e in zip(
            np.arange(1, len(corr) + 1) / len(corr), np.cumsum(1 - corr[np.argsort(-conf, kind="stable")]) /
            np.arange(1, len(corr) + 1))][::25]
        results[key] = r
        print(f"{key:26s} test top1 {r['test']['top1']:.3f} top5 {r['test']['top5']:.3f} indian {r['test']['indian_top1']:.3f} "
              f"ECE {r['calibration']['ece_after']:.3f} AURC {r['uncertainty']['aurc_max_prob']:.3f} "
              f"unseen {r['unseen_photos']['top1']:.3f}", flush=True)

    # photos of dishes outside the 45 training classes: only an open-vocabulary model can name them
    unseen_cls = ~seen
    open_vocab = {}
    if unseen_cls.any():
        full = [{enc.ids.index(a) for a in r["accept"]} for r, s in zip(d.unseen.rows, seen) if not s]
        E_u = emb["unseen"][unseen_cls]
        for key, W in zs_models:
            Pu = softmax(LOGIT_SCALE * E_u @ W.T)
            open_vocab[key] = {"top1": r4(topk(Pu, full, 1).mean()), "top5": r4(topk(Pu, full, 5).mean())}
        open_vocab.update({"n": int(unseen_cls.sum()), "label_space": len(enc.ids),
                           "classes": sorted({r["source_class"] for r, s in zip(d.unseen.rows, seen) if not s}),
                           "note": "CNN / transfer / probe models have a fixed 45-class output layer, so their accuracy "
                                   "on these photos is 0 by construction. CLIP zero-shot ranks all 188 foods."})

    # ------------------------------------------------ few-shot learning curve
    feats_path = cnn_dir / "resnet50_features.npz"
    rn = np.load(feats_path) if feats_path.exists() else None
    shots, curve = (1, 2, 4, 8, 16, 40), {"shots": [0, 1, 2, 4, 8, 16, 40]}
    by_class = {c: np.where(d.train.y == c)[0] for c in range(C)}
    zs_test = float(topk(P["clip_zero_shot_ensemble"]["test"], d.test.accept, 1).mean())
    curve["clip_zero_shot_ensemble"] = [zs_test] * 7
    zs32_test = None
    if W_ens32 is not None:
        zs32_test = float(topk(P["clip_zero_shot_fp32_text"]["test"], d.test.accept, 1).mean())
        curve["clip_zero_shot_fp32_text"] = [zs32_test] * 7
    for name in ("clip_linear_probe", "clip_tip_adapter", "clip_tip_adapter_fp32_text", "resnet50_frozen"):
        if (name == "resnet50_frozen" and rn is None) or (name.endswith("fp32_text") and W_ens32 is None):
            continue
        vals = [{"clip_tip_adapter": zs_test, "clip_tip_adapter_fp32_text": zs32_test}.get(name)]
        for k in shots:
            accs = []
            for rep in range(3 if k < 16 else 1):
                r_ = np.random.default_rng(args.seed + 1000 * rep + k)
                ix = np.concatenate([r_.choice(v, min(k, len(v)), replace=False) for v in by_class.values()])
                y = d.train.y[ix]
                if name.startswith("clip_tip_adapter"):
                    ad = TipAdapter.build((W_ens32 if name.endswith("fp32_text") else W_ens)[cols], emb["train"][ix], y)
                    ad.alpha, ad.beta, _ = search_alpha_beta(ad, emb["val"], d.val.y)
                    Pt = ad.predict_proba(emb["test"])
                else:
                    if name == "clip_linear_probe":
                        Xtr, Xva, Xte = emb["train"][ix], emb["val"], emb["test"]
                    else:
                        mu, sd = rn["train"][ix].mean(0), rn["train"].std(0) + 1e-6
                        Xtr, Xva, Xte = [(rn[s_] - mu) / sd for s_ in ("train", "val", "test")]
                        Xtr = Xtr[ix]
                    best_ = None
                    grid_ = (0.3, 1, 3, 10, 30) if name == "clip_linear_probe" else (0.003, 0.01, 0.03, 0.1, 0.3)
                    for Cr in grid_:
                        m_ = LogisticRegression(C=Cr, max_iter=2000, random_state=args.seed).fit(Xtr, y)
                        acc = topk(m_.predict_proba(Xva), d.val.accept, 1).mean()
                        if best_ is None or acc > best_[0]:
                            best_ = (acc, m_)
                    Pt = best_[1].predict_proba(Xte)
                accs.append(float(topk(Pt, d.test.accept, 1).mean()))
            vals.append(round(float(np.mean(accs)), 4))
            print(f"  few-shot {name} k={k}: {vals[-1]:.3f}", flush=True)
        curve[name] = vals
    curve["note"] = ("k labelled training photos per class (3 random draws averaged for k<16); hyper-parameters picked "
                     "on the full val split; test top-1 over 45 classes. k=0 is zero-shot.")

    # ------------------------------------------------ write + plots
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    metrics = {"run_id": run, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed,
               "data": {"classes": C, "train": len(d.train.y), "val": len(d.val.y), "test": len(d.test.y),
                        "unseen_photos_seen_classes": int(seen.sum()),
                        "test_photos_duplicating_train_or_val": int((~dedup).sum()), "unseen_photos_new_classes": int(unseen_cls.sum()),
                        "synthetic": False},
               "models": clean, "open_vocabulary_new_classes": open_vocab, "few_shot_curve": curve,
               "search": {"clip_linear_probe_C": probe_search, "tip_adapter_grid": cost["clip_tip_adapter"]["grid"]},
               "cnn_training": cnn_meta, "clip_embedding_ms_median": ms}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plots(out, results, curve)
    print("results in", out)


def plots(out, results, curve) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(results)
    short = {"cnn_scratch": "CNN\nscratch", "resnet50_frozen": "ResNet-50\nfrozen+LR", "resnet50_finetune": "ResNet-50\nfine-tuned",
             "clip_zero_shot_single": "CLIP ZS\n1 prompt", "clip_zero_shot_ensemble": "CLIP ZS\n7 prompts",
             "clip_zero_shot_fp32_text": "CLIP ZS\nfp32 text", "clip_tip_adapter_fp32_text": "CLIP+Tip\nfp32 text",
             "clip_linear_probe": "CLIP\n+probe", "clip_tip_adapter": "CLIP\n+Tip-Adapter"}
    colors = {"1": "#8a8a8a", "2": "#4c78a8", "3": "#e45756", "4": "#c8442b"}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        r = results[k]
        c = colors[r["family"][0]]
        ax.bar(i - 0.2, r["test"]["top1"], 0.38, color=c)
        ax.bar(i + 0.2, r["unseen_photos"]["top1"], 0.38, color=c, alpha=0.45, hatch="//", edgecolor="white")
        ax.text(i - 0.2, r["test"]["top1"] + 0.01, f"{r['test']['top1']:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, r["unseen_photos"]["top1"] + 0.01, f"{r['unseen_photos']['top1']:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [short[k] for k in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1 accuracy (45 classes)")
    ax.set_title("Test split (solid) vs unseen Commons photos (hatched)")
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=110)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = curve["shots"]
    style = {"clip_zero_shot_ensemble": ("CLIP zero-shot, int8 text (0 photos)", "--", "#e45756"),
             "clip_zero_shot_fp32_text": ("CLIP zero-shot, fp32 text (0 photos)", ":", "#8c2d04"),
             "clip_tip_adapter_fp32_text": ("CLIP + Tip-Adapter, fp32 text", "D-", "#8c2d04"),
             "clip_linear_probe": ("CLIP + linear probe", "o-", "#c8442b"),
             "clip_tip_adapter": ("CLIP + Tip-Adapter", "s-", "#f58518"),
             "resnet50_frozen": ("ResNet-50 ImageNet + LR", "^-", "#4c78a8")}
    for k, (lab, st, c) in style.items():
        if k in curve:
            pts = [(s, v) for s, v in zip(xs, curve[k]) if v is not None]
            ax.plot([p[0] + 0.5 for p in pts], [p[1] for p in pts], st, label=lab, color=c)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 40.5], ["0", "1", "2", "4", "8", "16", "40"])
    ax.set(xlabel="labelled training photos per class", ylabel="test top-1 (45 classes)", title="Few-shot learning curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "few_shot.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k in keys:
        r = results[k]
        tab = [b for b in r["_reliability_after"] if b["n"]]
        ax[0].plot([b["confidence"] for b in tab], [b["accuracy"] for b in tab], "o-", ms=3, label=short[k].replace("\n", " "))
        rc = r["_risk_coverage"]
        ax[1].plot([p[0] for p in rc], [p[1] for p in rc], label=f"{short[k].replace(chr(10), ' ')} ({r['uncertainty']['aurc_max_prob']:.3f})")
    ax[0].plot([0, 1], [0, 1], color="#999", lw=1)
    ax[0].set(title="Reliability after temperature scaling (test)", xlabel="confidence", ylabel="accuracy")
    ax[1].set(title="Risk-coverage (AURC, lower is better)", xlabel="coverage", ylabel="error rate of answered")
    ax[1].legend(fontsize=7)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "uncertainty.png", dpi=110)


if __name__ == "__main__":
    main()
