"""Run one evaluation experiment.

    cd backend
    python -m ml.datasets              # once: sample Food-101 + Indian Foods
    python -m ml.fetch_testset         # once: Commons OOD / not-food images
    python ../experiments/run.py --config ../experiments/configs/main.yaml

Writes experiments/results/<run_id>/ (metrics.json, config snapshot, env, errors, plots) and, if
the config says so, backend/artifacts/recognizer.json + the linear probe.

Everything that is *chosen* (probe C, hybrid weight, temperature, abstention threshold, not-food
threshold, zero-shot vs hybrid) is chosen on the validation split. The test split is only scored.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.ml.calibration import (apply_temperature, brier_top1, coverage_curve, fit_temperature, nll,  # noqa: E402
                                pick_threshold, reliability, softmax, top1)
from app.ml.clip_model import LOGIT_SCALE, ClipEncoder, labels_from_foods  # noqa: E402
from app.ml.personalize import blend, knn_probs  # noqa: E402
from app.ml.pipeline import hybrid  # noqa: E402
from app.ml.probe import LinearProbe, train_probe  # noqa: E402
from ml.datasets import PROCESSED  # noqa: E402
from ml.embeddings import embed_rows, load_set  # noqa: E402


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "--short", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def env_info() -> dict:
    import onnxruntime
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "onnxruntime": onnxruntime.__version__, "scikit_learn": sklearn.__version__, **git_info()}


def scores(P: np.ndarray, accept: list[set[int]]) -> dict:
    order = np.argsort(-P, axis=1)[:, :5]
    t1 = np.array([order[i, 0] in a for i, a in enumerate(accept)])
    t5 = np.array([bool(set(order[i]) & a) for i, a in enumerate(accept)])
    return {"top1": round(float(t1.mean()), 4), "top5": round(float(t5.mean()), 4), "n": len(accept), "_t1": t1}


def macro(t1: np.ndarray, groups: list[str]) -> float:
    by = defaultdict(list)
    for ok, g in zip(t1, groups):
        by[g].append(ok)
    return round(float(np.mean([np.mean(v) for v in by.values()])), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    random.seed(seed)
    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y-%m-%d}_{cfg['run_name']}"
    out = ROOT / "experiments" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)

    foods = json.loads((BACKEND / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    enc = ClipEncoder(BACKEND / "models", labels_from_foods(foods), BACKEND / "models" / "cache")
    ids = enc.ids
    index = {f: i for i, f in enumerate(ids)}
    M_single = enc.build_label_matrix(ensemble=False)
    M_ens = enc.label_matrix

    # ---------------- data ----------------
    rows, embs, latency = [], [], {}
    for name in cfg["datasets"]:
        r = load_set(name)
        e, ms = embed_rows(enc, r, name)
        rows += r
        embs.append(e)
        latency[name] = ms
    E = np.concatenate(embs)
    split = np.array([r["split"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    cls = [f"{r['dataset']}/{r['source_class']}" for r in rows]
    accept = [{index[a] for a in r["accept"]} for r in rows]
    primary = [r["accept"][0] for r in rows]

    ood_rows = load_set(cfg["ood_set"])
    E_ood, ms_ood = embed_rows(enc, ood_rows, cfg["ood_set"])
    latency[cfg["ood_set"]] = ms_ood
    ood_food = np.array([r["source_class"] != "not-food" for r in ood_rows])
    ood_accept = [{index[a] for a in r["accept"]} for r, f in zip(ood_rows, ood_food) if f]

    def sel(s):
        return np.where(split == s)[0]

    tr, va, te = sel("train"), sel("val"), sel("test")
    A = lambda ix: [accept[i] for i in ix]  # noqa: E731
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  ood food {ood_food.sum()}  ood non-food {(~ood_food).sum()}")

    # ---------------- models ----------------
    zs_single = softmax(LOGIT_SCALE * E @ M_single.T)
    zs_ens = softmax(LOGIT_SCALE * E @ M_ens.T)

    probe_classes = sorted(set(primary))
    pmask = np.isin(ids, probe_classes)
    closed_cols = np.where(pmask)[0]

    def closed(P):
        """Restrict a 188-way distribution to the probe's classes (same label space as the probe)."""
        Q = np.zeros_like(P)
        Q[:, closed_cols] = P[:, closed_cols]
        return Q / Q.sum(axis=1, keepdims=True)

    # majority-class baseline (most frequent primary label in train)
    maj = Counter(primary[i] for i in tr).most_common(1)[0][0]
    P_maj = np.zeros_like(zs_ens)
    P_maj[:, index[maj]] = 1.0
    others = [c for c in probe_classes if c != maj]
    P_maj[:, [index[c] for c in others[:4]]] = 1e-3  # arbitrary fixed runners-up for top-5

    # linear probe, C chosen on val
    Xtr, ytr = E[tr], [primary[i] for i in tr]
    probe_search = []
    best = None
    for C in cfg["probe"]["C_grid"]:
        pr = train_probe(Xtr, ytr, C, seed)
        s = scores(pr.expand(pr.predict_proba(E[va]), ids), A(va))
        probe_search.append({"C": C, "val_top1": s["top1"], "val_top5": s["top5"]})
        if best is None or s["top1"] > best[1]:
            best = (C, s["top1"], pr)
    C_best, _, probe = best
    P_probe = probe.expand(probe.predict_proba(E), ids)
    print("probe C search:", probe_search)

    # hybrid weight chosen on val
    hybrid_search = []
    for w in cfg["hybrid"]["weight_grid"]:
        s = scores(hybrid(zs_ens[va], P_probe[va], pmask, w), A(va))
        hybrid_search.append({"weight": w, "val_top1": s["top1"], "val_top5": s["top5"]})
    w_best = max(hybrid_search, key=lambda h: (h["val_top1"], -h["weight"]))["weight"]
    P_hyb = hybrid(zs_ens, P_probe, pmask, w_best)

    models = {
        "majority_class": P_maj,
        "zero_shot_single_prompt": zs_single,
        "zero_shot_ensemble": zs_ens,
        "zero_shot_ensemble_closed_set": closed(zs_ens),
        "linear_probe_closed_set": P_probe,
        "hybrid_ensemble_plus_probe": P_hyb,
    }
    table = {}
    for name, P in models.items():
        table[name] = {}
        for sname, ix in (("val", va), ("test", te)):
            s = scores(P[ix], A(ix))
            entry = {"top1": s["top1"], "top5": s["top5"], "macro_top1": macro(s["_t1"], [cls[i] for i in ix]), "n": s["n"]}
            for d in cfg["datasets"]:
                m = ds[ix] == d
                sd = scores(P[ix][m], [A(ix)[k] for k in np.where(m)[0]])
                entry[d] = {"top1": sd["top1"], "top5": sd["top5"], "n": sd["n"]}
            table[name][sname] = entry
    # OOD (Commons): classes partly unseen by the probe, 188-way models only
    zs_ood_single = softmax(LOGIT_SCALE * E_ood @ M_single.T)[ood_food]
    zs_ood = softmax(LOGIT_SCALE * E_ood @ M_ens.T)[ood_food]
    P_ood_probe = probe.expand(probe.predict_proba(E_ood[ood_food]), ids)
    ood = {
        "zero_shot_single_prompt": scores(zs_ood_single, ood_accept),
        "zero_shot_ensemble": scores(zs_ood, ood_accept),
        "hybrid_ensemble_plus_probe": scores(hybrid(zs_ood, P_ood_probe, pmask, w_best), ood_accept),
    }
    ood_seen = np.array([bool(a & set(np.where(pmask)[0])) for a in ood_accept])
    for k in ood:
        t1 = ood[k].pop("_t1")
        ood[k]["top1_classes_seen_by_probe"] = round(float(t1[ood_seen].mean()), 4)
        ood[k]["top1_classes_unseen_by_probe"] = round(float(t1[~ood_seen].mean()), 4)
    ood["n_seen"], ood["n_unseen"] = int(ood_seen.sum()), int((~ood_seen).sum())

    chosen = "hybrid_ensemble_plus_probe" if (
        table["hybrid_ensemble_plus_probe"]["val"]["top1"] > table["zero_shot_ensemble"]["val"]["top1"] and w_best > 0
    ) else "zero_shot_ensemble"
    P = models[chosen]
    print("chosen:", chosen, "hybrid weight", w_best)

    # ---------------- calibration ----------------
    T = fit_temperature(P[va], A(va))
    cal = {"temperature": round(T, 4)}
    rel_tables = {}
    for sname, ix in (("val", va), ("test", te)):
        for tag, Q in (("before", P[ix]), ("after", apply_temperature(P[ix], T))):
            conf, corr = top1(Q, A(ix))
            ece, tab = reliability(conf, corr)
            cal[f"{sname}_{tag}"] = {"ece": round(ece, 4), "brier_top1": round(brier_top1(conf, corr), 4),
                                     "nll": round(nll(Q, A(ix)), 4), "mean_confidence": round(float(conf.mean()), 4),
                                     "accuracy": round(float(corr.mean()), 4)}
            rel_tables[f"{sname}_{tag}"] = tab

    # ---------------- abstention ----------------
    Pc = apply_temperature(P, T)
    conf_va, corr_va = top1(Pc[va], A(va))
    conf_te, corr_te = top1(Pc[te], A(te))
    target = cfg["abstention"]["target_accuracy"]
    thr = pick_threshold(conf_va, corr_va, target)
    m_te = conf_te >= thr
    P_ood_final = hybrid(zs_ood, P_ood_probe, pmask, w_best) if chosen.startswith("hybrid") else zs_ood
    conf_o, corr_o = top1(apply_temperature(P_ood_final, T), ood_accept)
    m_o = conf_o >= thr
    abst = {
        "target_accuracy_on_val": target, "threshold": thr,
        "val": {"coverage": round(float((conf_va >= thr).mean()), 4),
                "accuracy_when_answering": round(float(corr_va[conf_va >= thr].mean()), 4)},
        "test": {"coverage": round(float(m_te.mean()), 4),
                 "accuracy_when_answering": round(float(corr_te[m_te].mean()), 4) if m_te.any() else None,
                 "accuracy_when_abstaining": round(float(corr_te[~m_te].mean()), 4) if (~m_te).any() else None,
                 "top5_when_abstaining": round(float(np.mean([bool(set(np.argsort(-Pc[te][i])[:5]) & A(te)[i])
                                                              for i in np.where(~m_te)[0]])), 4) if (~m_te).any() else None},
        "ood_commons": {"coverage": round(float(m_o.mean()), 4),
                        "accuracy_when_answering": round(float(corr_o[m_o].mean()), 4) if m_o.any() else None},
        "curve_val": coverage_curve(conf_va, corr_va),
        "curve_test": coverage_curve(conf_te, corr_te),
    }

    # ---------------- not-food guard ----------------
    nf = np.where(~ood_food)[0]
    perm = rng.permutation(nf)
    k = int(len(perm) * cfg["not_food"]["val_fraction"])
    nf_val, nf_test = perm[:k], perm[k:]
    fs_food_va, fs_food_te = enc.food_score(E[va]), enc.food_score(E[te])
    fs_nf = enc.food_score(E_ood)

    def guard_at(t, food_scores, nonfood_scores):
        return {"non_food_flagged": round(float((nonfood_scores < t).mean()), 4),
                "food_wrongly_flagged": round(float((food_scores < t).mean()), 4)}
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    bal = [((fs_nf[nf_val] < t).mean() + (fs_food_va >= t).mean()) / 2 for t in grid]
    t_nf = float(grid[int(np.argmax(bal))])
    guard = {"n_non_food_val": len(nf_val), "n_non_food_test": len(nf_test), "n_food_test": len(te),
             "threshold_chosen_on_val": t_nf,
             "test_at_chosen": guard_at(t_nf, fs_food_te, fs_nf[nf_test]),
             "test_at_0.5_old_default": guard_at(0.5, fs_food_te, fs_nf[nf_test])}
    try:
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones(len(te)), np.zeros(len(nf_test))]
        guard["test_roc_auc"] = round(float(roc_auc_score(y, np.r_[fs_food_te, fs_nf[nf_test]])), 4)
    except Exception:
        pass

    # ---------------- personalisation (simulated) ----------------
    base_tr = scores(P[tr], A(tr))["_t1"]
    wrong_tr = tr[~base_tr]
    pers = {"description": "memory = train-split photos the chosen model got wrong, labelled with their true class; "
                           "scored on the test split (different photos). Simulates a user who corrected those photos."}
    for label, cap in (("up_to_3_corrections_per_class", 3), ("all_corrections", None)):
        per = defaultdict(list)
        for i in wrong_tr:
            per[primary[i]].append(i)
        mem = [i for v in per.values() for i in (v if cap is None else v[:cap])]
        ME, ML = E[mem], np.array([index[primary[i]] for i in mem])
        Pp, fired = [], 0
        for i in te:
            kp, a = knn_probs(E[i], ME, ML, len(ids))
            fired += a > 0
            Pp.append(blend(Pc[i], kp, a))
        s = scores(np.stack(Pp), A(te))
        pers[label] = {"memory_size": len(mem), "fired_on": int(fired), "test_top1": s["top1"], "test_top5": s["top5"]}
    pers["test_top1_without"] = scores(Pc[te], A(te))["top1"]

    # ---------------- error analysis ----------------
    def confusions(Q, ix):
        c = Counter()
        pred = Q[ix].argmax(axis=1)
        for i, p in zip(ix, pred):
            if p not in accept[i]:
                c[(cls[i], ids[p])] += 1
        totals = Counter(cls[i] for i in ix)
        return [{"true_class": t, "predicted": p, "count": n, "share_of_class": round(n / totals[t], 3)}
                for (t, p), n in c.most_common(25)]
    conf_all = Pc[te].max(axis=1)
    worst = []
    for j in np.argsort(-conf_all):
        i = te[j]
        p = int(Pc[i].argmax())
        if p not in accept[i]:
            worst.append({"row_id": rows[i]["row_id"], "true_class": cls[i], "predicted": ids[p],
                          "confidence": round(float(conf_all[j]), 3)})
        if len(worst) == 30:
            break
    per_class = {}
    t1_te = scores(Pc[te], A(te))["_t1"]
    zs_t1 = scores(zs_ens[te], A(te))["_t1"]
    for c in sorted(set(cls[i] for i in te)):
        m = np.array([cls[i] == c for i in te])
        per_class[c] = {"n": int(m.sum()), "zero_shot_top1": round(float(zs_t1[m].mean()), 3),
                        "chosen_top1": round(float(t1_te[m].mean()), 3)}
    errors = {"model": chosen, "split": "test",
              "top_confusions_chosen": confusions(Pc, te),
              "top_confusions_zero_shot_ensemble": confusions(zs_ens, te),
              "most_confident_mistakes": worst, "per_class_top1": per_class}
    with open(out / "confusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "true_class", "predicted", "count", "share_of_class"])
        for key in ("top_confusions_chosen", "top_confusions_zero_shot_ensemble"):
            for r in errors[key]:
                w.writerow([key.replace("top_confusions_", ""), r["true_class"], r["predicted"], r["count"], r["share_of_class"]])

    # ---------------- plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for tag, style in (("before", "o--"), ("after", "o-")):
            tab = [b for b in rel_tables[f"test_{tag}"] if b["n"]]
            ax[0].plot([b["confidence"] for b in tab], [b["accuracy"] for b in tab], style,
                       label=f"{tag} (ECE {cal[f'test_{tag}']['ece']:.3f})")
        ax[0].plot([0, 1], [0, 1], color="#999", lw=1)
        ax[0].set(title="Reliability (test)", xlabel="confidence", ylabel="accuracy")
        ax[0].legend()
        cv = abst["curve_test"]
        ax[1].plot([c["coverage"] for c in cv], [c["accuracy"] or 0 for c in cv], "o-", color="#c8442b")
        ax[1].axvline(abst["test"]["coverage"], color="#555", ls=":", label=f"threshold {thr:.2f}")
        ax[1].set(title="Coverage vs accuracy (test)", xlabel="coverage", ylabel="accuracy when answering")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out / "calibration.png", dpi=110)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    # ---------------- write ----------------
    manifest = json.loads((PROCESSED / "manifest.json").read_text()) if (PROCESSED / "manifest.json").exists() else {}
    metrics = {
        "run_id": run_id, "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
        "data": {"splits": {"train": len(tr), "val": len(va), "test": len(te)},
                 "classes": len(set(cls)), "probe_classes": len(probe_classes), "food_list_size": len(ids),
                 "ood_commons": {"food": int(ood_food.sum()), "non_food": int((~ood_food).sum())},
                 "manifest": manifest, "synthetic": False},
        "models": table, "ood_commons": ood,
        "probe": {"C_search": probe_search, "C": C_best, "hybrid_search": hybrid_search, "hybrid_weight": w_best},
        "chosen_model": chosen, "calibration": cal, "reliability": rel_tables, "abstention": abst,
        "not_food_guard": guard, "personalisation_simulated": pers,
        "latency_ms_vision_median": latency,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "errors.json").write_text(json.dumps(errors, indent=2))
    (out / "config.yaml").write_text(Path(args.config).read_text())
    (out / "env.json").write_text(json.dumps(env_info(), indent=2))

    if cfg.get("export_artifacts"):
        art = BACKEND / "artifacts"
        art.mkdir(exist_ok=True)
        summary = {
            "test_top1": table[chosen]["test"]["top1"], "test_top5": table[chosen]["test"]["top5"],
            "test_top1_zero_shot_single_prompt": table["zero_shot_single_prompt"]["test"]["top1"],
            "test_ece_before": cal["test_before"]["ece"], "test_ece_after": cal["test_after"]["ece"],
            "abstain_threshold": thr, "test_coverage": abst["test"]["coverage"],
            "test_accuracy_when_answering": abst["test"]["accuracy_when_answering"],
            "ood_commons_top1": ood[chosen]["top1"] if chosen in ood else None,
            "not_food_test": guard["test_at_chosen"],
        }
        datasets = {"train/val/test": "Food-101 (30 mapped classes) + Indian Food Images (15 classes), real photos",
                    "sizes": metrics["data"]["splits"], "ood": "Wikimedia Commons hand-picked set", "manifest": manifest}
        probe_file = None
        if chosen.startswith("hybrid"):
            probe_file = "probe_v1.npz"
            probe.card = {"model": "multinomial logistic regression on frozen CLIP ViT-B/32 image embeddings",
                          "trained_at": started.isoformat(timespec="seconds"), "run_id": run_id,
                          "classes": probe.classes, "train_size": len(tr), "C": C_best, "seed": seed,
                          "dataset": datasets, "synthetic": False, "env": env_info(),
                          "val_top1_closed_set": table["linear_probe_closed_set"]["val"]["top1"],
                          "test_top1_closed_set": table["linear_probe_closed_set"]["test"]["top1"]}
            probe.save(art / probe_file)
        rec = {"run_id": run_id, "created_at": started.isoformat(timespec="seconds"),
               "settings": {"version": f"{run_id}", "mode": "hybrid" if probe_file else "zero_shot",
                            "temperature": round(T, 4), "abstain_below": thr, "not_food_below": t_nf,
                            "hybrid_weight": w_best if probe_file else 0.0, "probe_file": probe_file,
                            "calibrated": True},
               "metrics": summary, "datasets": datasets, "env": env_info()}
        (art / "recognizer.json").write_text(json.dumps(rec, indent=2))

    brief = {k: v["test"] for k, v in table.items()}
    print(json.dumps({"test": {k: {"top1": v["top1"], "top5": v["top5"]} for k, v in brief.items()},
                      "ood": {k: v for k, v in ood.items() if isinstance(v, dict)},
                      "calibration": cal, "abstention": {k: abst[k] for k in ("threshold", "val", "test", "ood_commons")},
                      "guard": guard, "personalisation": pers, "latency": latency}, indent=1))
    print("results in", out)


if __name__ == "__main__":
    main()
