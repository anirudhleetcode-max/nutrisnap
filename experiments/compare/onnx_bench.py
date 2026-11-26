"""ONNX Runtime optimisation study: precision, graph optimisation, threads, batching, and what each
costs in accuracy.

    cd backend
    python ../experiments/compare/onnx_bench.py            # after compare.py (uses its splits)

CLIP vision tower variants (same OpenAI ViT-B/32 weights, Xenova ONNX exports + one made here):
  fp32           onnx/vision_model.onnx                 reference
  fp16           onnx/vision_model_fp16.onnx            half-precision weights
  int8_xenova    onnx/vision_model_quantized.onnx       what the app ships (8-bit dynamic quantisation)
  int8_here      fp32 quantised in this script with onnxruntime.quantization.quantize_dynamic
                 (QInt8 weights, per-channel), to show the step rather than trust a download

Also: the text tower fp32 vs int8, and the fine-tuned ResNet-50 from train_cnn.py exported from
PyTorch to ONNX, then statically quantised to INT8 (QDQ, calibrated on 128 training photos).

Accuracy impact is measured on the val split (900 photos): cosine similarity of each variant's
embedding to the fp32 one, and zero-shot / linear-probe top-1 using that variant's embeddings.
Latency: median of 40 runs after 5 warm-up runs, single process, this machine (recorded in env).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from data import BACKEND, ROOT, load, pixels

from app.ml.calibration import softmax  # noqa: E402
from app.ml.clip_model import EOS, LOGIT_SCALE, MAX_TOKENS, ClipEncoder, l2n, labels_from_foods  # noqa: E402
from app.ml.preprocess import clip_tensor, decode_image  # noqa: E402

HF = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/"
MD = BACKEND / "models"
OPT = MD / "optim"
LEVELS = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def fetch(name: str) -> Path:
    OPT.mkdir(parents=True, exist_ok=True)
    p = OPT / name
    if not p.exists():
        print("downloading", name, flush=True)
        req = urllib.request.Request(HF + name, headers={"User-Agent": "NutriSnap/1.0 experiments"})
        with urllib.request.urlopen(req, timeout=120) as r, open(p.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        p.with_suffix(".part").rename(p)
    return p


def session(path: Path, threads: int = 2, level: str = "all") -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.graph_optimization_level = LEVELS[level]
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def bench(sess: ort.InferenceSession, feed: dict, n: int = 40) -> dict:
    for _ in range(5):
        sess.run(None, feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, feed)
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(float(np.median(ts)), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2)}


def mb(p: Path) -> float:
    return round(p.stat().st_size / 1e6, 1)


def vision_feed(sess, x):
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    return {inp.name: x.astype(dtype)}


def embed(sess, X: np.ndarray) -> np.ndarray:
    out_name = "image_embeds"
    return l2n(np.stack([sess.run([out_name], vision_feed(sess, X[i:i + 1]))[0][0].astype(np.float32) for i in range(len(X))]))


def main() -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    started = datetime.now(timezone.utc)
    out = ROOT / "experiments" / "results" / f"{started:%Y-%m-%d}_onnx"
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"measured_at": started.isoformat(timespec="seconds"),
                 "machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "onnxruntime": ort.__version__,
                             "note": "shared cloud VM; absolute numbers vary between runs, ratios are the point"}}

    # ---------------------------------------------------------------- CLIP vision variants
    variants = {"fp32": fetch("vision_model.onnx"), "fp16": fetch("vision_model_fp16.onnx"),
                "int8_xenova": MD / "vision_model_quantized.onnx"}
    mine = OPT / "vision_model_int8_here.onnx"
    if not mine.exists():
        t = time.time()
        quantize_dynamic(str(variants["fp32"]), str(mine), weight_type=QuantType.QInt8, per_channel=True)
        res["int8_here_quantize_seconds"] = round(time.time() - t, 1)
    variants["int8_here"] = mine

    d = load()
    X_va = np.stack([clip_tensor(decode_image(r["image"])) for r in d.val.rows]).astype(np.float32)
    x1 = X_va[:1]
    vis = {}
    for k, p in variants.items():
        t = time.time()
        s = session(p, 2)
        load_s = time.time() - t
        g = onnx.load(str(p), load_external_data=False)
        ops: dict[str, int] = {}
        for node in g.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        vis[k] = {"file_mb": mb(p), "load_seconds": round(load_s, 2), "nodes": len(g.graph.node),
                  "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:6]),
                  "latency_batch1": {f"threads_{n}": bench(session(p, n), vision_feed(s, x1)) for n in (1, 2, 4)}}
        vis[k]["graph_optimisation_threads_2"] = {lv: bench(session(p, 2, lv), vision_feed(s, x1), 20) for lv in LEVELS}
        if k in ("fp32", "fp16"):  # batching is only safe without dynamic (per-batch) activation quantisation
            xb = X_va[:8]
            r8 = bench(session(p, 4), vision_feed(s, xb), 15)
            vis[k]["batch8_threads_4"] = {**r8, "ms_per_image": round(r8["median_ms"] / 8, 2)}
        print(k, vis[k]["file_mb"], "MB", vis[k]["latency_batch1"], flush=True)
    res["clip_vision"] = vis

    # accuracy impact on val
    foods = json.loads((BACKEND / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    enc = ClipEncoder(MD, labels_from_foods(foods), MD / "cache")
    cols = [enc.ids.index(c) for c in d.classes]
    W = enc.label_matrix
    E = {k: embed(session(p, 4), X_va) for k, p in variants.items()}
    from sklearn.linear_model import LogisticRegression
    X_tr = np.stack([clip_tensor(decode_image(r["image"])) for r in d.train.rows]).astype(np.float32)
    acc = {}
    for k in variants:
        E_tr = embed(session(variants[k], 4), X_tr)
        clf = LogisticRegression(C=10, max_iter=3000, random_state=42).fit(E_tr, d.train.y)
        zs45 = softmax(LOGIT_SCALE * E[k] @ W[cols].T)
        zs188 = softmax(LOGIT_SCALE * E[k] @ W.T)
        acc188 = np.mean([p in {enc.ids.index(a) for a in r["accept"]} for p, r in zip(zs188.argmax(1), d.val.rows)])
        acc[k] = {"cosine_to_fp32_mean": round(float((E[k] * E["fp32"]).sum(1).mean()), 5),
                  "cosine_to_fp32_min": round(float((E[k] * E["fp32"]).sum(1).min()), 5),
                  "zero_shot_45_val_top1": round(float(np.mean([p in a for p, a in zip(zs45.argmax(1), d.val.accept)])), 4),
                  "zero_shot_188_val_top1": round(float(acc188), 4),
                  "zero_shot_top1_agrees_with_fp32": round(float((zs188.argmax(1) == softmax(LOGIT_SCALE * E["fp32"] @ W.T).argmax(1)).mean()), 4),
                  "linear_probe_val_top1": round(float(np.mean([p in a for p, a in zip(clf.predict_proba(E[k]).argmax(1), d.val.accept)])), 4)}
        print(k, acc[k], flush=True)
    res["clip_vision_accuracy_val"] = acc

    # ---------------------------------------------------------------- CLIP text tower
    text = {"fp32": fetch("text_model.onnx"), "int8_xenova": MD / "text_model_quantized.onnx"}
    prompts = [p for lab in enc.labels[:40] for p in enc.prompts_for(lab, True)][:120]

    def tok(p):
        ids = enc.tokenizer.encode(p).ids[:MAX_TOKENS]
        if ids[-1] != EOS:
            ids[-1] = EOS
        return {"input_ids": np.array([ids], dtype=np.int64)}
    tres, temb = {}, {}
    for k, p in text.items():
        s = session(p, 2)
        temb[k] = l2n(np.stack([s.run(["text_embeds"], tok(q))[0][0] for q in prompts]).astype(np.float32))
        tres[k] = {"file_mb": mb(p), "latency_per_prompt_threads_2": bench(s, tok(prompts[0]))}
    tres["int8_xenova"]["cosine_to_fp32_mean"] = round(float((temb["fp32"] * temb["int8_xenova"]).sum(1).mean()), 5)
    tres["note"] = ("the text tower runs once per food-list change (188 foods x 7 prompts) and is cached, "
                    "so its speed does not affect request latency")
    res["clip_text"] = tres

    # ---------------------------------------------------------------- ResNet-50 (ours): PyTorch -> ONNX -> INT8
    wt = MD / "cnn" / "resnet50_finetune.pt"
    if wt.exists():
        import torch
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        from train_cnn import center, latency_ms, resnet50
        m = resnet50(len(d.classes))
        m.load_state_dict(torch.load(wt, map_location="cpu"))
        m.eval()
        torch.set_num_threads(2)
        fp32_path, pre_path, int8_path = OPT / "resnet50_ft_fp32.onnx", OPT / "resnet50_ft_pre.onnx", OPT / "resnet50_ft_int8.onnx"
        U_va, U_tr = pixels(d.val, "val"), pixels(d.train, "train")
        torch.onnx.export(m, torch.randn(1, 3, 224, 224), str(fp32_path), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        quant_pre_process(str(fp32_path), str(pre_path))
        calib_ix = np.random.default_rng(0).choice(len(U_tr), 128, replace=False)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"x": center(U_tr[i:i + 1], 224).numpy()} for i in calib_ix])

            def get_next(self):
                return next(self.it, None)
        quantize_static(str(pre_path), str(int8_path), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        Xv = center(U_va, 224).numpy()
        with torch.no_grad():
            ref = np.concatenate([m(torch.from_numpy(Xv[i:i + 64])).numpy() for i in range(0, len(Xv), 64)])
        rn = {"pytorch_eager": {"latency_batch1_threads_2_ms": latency_ms(m, 224),
                                "val_top1": round(float(np.mean([p in a for p, a in zip(ref.argmax(1), d.val.accept)])), 4)}}
        for k, p in (("onnx_fp32", fp32_path), ("onnx_int8_static", int8_path)):
            s = session(p, 2)
            lg = np.concatenate([s.run(None, {"x": Xv[i:i + 64]})[0] for i in range(0, len(Xv), 64)])
            rn[k] = {"file_mb": mb(p), "latency_batch1_threads_2": bench(s, {"x": Xv[:1]}),
                     "val_top1": round(float(np.mean([p_ in a for p_, a in zip(lg.argmax(1), d.val.accept)])), 4),
                     "top1_agrees_with_pytorch": round(float((lg.argmax(1) == ref.argmax(1)).mean()), 4),
                     "max_abs_logit_diff": round(float(np.abs(lg - ref).max()), 4)}
        rn["calibration_photos"] = 128
        res["resnet50_finetuned_export"] = rn
        print(json.dumps(rn, indent=1), flush=True)

    res["request_stages_ms"] = stage_breakdown()
    print("stages", res["request_stages_ms"], flush=True)
    (out / "onnx.json").write_text(json.dumps(res, indent=2))
    plot(out, res)
    print("results in", out)


def stage_breakdown(n: int = 30) -> dict:
    """Where the time goes in one /api/recognize call (whole image, app settings, 2 threads)."""
    from app.ml.pipeline import FoodRecognizer, hybrid
    from app.ml.calibration import apply_temperature
    from app.ml.preprocess import quality_warnings
    foods = json.loads((BACKEND / "data" / "foods.json").read_text(encoding="utf-8"))["foods"]
    rec = FoodRecognizer(MD, foods, BACKEND / "artifacts", threads=2)
    data = (BACKEND / "tests" / "fixtures" / "pizza.jpg").read_bytes()
    stages: dict[str, list[float]] = {k: [] for k in ("decode_validate", "preprocess", "vision_encoder_int8",
                                                      "zero_shot_188", "probe_and_hybrid", "temperature_decision",
                                                      "not_food_guard", "quality_checks")}
    for i in range(n + 3):
        t = [time.perf_counter()]
        img = decode_image(data); t.append(time.perf_counter())
        x = clip_tensor(img)[None].astype(np.float32); t.append(time.perf_counter())
        emb = l2n(rec.encoder.vision.run(["image_embeds"], {"pixel_values": x})[0])[0]; t.append(time.perf_counter())
        zs = rec.encoder.zero_shot(emb); t.append(time.perf_counter())
        p = hybrid(zs, rec.probe.expand(rec.probe.predict_proba(emb), rec.ids), rec.probe_mask,
                   float(rec.cfg["hybrid_weight"])) if rec.probe else zs; t.append(time.perf_counter())
        p = apply_temperature(p, float(rec.cfg["temperature"])); _ = p.max() >= rec.cfg["abstain_below"]; t.append(time.perf_counter())
        rec.encoder.food_score(emb); t.append(time.perf_counter())
        quality_warnings(img); t.append(time.perf_counter())
        if i >= 3:
            for k, a, b in zip(stages, t, t[1:]):
                stages[k].append((b - a) * 1000)
    med = {k: round(float(np.median(v)), 3) for k, v in stages.items()}
    med["total"] = round(sum(med.values()), 2)
    med["image"] = "tests/fixtures/pizza.jpg (decoded size as stored)"
    return med


def plot(out: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = res["clip_vision"]
    ks = list(vis)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, n in enumerate((1, 2, 4)):
        ax[0].bar(np.arange(len(ks)) + (i - 1) * 0.27, [vis[k]["latency_batch1"][f"threads_{n}"]["median_ms"] for k in ks], 0.27,
                  label=f"{n} thread{'s' if n > 1 else ''}")
    ax[0].set_xticks(range(len(ks)), ks, fontsize=8)
    ax[0].set(title="CLIP vision latency, batch 1", ylabel="ms (median)")
    ax[0].legend(fontsize=7)
    ax[1].bar(ks, [vis[k]["file_mb"] for k in ks], color="#4c78a8")
    ax[1].set(title="Model file size", ylabel="MB")
    ax[1].tick_params(axis="x", labelsize=8)
    acc = res["clip_vision_accuracy_val"]
    ax[2].bar(np.arange(len(ks)) - 0.2, [acc[k]["zero_shot_188_val_top1"] for k in ks], 0.4, label="zero-shot (188)")
    ax[2].bar(np.arange(len(ks)) + 0.2, [acc[k]["linear_probe_val_top1"] for k in ks], 0.4, label="linear probe (45)")
    ax[2].set_xticks(range(len(ks)), ks, fontsize=8)
    ax[2].set(title="Accuracy on val by precision", ylim=(0, 1))
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "onnx.png", dpi=110)


if __name__ == "__main__":
    main()
