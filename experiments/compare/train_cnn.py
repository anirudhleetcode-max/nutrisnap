"""Train / fit the conventional image classifiers for the comparison.

    cd backend
    pip install -r ../experiments/compare/requirements.txt   # torch + torchvision (CPU is fine)
    python ../experiments/compare/train_cnn.py --model scratch
    python ../experiments/compare/train_cnn.py --model resnet50_frozen
    python ../experiments/compare/train_cnn.py --model resnet50_finetune

Models (all predict the same 45 classes):
  scratch            A small VGG-style CNN (5 conv blocks, batch norm, global average pooling),
                     random initialisation, trained only on our 1,800 training photos.
  resnet50_frozen    Transfer learning by feature extraction: ImageNet-pretrained ResNet-50 with
                     the classifier removed, 2048-d pooled features, logistic regression on top
                     (C chosen on val). The exact CNN counterpart of the CLIP linear probe.
  resnet50_finetune  Transfer learning by fine-tuning: same ResNet-50, new 45-way head, the last
                     residual stage (layer4) and the head trained with augmentation.

Model selection (epoch, C) uses val only. Writes
experiments/results/<date>_compare/cnn/<model>.npz  (45-way probabilities for val/test/unseen)
                                       <model>.json (training log, sizes, timings)
and keeps the fine-tuned weights in backend/models/cnn/ (not committed) for the ONNX study.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import BACKEND, ROOT, load, pixels

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- models
class SmallCNN(nn.Module):
    """Plain conv net: stride-2 stem, [conv-BN-ReLU x2 -> maxpool] x 4, global average pool, dropout, linear."""

    def __init__(self, n_classes: int, widths=(32, 64, 128, 256, 256), dropout: float = 0.3):
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)]
        c_in = widths[0]
        for w in widths[1:]:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet50(n_classes: int | None):
    from torchvision.models import ResNet50_Weights, resnet50 as rn50
    m = rn50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity() if n_classes is None else nn.Linear(2048, n_classes)
    return m


def n_params(m: nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)


# ---------------------------------------------------------------- tensors & augmentation
def to_tensor(u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def center(u8: np.ndarray, size: int) -> torch.Tensor:
    """Eval transform: centre crop 224 of the 256 image (the standard 0.875 crop), then resize."""
    s = u8.shape[1]
    c = int(round(s * 0.875))
    o = (s - c) // 2
    x = to_tensor(u8[:, o:o + c, o:o + c])
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True) if c != size else x


def augment(u8: np.ndarray, size: int, rng: np.random.Generator) -> torch.Tensor:
    """Random-resized crop (area 35-100%, aspect 3/4-4/3), horizontal flip, mild brightness/contrast."""
    out = []
    S = u8.shape[1]
    for im in u8:
        for _ in range(10):
            area = rng.uniform(0.35, 1.0) * S * S
            ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            w, h = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
            if w <= S and h <= S:
                break
        else:
            w = h = S
        x0, y0 = rng.integers(0, S - w + 1), rng.integers(0, S - h + 1)
        crop = im[y0:y0 + h, x0:x0 + w]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None].float() / 255.0
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        t = (t - t.mean()) * rng.uniform(0.8, 1.2) + t.mean() * rng.uniform(0.8, 1.2)
        out.append(t.clamp(0, 1))
    x = torch.cat(out)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def predict(model: nn.Module, u8: np.ndarray, size: int, bs: int = 64) -> np.ndarray:
    model.eval()
    return np.concatenate([F.softmax(model(center(u8[i:i + bs], size)), 1).numpy() for i in range(0, len(u8), bs)])


@torch.no_grad()
def latency_ms(model: nn.Module, size: int, n: int = 30) -> float:
    model.eval()
    x = torch.randn(1, 3, size, size)
    for _ in range(5):
        model(x)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model(x)
        ts.append(time.perf_counter() - t)
    return round(float(np.median(ts)) * 1000, 2)


def top1_accept(P: np.ndarray, accept: list[set[int]]) -> float:
    return float(np.mean([p in a for p, a in zip(P.argmax(1), accept)]))


# ---------------------------------------------------------------- training loop
def train(model, params, X_tr, y_tr, X_va, acc_va, *, size, epochs, lr, wd, bs, seed, log, label_smoothing=0.1):
    rng = np.random.default_rng(seed)
    if isinstance(lr, list):  # one learning rate per parameter group
        params = [{**g, "lr": v} for g, v in zip(params, lr)]
    opt = torch.optim.AdamW(params, lr=lr[0] if isinstance(lr, list) else lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(X_tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    y_t = torch.from_numpy(y_tr).long()
    best = (-1.0, None, -1)
    for ep in range(epochs):
        model.train()
        t0, perm, tot, n = time.time(), rng.permutation(len(X_tr)), 0.0, 0
        for i in range(0, len(perm), bs):
            ix = perm[i:i + bs]
            loss = F.cross_entropy(model(augment(X_tr[ix], size, rng)), y_t[ix], label_smoothing=label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
            n += len(ix)
        va = top1_accept(predict(model, X_va, size), acc_va)
        log.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "val_top1": round(va, 4), "seconds": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
        if va > best[0]:
            best = (va, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    return best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["scratch", "resnet50_frozen", "resnet50_finetune"])
    ap.add_argument("--run", default=None, help="results folder name (default <today>_compare)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = ROOT / "experiments" / "results" / (args.run or f"{datetime.now(timezone.utc):%Y-%m-%d}_compare") / "cnn"
    out.mkdir(parents=True, exist_ok=True)

    d = load()
    C = len(d.classes)
    X = {s: pixels(getattr(d, s), s) for s in ("train", "val", "test", "unseen")}
    print({k: v.shape for k, v in X.items()}, flush=True)
    meta = {"model": args.model, "classes": C, "seed": args.seed, "threads": args.threads, "torch": torch.__version__,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "train_size": len(d.train.y)}
    t_start = time.time()

    if args.model == "scratch":
        size = 128
        model = SmallCNN(C)
        log: list[dict] = []
        epochs = args.epochs or 40
        best_ep = train(model, model.parameters(), X["train"], d.train.y, X["val"], d.val.accept,
                        size=size, epochs=epochs, lr=3e-3, wd=5e-4, bs=64, seed=args.seed, log=log)
        meta.update({"input_size": size, "epochs": epochs, "best_epoch": best_ep, "log": log,
                     "optimizer": "AdamW lr 3e-3 one-cycle, wd 5e-4, batch 64, label smoothing 0.1",
                     "params": n_params(model), "trainable_params": n_params(model, True),
                     "pretrained": "none (random init)"})
        P = {s: predict(model, X[s], size) for s in ("val", "test", "unseen")}

    elif args.model == "resnet50_frozen":
        from sklearn.linear_model import LogisticRegression
        size = 224
        model = resnet50(None)
        model.eval()
        with torch.no_grad():
            feats = {s: np.concatenate([model(center(X[s][i:i + 64], size)).numpy()
                                        for i in range(0, len(X[s]), 64)]) for s in X}
        np.savez_compressed(out / "resnet50_features.npz", **feats)
        mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-6
        Z = {s: (f - mu) / sd for s, f in feats.items()}
        search, best = [], None
        for Cr in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            clf = LogisticRegression(C=Cr, max_iter=3000, random_state=args.seed).fit(Z["train"], d.train.y)
            va = top1_accept(clf.predict_proba(Z["val"]), d.val.accept)
            search.append({"C": Cr, "val_top1": round(va, 4)})
            print(search[-1], flush=True)
            if best is None or va > best[0]:
                best = (va, clf, Cr)
        clf = best[1]
        meta.update({"input_size": size, "C_search": search, "C": best[2], "feature_dim": 2048,
                     "pretrained": "torchvision ResNet50_Weights.IMAGENET1K_V2 (ImageNet-1k, 1.28M labelled photos)",
                     "params": n_params(model) + clf.coef_.size + clf.intercept_.size,
                     "trainable_params": clf.coef_.size + clf.intercept_.size})
        P = {s: clf.predict_proba(Z[s]) for s in ("val", "test", "unseen")}

    else:  # resnet50_finetune
        size = 224
        model = resnet50(C)
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("layer4", "fc"))
        # keep the frozen stages' batch-norm statistics fixed (ImageNet); only layer4's adapt
        orig_train = model.train

        def train_mode(mode: bool = True):
            orig_train(mode)
            for n_, m in model.named_modules():
                if isinstance(m, nn.BatchNorm2d) and not n_.startswith("layer4"):
                    m.eval()
            return model
        model.train = train_mode  # type: ignore[method-assign]
        log = []
        epochs = args.epochs or 10
        groups = [{"params": model.layer4.parameters()}, {"params": model.fc.parameters()}]
        best_ep = train(model, groups, X["train"], d.train.y, X["val"], d.val.accept,
                        size=size, epochs=epochs, lr=[3e-4, 3e-3], wd=1e-4, bs=32, seed=args.seed, log=log)
        meta.update({"input_size": size, "epochs": epochs, "best_epoch": best_ep, "log": log,
                     "optimizer": "AdamW one-cycle, lr 3e-4 (layer4) / 3e-3 (head), wd 1e-4, batch 32, label smoothing 0.1",
                     "pretrained": "torchvision ResNet50_Weights.IMAGENET1K_V2", "trained_layers": "layer4 + fc",
                     "params": n_params(model), "trainable_params": n_params(model, True)})
        P = {s: predict(model, X[s], size) for s in ("val", "test", "unseen")}
        wdir = BACKEND / "models" / "cnn"
        wdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), wdir / "resnet50_finetune.pt")

    meta["train_seconds"] = round(time.time() - t_start, 1)
    meta["latency_ms_torch_batch1"] = latency_ms(model, meta["input_size"])
    meta["val_top1"] = round(top1_accept(P["val"], d.val.accept), 4)
    meta["test_top1"] = round(top1_accept(P["test"], d.test.accept), 4)
    meta["classes_list"] = d.classes
    np.savez_compressed(out / f"{args.model}.npz", **P)
    (out / f"{args.model}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("log", "classes_list")}, indent=1))


if __name__ == "__main__":
    main()
