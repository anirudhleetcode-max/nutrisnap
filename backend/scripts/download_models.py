"""Fetch and prepare the CLIP ViT-B/32 ONNX files in backend/models/ (about 345 MB on disk).

    python -m scripts.download_models

Source: https://huggingface.co/Xenova/clip-vit-base-patch32 (MIT-licensed OpenAI CLIP weights,
exported to ONNX).

- Text tower: the full-precision export. It only runs once, to build the cached label embeddings,
  and the 8-bit export loses ~22 points of zero-shot accuracy.
- Vision tower (runs per request): the full-precision export (352 MB) is downloaded, quantised here
  with onnxruntime's dynamic quantisation (int8 weights, per-channel; ~15 s), and then deleted.
  The result is 2.2x faster on CPU than the 8-bit export Xenova publishes, and closer to fp32
  (experiments/results/2026-11-26_onnx). Quantisation is deterministic for a given onnx/onnxruntime
  version; the printed SHA-256 can be compared with `vision_sha256` in artifacts/recognizer.json.

Safe to re-run; finished files are skipped.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from app.config import get_settings

BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {
    "text_model.onnx": "onnx/text_model.onnx",
    "tokenizer.json": "tokenizer.json",
}
VISION = "vision_model_int8.onnx"
VISION_FP32 = "onnx/vision_model.onnx"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NutriSnap/1.0 model-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(done * 10 / total) if total else -1
            if pct != last:  # one line per 10%, so CI logs stay short
                print(f"  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", flush=True)
                last = pct
    shutil.move(tmp, dest)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def quantise_vision(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    fp32 = out / "vision_model_fp32.onnx"
    if not fp32.exists():
        print("downloading the fp32 vision model (deleted after quantisation)")
        download(BASE + VISION_FP32, fp32)
    print("quantising the vision model (int8, per-channel)")
    tmp = out / (VISION + ".part")
    quantize_dynamic(str(fp32), str(tmp), weight_type=QuantType.QInt8, per_channel=True)
    shutil.move(tmp, out / VISION)
    fp32.unlink()


def main() -> int:
    out = get_settings().model_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name, remote in FILES.items():
            dest = out / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {name}: already there")
                continue
            print(f"downloading {name}")
            download(BASE + remote, dest)
        if (out / VISION).exists():
            print(f"  {VISION}: already there")
        else:
            quantise_vision(out)
    except Exception as e:  # noqa: BLE001
        print(f"failed: {e}")
        return 1
    print(f"  {VISION} sha256 {sha16(out / VISION)}")
    print(f"models ready in {out}. The label-embedding cache is built on first start (~15 s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
