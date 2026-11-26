"""Fetch the CLIP ViT-B/32 ONNX files (about 345 MB) into backend/models/.

    python -m scripts.download_models

Source: https://huggingface.co/Xenova/clip-vit-base-patch32 (MIT-licensed OpenAI CLIP weights,
exported to ONNX). The vision tower is the 8-bit dynamically quantised export (runs per request);
the text tower is the full-precision export, because the 8-bit one loses ~22 points of zero-shot
accuracy and it only runs once to build the cached label embeddings. Safe to re-run; finished files
are skipped.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from app.config import get_settings

BASE = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {
    "vision_model_quantized.onnx": "onnx/vision_model_quantized.onnx",
    "text_model.onnx": "onnx/text_model.onnx",
    "tokenizer.json": "tokenizer.json",
}


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "NutriSnap/1.0 model-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    shutil.move(tmp, dest)


def main() -> int:
    out = get_settings().model_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {name}: already there")
            continue
        print(f"downloading {name}")
        try:
            download(BASE + remote, dest)
        except Exception as e:  # noqa: BLE001
            print(f"failed: {e}")
            return 1
    print(f"models ready in {out}. The label-embedding cache is built on first start (~15 s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
