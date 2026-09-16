"""Input validation and CLIP preprocessing.

Validation happens before any heavy decoding:
  1. size limit (bytes) - enforced by the API
  2. magic bytes: only JPEG, PNG and WebP signatures are accepted, whatever the client claims
  3. pixel limit: the header is parsed (lazy open) and images above MAX_PIXELS are refused before
     the pixel data is decompressed (protects against decompression bombs)
  4. minimum size
"""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224
PREPROCESS_VERSION = "clip-224-bicubic-centercrop-v1"
MAX_PIXELS = 40_000_000   # a 48 MP phone photo is ~48M; 40M covers common phones after downscaling
MIN_SIDE = 32


class ImageRejected(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def sniff(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(data: bytes, max_pixels: int = MAX_PIXELS) -> Image.Image:
    if not data:
        raise ImageRejected(400, "The uploaded file is empty")
    kind = sniff(data)
    if kind is None:
        raise ImageRejected(415, "Only JPEG, PNG or WebP photos are supported")
    try:
        img = Image.open(io.BytesIO(data), formats=[kind.upper()])
        w, h = img.size
        if w * h > max_pixels:
            raise ImageRejected(413, f"Image is {w}x{h}; the limit is {max_pixels // 1_000_000} megapixels")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img.load()
    except ImageRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageRejected(413, "Image has too many pixels")
    except Exception:  # truncated / corrupt files raise OSError, SyntaxError, ValueError...
        raise ImageRejected(400, "That file doesn't look like an image we can read")
    img = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF
    if min(img.size) < MIN_SIDE:
        raise ImageRejected(400, f"Image is too small (minimum {MIN_SIDE} px per side)")
    return img.convert("RGB")


def clip_tensor(img: Image.Image) -> np.ndarray:
    """CLIP preprocessing: shortest side -> 224 (bicubic), centre crop, scale to 0-1, normalise, CHW."""
    img = img.convert("RGB")
    w, h = img.size
    scale = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * scale)), max(SIZE, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


def quality_warnings(img: Image.Image) -> list[str]:
    """Cheap, rule-based photo checks (not ML). Thresholds picked by eye on a few photos."""
    g = np.asarray(img.convert("L").resize((224, 224)), dtype=np.float32)
    out = []
    if g.mean() < 35:
        out.append("too_dark")
    elif g.mean() > 235:
        out.append("overexposed")
    # variance of a discrete Laplacian: low values mean little edge detail (blur)
    lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    if lap.var() < 20:
        out.append("blurry")
    return out


def crops(img: Image.Image) -> list[Image.Image]:
    """2x2 grid + a centre crop, used for the multi-item hint."""
    w, h = img.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h),
             (w // 5, h // 5, w - w // 5, h - h // 5)]
    return [img.crop(b) for b in boxes]
