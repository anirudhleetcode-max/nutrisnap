import io
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from bson import Binary
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from .config import get_settings
from .db import get_db
from .foods import by_id, catalogue, public_food
from .meals import oid
from .security import current_user
from .services.clip import ClipRecognizer, labels_from_foods, load_image
from .services.nutrition import meal_type_for_hour
from .services.personalize import blend, knn_probs

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recognition"])
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
NOT_FOOD_THRESHOLD = 0.5
THUMB_PX = 480

_recognizer: ClipRecognizer | None = None
_load_error: str | None = None


def load_model() -> None:
    global _recognizer, _load_error
    s = get_settings()
    try:
        _recognizer = ClipRecognizer(s.model_dir, labels_from_foods(catalogue()), s.model_dir / "cache", s.onnx_threads)
        _load_error = None
    except Exception as e:  # missing model files etc. - the rest of the app still works
        _recognizer, _load_error = None, str(e)
        log.warning("food recognition disabled: %s (run `python -m scripts.download_models`)", e)


def recognizer() -> ClipRecognizer:
    if _recognizer is None:
        raise HTTPException(503, "Food recognition model is not loaded. Run `python -m scripts.download_models` and restart.")
    return _recognizer


def make_thumb(img: Image.Image) -> bytes:
    t = img.copy()
    t.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=82, optimize=True)
    return buf.getvalue()


def analyse_sync(rec: ClipRecognizer, data: bytes, multi: bool):
    img = load_image(data)
    return img, rec.analyse(img, multi=multi, top_k=len(rec.labels)), make_thumb(img)


@router.get("/recognize/status")
async def model_status():
    return {"loaded": _recognizer is not None, "classes": len(_recognizer.labels) if _recognizer else 0,
            "error": _load_error}


@router.post("/recognize")
async def recognize(request: Request, image: UploadFile = File(...), multi: bool = Query(True),
                    hour: int | None = Query(None, ge=0, le=23, description="client local hour, for the meal-type hint"),
                    user: dict = Depends(current_user)):
    rec = recognizer()
    s = get_settings()
    limit = int(s.max_upload_mb * 1024 * 1024)
    if int(request.headers.get("content-length") or 0) > limit + 1024 * 64:
        raise HTTPException(413, f"Image is larger than {s.max_upload_mb:g} MB")
    if image.content_type not in ALLOWED:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP photo")
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image is larger than {s.max_upload_mb:g} MB")
    t0 = time.perf_counter()
    try:
        img, res, thumb = await run_in_threadpool(analyse_sync, rec, data, multi)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file doesn't look like an image we can read")
    if min(img.size) < 32:
        raise HTTPException(400, "Image is too small")

    db = get_db()
    emb: np.ndarray = res["embedding"]
    zero_shot: np.ndarray = res["probs"]

    # few-shot personalisation from this user's past corrections
    mem = await db.corrections.find({"user_id": user["_id"]}, {"embedding": 1, "food_id": 1}) \
        .sort("created_at", -1).limit(s.knn_memory_limit).to_list(s.knn_memory_limit)
    ids = [lab.id for lab in rec.labels]
    index = {fid: i for i, fid in enumerate(ids)}
    mem = [m for m in mem if m["food_id"] in index]
    alpha = 0.0
    probs = zero_shot
    if mem:
        E = np.stack([np.frombuffer(m["embedding"], dtype=np.float32) for m in mem])
        L = np.array([index[m["food_id"]] for m in mem])
        kp, alpha = knn_probs(emb, E, L, len(ids))
        probs = blend(zero_shot, kp, alpha)
    order = np.argsort(-probs)[:5]

    now = datetime.now(timezone.utc)
    top_ids = [ids[i] for i in order]
    photo = {"user_id": user["_id"], "thumb": Binary(thumb), "embedding": Binary(emb.astype(np.float32).tobytes()),
             "top": top_ids, "created_at": now, "expires_at": now + timedelta(hours=24)}
    pid = (await db.photos.insert_one(photo)).inserted_id

    food_score = res["food_score"]
    return {
        "photo_id": str(pid),
        "candidates": [{"food": public_food(by_id()[ids[i]]), "confidence": round(float(probs[i]), 4),
                        "zero_shot": round(float(zero_shot[i]), 4)} for i in order],
        "not_food": food_score < NOT_FOOD_THRESHOLD,
        "food_score": round(food_score, 3),
        "also_detected": [{"food": public_food(by_id()[ids[i]]), "confidence": round(p, 3)}
                          for i, p in res["also_detected"] if ids[i] not in top_ids[:1]],
        "personalized": alpha > 0,
        "personal_weight": round(alpha, 3),
        "suggested_meal_type": meal_type_for_hour(hour if hour is not None else datetime.now().hour),
        "timing_ms": round((time.perf_counter() - t0) * 1000),
    }


@router.get("/photos/{photo_id}")
async def photo(photo_id: str, user: dict = Depends(current_user)):
    p = await get_db().photos.find_one({"_id": oid(photo_id, "photo"), "user_id": user["_id"]}, {"thumb": 1})
    if not p:
        raise HTTPException(404, "No such photo")
    return Response(bytes(p["thumb"]), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/corrections")
async def corrections(user: dict = Depends(current_user)):
    """What the personalisation layer has learned for this user."""
    rows = await get_db().corrections.aggregate([
        {"$match": {"user_id": user["_id"]}},
        {"$group": {"_id": {"food": "$food_id", "predicted": "$predicted"}, "n": {"$sum": 1}}},
        {"$sort": {"n": -1}}, {"$limit": 20},
    ]).to_list(20)
    names = by_id()
    return {"items": [{"food": names[r["_id"]["food"]]["name"] if r["_id"]["food"] in names else r["_id"]["food"],
                       "was_predicted_as": names.get(r["_id"]["predicted"] or "", {}).get("name"),
                       "count": r["n"]} for r in rows]}
