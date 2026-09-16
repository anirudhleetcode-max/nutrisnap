import asyncio
import io
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from bson import Binary
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from .config import get_settings
from .db import get_db
from .foods import by_id, catalogue, public_food
from .meals import oid
from .ml.pipeline import FoodRecognizer
from .ml.preprocess import ImageRejected, decode_image
from .security import current_user
from .services.nutrition import meal_type_for_hour

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["recognition"])
THUMB_PX = 480
INFERENCE_TIMEOUT_S = 20

_recognizer: FoodRecognizer | None = None
_load_error: str | None = None


def load_model() -> None:
    global _recognizer, _load_error
    s = get_settings()
    try:
        _recognizer = FoodRecognizer(s.model_dir, catalogue(), s.artifacts_dir, s.onnx_threads)
        _load_error = None
    except Exception as e:  # missing/corrupt model files - the rest of the app still works
        _recognizer, _load_error = None, f"{type(e).__name__}: {e}"
        log.warning("food recognition disabled: %s (run `python -m scripts.download_models`)", _load_error)


def model_state() -> dict:
    return {"loaded": _recognizer is not None, "error": _load_error}


def recognizer() -> FoodRecognizer:
    if _recognizer is None:
        raise HTTPException(503, "Food recognition model is not loaded. Run `python -m scripts.download_models` and restart.")
    return _recognizer


def make_thumb(img: Image.Image) -> bytes:
    t = img.copy()
    t.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=82, optimize=True)  # re-encoding also drops EXIF (location etc.)
    return buf.getvalue()


def analyse_sync(rec: FoodRecognizer, data: bytes, multi: bool, memory):
    img = decode_image(data)
    return img, rec.analyse(img, memory=memory, multi=multi), make_thumb(img)


@router.get("/recognize/status")
async def model_status():
    return {"loaded": _recognizer is not None, "classes": len(_recognizer.ids) if _recognizer else 0,
            "error": _load_error}


@router.get("/model")
async def model_info():
    """Model card: what the recognizer is, how it was evaluated, and its limits."""
    if _recognizer is None:
        return {"loaded": False, "error": _load_error}
    return {"loaded": True, **_recognizer.model_card()}


@router.post("/recognize")
async def recognize(request: Request, image: UploadFile = File(...), multi: bool = Query(True),
                    hour: int | None = Query(None, ge=0, le=23, description="client local hour, for the meal-type hint"),
                    user: dict = Depends(current_user)):
    rec = recognizer()
    s = get_settings()
    limit = int(s.max_upload_mb * 1024 * 1024)
    if int(request.headers.get("content-length") or 0) > limit + 1024 * 64:
        raise HTTPException(413, f"Image is larger than {s.max_upload_mb:g} MB")
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image is larger than {s.max_upload_mb:g} MB")

    db = get_db()
    # few-shot personalisation memory: this user's past corrections only
    mem = await db.corrections.find({"user_id": user["_id"]}, {"embedding": 1, "food_id": 1}) \
        .sort("created_at", -1).limit(s.knn_memory_limit).to_list(s.knn_memory_limit)
    index = {fid: i for i, fid in enumerate(rec.ids)}
    mem = [m for m in mem if m["food_id"] in index]
    memory = None
    if mem:
        memory = (np.stack([np.frombuffer(m["embedding"], dtype=np.float32) for m in mem]),
                  np.array([index[m["food_id"]] for m in mem]))

    t0 = time.perf_counter()
    try:
        img, a, thumb = await asyncio.wait_for(run_in_threadpool(analyse_sync, rec, data, multi, memory),
                                               timeout=INFERENCE_TIMEOUT_S)
    except ImageRejected as e:
        raise HTTPException(e.status, str(e))
    except asyncio.TimeoutError:
        raise HTTPException(503, "The model took too long on this image. Try a smaller photo.")

    now = datetime.now(timezone.utc)
    top_ids = [rec.ids[i] for i in a.order]
    photo = {"user_id": user["_id"], "thumb": Binary(thumb), "embedding": Binary(a.embedding.astype(np.float32).tobytes()),
             "top": top_ids, "created_at": now, "expires_at": now + timedelta(hours=24)}
    pid = (await db.photos.insert_one(photo)).inserted_id

    return {
        "photo_id": str(pid),
        "candidates": [{"food": public_food(by_id()[rec.ids[i]]), "confidence": round(float(a.probs[i]), 4),
                        "zero_shot": round(float(a.zero_shot[i]), 4)} for i in a.order],
        "not_food": a.decision == "not_food",
        "food_score": round(a.food_score, 3),
        "also_detected": [{"food": public_food(by_id()[rec.ids[i]]), "confidence": round(p, 3)}
                          for i, p in a.also_detected if rec.ids[i] not in top_ids[:1]],
        "personalized": a.personal_weight > 0,
        "personal_weight": round(a.personal_weight, 3),
        "suggested_meal_type": meal_type_for_hour(hour if hour is not None else datetime.now().hour),
        "timing_ms": round((time.perf_counter() - t0) * 1000),
        # phase 2 additions
        "decision": a.decision,
        "confidence": round(a.confidence, 4),
        "abstain_below": rec.cfg["abstain_below"],
        "calibrated": bool(rec.cfg["calibrated"]) and a.personal_weight == 0,
        "model_version": rec.cfg["version"],
        "probe_used": a.probe_used,
        "quality_warnings": a.warnings,
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
    total = await get_db().corrections.count_documents({"user_id": user["_id"]})
    return {"total": total,
            "items": [{"food": names[r["_id"]["food"]]["name"] if r["_id"]["food"] in names else r["_id"]["food"],
                       "was_predicted_as": names.get(r["_id"]["predicted"] or "", {}).get("name"),
                       "count": r["n"]} for r in rows]}


@router.delete("/corrections", status_code=204)
async def forget_corrections(user: dict = Depends(current_user)):
    """Delete every stored image embedding used for this user's personalisation."""
    await get_db().corrections.delete_many({"user_id": user["_id"]})
    return Response(status_code=204)
