from datetime import date as Date, datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .db import get_db
from .foods import get_food
from .schemas import MealIn, MealPatch
from .security import current_user
from .services.nutrition import NUTRIENTS, grams_for, nutrients_for

router = APIRouter(prefix="/api/meals", tags=["meals"])
MEAL_ORDER = ["breakfast", "lunch", "snacks", "dinner"]


def oid(value: str, what: str = "meal") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(404, f"No such {what}")


def meal_out(m: dict) -> dict:
    return {
        "id": str(m["_id"]), "date": m["date"], "meal_type": m["meal_type"],
        "food_id": m["food_id"], "food_name": m["food_name"],
        "serving_label": m["serving_label"], "quantity": m["quantity"], "grams": m["grams"],
        "nutrients": m["nutrients"], "photo_id": str(m["photo_id"]) if m.get("photo_id") else None,
        "servings": get_food(m["food_id"])["servings"],
        "nutrition_verified": bool(get_food(m["food_id"]).get("verified")),
        "created_at": m["created_at"].isoformat(),
    }


def computed_fields(food_id: str, serving_label: str, quantity: float) -> dict:
    food = get_food(food_id)
    try:
        grams = grams_for(food, serving_label, quantity)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if grams > 3000:
        raise HTTPException(422, "That portion is over 3 kg - check the quantity")
    return {"food_id": food_id, "food_name": food["name"], "serving_label": serving_label,
            "quantity": quantity, "grams": grams, "nutrients": nutrients_for(food, grams)}


async def record_correction(user_id: ObjectId, photo: dict, food_id: str) -> None:
    """Store (embedding, label) when the user picked something other than our top guess."""
    db = get_db()
    if photo.get("top") and photo["top"][0] == food_id:
        await db.corrections.delete_one({"user_id": user_id, "photo_id": photo["_id"]})
        return
    await db.corrections.update_one(
        {"user_id": user_id, "photo_id": photo["_id"]},
        {"$set": {"food_id": food_id, "embedding": photo["embedding"], "predicted": (photo.get("top") or [None])[0],
                  "created_at": datetime.now(timezone.utc)}},
        upsert=True)


@router.post("", status_code=201)
async def create_meal(body: MealIn, user: dict = Depends(current_user)):
    db = get_db()
    doc = {"user_id": user["_id"], "date": body.date.isoformat(), "meal_type": body.meal_type,
           **computed_fields(body.food_id, body.serving_label, body.quantity),
           "source": "search", "created_at": datetime.now(timezone.utc)}
    if body.photo_id:
        pid = oid(body.photo_id, "photo")
        photo = await db.photos.find_one({"_id": pid, "user_id": user["_id"]}, {"thumb": 0})
        if not photo:
            raise HTTPException(404, "Photo not found or expired - snap it again")
        doc["photo_id"] = pid
        doc["source"] = "photo"
        doc["predicted"] = (photo.get("top") or [None])[0]
        await db.photos.update_one({"_id": pid}, {"$unset": {"expires_at": ""}})
        await record_correction(user["_id"], photo, body.food_id)
    res = await db.meals.insert_one(doc)
    doc["_id"] = res.inserted_id
    return meal_out(doc)


@router.get("")
async def meals_for_day(date: Date, user: dict = Depends(current_user)):
    db = get_db()
    d = date.isoformat()
    meals = await db.meals.find({"user_id": user["_id"], "date": d}).sort("created_at", 1).to_list(200)
    group = {"_id": "$meal_type", **{k: {"$sum": f"$nutrients.{k}"} for k in NUTRIENTS}}
    rows = await db.meals.aggregate([{"$match": {"user_id": user["_id"], "date": d}}, {"$group": group}]).to_list(10)
    by_type = {r["_id"]: {k: round(r[k], 1) for k in NUTRIENTS} for r in rows}
    totals = {k: round(sum(v[k] for v in by_type.values()), 1) for k in NUTRIENTS}
    return {
        "date": d,
        "groups": [{"meal_type": t, "totals": by_type.get(t, {k: 0 for k in NUTRIENTS}),
                    "items": [meal_out(m) for m in meals if m["meal_type"] == t]} for t in MEAL_ORDER],
        "totals": totals,
    }


@router.get("/history")
async def history(limit: int = Query(20, ge=1, le=100), before: str | None = Query(None, description="cursor: meal id"),
                  user: dict = Depends(current_user)):
    """Newest-first list of everything logged, cursor-paginated by meal id."""
    q: dict = {"user_id": user["_id"]}
    if before:
        q["_id"] = {"$lt": oid(before)}
    docs = await get_db().meals.find(q).sort("_id", -1).limit(limit + 1).to_list(limit + 1)
    more = len(docs) > limit
    docs = docs[:limit]
    return {"items": [meal_out(m) for m in docs], "next_cursor": str(docs[-1]["_id"]) if more else None}


@router.patch("/{meal_id}")
async def update_meal(meal_id: str, body: MealPatch, user: dict = Depends(current_user)):
    db = get_db()
    m = await db.meals.find_one({"_id": oid(meal_id), "user_id": user["_id"]})
    if not m:
        raise HTTPException(404, "No such meal")
    changes = body.model_dump(exclude_none=True)
    upd: dict = {}
    if "meal_type" in changes:
        upd["meal_type"] = changes["meal_type"]
    if "date" in changes:
        upd["date"] = changes["date"].isoformat()
    if {"food_id", "serving_label", "quantity"} & changes.keys():
        food_id = changes.get("food_id", m["food_id"])
        label = changes.get("serving_label", m["serving_label"] if food_id == m["food_id"] else None)
        if label is None:
            raise HTTPException(422, "Pick a serving size for the new food")
        upd.update(computed_fields(food_id, label, changes.get("quantity", m["quantity"])))
    if not upd:
        return meal_out(m)
    await db.meals.update_one({"_id": m["_id"]}, {"$set": upd})
    if "food_id" in changes and changes["food_id"] != m["food_id"] and m.get("photo_id"):
        photo = await db.photos.find_one({"_id": m["photo_id"], "user_id": user["_id"]}, {"thumb": 0})
        if photo:
            await record_correction(user["_id"], photo, changes["food_id"])
    m.update(upd)
    return meal_out(m)


@router.delete("/{meal_id}", status_code=204)
async def delete_meal(meal_id: str, user: dict = Depends(current_user)):
    db = get_db()
    m = await db.meals.find_one_and_delete({"_id": oid(meal_id), "user_id": user["_id"]})
    if not m:
        raise HTTPException(404, "No such meal")
    if m.get("photo_id"):
        # keep the correction (it is still a valid labelled example); the photo itself goes
        await db.photos.delete_one({"_id": m["photo_id"], "user_id": user["_id"]})
    return Response(status_code=204)
