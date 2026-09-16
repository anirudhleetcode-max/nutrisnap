"""Food catalogue: loaded from data/foods.json into memory (for O(1) lookups) and upserted into
Mongo (for text search)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from pymongo import UpdateOne

from .config import get_settings
from .db import get_db

router = APIRouter(prefix="/api/foods", tags=["foods"])


@lru_cache
def catalogue() -> list[dict]:
    return json.loads(get_settings().foods_file.read_text(encoding="utf-8"))["foods"]


@lru_cache
def by_id() -> dict[str, dict]:
    return {f["id"]: f for f in catalogue()}


def get_food(food_id: str) -> dict:
    f = by_id().get(food_id)
    if f is None:
        raise HTTPException(404, f"Unknown food '{food_id}'")
    return f


PUBLIC_FIELDS = ("id", "name", "cuisine", "category", "per100g", "servings", "aliases", "source", "verified",
                 "source_id", "source_url", "source_note", "servings_source")


def public_food(f: dict) -> dict:
    return {k: f[k] for k in PUBLIC_FIELDS if k in f}


async def seed_foods() -> int:
    ops = [UpdateOne({"_id": f["id"]}, {"$set": {**{k: v for k, v in f.items() if k != "id"},
                                                 "name_lc": f["name"].lower()}}, upsert=True)
           for f in catalogue()]
    await get_db().foods.bulk_write(ops, ordered=False)
    await get_db().foods.delete_many({"_id": {"$nin": list(by_id())}})
    return len(ops)


@router.get("")
async def search_foods(q: str = Query("", max_length=60), limit: int = Query(20, ge=1, le=50),
                       skip: int = Query(0, ge=0, le=500)):
    db = get_db()
    q = q.strip()
    if not q:
        docs = await db.foods.find({}, {"name_lc": 0}).sort("name_lc", 1).skip(skip).limit(limit).to_list(limit)
        total = await db.foods.count_documents({})
    else:
        # 1) prefix / substring matches on name and aliases (text indexes don't do prefixes: "pan" -> paneer)
        rx = {"$regex": re.escape(q.lower()), "$options": "i"}
        prefix = await db.foods.find({"$or": [{"name_lc": rx}, {"aliases": rx}]}, {"_id": 1, "name_lc": 1}).to_list(200)
        starts = [d["_id"] for d in sorted(prefix, key=lambda d: (not d["name_lc"].startswith(q.lower()), d["name_lc"]))]
        # 2) full-text matches (stemming, multi-word) ranked by score
        text = await db.foods.find({"$text": {"$search": q}}, {"score": {"$meta": "textScore"}}) \
            .sort([("score", {"$meta": "textScore"})]).limit(100).to_list(100)
        ranked = list(dict.fromkeys(starts + [d["_id"] for d in text]))
        total = len(ranked)
        page = ranked[skip:skip + limit]
        docs = [{"_id": i} for i in page]
    items = [public_food(by_id()[d["_id"]]) for d in docs if d["_id"] in by_id()]
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.get("/{food_id}")
async def food_detail(food_id: str):
    return public_food(get_food(food_id))
