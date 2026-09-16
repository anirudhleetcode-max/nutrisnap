from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, TEXT

from .config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(get_settings().mongo_uri, serverSelectionTimeoutMS=4000, tz_aware=True)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().mongo_db]


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def ensure_indexes() -> None:
    db = get_db()
    await db.users.create_index("email", unique=True)
    # full-text search over food names/aliases; name matches rank higher
    await db.foods.create_index([("name", TEXT), ("aliases", TEXT), ("category", TEXT)],
                                weights={"name": 10, "aliases": 5, "category": 1}, name="food_text")
    await db.foods.create_index("name_lc")
    await db.meals.create_index([("user_id", ASCENDING), ("date", ASCENDING), ("created_at", ASCENDING)])
    await db.meals.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    await db.photos.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    # photos never attached to a meal expire after a day
    await db.photos.create_index("expires_at", expireAfterSeconds=0)
    await db.corrections.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    await db.corrections.create_index([("user_id", ASCENDING), ("photo_id", ASCENDING)], unique=True)
