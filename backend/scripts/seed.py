"""Seed the food catalogue and a demo account with ~30 days of realistic Indian meals.

    python -m scripts.seed

Demo login: demo@nutrisnap.app / demo1234  (re-running replaces the demo user's meals)
"""
from __future__ import annotations

import asyncio
import random
from datetime import date, datetime, timedelta, timezone

from app.db import close_client, ensure_indexes, get_db
from app.foods import by_id, seed_foods
from app.security import hash_password
from app.services.nutrition import grams_for, nutrients_for

EMAIL = "demo@nutrisnap.app"
PROFILE = {"age": 24, "sex": "male", "height_cm": 174, "weight_kg": 72, "activity": "light", "goal": "lose"}

# each option: list of (food_id, serving_label, quantity)
BREAKFAST = [
    [("poha", "1 plate", 1), ("masala-chai", "1 cup", 1)],
    [("idli", "1 idli", 3), ("sambar", "1 katori", 1), ("coconut-chutney", "1 tbsp", 2)],
    [("aloo-paratha", "1 paratha", 2), ("curd", "1 katori", 1)],
    [("upma", "1 bowl", 1), ("filter-coffee", "1 tumbler", 1)],
    [("oatmeal", "1 bowl", 1), ("banana", "1 medium", 1)],
    [("masala-dosa", "1 dosa", 1), ("sambar", "1 katori", 1)],
    [("omelette", "1 omelette (2 eggs)", 1), ("brown-bread", "1 slice", 2), ("masala-chai", "1 cup", 1)],
]
LUNCH = [
    [("roti", "1 roti", 3), ("dal-tadka", "1 katori", 1), ("bhindi-masala", "1 katori", 1), ("plain-rice", "1 katori", 1)],
    [("rajma", "1 katori", 1), ("plain-rice", "1 plate", 1), ("raita", "1 katori", 1)],
    [("chole", "1 katori", 1), ("roti", "1 roti", 3), ("green-salad", "1 bowl", 1)],
    [("curd-rice", "1 bowl", 1), ("sambar", "1 katori", 1)],
    [("chicken-biryani", "1 plate", 1), ("raita", "1 katori", 1)],
    [("roti", "1 roti", 2), ("palak-paneer", "1 katori", 1), ("dal-tadka", "1 katori", 1)],
    [("veg-burger", "1 burger", 1), ("french-fries", "medium", 1)],
]
SNACKS = [
    [("masala-chai", "1 cup", 1), ("samosa", "1 samosa", 1)],
    [("apple", "1 medium", 1)],
    [("sprouts-salad", "1 bowl", 1)],
    [("masala-chai", "1 cup", 1), ("dhokla", "1 piece", 3)],
    [("almonds", "1 handful", 1)],
    [("vada-pav", "1 vada pav", 1)],
    [("chaas", "1 glass", 1)],
    [("pani-puri", "1 plate (6)", 1)],
]
DINNER = [
    [("roti", "1 roti", 3), ("paneer-butter-masala", "1 katori", 1), ("dal-makhani", "1 katori", 1)],
    [("khichdi", "1 bowl", 1), ("curd", "1 katori", 1)],
    [("roti", "1 roti", 3), ("chicken-curry", "1 katori", 1), ("green-salad", "1 bowl", 1)],
    [("pizza", "1 slice", 3)],
    [("roti", "1 roti", 2), ("mixed-veg", "1 katori", 1), ("dal-tadka", "1 katori", 1)],
    [("hakka-noodles", "1 plate", 1), ("manchurian", "1 bowl", 1)],
    [("pav-bhaji", "1 plate", 1)],
    [("fish-curry", "1 katori", 1), ("plain-rice", "1 plate", 1)],
]
MEALS = [("breakfast", BREAKFAST, 8), ("lunch", LUNCH, 13), ("snacks", SNACKS, 17), ("dinner", DINNER, 20)]


async def main() -> None:
    await ensure_indexes()
    n = await seed_foods()
    db = get_db()
    user = await db.users.find_one({"email": EMAIL})
    if user:
        uid = user["_id"]
        await db.users.update_one({"_id": uid}, {"$set": {"profile": PROFILE}})
        for coll in ("meals", "photos", "corrections"):
            await db[coll].delete_many({"user_id": uid})
    else:
        uid = (await db.users.insert_one({"name": "Demo Rohan", "email": EMAIL, "password_hash": hash_password("demo1234"),
                                          "profile": PROFILE, "created_at": datetime.now(timezone.utc)})).inserted_id
    rng = random.Random(42)
    today = date.today()
    docs = []
    foods = by_id()
    for back in range(30, -1, -1):
        day = today - timedelta(days=back)
        if back > 10 and rng.random() < 0.12:
            continue  # a few missed days, but a live streak for the last ~10
        for meal_type, options, hour in MEALS:
            if back == 0 and meal_type != "breakfast":
                continue  # today: only breakfast so far
            if meal_type == "snacks" and rng.random() < 0.3:
                continue
            for food_id, label, qty in rng.choice(options):
                if food_id == "roti":
                    qty = rng.choice([2, 3, 3, 4])
                food = foods[food_id]
                grams = grams_for(food, label, qty)
                ts = datetime(day.year, day.month, day.day, hour, rng.randint(0, 50), tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)
                docs.append({"user_id": uid, "date": day.isoformat(), "meal_type": meal_type, "food_id": food_id,
                             "food_name": food["name"], "serving_label": label, "quantity": qty, "grams": grams,
                             "nutrients": nutrients_for(food, grams), "source": "search", "created_at": ts})
    await db.meals.insert_many(docs)
    print(f"foods: {n}; demo user {EMAIL} / demo1234 with {len(docs)} logged items")
    await close_client()


if __name__ == "__main__":
    asyncio.run(main())
