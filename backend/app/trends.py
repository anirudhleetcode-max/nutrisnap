from datetime import date as Date, timedelta

from fastapi import APIRouter, Depends, Query

from .db import get_db
from .security import current_user
from .services.nutrition import NUTRIENTS, targets

router = APIRouter(prefix="/api/trends", tags=["trends"])


def streak_from(dates: list[str], end: Date) -> int:
    """Consecutive logged days ending today (or yesterday, if today is still empty)."""
    have = set(dates)
    day = end if end.isoformat() in have else end - timedelta(days=1)
    n = 0
    while day.isoformat() in have:
        n += 1
        day -= timedelta(days=1)
    return n


@router.get("")
async def trends(days: int = Query(7, ge=7, le=90), end: Date = Query(..., description="client's local today"),
                 user: dict = Depends(current_user)):
    start = end - timedelta(days=days - 1)
    match = {"user_id": user["_id"], "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}
    sums = {k: {"$sum": f"$nutrients.{k}"} for k in NUTRIENTS}
    pipeline = [
        {"$match": match},
        {"$facet": {
            "daily": [{"$group": {"_id": "$date", **sums, "meals": {"$sum": 1}}}, {"$sort": {"_id": 1}}],
            "top_foods": [
                {"$group": {"_id": "$food_id", "name": {"$first": "$food_name"}, "count": {"$sum": 1},
                            "kcal": {"$sum": "$nutrients.kcal"}}},
                {"$sort": {"count": -1, "kcal": -1}}, {"$limit": 6}],
            "by_meal": [{"$group": {"_id": "$meal_type", "kcal": {"$sum": "$nutrients.kcal"}}}],
        }},
    ]
    facet = (await get_db().meals.aggregate(pipeline).to_list(1))[0]
    daily = {r["_id"]: r for r in facet["daily"]}
    series = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        r = daily.get(d)
        series.append({"date": d, "meals": r["meals"] if r else 0,
                       **{k: round(r[k], 1) if r else 0 for k in NUTRIENTS}})
    logged = [s for s in series if s["meals"]]
    avg = {k: round(sum(s[k] for s in logged) / len(logged), 1) if logged else 0 for k in NUTRIENTS}
    # streak looks further back than the window
    dates = await get_db().meals.distinct("date", {"user_id": user["_id"], "date": {"$lte": end.isoformat(),
                                                   "$gte": (end - timedelta(days=366)).isoformat()}})
    profile = user.get("profile")
    t = targets(profile) if profile else None
    target_kcal = t["kcal"] if t else None
    macro_kcal = avg["protein"] * 4 + avg["carbs"] * 4 + avg["fat"] * 9
    return {
        "start": start.isoformat(), "end": end.isoformat(), "days": series,
        "targets": {k: t[k] for k in NUTRIENTS} if t else None,
        "averages": avg,
        "macro_split": {k: round(avg[k] * m / macro_kcal * 100) if macro_kcal else 0
                        for k, m in (("protein", 4), ("carbs", 4), ("fat", 9))},
        "logged_days": len(logged),
        "days_on_target": sum(1 for s in logged if target_kcal and abs(s["kcal"] - target_kcal) <= 0.1 * target_kcal),
        "streak": streak_from(dates, end),
        "top_foods": [{"food_id": r["_id"], "name": r["name"], "count": r["count"], "kcal": round(r["kcal"])}
                      for r in facet["top_foods"]],
        "by_meal_type": {r["_id"]: round(r["kcal"]) for r in facet["by_meal"]},
    }
