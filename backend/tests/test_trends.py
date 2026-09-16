from datetime import date, timedelta

from app.foods import by_id
from app.trends import streak_from


def test_streak_logic():
    end = date(2026, 5, 10)
    assert streak_from(["2026-05-10", "2026-05-09", "2026-05-07"], end) == 2
    assert streak_from(["2026-05-09", "2026-05-08"], end) == 2  # today not logged yet
    assert streak_from(["2026-05-07"], end) == 0


def test_trends_endpoint(client, new_user):
    h = new_user
    client.put("/api/profile", json={"age": 30, "sex": "male", "height_cm": 175, "weight_kg": 70,
                                     "activity": "moderate", "goal": "maintain"}, headers=h)
    end = date(2026, 4, 20)
    for back in range(3):
        d = (end - timedelta(days=back)).isoformat()
        client.post("/api/meals", json={"food_id": "idli", "date": d, "meal_type": "breakfast",
                                        "serving_label": "1 idli", "quantity": 4}, headers=h)
    client.post("/api/meals", json={"food_id": "chicken-biryani", "date": end.isoformat(), "meal_type": "dinner",
                                    "serving_label": "1 plate", "quantity": 1}, headers=h)
    t = client.get("/api/trends", params={"days": 7, "end": end.isoformat()}, headers=h).json()
    assert len(t["days"]) == 7 and t["days"][-1]["date"] == "2026-04-20"
    expected = by_id()["idli"]["per100g"]["kcal"] * 1.6 + by_id()["chicken-biryani"]["per100g"]["kcal"] * 3.5
    assert abs(t["days"][-1]["kcal"] - expected) < 0.2 and t["days"][0]["meals"] == 0
    assert t["logged_days"] == 3 and t["streak"] == 3
    assert t["top_foods"][0]["food_id"] == "idli" and t["top_foods"][0]["count"] == 3
    assert t["targets"]["kcal"] == 2560  # 30 y male, 175 cm, 70 kg, moderate
    assert sum(t["macro_split"].values()) in (99, 100, 101)
    assert client.get("/api/trends", params={"days": 3, "end": "2026-04-20"}, headers=h).status_code == 422
