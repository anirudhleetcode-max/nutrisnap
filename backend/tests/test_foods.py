import json

from app.config import get_settings


def test_catalogue_is_complete():
    foods = json.loads(get_settings().foods_file.read_text(encoding="utf-8"))["foods"]
    assert len(foods) >= 120
    must = ["roti", "paratha", "dal-tadka", "rajma", "chole", "paneer-butter-masala", "chicken-biryani", "masala-dosa",
            "idli", "medu-vada", "sambar", "poha", "upma", "pav-bhaji", "vada-pav", "samosa", "dhokla", "khichdi",
            "curd-rice", "gulab-jamun", "jalebi", "pizza", "burger", "pasta-red", "green-salad", "sushi",
            "french-fries", "boiled-egg", "oatmeal", "banana", "apple"]
    ids = {f["id"] for f in foods}
    assert not set(must) - ids
    for f in foods:
        n = f["per100g"]
        atwater = 4 * n["protein"] + 4 * n["carbs"] + 9 * n["fat"]
        assert abs(atwater - n["kcal"]) <= max(25, 0.15 * n["kcal"]), f["id"]
        assert any(s["label"] == "gram" for s in f["servings"])
        assert all(s["grams"] > 0 for s in f["servings"])


def test_search_prefix_alias_and_text(client):
    r = client.get("/api/foods", params={"q": "pane"}).json()
    names = [f["name"] for f in r["items"]]
    assert names and all("paneer" in n.lower() for n in names[:3])
    # alias match: golgappa -> Pani Puri
    assert client.get("/api/foods", params={"q": "golgappa"}).json()["items"][0]["id"] == "pani-puri"
    # multi-word text search
    ids = [f["id"] for f in client.get("/api/foods", params={"q": "chicken curry"}).json()["items"]]
    assert "chicken-curry" in ids[:3]


def test_list_pagination_and_detail(client):
    a = client.get("/api/foods", params={"limit": 5}).json()
    b = client.get("/api/foods", params={"limit": 5, "skip": 5}).json()
    assert len(a["items"]) == 5 and a["total"] >= 120
    assert not {f["id"] for f in a["items"]} & {f["id"] for f in b["items"]}
    d = client.get("/api/foods/idli").json()
    assert d["servings"][0]["label"] == "1 idli"
    r = client.get("/api/foods/not-a-food")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
