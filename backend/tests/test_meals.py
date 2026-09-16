from app.foods import by_id


def kcal(food_id: str, grams: float) -> float:
    return round(by_id()[food_id]["per100g"]["kcal"] * grams / 100, 1)


def add(client, h, **kw):
    body = {"food_id": "roti", "date": "2026-03-10", "meal_type": "lunch", "serving_label": "1 roti", "quantity": 2, **kw}
    return client.post("/api/meals", json=body, headers=h)


def test_create_list_update_delete(client, new_user):
    h = new_user
    r = add(client, h)
    assert r.status_code == 201, r.text
    roti = r.json()
    assert roti["grams"] == 80 and roti["nutrients"]["kcal"] == kcal("roti", 80)
    add(client, h, food_id="dal-tadka", serving_label="1 katori", quantity=1)
    add(client, h, food_id="poha", meal_type="breakfast", serving_label="1 plate", quantity=1)
    add(client, h, food_id="poha", date="2026-03-11", serving_label="1 plate", quantity=1)

    day = client.get("/api/meals", params={"date": "2026-03-10"}, headers=h).json()
    assert [g["meal_type"] for g in day["groups"]] == ["breakfast", "lunch", "snacks", "dinner"]
    lunch = day["groups"][1]
    lunch_kcal = kcal("roti", 80) + kcal("dal-tadka", 150)
    assert len(lunch["items"]) == 2 and abs(lunch["totals"]["kcal"] - lunch_kcal) < 0.2
    assert abs(day["totals"]["kcal"] - (lunch_kcal + kcal("poha", 200))) < 0.2

    # edit portion
    r = client.patch(f"/api/meals/{roti['id']}", json={"quantity": 3}, headers=h)
    assert r.json()["nutrients"]["kcal"] == kcal("roti", 120)
    # switch food requires a valid serving for the new food
    assert client.patch(f"/api/meals/{roti['id']}", json={"food_id": "naan"}, headers=h).status_code == 422
    r = client.patch(f"/api/meals/{roti['id']}", json={"food_id": "naan", "serving_label": "1 naan", "quantity": 1}, headers=h)
    assert r.json()["food_name"] == "Naan" and r.json()["grams"] == 90

    assert client.delete(f"/api/meals/{roti['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/meals/{roti['id']}", headers=h).status_code == 404
    day = client.get("/api/meals", params={"date": "2026-03-10"}, headers=h).json()
    assert len(day["groups"][1]["items"]) == 1


def test_validation(client, auth_headers):
    h = auth_headers
    assert add(client, h, serving_label="1 bucket").status_code == 422
    assert add(client, h, food_id="unobtainium").status_code == 404
    assert add(client, h, quantity=0).status_code == 422
    assert add(client, h, meal_type="brunch").status_code == 422
    assert add(client, h, date="10-03-2026").status_code == 422
    assert add(client, h, serving_label="gram", quantity=5000).status_code == 422
    assert add(client, h, photo_id="000000000000000000000000").status_code == 404
    assert client.patch("/api/meals/not-an-id", json={"quantity": 1}, headers=h).status_code == 404


def test_users_are_isolated(client, new_user, auth_headers):
    m = add(client, new_user).json()
    assert client.patch(f"/api/meals/{m['id']}", json={"quantity": 1}, headers=auth_headers).status_code == 404
    assert client.delete(f"/api/meals/{m['id']}", headers=auth_headers).status_code == 404
    other = client.get("/api/meals", params={"date": "2026-03-10"}, headers=auth_headers).json()
    assert m["id"] not in [i["id"] for g in other["groups"] for i in g["items"]]


def test_history_cursor(client, new_user):
    for q in range(1, 6):
        add(client, new_user, quantity=q)
    page1 = client.get("/api/meals/history", params={"limit": 3}, headers=new_user).json()
    assert len(page1["items"]) == 3 and page1["next_cursor"]
    page2 = client.get("/api/meals/history", params={"limit": 3, "before": page1["next_cursor"]}, headers=new_user).json()
    assert len(page2["items"]) == 2 and page2["next_cursor"] is None
    assert [i["quantity"] for i in page1["items"] + page2["items"]] == [5, 4, 3, 2, 1]
