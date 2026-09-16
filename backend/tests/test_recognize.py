import io

from PIL import Image, ImageDraw

from .conftest import FIXTURES, needs_model


def upload(client, h, path=None, data=None, ctype="image/jpeg", **params):
    data = data if data is not None else path.read_bytes()
    return client.post("/api/recognize", files={"image": ("photo.jpg", data, ctype)}, params=params, headers=h)


def document_image() -> bytes:
    img = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(img)
    for y in range(40, 760, 28):
        d.text((40, y), "Invoice line item, quantity, unit price, total amount due " * 2, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@needs_model
def test_recognize_pizza(client, new_user):
    r = upload(client, new_user, FIXTURES / "pizza.jpg", hour=20)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["candidates"]) == 5
    assert body["candidates"][0]["food"]["id"] in {"pizza", "pepperoni-pizza"}
    assert body["not_food"] is False and body["personalized"] is False
    assert body["suggested_meal_type"] == "dinner"
    confs = [c["confidence"] for c in body["candidates"]]
    assert confs == sorted(confs, reverse=True)
    thumb = client.get(f"/api/photos/{body['photo_id']}", headers=new_user)
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


@needs_model
def test_not_food_guard(client, auth_headers):
    body = upload(client, auth_headers, data=document_image(), ctype="image/png", multi=False).json()
    assert body["not_food"] is True


@needs_model
def test_upload_validation(client, auth_headers):
    assert upload(client, auth_headers, data=b"not an image").status_code == 400
    assert upload(client, auth_headers, data=b"hello", ctype="text/plain").status_code == 415
    big = b"\xff\xd8" + b"0" * (9 * 1024 * 1024)
    r = upload(client, auth_headers, data=big)
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"
    assert client.post("/api/recognize", files={"image": ("a.jpg", b"x", "image/jpeg")}).status_code == 401


@needs_model
def test_correction_personalises_next_prediction(client, new_user, auth_headers):
    h = new_user
    first = upload(client, h, FIXTURES / "dosa.jpg", multi=False).json()
    assert first["personalized"] is False
    guess = first["candidates"][0]["food"]["id"]
    # the user says it's masala dosa (the zero-shot model tends to call this photo a paratha)
    label = "masala-dosa" if guess != "masala-dosa" else "plain-dosa"
    serving = client.get(f"/api/foods/{label}").json()["servings"][0]["label"]
    r = client.post("/api/meals", json={"food_id": label, "date": "2026-02-01", "meal_type": "breakfast",
                                        "serving_label": serving, "quantity": 1, "photo_id": first["photo_id"]}, headers=h)
    assert r.status_code == 201
    learned = client.get("/api/corrections", headers=h).json()["items"]
    assert learned and learned[0]["count"] == 1

    second = upload(client, h, FIXTURES / "dosa.jpg", multi=False).json()
    assert second["personalized"] is True
    assert second["candidates"][0]["food"]["id"] == label
    # other users are unaffected
    third = upload(client, auth_headers, FIXTURES / "dosa.jpg", multi=False).json()
    assert third["candidates"][0]["food"]["id"] == guess
    # photos are private
    assert client.get(f"/api/photos/{first['photo_id']}", headers=auth_headers).status_code == 404
