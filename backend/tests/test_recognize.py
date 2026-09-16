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


def png_bytes(w: int, h: int, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


@needs_model
def test_upload_validation(client, auth_headers):
    # not an image at all (magic bytes are checked, not the client's content type)
    r = upload(client, auth_headers, data=b"not an image")
    assert r.status_code == 415 and r.json()["code"] == "unsupported_media_type"
    assert upload(client, auth_headers, data=b"hello", ctype="text/plain").status_code == 415
    # a script renamed to .jpg is still refused
    assert upload(client, auth_headers, data=b"<?php echo 1; ?>" * 10).status_code == 415
    # empty file
    assert upload(client, auth_headers, data=b"").status_code in (400, 415)
    # truncated JPEG: right signature, broken body
    jpeg = (FIXTURES / "pizza.jpg").read_bytes()
    r = upload(client, auth_headers, data=jpeg[:300])
    assert r.status_code == 400 and "can read" in r.json()["detail"]
    # tiny image
    assert upload(client, auth_headers, data=png_bytes(10, 10), ctype="image/png").status_code == 400
    # decompression bomb: small file, 81 megapixels
    bomb = png_bytes(9000, 9000)
    assert len(bomb) < 8 * 1024 * 1024
    r = upload(client, auth_headers, data=bomb, ctype="image/png")
    assert r.status_code == 413 and "megapixels" in r.json()["detail"]
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


@needs_model
def test_low_quality_photo_is_flagged(client, auth_headers):
    body = upload(client, auth_headers, data=png_bytes(400, 300, (5, 5, 5)), ctype="image/png", multi=False).json()
    assert "too_dark" in body["quality_warnings"]
    assert body["decision"] in {"not_food", "uncertain", "confident"}


@needs_model
def test_low_confidence_path_abstains(client, auth_headers):
    from app import recognize
    rec = recognize._recognizer
    old = dict(rec.cfg)
    rec.cfg.update(abstain_below=0.999, not_food_below=0.0)
    try:
        body = upload(client, auth_headers, FIXTURES / "dosa.jpg", multi=False).json()
    finally:
        rec.cfg.clear()
        rec.cfg.update(old)
    assert body["decision"] == "uncertain" and body["abstain_below"] == 0.999
    assert len(body["candidates"]) == 5  # the user still gets choices


@needs_model
def test_model_card(client):
    card = client.get("/api/model").json()
    assert card["loaded"] is True
    assert "not trained by this project" in card["backbone"]["model"]
    assert card["preprocessing"].startswith("clip-224")
    assert card["classes"] >= 120 and card["limitations"]


def test_model_missing_degrades_gracefully(client, auth_headers, tmp_path, monkeypatch):
    from app import recognize
    from app.config import get_settings

    saved = recognize._recognizer, recognize._load_error
    monkeypatch.setattr(get_settings(), "model_dir", tmp_path)  # empty directory: no ONNX files
    try:
        recognize.load_model()
        assert recognize._recognizer is None and "missing model files" in recognize._load_error
        r = upload(client, auth_headers, FIXTURES / "pizza.jpg")
        assert r.status_code == 503 and r.json()["code"] == "unavailable"
        assert client.get("/api/model").json()["loaded"] is False
        health = client.get("/api/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        # the rest of the app keeps working
        assert client.get("/api/foods", params={"q": "dal"}).status_code == 200
    finally:
        recognize._recognizer, recognize._load_error = saved


def test_model_load_failure_is_reported(client, tmp_path, monkeypatch):
    from app import recognize
    from app.config import get_settings

    for name in ("vision_model_quantized.onnx", "text_model_quantized.onnx", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"corrupt")
    saved = recognize._recognizer, recognize._load_error
    monkeypatch.setattr(get_settings(), "model_dir", tmp_path)
    try:
        recognize.load_model()
        assert recognize._recognizer is None and recognize._load_error
    finally:
        recognize._recognizer, recognize._load_error = saved


@needs_model
def test_forget_corrections(client, new_user):
    first = upload(client, new_user, FIXTURES / "pizza.jpg", multi=False).json()
    other = next(c for c in ["sushi", "ramen"] if c != first["candidates"][0]["food"]["id"])
    client.post("/api/meals", json={"food_id": other, "date": "2026-02-01", "meal_type": "lunch",
                                    "serving_label": "gram", "quantity": 100, "photo_id": first["photo_id"]},
                headers=new_user)
    assert client.get("/api/corrections", headers=new_user).json()["total"] == 1
    assert client.delete("/api/corrections", headers=new_user).status_code == 204
    assert client.get("/api/corrections", headers=new_user).json()["total"] == 0
