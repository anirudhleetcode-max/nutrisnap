def test_health(client):
    body = client.get("/api/health").json()
    assert body["db"] == "connected"


def test_register_login_me(client):
    body = {"name": "Asha", "email": "asha@example.com", "password": "secret123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    r = client.post("/api/auth/register", json=body)
    assert r.status_code == 409 and r.json()["code"] == "conflict"
    assert client.post("/api/auth/login", json={"email": body["email"], "password": "wrong"}).status_code == 401
    tok = client.post("/api/auth/login", json={"email": body["email"], "password": "secret123"}).json()["token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert me["email"] == "asha@example.com" and me["has_profile"] is False
    assert client.get("/api/auth/me").status_code == 401


def test_validation_error_shape(client):
    r = client.post("/api/auth/register", json={"name": "", "email": "nope", "password": "1"})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error" and isinstance(r.json()["detail"], list)
