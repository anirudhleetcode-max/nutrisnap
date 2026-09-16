import pytest

from app.config import Settings


def test_placeholder_secret_refused_in_production():
    with pytest.raises(ValueError):
        Settings(env="production", jwt_secret="change-me-in-production", _env_file=None)
    with pytest.raises(ValueError):
        Settings(env="production", jwt_secret="short", _env_file=None)


def test_placeholder_secret_replaced_in_development():
    s = Settings(env="development", jwt_secret="", _env_file=None)
    assert s.jwt_secret_ephemeral and len(s.jwt_secret) >= 48
    real = "x" * 40
    assert Settings(env="production", jwt_secret=real, _env_file=None).jwt_secret == real


def test_login_rate_limit(client):
    email = "ratelimit@example.com"
    client.post("/api/auth/register", json={"name": "R", "email": email, "password": "secret123"})
    for _ in range(5):
        assert client.post("/api/auth/login", json={"email": email, "password": "wrong"}).status_code == 401
    r = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert r.status_code == 429 and r.json()["code"] == "rate_limited" and "Retry-After" in r.headers
    # another account from the same client is not locked out
    assert client.post("/api/auth/login", json={"email": "someone@example.com", "password": "x"}).status_code == 401


def test_cors_only_configured_origins(client):
    ok = client.options("/api/health", headers={"Origin": "http://localhost:5176", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5176"
    bad = client.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_protected_routes_need_auth(client):
    for method, path in [("get", "/api/meals?date=2026-01-01"), ("get", "/api/profile"), ("get", "/api/trends?end=2026-01-01"),
                         ("delete", "/api/corrections"), ("get", "/api/photos/000000000000000000000000")]:
        assert getattr(client, method)(path).status_code == 401
