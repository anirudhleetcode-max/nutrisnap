import os
import uuid
from pathlib import Path

os.environ["MONGO_DB"] = f"test_nutrisnap_{uuid.uuid4().hex[:8]}"

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"
MODELS_PRESENT = (get_settings().model_dir / "vision_model_quantized.onnx").exists()
needs_model = pytest.mark.skipif(not MODELS_PRESENT, reason="run `python -m scripts.download_models` first")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
    import pymongo
    pymongo.MongoClient(get_settings().mongo_uri).drop_database(os.environ["MONGO_DB"])


def _register(client) -> dict:
    r = client.post("/api/auth/register", json={"name": "Test", "email": f"t{uuid.uuid4().hex[:8]}@example.com",
                                                "password": "secret123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture(scope="session")
def auth_headers(client):
    return _register(client)


@pytest.fixture
def new_user(client):
    """A fresh user per test, for tests that need an empty diary."""
    return _register(client)
