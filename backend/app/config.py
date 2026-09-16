from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "NutriSnap"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "nutrisnap"
    jwt_secret: str = "dev-only-secret-change-me-in-production-0123456789"
    jwt_expire_minutes: int = 60 * 24 * 7
    cors_origins: str = "http://localhost:5176,http://127.0.0.1:5176"

    model_dir: Path = BACKEND_DIR / "models"
    foods_file: Path = BACKEND_DIR / "data" / "foods.json"
    max_upload_mb: float = 8.0
    onnx_threads: int = 2
    knn_memory_limit: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
