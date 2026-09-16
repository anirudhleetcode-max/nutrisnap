import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

# values that must never sign real tokens
PLACEHOLDER_SECRETS = {"", "change-me", "change-me-in-production", "replace-with-a-long-random-string",
                       "dev-only-secret-change-me-in-production-0123456789", "secret"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"            # development | production
    app_name: str = "NutriSnap"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "nutrisnap"
    jwt_secret: str = ""
    jwt_expire_minutes: int = 60 * 24 * 7
    cors_origins: str = "http://localhost:5176,http://127.0.0.1:5176"

    model_dir: Path = BACKEND_DIR / "models"
    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    foods_file: Path = BACKEND_DIR / "data" / "foods.json"
    max_upload_mb: float = 8.0
    onnx_threads: int = 2
    knn_memory_limit: int = 500
    login_max_failures: int = 5
    login_window_seconds: int = 300

    jwt_secret_ephemeral: bool = False

    @model_validator(mode="after")
    def check_secret(self):
        if self.jwt_secret in PLACEHOLDER_SECRETS or len(self.jwt_secret) < 32:
            if self.env != "development":
                raise ValueError("JWT_SECRET is missing, a placeholder or shorter than 32 characters. Generate one with "
                                 "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")
            log.warning("JWT_SECRET not set (or a placeholder): using a random secret for this process only. "
                        "Sessions end when the server restarts. Set JWT_SECRET in backend/.env.")
            self.jwt_secret = secrets.token_urlsafe(48)
            self.jwt_secret_ephemeral = True
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
