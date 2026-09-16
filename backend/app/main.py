import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from . import auth, errors, foods, meals, profile, recognize, trends
from .config import get_settings
from .db import close_client, ensure_indexes, get_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    await foods.seed_foods()
    await run_in_threadpool(recognize.load_model)  # CLIP loads once, at startup
    yield
    await close_client()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
errors.install(app)
for r in (auth.router, foods.router, recognize.router, meals.router, profile.router, trends.router):
    app.include_router(r)


@app.get("/api/health")
async def health():
    await get_db().command("ping")
    return {"status": "ok", "db": "connected", "model_loaded": recognize._recognizer is not None}
