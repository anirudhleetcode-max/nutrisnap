import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from pymongo.errors import DuplicateKeyError

from .config import get_settings
from .db import get_db
from .security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


def public_user(u: dict) -> dict:
    return {"id": str(u["_id"]), "name": u["name"], "email": u["email"], "has_profile": bool(u.get("profile"))}


@router.post("/register", status_code=201)
async def register(body: RegisterIn):
    doc = {
        "name": body.name.strip(),
        "email": body.email.lower(),
        "password_hash": hash_password(body.password),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        res = await get_db().users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists")
    doc["_id"] = res.inserted_id
    return {"token": create_token(str(res.inserted_id)), "user": public_user(doc)}


# In-memory failed-login limiter: per (client IP, email), sliding window. Good enough for a single
# process; a multi-worker deployment would need a shared store (e.g. Redis).
_failures: dict[tuple[str, str], deque] = defaultdict(deque)


def _recent_failures(key: tuple[str, str]) -> deque:
    s = get_settings()
    q = _failures[key]
    while q and q[0] < time.monotonic() - s.login_window_seconds:
        q.popleft()
    return q


@router.post("/login")
async def login(body: LoginIn, request: Request):
    s = get_settings()
    key = (request.client.host if request.client else "?", body.email.lower())
    q = _recent_failures(key)
    if len(q) >= s.login_max_failures:
        retry = int(q[0] + s.login_window_seconds - time.monotonic()) + 1
        raise HTTPException(429, f"Too many failed attempts. Try again in {retry} s.", headers={"Retry-After": str(retry)})
    u = await get_db().users.find_one({"email": body.email.lower()})
    if not u or not verify_password(body.password, u["password_hash"]):
        q.append(time.monotonic())
        raise HTTPException(401, "Incorrect email or password")
    _failures.pop(key, None)
    return {"token": create_token(str(u["_id"])), "user": public_user(u)}


@router.get("/me")
async def me(user: dict = Depends(current_user)):
    return public_user(user)
