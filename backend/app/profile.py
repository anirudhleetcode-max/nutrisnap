from fastapi import APIRouter, Depends

from .db import get_db
from .schemas import ProfileIn
from .security import current_user
from .services.nutrition import ACTIVITY, targets

router = APIRouter(prefix="/api/profile", tags=["profile"])


def profile_out(profile: dict | None) -> dict:
    return {
        "profile": profile,
        "targets": targets(profile) if profile else None,
        "activity_levels": [{"id": k, "factor": v[0], "label": v[1]} for k, v in ACTIVITY.items()],
    }


@router.get("")
async def get_profile(user: dict = Depends(current_user)):
    return profile_out(user.get("profile"))


@router.put("")
async def put_profile(body: ProfileIn, user: dict = Depends(current_user)):
    p = body.model_dump()
    await get_db().users.update_one({"_id": user["_id"]}, {"$set": {"profile": p}})
    return profile_out(p)


@router.post("/estimate")
async def estimate(body: ProfileIn, _: dict = Depends(current_user)):
    """Preview targets without saving (the form updates live)."""
    return {"targets": targets(body.model_dump())}
