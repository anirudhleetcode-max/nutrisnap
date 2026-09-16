from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field

MealType = Literal["breakfast", "lunch", "snacks", "dinner"]
Sex = Literal["male", "female"]
Activity = Literal["sedentary", "light", "moderate", "active", "very_active"]
Goal = Literal["lose", "maintain", "gain"]


class ProfileIn(BaseModel):
    age: int = Field(ge=13, le=100)
    sex: Sex
    height_cm: float = Field(ge=120, le=230)
    weight_kg: float = Field(ge=30, le=250)
    activity: Activity
    goal: Goal


class MealIn(BaseModel):
    food_id: str = Field(min_length=1, max_length=60)
    date: Date
    meal_type: MealType
    serving_label: str = Field(min_length=1, max_length=40)
    quantity: float = Field(gt=0, le=2000)
    photo_id: str | None = Field(default=None, max_length=40)


class MealPatch(BaseModel):
    food_id: str | None = Field(default=None, min_length=1, max_length=60)
    meal_type: MealType | None = None
    serving_label: str | None = Field(default=None, min_length=1, max_length=40)
    quantity: float | None = Field(default=None, gt=0, le=2000)
    date: Date | None = None
