"""Pure nutrition maths (no I/O) - easy to unit test and to explain."""
from __future__ import annotations

NUTRIENTS = ("kcal", "protein", "carbs", "fat", "fibre")

ACTIVITY = {
    "sedentary": (1.2, "desk job, little exercise"),
    "light": (1.375, "exercise 1-3 days/week"),
    "moderate": (1.55, "exercise 3-5 days/week"),
    "active": (1.725, "exercise 6-7 days/week"),
    "very_active": (1.9, "physical job or twice-a-day training"),
}
GOAL_DELTA = {"lose": -500, "maintain": 0, "gain": 300}
PROTEIN_G_PER_KG = {"lose": 1.6, "maintain": 1.2, "gain": 1.6}
FAT_SHARE = 0.25
MIN_KCAL = {"male": 1500, "female": 1200}


def grams_for(food: dict, serving_label: str, quantity: float) -> float:
    if serving_label == "gram":
        return round(quantity, 1)
    for s in food["servings"]:
        if s["label"] == serving_label:
            return round(s["grams"] * quantity, 1)
    raise ValueError(f"'{serving_label}' is not a serving size for {food['name']}")


def nutrients_for(food: dict, grams: float) -> dict:
    f = grams / 100.0
    return {k: round(food["per100g"][k] * f, 1) for k in NUTRIENTS}


def bmr_mifflin(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    """Mifflin-St Jeor (1990): 10W + 6.25H - 5A + s, s = +5 (male) / -161 (female)."""
    s = 5 if sex == "male" else -161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + s


def targets(profile: dict) -> dict:
    sex, w, h, a = profile["sex"], profile["weight_kg"], profile["height_cm"], profile["age"]
    goal, activity = profile["goal"], profile["activity"]
    bmr = bmr_mifflin(sex, w, h, a)
    factor = ACTIVITY[activity][0]
    tdee = bmr * factor
    kcal = max(tdee + GOAL_DELTA[goal], MIN_KCAL[sex])
    kcal = round(kcal / 10) * 10
    protein = round(PROTEIN_G_PER_KG[goal] * w)
    fat = round(kcal * FAT_SHARE / 9)
    carbs = max(0, round((kcal - protein * 4 - fat * 9) / 4))
    fibre = round(kcal / 1000 * 14)
    s = "+ 5" if sex == "male" else "- 161"
    return {
        "kcal": kcal, "protein": protein, "carbs": carbs, "fat": fat, "fibre": fibre,
        "breakdown": {
            "bmr": round(bmr), "activity_factor": factor, "tdee": round(tdee),
            "goal_adjustment": GOAL_DELTA[goal], "floored": kcal > round(tdee + GOAL_DELTA[goal]) + 5,
            "bmr_formula": f"10 x {w:g} kg + 6.25 x {h:g} cm - 5 x {a} y {s} = {round(bmr)} kcal",
            "tdee_formula": f"{round(bmr)} x {factor} ({activity.replace('_', ' ')}) = {round(tdee)} kcal",
            "protein_rule": f"{PROTEIN_G_PER_KG[goal]} g per kg body weight",
            "fat_rule": f"{int(FAT_SHARE * 100)}% of calories",
            "carbs_rule": "remaining calories",
            "fibre_rule": "14 g per 1000 kcal",
        },
    }


def meal_type_for_hour(hour: int) -> str:
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 16 <= hour < 19:
        return "snacks"
    return "dinner"
