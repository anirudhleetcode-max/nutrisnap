import pytest

from app.services.nutrition import bmr_mifflin, grams_for, meal_type_for_hour, nutrients_for, targets

FOOD = {"name": "Roti", "per100g": {"kcal": 280, "protein": 9, "carbs": 50, "fat": 4.5, "fibre": 6},
        "servings": [{"label": "1 roti", "grams": 40}, {"label": "gram", "grams": 1}]}


def test_mifflin_st_jeor_reference_values():
    # 10*70 + 6.25*175 - 5*25 + 5 = 1673.75
    assert bmr_mifflin("male", 70, 175, 25) == pytest.approx(1673.75)
    # 10*60 + 6.25*160 - 5*30 - 161 = 1289
    assert bmr_mifflin("female", 60, 160, 30) == pytest.approx(1289)


def test_targets_split_adds_up():
    t = targets({"sex": "male", "weight_kg": 70, "height_cm": 175, "age": 25, "activity": "moderate", "goal": "maintain"})
    assert t["breakdown"]["tdee"] == round(1673.75 * 1.55)
    assert t["kcal"] == 2590
    assert t["protein"] == 84  # 1.2 g/kg
    assert abs(t["protein"] * 4 + t["carbs"] * 4 + t["fat"] * 9 - t["kcal"]) < 10
    lose = targets({"sex": "female", "weight_kg": 50, "height_cm": 150, "age": 60, "activity": "sedentary", "goal": "lose"})
    assert lose["kcal"] == 1200 and lose["breakdown"]["floored"]


def test_portions():
    assert grams_for(FOOD, "1 roti", 3) == 120
    assert grams_for(FOOD, "gram", 55) == 55
    with pytest.raises(ValueError):
        grams_for(FOOD, "1 plate", 1)
    n = nutrients_for(FOOD, 80)
    assert n == {"kcal": 224.0, "protein": 7.2, "carbs": 40.0, "fat": 3.6, "fibre": 4.8}


@pytest.mark.parametrize("hour,meal", [(7, "breakfast"), (13, "lunch"), (17, "snacks"), (21, "dinner"), (2, "dinner")])
def test_meal_type_hint(hour, meal):
    assert meal_type_for_hour(hour) == meal
