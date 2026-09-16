import type { Food, Recognition } from "../lib/types";

export const food = (over: Partial<Food> = {}): Food => ({
  id: "idli", name: "Idli", cuisine: "indian", category: "south-indian",
  per100g: { kcal: 128, protein: 6.4, carbs: 25, fat: 0.3, fibre: 5.8 },
  servings: [{ label: "1 idli", grams: 40 }, { label: "gram", grams: 1 }],
  aliases: [], source: "USDA", verified: true, source_id: "fdc:2708346",
  source_url: "https://fdc.nal.usda.gov/food-details/2708346/nutrients", ...over,
});

export const recognition = (over: Partial<Recognition> = {}): Recognition => ({
  photo_id: "p1",
  candidates: [
    { food: food(), confidence: 0.82, zero_shot: 0.7 },
    { food: food({ id: "plain-dosa", name: "Plain Dosa" }), confidence: 0.1, zero_shot: 0.2 },
  ],
  not_food: false, food_score: 0.99, also_detected: [], personalized: false, personal_weight: 0,
  suggested_meal_type: "breakfast", timing_ms: 120,
  decision: "confident", confidence: 0.82, abstain_below: 0.4, calibrated: true, ...over,
});
