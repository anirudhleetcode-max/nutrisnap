export type NutrientKey = "kcal" | "protein" | "carbs" | "fat" | "fibre";
export type Nutrients = Record<NutrientKey, number>;
export type MealType = "breakfast" | "lunch" | "snacks" | "dinner";
export type Serving = { label: string; grams: number };

export type Food = {
  id: string;
  name: string;
  cuisine: "indian" | "global";
  category: string;
  per100g: Nutrients;
  servings: Serving[];
  aliases: string[];
  source: string;
};

export type Candidate = { food: Food; confidence: number; zero_shot: number };

export type Recognition = {
  photo_id: string;
  candidates: Candidate[];
  not_food: boolean;
  food_score: number;
  also_detected: { food: Food; confidence: number }[];
  personalized: boolean;
  personal_weight: number;
  suggested_meal_type: MealType;
  timing_ms: number;
};

export type Meal = {
  id: string;
  date: string;
  meal_type: MealType;
  food_id: string;
  food_name: string;
  serving_label: string;
  quantity: number;
  grams: number;
  nutrients: Nutrients;
  photo_id: string | null;
  servings: Serving[];
  created_at: string;
};

export type DayLog = {
  date: string;
  groups: { meal_type: MealType; totals: Nutrients; items: Meal[] }[];
  totals: Nutrients;
};

export type Profile = {
  age: number;
  sex: "male" | "female";
  height_cm: number;
  weight_kg: number;
  activity: "sedentary" | "light" | "moderate" | "active" | "very_active";
  goal: "lose" | "maintain" | "gain";
};

export type Targets = Nutrients & {
  breakdown: {
    bmr: number; activity_factor: number; tdee: number; goal_adjustment: number; floored: boolean;
    bmr_formula: string; tdee_formula: string; protein_rule: string; fat_rule: string; carbs_rule: string; fibre_rule: string;
  };
};

export type ProfileResp = {
  profile: Profile | null;
  targets: Targets | null;
  activity_levels: { id: Profile["activity"]; factor: number; label: string }[];
};

export type Trends = {
  start: string;
  end: string;
  days: (Nutrients & { date: string; meals: number })[];
  targets: Nutrients | null;
  averages: Nutrients;
  macro_split: { protein: number; carbs: number; fat: number };
  logged_days: number;
  days_on_target: number;
  streak: number;
  top_foods: { food_id: string; name: string; count: number; kcal: number }[];
  by_meal_type: Partial<Record<MealType, number>>;
};
