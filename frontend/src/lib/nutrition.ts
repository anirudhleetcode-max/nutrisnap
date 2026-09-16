import type { Food, MealType, NutrientKey, Nutrients } from "./types";

export const MACROS: { key: Exclude<NutrientKey, "kcal">; label: string; unit: string }[] = [
  { key: "protein", label: "Protein", unit: "g" },
  { key: "carbs", label: "Carbs", unit: "g" },
  { key: "fat", label: "Fat", unit: "g" },
  { key: "fibre", label: "Fibre", unit: "g" },
];

export const MEAL_TYPES: { key: MealType; label: string }[] = [
  { key: "breakfast", label: "Breakfast" },
  { key: "lunch", label: "Lunch" },
  { key: "snacks", label: "Snacks" },
  { key: "dinner", label: "Dinner" },
];

export function gramsFor(food: Food, servingLabel: string, qty: number): number {
  if (servingLabel === "gram") return Math.round(qty * 10) / 10;
  const s = food.servings.find((x) => x.label === servingLabel);
  return s ? Math.round(s.grams * qty * 10) / 10 : 0;
}

export function nutrientsFor(food: Food, grams: number): Nutrients {
  const f = grams / 100;
  const r = (v: number) => Math.round(v * f * 10) / 10;
  const p = food.per100g;
  return { kcal: r(p.kcal), protein: r(p.protein), carbs: r(p.carbs), fat: r(p.fat), fibre: r(p.fibre) };
}

export function mealTypeForHour(h: number): MealType {
  if (h >= 5 && h < 11) return "breakfast";
  if (h >= 11 && h < 16) return "lunch";
  if (h >= 16 && h < 19) return "snacks";
  return "dinner";
}

export const servingText = (label: string) => (label === "gram" ? "grams" : label);

export function portionText(label: string, qty: number, grams: number) {
  if (label === "gram") return `${fmt(grams)} g`;
  const q = qty === 1 ? "" : `${fmt(qty)} × `;
  return `${q}${label} · ${fmt(grams)} g`;
}

export const fmt = (n: number, digits = 1) => {
  const r = Math.round(n * 10 ** digits) / 10 ** digits;
  return Number.isInteger(r) ? String(r) : r.toFixed(digits);
};
export const kcal = (n: number) => Math.round(n).toLocaleString("en-IN");
