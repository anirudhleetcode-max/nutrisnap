import type { Recognition } from "./types";

export type ResultState = {
  kind: "confident" | "uncertain" | "not_food";
  title: string;
  detail: string;
  preselect: boolean;
};

export const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);

/** Turns the API's decision into what the Snap screen says and whether it preselects a food. */
export function describeResult(r: Recognition): ResultState {
  const top = r.candidates[0];
  const conf = r.confidence ?? top.confidence;
  // older API versions had no decision field: fall back to the raw flags
  const kind = r.decision ?? (r.not_food ? "not_food" : "confident");
  const how = r.calibrated ? "calibrated confidence" : r.personalized ? "confidence (includes your corrections)" : "confidence";
  if (kind === "not_food") {
    return {
      kind, preselect: false,
      title: "This doesn’t look like food",
      detail: "If it is, pick the dish below or search for it.",
    };
  }
  if (kind === "uncertain") {
    return {
      kind, preselect: false,
      title: "Not sure — pick from the list",
      detail: `Best guess is ${top.food.name} at ${pct(conf)} ${how}, below the ${pct(r.abstain_below ?? 0)} we need to choose for you.`,
    };
  }
  return { kind, preselect: true, title: top.food.name, detail: `${pct(conf)} ${how}` };
}

export const QUALITY_TEXT: Record<string, string> = {
  too_dark: "The photo is very dark.",
  overexposed: "The photo is very bright.",
  blurry: "The photo looks blurry.",
};
