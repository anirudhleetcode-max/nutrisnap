import { BadgeCheck, CircleDashed } from "lucide-react";
import type { Food } from "../lib/types";

/** Shows where a food's nutrition numbers come from. Unverified estimates must look different. */
export default function NutritionSource({ food, compact }: { food: Pick<Food, "verified" | "source" | "source_id" | "source_url" | "source_note">; compact?: boolean }) {
  if (food.verified) {
    const id = food.source_id?.replace("fdc:", "") ?? "";
    return (
      <span className="src src-verified" data-testid="nutrition-source">
        <BadgeCheck aria-hidden="true" />
        <span>
          USDA verified{!compact && <> · <a href={food.source_url ?? "#"} target="_blank" rel="noreferrer">FDC {id}</a></>}
          {!compact && food.source_note && <span className="faint"> · {food.source_note}</span>}
        </span>
      </span>
    );
  }
  return (
    <span className="src src-estimate" data-testid="nutrition-source" title={food.source}>
      <CircleDashed aria-hidden="true" />
      <span>Estimate — unverified{!compact && <span className="faint"> · typical recipe, not checked against a database</span>}</span>
    </span>
  );
}
