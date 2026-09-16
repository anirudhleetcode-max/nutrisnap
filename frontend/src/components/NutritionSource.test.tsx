import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { food } from "../test/fixtures";
import NutritionSource from "./NutritionSource";

describe("NutritionSource", () => {
  it("links verified foods to their USDA record", () => {
    render(<NutritionSource food={food()} />);
    expect(screen.getByText(/USDA verified/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "FDC 2708346" })).toHaveAttribute(
      "href", "https://fdc.nal.usda.gov/food-details/2708346/nutrients");
  });

  it("labels unverified foods as estimates, with no fake source link", () => {
    render(<NutritionSource food={food({ verified: false, source_id: null, source_url: null, source: "Estimate - typical recipe" })} />);
    const badge = screen.getByTestId("nutrition-source");
    expect(badge).toHaveTextContent("Estimate — unverified");
    expect(badge).toHaveClass("src-estimate");
    expect(screen.queryByRole("link")).toBeNull();
  });
});
