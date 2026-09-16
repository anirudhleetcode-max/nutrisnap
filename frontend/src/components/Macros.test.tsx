import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KcalRing, NutrientReadout } from "./Macros";

describe("KcalRing", () => {
  it("shows calories left under target", () => {
    render(<KcalRing value={1500} target={2000} />);
    expect(screen.getByText("500 left")).toBeInTheDocument();
  });
  it("shows the overage above target", () => {
    render(<KcalRing value={2350} target={2000} />);
    expect(screen.getByText("350 over")).toBeInTheDocument();
  });
  it("works without a target", () => {
    render(<KcalRing value={800} target={null} />);
    expect(screen.queryByText(/left|over/)).toBeNull();
  });
});

describe("NutrientReadout", () => {
  const n = { kcal: 256, protein: 12.8, carbs: 50, fat: 0.6, fibre: 11.6 };
  it("marks estimated values with an approximately sign", () => {
    render(<NutrientReadout n={n} estimated />);
    expect(screen.getByLabelText("approximately")).toBeInTheDocument();
    expect(screen.getByTestId("portion-kcal")).toHaveTextContent("256");
  });
  it("shows verified values plainly", () => {
    render(<NutrientReadout n={n} />);
    expect(screen.queryByLabelText("approximately")).toBeNull();
  });
});
