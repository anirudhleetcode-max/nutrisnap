import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { food } from "../test/fixtures";
import PortionControl from "./PortionControl";

describe("PortionControl", () => {
  it("steps servings by half and never goes below half a serving", () => {
    const onChange = vi.fn();
    const { rerender } = render(<PortionControl food={food()} servingLabel="1 idli" quantity={1} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "More" }));
    expect(onChange).toHaveBeenLastCalledWith("1 idli", 1.5);
    rerender(<PortionControl food={food()} servingLabel="1 idli" quantity={0.5} onChange={onChange} />);
    expect(screen.getByRole("button", { name: "Less" })).toBeDisabled();
  });

  it("uses 10 g steps and resets quantity when switching to grams", () => {
    const onChange = vi.fn();
    render(<PortionControl food={food()} servingLabel="1 idli" quantity={2} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Serving"), { target: { value: "gram" } });
    expect(onChange).toHaveBeenLastCalledWith("gram", 100);
  });

  it("ignores invalid typed quantities", () => {
    const onChange = vi.fn();
    render(<PortionControl food={food()} servingLabel="1 idli" quantity={2} onChange={onChange} />);
    const input = screen.getByTestId("qty");
    fireEvent.change(input, { target: { value: "-3" } });
    fireEvent.change(input, { target: { value: "abc" } });
    fireEvent.change(input, { target: { value: "5000" } });
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: "3" } });
    expect(onChange).toHaveBeenCalledWith("1 idli", 3);
  });
});
