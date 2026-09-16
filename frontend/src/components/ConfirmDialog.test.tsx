import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ConfirmDialog from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  const props = { title: "Remove Idli?", body: "This deletes the entry.", confirmLabel: "Remove" };

  it("renders nothing when closed", () => {
    render(<ConfirmDialog open={false} {...props} onConfirm={vi.fn()} onCancel={vi.fn()} />);
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("focuses Cancel first and closes on Escape", () => {
    const onCancel = vi.fn();
    const onConfirm = vi.fn();
    render(<ConfirmDialog open {...props} onConfirm={onConfirm} onCancel={onCancel} />);
    expect(screen.getByRole("alertdialog", { name: "Remove Idli?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onCancel).toHaveBeenCalled();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("confirms only on the destructive button", () => {
    const onConfirm = vi.fn();
    render(<ConfirmDialog open {...props} onConfirm={onConfirm} onCancel={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });
});
