import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../lib/auth";
import Auth from "./Auth";

function setup() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <Auth mode="login" />
      </AuthProvider>
    </MemoryRouter>,
  );
}

describe("Sign-in page", () => {
  it("fills the demo credentials", () => {
    setup();
    fireEvent.click(screen.getByRole("button", { name: "Use demo account" }));
    expect(screen.getByLabelText("Email")).toHaveValue("demo@nutrisnap.app");
    expect(screen.getByLabelText("Password")).toHaveValue("demo1234");
  });

  it("shows the server's error message on failed login", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Incorrect email or password", code: "unauthorized" }), { status: 401 }),
    );
    setup();
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.co" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpass" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Incorrect email or password")).toBeInTheDocument();
  });

  it("explains rate limiting", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Too many failed attempts. Try again in 120 s.", code: "rate_limited" }), { status: 429 }),
    );
    setup();
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.co" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpass" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many failed attempts");
  });
});
