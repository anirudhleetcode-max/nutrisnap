import { describe, expect, it } from "vitest";
import { recognition } from "../test/fixtures";
import { describeResult, pct } from "./recognition";

describe("describeResult", () => {
  it("preselects the top food only when confident", () => {
    const s = describeResult(recognition());
    expect(s.kind).toBe("confident");
    expect(s.preselect).toBe(true);
    expect(s.detail).toBe("82% calibrated confidence");
  });

  it("abstains below the threshold and explains why", () => {
    const s = describeResult(recognition({ decision: "uncertain", confidence: 0.21, abstain_below: 0.4 }));
    expect(s.preselect).toBe(false);
    expect(s.title).toBe("Not sure — pick from the list");
    expect(s.detail).toContain("21%");
    expect(s.detail).toContain("40%");
  });

  it("flags non-food photos", () => {
    const s = describeResult(recognition({ decision: "not_food", not_food: true }));
    expect(s.kind).toBe("not_food");
    expect(s.preselect).toBe(false);
  });

  it("falls back to legacy fields when the API has no decision", () => {
    const r = recognition({ decision: undefined, calibrated: undefined, not_food: true });
    expect(describeResult(r).kind).toBe("not_food");
  });

  it("formats edge percentages", () => {
    expect(pct(0.999)).toBe("99%");
    expect(pct(0.004)).toBe("<1%");
  });
});
