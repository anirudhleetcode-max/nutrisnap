export function isoDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export const today = () => isoDate(new Date());

export function parseIso(s: string): Date {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(y, m - 1, d);
}

export function addDays(s: string, n: number): string {
  const d = parseIso(s);
  d.setDate(d.getDate() + n);
  return isoDate(d);
}

export function dayLabel(s: string): string {
  const t = today();
  if (s === t) return "Today";
  if (s === addDays(t, -1)) return "Yesterday";
  return parseIso(s).toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" });
}

export const longDate = (s: string) =>
  parseIso(s).toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
