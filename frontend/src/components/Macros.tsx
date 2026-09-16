import { MACROS, fmt, kcal } from "../lib/nutrition";
import type { Nutrients } from "../lib/types";

/** Compact readout for a single food/portion. */
export function NutrientReadout({ n }: { n: Nutrients }) {
  return (
    <div className="readout" aria-live="polite">
      <div className="readout-kcal">
        <span className="num big" data-testid="portion-kcal">{kcal(n.kcal)}</span>
        <span className="muted">kcal</span>
      </div>
      <dl className="readout-macros">
        {MACROS.map((m) => (
          <div key={m.key} className={`rm rm-${m.key}`}>
            <dt>{m.label}</dt>
            <dd className="num">{fmt(n[m.key])}<small> g</small></dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Progress bars against targets. */
export function MacroBars({ totals, targets, compact }: { totals: Nutrients; targets: Nutrients | null; compact?: boolean }) {
  return (
    <div className={`mbars ${compact ? "compact" : ""}`}>
      {MACROS.map((m) => {
        const t = targets?.[m.key] ?? 0;
        const pct = t ? Math.min(100, (totals[m.key] / t) * 100) : 0;
        const over = t > 0 && totals[m.key] > t * 1.05 && m.key !== "fibre" && m.key !== "protein";
        return (
          <div className="mbar" key={m.key}>
            <div className="mbar-head">
              <span className={`dot dot-${m.key}`} />
              <span className="mbar-label">{m.label}</span>
              <span className={`num mbar-val ${over ? "is-over" : ""}`}>
                {fmt(totals[m.key], 0)}{t ? <span className="faint"> / {t} g</span> : " g"}
              </span>
            </div>
            <div className="track" role="progressbar" aria-label={m.label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
              <div className={`fill fill-${m.key} ${over ? "over" : ""}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function KcalRing({ value, target, size = 148 }: { value: number; target: number | null; size?: number }) {
  const stroke = 11;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = target ? Math.min(1, value / target) : 0;
  const over = target ? value > target : false;
  const left = target ? Math.round(target - value) : null;
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-2)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={over ? "var(--over)" : "var(--kcal)"} strokeWidth={stroke}
          strokeDasharray={`${c * pct} ${c}`} strokeLinecap="butt" transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      </svg>
      <div className="ring-center">
        <span className="num ring-value" data-testid="day-kcal">{kcal(value)}</span>
        <span className="xs faint">{target ? `of ${kcal(target)} kcal` : "kcal"}</span>
        {left !== null && (
          <span className={`xs ring-left ${over ? "over" : ""}`}>{over ? `${kcal(-left)} over` : `${kcal(left)} left`}</span>
        )}
      </div>
    </div>
  );
}
