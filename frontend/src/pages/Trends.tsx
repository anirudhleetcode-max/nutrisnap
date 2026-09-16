import { AlertTriangle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import { parseIso, today } from "../lib/dates";
import { MACROS, MEAL_TYPES, fmt, kcal } from "../lib/nutrition";
import type { Trends as T } from "../lib/types";

function CalorieChart({ data }: { data: T }) {
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(720);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = W < 500 ? 200 : 240, padL = 36, padR = 4, padT = 22, padB = 26;
  const target = data.targets?.kcal ?? 0;
  const max = Math.max(target * 1.25, ...data.days.map((d) => d.kcal), 500);
  const yMax = Math.ceil(max / 500) * 500;
  const n = data.days.length;
  const bw = (W - padL - padR) / n;
  const y = (v: number) => padT + (H - padT - padB) * (1 - v / yMax);
  const ticks = Array.from({ length: yMax / 500 + 1 }, (_, i) => i * 500).filter((t) => yMax <= 3000 || t % 1000 === 0);
  const every = n > 14 ? (W < 500 ? 7 : 5) : 1;
  const h = hover !== null ? data.days[hover] : null;
  return (
    <div className="chart" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Calories per day for the last ${n} days`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={padL - 6} y={y(t) + 4} textAnchor="end" className="axis">{t === 0 ? "0" : `${t / 1000}k`}</text>
          </g>
        ))}
        {data.days.map((d, i) => {
          const x = padL + i * bw;
          const over = target > 0 && d.kcal > target * 1.1;
          return (
            <g key={d.date} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={x} y={padT} width={bw} height={H - padT - padB} fill={hover === i ? "var(--surface-2)" : "transparent"} />
              {d.kcal > 0 && (
                <rect x={x + bw * 0.18} width={bw * 0.64} y={y(d.kcal)} height={y(0) - y(d.kcal)}
                  fill={over ? "var(--over)" : "var(--kcal)"} opacity={d.date === today() ? 0.55 : 1} rx={1.5} />
              )}
              {i % every === 0 && (
                <text x={x + bw / 2} y={H - 8} textAnchor="middle" className="axis">
                  {n > 7 ? parseIso(d.date).getDate() : parseIso(d.date).toLocaleDateString("en-IN", { weekday: "short" })}
                </text>
              )}
            </g>
          );
        })}
        {target > 0 && (
          <g>
            <line x1={padL} x2={W - padR} y1={y(target)} y2={y(target)} stroke="var(--ink)" strokeDasharray="4 4" />
            <rect x={padL + 2} y={y(target) - 17} width={82} height={14} fill="var(--surface)" />
            <text x={padL + 6} y={y(target) - 6} className="axis strong">target {kcal(target)}</text>
          </g>
        )}
      </svg>
      <p className="chart-caption small num" aria-live="polite">
        {h ? (
          <>
            <strong>{parseIso(h.date).toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" })}</strong>
            {" · "}{h.meals ? `${kcal(h.kcal)} kcal · P ${fmt(h.protein, 0)} g · C ${fmt(h.carbs, 0)} g · F ${fmt(h.fat, 0)} g` : "nothing logged"}
          </>
        ) : (
          <span className="faint">Hover a day for details. Faded bar = today (still in progress). Dark bars are more than 10% over target.</span>
        )}
      </p>
    </div>
  );
}

export default function Trends() {
  const [days, setDays] = useState(7);
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    api<T>(`/api/trends?days=${days}&end=${today()}`).then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Couldn't load trends."));
  }, [days]);

  const target = data?.targets?.kcal ?? null;
  const avgDiff = data && target ? Math.round(data.averages.kcal - target) : null;
  const mealTotal = data ? Object.values(data.by_meal_type).reduce((a, b) => a + (b ?? 0), 0) : 0;

  return (
    <div className="page trends">
      <div className="page-head">
        <div>
          <p className="eyebrow">Trends</p>
          <h1>How the last {days} days went</h1>
        </div>
        <div className="seg" role="group" aria-label="Range">
          {[7, 30].map((d) => <button key={d} aria-pressed={days === d} onClick={() => setDays(d)}>{d} days</button>)}
        </div>
      </div>

      {error && <div className="alert alert-error" role="alert"><AlertTriangle aria-hidden="true" />{error}</div>}
      {!data && !error && <div className="skeleton" style={{ height: 320 }} />}

      {data && data.logged_days === 0 && (
        <div className="empty">
          <p><strong>No meals in this period.</strong></p>
          <p className="small muted">Trends fill in once you’ve logged a couple of days.</p>
          <Link className="btn btn-primary" to="/">Log a meal</Link>
        </div>
      )}

      {data && data.logged_days > 0 && (
        <>
          <dl className="stats">
            <div>
              <dt>Average intake</dt>
              <dd className="num" data-testid="avg-kcal">{kcal(data.averages.kcal)} <small>kcal/day</small></dd>
              {avgDiff !== null && (
                <p className={`xs ${Math.abs(avgDiff) <= (target ?? 0) * 0.1 ? "ok" : "muted"}`}>
                  {avgDiff === 0 ? "right on target" : `${kcal(Math.abs(avgDiff))} ${avgDiff > 0 ? "above" : "below"} target`}
                </p>
              )}
            </div>
            <div>
              <dt>Streak</dt>
              <dd className="num">{data.streak} <small>{data.streak === 1 ? "day" : "days"}</small></dd>
              <p className="xs muted">consecutive days logged</p>
            </div>
            <div>
              <dt>Days logged</dt>
              <dd className="num">{data.logged_days}<small> / {days}</small></dd>
              <p className="xs muted">{target ? `${data.days_on_target} within ±10% of target` : "set a target in Profile"}</p>
            </div>
            <div>
              <dt>Macro split</dt>
              <dd className="split-bar" aria-label={`Protein ${data.macro_split.protein}%, carbs ${data.macro_split.carbs}%, fat ${data.macro_split.fat}%`}>
                <span className="fill-protein" style={{ width: `${data.macro_split.protein}%` }} />
                <span className="fill-carbs" style={{ width: `${data.macro_split.carbs}%` }} />
                <span className="fill-fat" style={{ width: `${data.macro_split.fat}%` }} />
              </dd>
              <p className="xs muted num">P {data.macro_split.protein}% · C {data.macro_split.carbs}% · F {data.macro_split.fat}% of kcal</p>
            </div>
          </dl>

          <section className="panel">
            <div className="panel-head"><h2>Calories per day</h2></div>
            <CalorieChart data={data} />
          </section>

          <div className="two-col">
            <section className="panel">
              <div className="panel-head"><h2>Daily averages</h2><span className="xs faint">on logged days</span></div>
              <table className="table num">
                <thead><tr><th>Nutrient</th><th>Average</th><th>Target</th><th /></tr></thead>
                <tbody>
                  {MACROS.map((m) => {
                    const avg = data.averages[m.key];
                    const t = data.targets?.[m.key];
                    return (
                      <tr key={m.key}>
                        <td><span className={`dot dot-${m.key}`} /> {m.label}</td>
                        <td>{fmt(avg, 0)} g</td>
                        <td className="faint">{t ? `${t} g` : "—"}</td>
                        <td className="bar-cell">
                          <span className="track"><span className={`fill fill-${m.key}`} style={{ width: `${t ? Math.min(100, (avg / t) * 100) : 0}%` }} /></span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <h3 className="sub">Where the calories come from</h3>
              <ul className="meal-split">
                {MEAL_TYPES.map((m) => {
                  const v = data.by_meal_type[m.key] ?? 0;
                  const share = mealTotal ? Math.round((v / mealTotal) * 100) : 0;
                  return (
                    <li key={m.key}>
                      <span>{m.label}</span>
                      <span className="track"><span className="fill fill-kcal" style={{ width: `${share}%` }} /></span>
                      <span className="num small">{share}%</span>
                    </li>
                  );
                })}
              </ul>
            </section>
            <section className="panel">
              <div className="panel-head"><h2>Most logged</h2></div>
              <ol className="top-foods">
                {data.top_foods.map((f, i) => (
                  <li key={f.food_id}>
                    <span className="rank num">{i + 1}</span>
                    <span className="grow">{f.name}</span>
                    <span className="num small muted">{f.count}×</span>
                    <span className="num small">{kcal(f.kcal)} kcal</span>
                  </li>
                ))}
              </ol>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
