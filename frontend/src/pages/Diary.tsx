import { AlertTriangle, ChevronLeft, ChevronRight, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import FoodThumb from "../components/FoodThumb";
import { KcalRing, MacroBars } from "../components/Macros";
import PortionControl from "../components/PortionControl";
import { api, ApiError } from "../lib/api";
import { addDays, dayLabel, longDate, today } from "../lib/dates";
import { MEAL_TYPES, gramsFor, kcal, nutrientsFor, portionText } from "../lib/nutrition";
import type { DayLog, Food, Meal, MealType, ProfileResp, Targets } from "../lib/types";

function MealRow({ meal, onChanged }: { meal: Meal; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [food, setFood] = useState<Food | null>(null);
  const [serving, setServing] = useState(meal.serving_label);
  const [qty, setQty] = useState(meal.quantity);
  const [type, setType] = useState<MealType>(meal.meal_type);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (open && !food) api<Food>(`/api/foods/${meal.food_id}`).then(setFood).catch(() => setErr("Couldn't load this food."));
  }, [open, food, meal.food_id]);

  const startEdit = () => { setServing(meal.serving_label); setQty(meal.quantity); setType(meal.meal_type); setErr(null); setOpen((o) => !o); };

  const save = async () => {
    setBusy(true); setErr(null);
    try {
      await api(`/api/meals/${meal.id}`, { method: "PATCH", json: { serving_label: serving, quantity: qty, meal_type: type } });
      setOpen(false); onChanged();
    } catch (e) { setErr(e instanceof ApiError ? e.message : "Couldn't save."); }
    finally { setBusy(false); }
  };
  const remove = async () => {
    if (!window.confirm(`Remove ${meal.food_name} from your diary?`)) return;
    setBusy(true);
    try { await api(`/api/meals/${meal.id}`, { method: "DELETE" }); onChanged(); }
    catch (e) { setErr(e instanceof ApiError ? e.message : "Couldn't delete."); setBusy(false); }
  };

  const preview = food ? nutrientsFor(food, gramsFor(food, serving, qty)).kcal : meal.nutrients.kcal;

  return (
    <li className={`entry ${open ? "open" : ""}`}>
      <button className="entry-main" onClick={startEdit} aria-expanded={open} data-testid="meal-row">
        <FoodThumb photoId={meal.photo_id} name={meal.food_name} />
        <span className="entry-text">
          <span className="entry-name">{meal.food_name}</span>
          <span className="xs muted">{portionText(meal.serving_label, meal.quantity, meal.grams)}</span>
        </span>
        <span className="entry-macros xs faint num">
          P {Math.round(meal.nutrients.protein)} · C {Math.round(meal.nutrients.carbs)} · F {Math.round(meal.nutrients.fat)}
        </span>
        <span className="entry-kcal num" data-testid="meal-kcal">{kcal(meal.nutrients.kcal)}</span>
      </button>
      {open && (
        <div className="entry-edit">
          {!food && !err && <span className="spinner" />}
          {food && (
            <>
              <PortionControl food={food} servingLabel={serving} quantity={qty} idPrefix={`e-${meal.id}`}
                onChange={(s, q) => { setServing(s); setQty(q); }} />
              <div className="seg" role="group" aria-label="Move to meal">
                {MEAL_TYPES.map((m) => (
                  <button key={m.key} type="button" aria-pressed={type === m.key} onClick={() => setType(m.key)}>{m.label}</button>
                ))}
              </div>
              <div className="entry-edit-actions">
                <button className="btn btn-ghost btn-sm btn-danger" onClick={remove} disabled={busy}><Trash2 aria-hidden="true" /> Remove</button>
                <span className="grow" />
                <button className="btn btn-sm" onClick={() => setOpen(false)}>Cancel</button>
                <button className="btn btn-sm btn-primary" onClick={save} disabled={busy} data-testid="update-meal">
                  Save · <span className="num">{kcal(preview)}</span> kcal
                </button>
              </div>
            </>
          )}
          {err && <div className="alert alert-error"><AlertTriangle aria-hidden="true" />{err}</div>}
        </div>
      )}
    </li>
  );
}

export default function Diary() {
  const [params, setParams] = useSearchParams();
  const date = params.get("date") ?? today();
  const [day, setDay] = useState<DayLog | null>(null);
  const [targets, setTargets] = useState<Targets | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api<DayLog>(`/api/meals?date=${date}`).then(setDay).catch((e) => setError(e instanceof ApiError ? e.message : "Couldn't load your diary."));
  }, [date]);
  useEffect(() => { setDay(null); load(); }, [load]);
  useEffect(() => { api<ProfileResp>("/api/profile").then((p) => setTargets(p.targets)).catch(() => undefined); }, []);

  const go = (d: string) => setParams(d === today() ? {} : { date: d });
  const empty = day && day.groups.every((g) => g.items.length === 0);

  return (
    <div className="page diary">
      <div className="page-head">
        <div>
          <p className="eyebrow">Diary</p>
          <h1>{dayLabel(date)}</h1>
          <p className="small muted">{longDate(date)}</p>
        </div>
        <div className="datenav">
          <button className="btn icon-btn" onClick={() => go(addDays(date, -1))} aria-label="Previous day"><ChevronLeft aria-hidden="true" /></button>
          <input className="input" type="date" value={date} max={today()} aria-label="Pick a date"
            onChange={(e) => e.target.value && go(e.target.value)} />
          <button className="btn icon-btn" onClick={() => go(addDays(date, 1))} disabled={date >= today()} aria-label="Next day"><ChevronRight aria-hidden="true" /></button>
          {date !== today() && <button className="btn btn-sm" onClick={() => go(today())}>Today</button>}
        </div>
      </div>

      {error && <div className="alert alert-error"><AlertTriangle aria-hidden="true" />{error}</div>}

      <div className="diary-grid">
        <section className="day-summary" aria-label="Day summary">
          <KcalRing value={day?.totals.kcal ?? 0} target={targets?.kcal ?? null} />
          <MacroBars totals={day?.totals ?? { kcal: 0, protein: 0, carbs: 0, fat: 0, fibre: 0 }} targets={targets} />
          {!targets && <p className="xs muted"><Link to="/profile">Set up your profile</Link> to see targets.</p>}
        </section>

        <section className="notebook" aria-label="Meals">
          {!day && !error && [0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 56, marginBottom: 12 }} />)}
          {empty && (
            <div className="empty">
              <p><strong>Nothing logged {date === today() ? "yet today" : "on this day"}.</strong></p>
              <p className="small muted">Snap a photo of your next meal, or add something you already ate.</p>
              <Link className="btn btn-primary" to={date === today() ? "/" : `/?date=${date}`}>Snap a meal</Link>
            </div>
          )}
          {day && day.groups.map((g) => (
            <div className="meal-group" key={g.meal_type}>
              <div className="meal-head">
                <h2>{MEAL_TYPES.find((m) => m.key === g.meal_type)?.label}</h2>
                <span className="num small muted">{g.items.length ? `${kcal(g.totals.kcal)} kcal` : ""}</span>
                <Link className="btn btn-ghost btn-sm add-link" to={`/?manual=1&meal=${g.meal_type}${date !== today() ? `&date=${date}` : ""}`}
                  aria-label={`Add to ${g.meal_type}`}>
                  <Plus aria-hidden="true" /> Add
                </Link>
              </div>
              {g.items.length > 0 ? (
                <ul className="entries">{g.items.map((m) => <MealRow key={m.id} meal={m} onChanged={load} />)}</ul>
              ) : (
                <p className="xs faint none">Nothing here.</p>
              )}
            </div>
          ))}
          {day && !empty && (
            <div className="day-total num">
              <span>Day total</span>
              <span>{kcal(day.totals.kcal)} kcal</span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
