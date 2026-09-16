import { AlertTriangle, Camera, Check, ImageUp, Info, RotateCcw, Sparkles, Video } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import FoodSearch from "../components/FoodSearch";
import { KcalRing, NutrientReadout } from "../components/Macros";
import PortionControl, { defaultQty } from "../components/PortionControl";
import Webcam from "../components/Webcam";
import { api, ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { dayLabel, today } from "../lib/dates";
import { MEAL_TYPES, gramsFor, kcal, mealTypeForHour, nutrientsFor } from "../lib/nutrition";
import type { DayLog, Food, Meal, MealType, ProfileResp, Recognition } from "../lib/types";

type Phase = "idle" | "analysing" | "ready" | "error";
const MAX_MB = 8;

const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);

function confidenceWord(p: number) {
  if (p >= 0.6) return "Confident";
  if (p >= 0.3) return "Likely";
  return "Unsure";
}

export default function Snap() {
  const { user } = useAuth();
  const [params, setParams] = useSearchParams();
  const date = params.get("date") ?? today();
  const presetMeal = params.get("meal") as MealType | null;

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [rec, setRec] = useState<Recognition | null>(null);
  const [food, setFood] = useState<Food | null>(null);
  const [serving, setServing] = useState("");
  const [qty, setQty] = useState(1);
  const [mealType, setMealType] = useState<MealType>(presetMeal ?? mealTypeForHour(new Date().getHours()));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<Meal | null>(null);
  const [webcam, setWebcam] = useState(false);
  const [manual, setManual] = useState(params.get("manual") === "1");
  const [drag, setDrag] = useState(false);
  const [day, setDay] = useState<DayLog | null>(null);
  const [target, setTarget] = useState<number | null>(null);
  const closeWebcam = useCallback(() => setWebcam(false), []);
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraInput = useRef<HTMLInputElement>(null);

  const loadDay = useCallback(() => {
    api<DayLog>(`/api/meals?date=${date}`).then(setDay).catch(() => undefined);
  }, [date]);
  useEffect(() => {
    loadDay();
    api<ProfileResp>("/api/profile").then((p) => setTarget(p.targets?.kcal ?? null)).catch(() => undefined);
  }, [loadDay]);

  const chooseFood = (f: Food) => {
    setFood(f);
    const first = f.servings[0].label;
    setServing(first);
    setQty(defaultQty(first));
  };

  const reset = () => {
    setPhase("idle"); setRec(null); setFood(null); setError(null); setSaved(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null); setManual(false);
    if (params.has("manual")) { params.delete("manual"); setParams(params, { replace: true }); }
  };

  const analyse = async (file: File) => {
    setWebcam(false); setSaved(null); setManual(false); setError(null);
    if (!file.type.startsWith("image/")) { setPhase("error"); setError("That isn't an image. Pick a JPEG, PNG or WebP photo."); return; }
    if (file.size > MAX_MB * 1024 * 1024) { setPhase("error"); setError(`That photo is over ${MAX_MB} MB. Try a smaller one.`); return; }
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setPhase("analysing"); setRec(null); setFood(null);
    const fd = new FormData();
    fd.append("image", file);
    try {
      const r = await api<Recognition>(`/api/recognize?hour=${new Date().getHours()}`, { method: "POST", body: fd });
      setRec(r);
      chooseFood(r.candidates[0].food);
      if (!presetMeal) setMealType(r.suggested_meal_type);
      setPhase("ready");
    } catch (e) {
      setPhase("error");
      setError(e instanceof ApiError ? e.message : "Couldn't reach the server. Check your connection and try again.");
    }
  };

  const save = async () => {
    if (!food) return;
    setSaving(true); setError(null);
    try {
      const m = await api<Meal>("/api/meals", {
        method: "POST",
        json: { food_id: food.id, date, meal_type: mealType, serving_label: serving, quantity: qty, photo_id: rec?.photo_id ?? null },
      });
      setSaved(m);
      loadDay();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Saving failed. Try again.");
    } finally {
      setSaving(false);
    }
  };

  const grams = food ? gramsFor(food, serving, qty) : 0;
  const n = food ? nutrientsFor(food, grams) : null;
  const top = rec?.candidates[0];
  const pickedIsTop = food && top && food.id === top.food.id;
  const inCandidates = food && rec?.candidates.some((c) => c.food.id === food.id);
  const showWorkbench = phase === "analysing" || phase === "ready" || manual || (phase === "error" && !!preview);

  return (
    <div className="page snap">
      <div className="page-head">
        <div>
          <p className="eyebrow">{dayLabel(date)}{date !== today() && " · logging for a past day"}</p>
          <h1>What’s on your plate?</h1>
        </div>
        <div className="today-strip" aria-label="Today so far">
          <KcalRing value={day?.totals.kcal ?? 0} target={target} size={64} />
          <div>
            <p className="small"><strong className="num">{kcal(day?.totals.kcal ?? 0)}</strong> kcal logged</p>
            <Link to={date === today() ? "/diary" : `/diary?date=${date}`} className="small">Open diary</Link>
          </div>
        </div>
      </div>

      {!user?.has_profile && (
        <div className="alert alert-info onboarding">
          <Info aria-hidden="true" />
          <span>Add your height, weight and goal to get a daily calorie target. <Link to="/profile">Set up profile</Link></span>
        </div>
      )}

      <input ref={fileInput} type="file" accept="image/*" hidden data-testid="file-input"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) analyse(f); e.target.value = ""; }} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden
        onChange={(e) => { const f = e.target.files?.[0]; if (f) analyse(f); e.target.value = ""; }} />

      {!showWorkbench && (
        <div
          className={`dropzone ${drag ? "drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) analyse(f); }}
        >
          <div className="dz-art" aria-hidden="true">
            <svg viewBox="0 0 120 80" width="120" height="80">
              <ellipse cx="60" cy="44" rx="50" ry="26" fill="#fff" stroke="#d4cabb" />
              <ellipse cx="60" cy="44" rx="34" ry="16" fill="none" stroke="#e6dfd3" />
              <circle cx="48" cy="41" r="7" fill="#d19a12" opacity=".85" />
              <circle cx="66" cy="46" r="8" fill="#c8442b" opacity=".85" />
              <path d="M70 34c6-6 12-6 16-2-5 5-11 6-16 2Z" fill="#6b7d2a" />
            </svg>
          </div>
          <h2>Snap your meal</h2>
          <p className="muted">We’ll guess the dish, you pick the portion. Works best with one dish, shot from above in good light.</p>
          <div className="dz-actions">
            <button className="btn btn-primary btn-lg only-touch" onClick={() => cameraInput.current?.click()}>
              <Camera aria-hidden="true" /> Take photo
            </button>
            <button className="btn btn-primary btn-lg only-pointer" onClick={() => fileInput.current?.click()}>
              <ImageUp aria-hidden="true" /> Upload photo
            </button>
            <button className="btn btn-lg only-touch" onClick={() => fileInput.current?.click()}>
              <ImageUp aria-hidden="true" /> From gallery
            </button>
            <button className="btn btn-lg" onClick={() => setWebcam(true)}>
              <Video aria-hidden="true" /> Use webcam
            </button>
          </div>
          <p className="small faint dz-foot">
            or drop an image here · no photo? <button className="linklike" onClick={() => setManual(true)}>search and log by name</button>
          </p>
          {phase === "error" && error && <div className="alert alert-error"><AlertTriangle aria-hidden="true" />{error}</div>}
        </div>
      )}

      {showWorkbench && (
        <div className={`workbench ${manual && !preview ? "no-photo" : ""}`}>
          {preview && (
            <section className="photo-panel" aria-label="Your photo">
              <div className="photo-frame">
                <img src={preview} alt="Your meal" />
                {phase === "analysing" && (
                  <div className="scan"><span className="spinner" /> Looking at your plate…</div>
                )}
              </div>
              <div className="photo-actions">
                <button className="btn btn-sm" onClick={reset}><RotateCcw aria-hidden="true" /> New photo</button>
                {rec && <span className="xs faint num">Analysed in {rec.timing_ms} ms · CLIP ViT-B/32</span>}
              </div>
            </section>
          )}

          <section className="result-panel" aria-label="Result">
            {phase === "analysing" && (
              <div className="result-loading">
                <div className="skeleton" style={{ height: 14, width: 90 }} />
                <div className="skeleton" style={{ height: 34, width: "70%" }} />
                <div className="skeleton" style={{ height: 32, width: "100%" }} />
                <div className="skeleton" style={{ height: 120, width: "100%" }} />
              </div>
            )}

            {phase === "error" && error && (
              <div className="alert alert-error"><AlertTriangle aria-hidden="true" /><span>{error}</span></div>
            )}

            {rec?.not_food && phase === "ready" && (
              <div className="alert alert-warn">
                <AlertTriangle aria-hidden="true" />
                <span>This doesn’t look like food to us. If it is, pick the dish below or search for it.</span>
              </div>
            )}

            {(phase === "ready" || manual) && (
              <>
                {food ? (
                  <div className="guess">
                    <p className="eyebrow">
                      {rec && pickedIsTop ? `${confidenceWord(top!.confidence)} · ${pct(top!.confidence)}` : rec ? "Your pick" : "Logging by name"}
                      {rec?.personalized && pickedIsTop && <span className="tag"><Sparkles aria-hidden="true" /> learned from your fixes</span>}
                    </p>
                    <h2 className="guess-name" data-testid="picked-food">{food.name}</h2>
                    <p className="small muted">{food.cuisine === "indian" ? "Indian" : "International"} · {food.category.replace("-", " ")} · {kcal(food.per100g.kcal)} kcal per 100 g</p>
                  </div>
                ) : (
                  <div className="guess"><h2>Find your food</h2><p className="small muted">Search by any name — “golgappa”, “chapati”, “dal fry”.</p></div>
                )}

                {rec && (
                  <div className="alts">
                    <p className="label-row"><span className="small strong">Top matches</span></p>
                    <div className="chips" role="group" aria-label="Top matches">
                      {rec.candidates.map((c) => (
                        <button key={c.food.id} className="chip" aria-pressed={food?.id === c.food.id} onClick={() => chooseFood(c.food)}>
                          {c.food.name} <span className="pct">{pct(c.confidence)}</span>
                        </button>
                      ))}
                      {food && !inCandidates && (
                        <button className="chip" aria-pressed="true">{food.name}</button>
                      )}
                    </div>
                    {rec.also_detected.length > 0 && (
                      <p className="xs muted multi">
                        Parts of the photo also look like {rec.also_detected.map((a) => a.food.name).join(", ")}. Log them as separate items after saving this one.
                      </p>
                    )}
                  </div>
                )}

                <div className="field">
                  <span>{rec ? "None of these? Search" : "Food"}</span>
                  <FoodSearch onPick={chooseFood} autoFocus={manual && !food} />
                </div>

                {food && n && (
                  <>
                    <PortionControl food={food} servingLabel={serving} quantity={qty}
                      onChange={(s, q) => { setServing(s); setQty(q); }} />
                    <NutrientReadout n={n} />
                    <p className="xs faint">{grams} g · approximate values ({food.source})</p>

                    <div className="field">
                      <span>Meal</span>
                      <div className="seg block" role="group" aria-label="Meal">
                        {MEAL_TYPES.map((m) => (
                          <button key={m.key} type="button" aria-pressed={mealType === m.key} onClick={() => setMealType(m.key)}>{m.label}</button>
                        ))}
                      </div>
                    </div>

                    {saved ? (
                      <div className="saved" role="status">
                        <Check aria-hidden="true" />
                        <div>
                          <strong>Logged {saved.food_name}</strong>
                          <p className="small muted num">{kcal(saved.nutrients.kcal)} kcal added to {saved.meal_type}.</p>
                        </div>
                        <div className="saved-actions">
                          <Link className="btn btn-sm" to={date === today() ? "/diary" : `/diary?date=${date}`}>View diary</Link>
                          <button className="btn btn-sm btn-primary" onClick={reset}>Snap another</button>
                        </div>
                      </div>
                    ) : (
                      <button className="btn btn-primary btn-lg btn-block" onClick={save} disabled={saving} data-testid="save-meal">
                        {saving ? <span className="spinner" /> : <Check aria-hidden="true" />}
                        Log {kcal(n.kcal)} kcal to {MEAL_TYPES.find((m) => m.key === mealType)?.label.toLowerCase()}
                      </button>
                    )}
                    {error && phase !== "error" && <div className="alert alert-error"><AlertTriangle aria-hidden="true" />{error}</div>}
                  </>
                )}
                {manual && !preview && (
                  <button className="btn btn-ghost btn-sm" onClick={reset}>Use a photo instead</button>
                )}
              </>
            )}
          </section>
        </div>
      )}

      {webcam && <Webcam onCapture={analyse} onClose={closeWebcam} />}
    </div>
  );
}
