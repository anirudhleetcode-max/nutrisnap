import { AlertTriangle, Check, LogOut } from "lucide-react";
import { Link } from "react-router-dom";
import ConfirmDialog from "../components/ConfirmDialog";
import { useToast } from "../components/Toast";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { kcal } from "../lib/nutrition";
import type { Profile as P, ProfileResp, Targets } from "../lib/types";

const DEFAULTS: P = { age: 25, sex: "female", height_cm: 160, weight_kg: 60, activity: "light", goal: "maintain" };
const GOALS: { key: P["goal"]; label: string }[] = [
  { key: "lose", label: "Lose weight" }, { key: "maintain", label: "Maintain" }, { key: "gain", label: "Gain" },
];

export default function Profile() {
  const { user, refresh, logout } = useAuth();
  const nav = useNavigate();
  const [form, setForm] = useState<P>(DEFAULTS);
  const [levels, setLevels] = useState<ProfileResp["activity_levels"]>([]);
  const [targets, setTargets] = useState<Targets | null>(null);
  const [saved, setSaved] = useState(false);
  const [hasSaved, setHasSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [learned, setLearned] = useState<{ food: string; was_predicted_as: string | null; count: number }[]>([]);
  const [learnedTotal, setLearnedTotal] = useState(0);
  const [forgetting, setForgetting] = useState(false);
  const toast = useToast();

  useEffect(() => {
    api<ProfileResp>("/api/profile").then((r) => {
      setLevels(r.activity_levels);
      if (r.profile) { setForm(r.profile); setHasSaved(true); }
      setTargets(r.targets);
    }).catch(() => setError("Couldn't load your profile."));
    api<{ items: typeof learned; total: number }>("/api/corrections").then((r) => { setLearned(r.items); setLearnedTotal(r.total ?? 0); }).catch(() => undefined);
  }, []);

  // live preview
  const valid = form.age >= 13 && form.age <= 100 && form.height_cm >= 120 && form.height_cm <= 230 && form.weight_kg >= 30 && form.weight_kg <= 250;
  useEffect(() => {
    if (!valid) return;
    const t = setTimeout(() => {
      api<{ targets: Targets }>("/api/profile/estimate", { method: "POST", json: form }).then((r) => setTargets(r.targets)).catch(() => undefined);
    }, 250);
    return () => clearTimeout(t);
  }, [form, valid]);

  const set = <K extends keyof P>(k: K, v: P[K]) => { setForm((f) => ({ ...f, [k]: v })); setSaved(false); };
  const num = (k: "age" | "height_cm" | "weight_kg") => (e: React.ChangeEvent<HTMLInputElement>) => set(k, Number(e.target.value));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const r = await api<ProfileResp>("/api/profile", { method: "PUT", json: form });
      setTargets(r.targets); setSaved(true);
      const first = !user?.has_profile;
      setHasSaved(true);
      await refresh();
      if (first) setTimeout(() => nav("/"), 1200);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't save.");
    } finally { setBusy(false); }
  };

  const forget = async () => {
    try {
      await api("/api/corrections", { method: "DELETE" });
      setLearned([]); setLearnedTotal(0);
      toast("Deleted what the app learned from your corrections");
    } catch { toast("Couldn't delete — try again", "error"); }
    setForgetting(false);
  };

  const b = targets?.breakdown;
  return (
    <div className="page profile">
      <div className="page-head">
        <div>
          <p className="eyebrow">Profile</p>
          <h1>{hasSaved ? "Your targets" : "Set your daily target"}</h1>
          {!hasSaved && <p className="small muted">Takes 20 seconds. We use it to work out calories and macros — nothing else.</p>}
        </div>
        <button className="btn btn-ghost btn-sm mobile-only" onClick={logout}><LogOut aria-hidden="true" /> Sign out</button>
      </div>

      <div className="profile-grid">
        <form className="panel profile-form" onSubmit={submit}>
          <div className="row3">
            <label className="field"><span>Age</span>
              <div className="input-suffix"><input className="input num" type="number" min={13} max={100} value={form.age} onChange={num("age")} name="age" required /><em>yrs</em></div>
            </label>
            <label className="field"><span>Height</span>
              <div className="input-suffix"><input className="input num" type="number" min={120} max={230} step="0.5" value={form.height_cm} onChange={num("height_cm")} name="height" required /><em>cm</em></div>
            </label>
            <label className="field"><span>Weight</span>
              <div className="input-suffix"><input className="input num" type="number" min={30} max={250} step="0.1" value={form.weight_kg} onChange={num("weight_kg")} name="weight" required /><em>kg</em></div>
            </label>
          </div>
          <div className="field"><span id="sex-l">Sex (for the BMR formula)</span>
            <div className="seg" role="group" aria-labelledby="sex-l">
              {(["female", "male"] as const).map((s) => (
                <button type="button" key={s} aria-pressed={form.sex === s} onClick={() => set("sex", s)}>{s === "female" ? "Female" : "Male"}</button>
              ))}
            </div>
          </div>
          <label className="field"><span>Activity</span>
            <select className="select" value={form.activity} name="activity" onChange={(e) => set("activity", e.target.value as P["activity"])}>
              {levels.map((l) => (
                <option key={l.id} value={l.id}>{l.id.replace("_", " ").replace(/^./, (c) => c.toUpperCase())} — {l.label}</option>
              ))}
            </select>
          </label>
          <div className="field"><span id="goal-l">Goal</span>
            <div className="seg block" role="group" aria-labelledby="goal-l">
              {GOALS.map((g) => (
                <button type="button" key={g.key} aria-pressed={form.goal === g.key} onClick={() => set("goal", g.key)}>{g.label}</button>
              ))}
            </div>
          </div>
          {error && <div className="alert alert-error" role="alert"><AlertTriangle aria-hidden="true" />{error}</div>}
          <button className="btn btn-primary btn-lg" type="submit" disabled={busy || !valid} data-testid="save-profile">
            {saved ? <><Check aria-hidden="true" /> Saved</> : hasSaved ? "Update targets" : "Save and start logging"}
          </button>
        </form>

        <section className="panel targets" aria-label="Targets" data-testid="targets">
          {targets && b ? (
            <>
              <p className="eyebrow">Daily target</p>
              <p className="target-kcal num"><span data-testid="target-kcal">{kcal(targets.kcal)}</span> <small>kcal</small></p>
              <dl className="target-macros">
                <div className="rm rm-protein"><dt>Protein</dt><dd className="num">{targets.protein} g</dd></div>
                <div className="rm rm-carbs"><dt>Carbs</dt><dd className="num">{targets.carbs} g</dd></div>
                <div className="rm rm-fat"><dt>Fat</dt><dd className="num">{targets.fat} g</dd></div>
                <div className="rm rm-fibre"><dt>Fibre</dt><dd className="num">{targets.fibre} g</dd></div>
              </dl>
              <h3 className="sub">How we got there</h3>
              <ol className="formula">
                <li><span className="step">BMR</span><span>Mifflin–St Jeor<code>{b.bmr_formula}</code></span></li>
                <li><span className="step">TDEE</span><span>BMR × activity factor<code>{b.tdee_formula}</code></span></li>
                <li><span className="step">Goal</span><span>{b.goal_adjustment === 0 ? "No adjustment to maintain weight" : `${b.goal_adjustment > 0 ? "+" : "−"}${Math.abs(b.goal_adjustment)} kcal/day ${b.goal_adjustment < 0 ? "(about 0.5 kg a week)" : "(lean gain)"}`}
                  {b.floored && <em className="xs"> · raised to a safe minimum</em>}
                  <code>= {kcal(targets.kcal)} kcal (rounded)</code></span></li>
                <li><span className="step">Macros</span><span>Protein {b.protein_rule}; fat {b.fat_rule}; carbs {b.carbs_rule}; fibre {b.fibre_rule}.</span></li>
              </ol>
              <p className="xs faint">Estimates for healthy adults, not medical advice. Food values are approximate (IFCT 2017 / USDA).</p>
            </>
          ) : (
            <p className="muted">Fill in the form to see your numbers.</p>
          )}
        </section>
      </div>

      <section className="panel learned">
        <div className="panel-head"><h2>Personalisation and privacy</h2></div>
        <p className="small muted">
          When you pick a different dish than our first guess, we keep that photo’s CLIP embedding (512 numbers, not the photo) with the dish you chose.
          Similar photos later lean towards your answer. Only your account uses it. <Link to="/model">More about the model</Link>
        </p>
        {learned.length > 0 ? (
          <>
            <ul>
              {learned.map((l, i) => (
                <li key={i} className="small"><strong>{l.food}</strong>{l.was_predicted_as && <span className="muted"> (we had guessed {l.was_predicted_as})</span>} <span className="faint num">× {l.count}</span></li>
              ))}
            </ul>
            <button className="btn btn-sm btn-danger" onClick={() => setForgetting(true)}>Delete {learnedTotal} stored corrections</button>
          </>
        ) : <p className="small faint">Nothing stored yet.</p>}
      </section>
      <ConfirmDialog open={forgetting} title="Delete learned corrections?"
        body="Your future photos will be scored by the base model only. Your diary is not affected."
        confirmLabel="Delete" onConfirm={forget} onCancel={() => setForgetting(false)} />
    </div>
  );
}
