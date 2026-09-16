import { AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import { pct } from "../lib/recognition";
import type { FoodsMeta, ModelCard } from "../lib/types";

type Summary = {
  test_top1?: number; test_top5?: number; test_top1_zero_shot_single_prompt?: number;
  test_ece_before?: number; test_ece_after?: number; abstain_threshold?: number;
  test_coverage?: number; test_accuracy_when_answering?: number; ood_commons_top1?: number | null;
  not_food_test?: { non_food_flagged: number; food_wrongly_flagged: number };
};

const PIPELINE = [
  ["Check", "The upload must be a real JPEG, PNG or WebP (checked from the file bytes), at most 8 MB and 40 megapixels."],
  ["Prepare", "Rotate using EXIF, resize the short side to 224 px, centre-crop, normalise with CLIP’s mean and std."],
  ["Embed", "The pretrained CLIP image encoder turns the photo into 512 numbers."],
  ["Score", "Compare that vector with pre-computed text vectors for every food (7 phrasings each). Optionally, a small linear model trained on public food photos refines the ranking among the dishes it knows."],
  ["Personalise", "If you corrected similar photos before, their labels are blended in (your account only)."],
  ["Calibrate", "A temperature fitted on held-out photos rescales the probabilities so “70%” means about 70% right."],
  ["Decide", "Below the confidence threshold the app does not pick for you; if the photo looks more like a non-food scene, it says so."],
];

function Num({ v, p }: { v: number | null | undefined; p?: boolean }) {
  if (v === null || v === undefined) return <span className="faint">Not measured yet</span>;
  return <span className="num">{p ? pct(v) : v}</span>;
}

export default function ModelInfo() {
  const [card, setCard] = useState<ModelCard | null>(null);
  const [meta, setMeta] = useState<FoodsMeta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<ModelCard>("/api/model").then(setCard).catch((e) => setError(e instanceof ApiError ? e.message : "Couldn't load the model card."));
    api<FoodsMeta>("/api/foods/meta").then(setMeta).catch(() => undefined);
  }, []);

  const m = (card?.metrics ?? null) as Summary | null;
  const s = card?.settings;

  return (
    <div className="page model-page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Model</p>
          <h1>How recognition works</h1>
          <p className="small muted">What runs on your photo, how well it does on public test photos, and where it fails.</p>
        </div>
        {card?.version && <span className="xs faint num">version {card.version}</span>}
      </div>

      {error && <div className="alert alert-error" role="alert"><AlertTriangle aria-hidden="true" />{error}</div>}
      {!card && !error && <div className="skeleton" style={{ height: 240 }} />}
      {card && !card.loaded && (
        <div className="alert alert-warn" role="alert"><AlertTriangle aria-hidden="true" />
          <span>The recognition model isn’t loaded on this server ({card.error}). You can still search and log food by name.</span>
        </div>
      )}

      {card?.loaded && (
        <div className="model-grid">
          <section className="panel">
            <div className="panel-head"><h2>Pipeline</h2></div>
            <ol className="formula pipeline">
              {PIPELINE.map(([k, v]) => <li key={k}><span className="step">{k}</span><span>{v}</span></li>)}
            </ol>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>The model</h2></div>
            <dl className="kv">
              <dt>Backbone</dt><dd>{card.backbone?.model}</dd>
              <dt>Weights</dt><dd>{card.backbone?.weights} (<a href={card.backbone?.source} target="_blank" rel="noreferrer">source</a>, {card.backbone?.license})</dd>
              <dt>Runs on</dt><dd>{card.backbone?.runtime}</dd>
              <dt>Mode</dt><dd>{card.mode}</dd>
              <dt>Foods it can name</dt><dd className="num">{card.classes}</dd>
              {card.probe && <><dt>Trained here</dt><dd>{card.probe.model}; {card.probe.train_size} real photos, {card.probe.classes.length} classes, C = {card.probe.C} (not synthetic)</dd></>}
              <dt>Temperature</dt><dd className="num">{s?.temperature}</dd>
              <dt>Abstain below</dt><dd><Num v={s?.abstain_below} p /></dd>
              <dt>Not-food below</dt><dd><Num v={s?.not_food_below} p /> food score</dd>
            </dl>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Measured quality</h2><span className="xs faint">{card.experiment ?? ""}</span></div>
            {m ? (
              <>
                <p className="small muted">{card.evaluated_on?.["train/val/test"]}. Test split: {card.evaluated_on?.sizes?.test} photos never used for training or tuning.</p>
                <table className="table num">
                  <tbody>
                    <tr><td>Top-1 accuracy (app model)</td><td><Num v={m.test_top1} p /></td></tr>
                    <tr><td>Top-5 accuracy</td><td><Num v={m.test_top5} p /></td></tr>
                    <tr><td>Top-1, single-prompt baseline</td><td><Num v={m.test_top1_zero_shot_single_prompt} p /></td></tr>
                    <tr><td>Calibration error (ECE) before → after</td><td><Num v={m.test_ece_before} /> → <Num v={m.test_ece_after} /></td></tr>
                    <tr><td>Answers given (coverage)</td><td><Num v={m.test_coverage} p /></td></tr>
                    <tr><td>Accuracy when it answers</td><td><Num v={m.test_accuracy_when_answering} p /></td></tr>
                    <tr><td>Hand-picked Commons photos, top-1</td><td><Num v={m.ood_commons_top1} p /></td></tr>
                    <tr><td>Non-food photos caught</td><td><Num v={m.not_food_test?.non_food_flagged} p /></td></tr>
                    <tr><td>Food photos wrongly flagged</td><td><Num v={m.not_food_test?.food_wrongly_flagged} p /></td></tr>
                  </tbody>
                </table>
              </>
            ) : <p className="muted">Not measured yet — run the evaluation experiment to fill this in.</p>}
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Nutrition data</h2></div>
            {meta ? (
              <p className="small">
                <strong className="num">{meta.verified}</strong> of <span className="num">{meta.total}</span> foods have values copied from {meta.verified_source}
                {" "}(<span className="num">{meta.indian_verified}</span> of <span className="num">{meta.indian_total}</span> Indian dishes).
                The rest are marked <em>Estimate — unverified</em>: typical-recipe values written by the author. Serving weights (1 katori, 1 roti…) are estimates for every food.
              </p>
            ) : <div className="skeleton" style={{ height: 60 }} />}
            <h3 className="sub">Your data</h3>
            <p className="small muted">
              Photos are stored as a 480 px thumbnail (EXIF location removed) and a 512-number CLIP embedding. Photos you don’t log are deleted after 24 hours.
              When you correct a guess, that photo’s embedding is kept to personalise your future results; you can delete these on the Profile page.
            </p>
          </section>

          <section className="panel span-2">
            <div className="panel-head"><h2>Limits</h2></div>
            <ul className="limits">{card.limitations?.map((l) => <li key={l}>{l}</li>)}</ul>
          </section>
        </div>
      )}
    </div>
  );
}
