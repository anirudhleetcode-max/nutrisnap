# NutriSnap audit (phase 2, before changes)

Written after re-reading the whole repo at commit `75efdf3`. Severity: **H** high, **M** medium, **L** low.
The Resolution section at the end records what was done.

## 1. Data honesty

| # | Sev | Finding |
|---|---|---|
| D1 | H | `backend/data/foods.json`: all 188 per-100 g values were typed in by hand as "typical recipe" estimates. Each food carries `source: "IFCT 2017 (approx.)"` or `"USDA FoodData Central (approx.)"`, but no item links to a record in either database. The UI shows that source string under the portion, so it reads as if the values were looked up. Nothing is verified. |
| D2 | M | Only an Atwater energy check (`test_catalogue_is_complete`) guards the values, so there is no check against any reference. |
| D3 | M | The demo account (`demo@nutrisnap.app`, "Demo Rohan") shows 30 days of generated meals, and nothing in the UI says the data is demo/seeded. |
| D4 | L | The sign-in page shows an illustrative "lunch" card with kcal numbers. They are computed from foods.json, but the card isn't labelled as an example. |

## 2. ML weaknesses

| # | Sev | Finding |
|---|---|---|
| M1 | H | Evaluation uses 79 hand-picked Commons images across 20 classes with no split. It is fine as a sanity check but too small to support any claim (per-class n = 3–4). |
| M2 | H | The confidence shown in the UI ("Confident · 80%") is the raw softmax at CLIP's logit scale of 100. It is uncalibrated, and the "Confident/Likely/Unsure" cut-offs (0.6/0.3) were chosen arbitrarily. |
| M3 | H | There is no abstention: the top-1 guess is always preselected, even when it is 12% likely. |
| M4 | M | There are no supervised baselines. Only zero-shot single-prompt and ensemble results exist; no simple trained model (e.g. a linear probe on frozen embeddings) is compared. |
| M5 | M | The not-food threshold (0.5) was not tuned and was evaluated on only 9 images. |
| M6 | M | The kNN personalisation constants (k=5, 0.72/0.90 similarity, α≤0.7) are hand-set, and the only evaluation is an optimistic leave-one-out on the same 79 images. |
| M7 | M | Error analysis is just a list of misclassified file names in metrics.json, with no confusion-pair aggregation. |
| M8 | L | Batch inference with the dynamically quantised model changes the numbers slightly compared with batch=1 (activation ranges are computed per batch). The code works around this by running the full image alone, but it isn't documented. |
| M9 | L | The model is described as "CLIP ViT-B/32 quantized", with no model card (source repository, quantisation type, licence, the fact that it was not trained here). |
| M10 | L | There is no experiment tracking: metrics.json has no git commit, library versions or dataset hash. |

## 3. Security / backend

| # | Sev | Finding |
|---|---|---|
| S1 | H | The JWT secret silently defaults to a placeholder string (`dev-only-secret-...`). A deployment that forgets `.env` signs tokens with a public secret. |
| S2 | H | Upload validation trusts the client `Content-Type`. The real check is simply whether Pillow can decode the file, and there is no pixel limit: a highly compressed 20 000 × 20 000 PNG passes the 8 MB limit and is decoded in full (decompression bomb; Pillow only warns up to about 179 MP). HEIC is listed as allowed but Pillow cannot decode it here. |
| S3 | M | No login rate limiting, so password guessing is unthrottled. |
| S4 | M | Privacy: `corrections` keeps image embeddings indefinitely, and nothing documents this or lets the user delete them. |
| S5 | L | `/api/health` reports `model_loaded` but still returns `status: ok` when recognition is disabled. |
| S6 | L | `recognize.py` mixes model lifecycle, personalisation and HTTP handling. It works but is hard to test in isolation. |
| S7 | L | No request timeout around inference. A pathological image could hold a worker thread (mitigated once S2 is fixed). |
| OK | – | CORS is limited to configured origins, every query filters by `user_id`, photos are served from Mongo (no user-controlled file paths), passwords use bcrypt, and nothing sensitive is logged. |

## 4. UI / UX

| # | Sev | Finding |
|---|---|---|
| U1 | H | Nutrition shown as fact: there is no visual difference between verified and estimated values (see D1). |
| U2 | M | No low-confidence state and no explanation of what the percentage means. |
| U3 | M | No "Model" panel describing the model, its data, its metrics and its limits. |
| U4 | M | Delete uses `window.confirm`, and there are no toasts. |
| U5 | L | Demo data is not labelled (D3). |

## 5. Tests

| # | Sev | Finding |
|---|---|---|
| T1 | M | No frontend unit tests. |
| T2 | M | No robustness tests for missing model files or model load failure, truncated or zero-byte images, pixel bombs, or the low-confidence path. |
| T3 | L | No CI. |

## Plan (priority order)

1. **Nutrition verification (D1, D2, U1):** download USDA FoodData Central bulk CSVs (SR Legacy + FNDDS) with a script, map foods to fdcIds by hand where a genuine match exists, replace values from the record, and add `source`, `source_id`, `source_url` and `verified` fields. Everything else becomes `verified: false` / "Estimate — unverified" in the UI.
2. **Real evaluation (M1, M4, M10):** Food-101 and an Indian food dataset from Hugging Face (parquet, sampled per class), with the class mapping documented. Compare single prompt, ensemble, and a logistic-regression linear probe on frozen CLIP embeddings, using train/val/test splits and an `experiments/` runner.
3. **Calibration and abstention (M2, M3, M5):** fit a temperature on the validation split and report ECE and a reliability table before and after. Choose the abstention threshold on validation (coverage–accuracy curve), then return and display it. Evaluate the not-food guard on more images.
4. **Error analysis (M7):** confusion pairs written to `experiments/results/<run>/errors.*`, plus a README analysis.
5. **Security (S1–S5):** JWT placeholder guard, magic bytes and pixel limit, login rate limit, degraded health status, and a privacy note plus a delete endpoint for learned embeddings.
6. **Model card (M9):** `/api/model`, a UI Model panel, demo labelling, vitest tests, CI, README, screenshots.

## Resolution

_Filled in at the end of phase 2._
