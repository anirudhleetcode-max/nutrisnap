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

| # | Status | What was done |
|---|---|---|
| D1 | Done | `scripts/usda_verify.py` downloads the USDA FDC bulk CSVs (FNDDS 2024-10-31, SR Legacy 2018-04). 95/188 foods (29/113 Indian) are mapped by hand to same-dish records (`data/usda_mapping.json`); their values were replaced and `verified`, `source_id` (`fdc:<id>`) and `source_url` added. The other 93 are `verified: false` with an honest source string. The old estimates had an 11% median kcal error on the verified items. The DEMO_KEY REST API was rate-limited (10 requests), so the bulk files were used. |
| D2 | Done | Tests check provenance fields, the mapping ↔ catalogue consistency, and `/api/foods/meta`. |
| D3 | Done | Demo banner on every page for the demo account; README calls the data generated. |
| D4 | Done | Card labelled "Example lunch" and its numbers recomputed from the current catalogue (roti and dal are USDA-verified; bhindi masala shows ≈ as an estimate). |
| M1 | Done | Food-101 (30 classes) + Indian Food Images (15 classes): train 1800 / val 900 / test 2461 real photos; Commons set kept as an OOD check (79 food + 36 non-food). |
| M2 | Done | Temperature scaling fitted on val (T = 0.446); test ECE 0.443 → 0.062; reliability tables and plot committed. |
| M3 | Done | Threshold 0.48 chosen on val (85% target); API returns `decision`/`abstain_below`; UI shows "Not sure — pick from the list" without preselecting. |
| M4 | Done | Majority, single prompt, ensemble, closed-set ensemble, logistic-regression probe and hybrid, all on the same splits. |
| M5 | Done | Guard threshold chosen on val (0.95) and evaluated on test (88.9% non-food caught, 0.73% food wrongly flagged). The non-food sample is still small (36). |
| M6 | Partly | Personalisation evaluated on held-out test photos with train-split corrections (79.4 → 83.9%). The constants are still hand-set; the gate fires on 95% of photos (documented). |
| M7 | Done | `errors.json` / `confusions.csv` with confusion pairs, confident mistakes and per-class accuracy; README analysis. |
| M8 | Done | Batch = 1 everywhere (crops too), documented in code and model card. |
| M9 | Done | `docs/MODEL_CARD.md`, `/api/model`, Model page. |
| M10 | Done | `experiments/` with configs, results (git commit, versions, data hashes) and reports. |
| S1 | Done | Placeholder or short secret: refuses to start in production, uses a random per-process secret with a warning in development; `.env.example` explains generation. |
| S2 | Done | Magic-byte sniffing (JPEG/PNG/WebP), 40 MP limit checked before decoding, decompression-bomb warnings treated as errors, 20 s inference timeout; thumbnails re-encoded (EXIF dropped). HEIC removed. |
| S3 | Done | Failed-login limit per IP + email (5 per 5 min → 429 with Retry-After). In-memory, single process (documented). |
| S4 | Done | Privacy text in the UI and model card; `DELETE /api/corrections` + Profile button. |
| S5 | Done | `/api/health` returns `degraded` with the model error. |
| S6 | Done | Model logic moved to `app/ml/pipeline.py`; the router only handles HTTP and storage. |
| S7 | Done | `asyncio.wait_for` timeout around inference. |
| U1 | Done | Verified vs estimate badge (dashed border, ≈ marker) on Snap, Diary and search. |
| U2 | Done | Decision banner explaining the threshold; "calibrated confidence" wording; quality warnings. |
| U3 | Done | Model page. |
| U4 | Done | Confirm dialog component and toasts. |
| U5 | Done | See D3. |
| T1 | Done | vitest + Testing Library: 21 tests (decision logic, provenance badge, portion control, confirm dialog, sign-in errors incl. rate limit, ring/readout). It found a missing `role="alert"` on error messages, now fixed. |
| T2 | Done | Robustness tests for empty, non-image, truncated, tiny, pixel-bomb, dark, low-confidence, missing and corrupt model; a binary-class bug in the probe export was found and fixed. |
| T3 | Done (not run) | `.github/workflows/ci.yml`: pytest with a Mongo service, frontend build + vitest. Not executed on GitHub from here. |
