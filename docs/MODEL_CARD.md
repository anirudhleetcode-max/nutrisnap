# Model card: NutriSnap food recognizer (`2026-11-26_main`)

The same information is served at `GET /api/model` and shown on the app's **Model** page.

## What it is

| | |
|---|---|
| Task | Name the dish in a photo, from a fixed list of 188 foods (113 Indian) |
| Backbone | **CLIP ViT-B/32** (Radford et al., OpenAI, 2021). Pretrained and **not trained by this project** |
| Weights | [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) ONNX export: `vision_model_quantized.onnx` (89 MB, **8-bit dynamic quantisation**, runs per request) and `text_model.onnx` (254 MB, **full precision**, runs only to build the cached label embeddings); MIT licence |
| Runtime | onnxruntime (CPU); no PyTorch |
| Trained by this project | A **multinomial logistic regression** (linear probe) on frozen 512-d image embeddings. 45 classes, 1,800 real photos (Food-101 + Indian Food Images), C = 10 chosen on val, seed 42. Artifact: `backend/artifacts/probe_v1.npz` (87 KB) + `probe_v1.json` |
| Fitted on validation data | Temperature T = 0.821, abstention threshold 0.52, not-food threshold 0.95, hybrid weight 0.25 (`backend/artifacts/recognizer.json`, which also records `text_tower: fp32`; the app logs a warning if a different text tower is loaded) |

## Inference

1. **Validate:** magic bytes must be JPEG, PNG or WebP; at most 8 MB; at most 40 MP (checked from the header before decoding); at least 32 px per side.
2. **Preprocess** (`clip-224-bicubic-centercrop-v1`): apply EXIF rotation → RGB → resize the shortest side to 224 (bicubic) → centre crop 224×224 → scale to [0, 1] → normalise with mean (0.4815, 0.4578, 0.4082) and std (0.2686, 0.2613, 0.2758) → NCHW float32, batch size 1. Batching changes results with dynamic quantisation, because activation ranges are computed per batch.
3. **Embed:** the vision model outputs `image_embeds` (512), which is L2-normalised.
4. **Zero-shot:** each food has 7 prompt templates (plain and descriptive names). Each prompt goes through the text model separately, with no padding, because this export pools at EOS and takes no attention mask. The prompt embeddings are averaged and re-normalised, giving a 188×512 label matrix that is cached as `.npy`. Scores are `softmax(100 · cos)`.
5. **Probe (hybrid):** zero-shot decides the total probability M of the 45 probe classes. Inside that group, the probe's softmax is mixed with zero-shot's (probe weight 0.25). The other 143 classes keep their zero-shot probability.
6. **Calibrate:** `softmax(log p / T)` with T = 0.821.
7. **Personalise (optional, per user):** a kNN over the user's corrected photos (cosine ≥ 0.72, k = 5) is blended in with weight α ≤ 0.7. Probabilities after this step are **not** calibrated, and the API reports `calibrated: false`.
8. **Decide:** food score < 0.95 → `not_food`; top-1 < 0.52 → `uncertain` (no preselection in the UI); otherwise `confident`.
9. **Extras:** a multi-item hint from 5 crops (2×2 grid + centre), and rule-based quality warnings (dark, overexposed, blurry).

## Evaluation (test split, 2,461 real photos; see `experiments/reports/2026-11-26_main.md`)

| | Value |
|---|---|
| Top-1 / top-5 (188-way) | 77.3% / 95.2% (single-prompt zero-shot: 74.8% / 94.0%) |
| Indian subset top-1 | 76.1% (zero-shot ensemble: 75.9%) |
| ECE before → after temperature | 0.118 → 0.046 |
| Coverage / accuracy at threshold 0.52 | 75.8% / 87.5% |
| Commons OOD photos (79) top-1 / top-5 | 86.1% / 100% (82.6% on the 23 photos of dishes the probe never saw) |
| Not-food: non-food caught / food wrongly flagged | 94.4% (n = 18) / 0.24% (n = 2461) |
| Latency, `/api/recognize` median (2 CPU threads) | 70.6 ms whole image; 352 ms with multi-item crops (measured 2026-09-16; the request path is unchanged) |

## Intended use and limits

- Intended for logging personal meals, with a human confirming every entry. It is not for medical or dietary decisions.
- The list is closed: anything not among the 188 foods is forced onto the nearest one.
- It classifies the whole photo. A thali or mixed plate gets one label, plus a crop hint.
- Portion size is not estimated. Nutrition values are USDA-verified for 95 foods and **unverified estimates** for 93.
- It was trained and evaluated on web and restaurant photos. It has not been evaluated on real users' phone photos (**Dataset required before evaluation**).
- Known confusions: wraps → shawarma, chicken curry / paneer curries → butter chicken, chocolate cake → brownie, and near-synonyms in the food list (Greek salad → green salad, club sandwich → sandwich).
- Privacy: stored per user are a 480 px thumbnail (EXIF dropped; deleted after 24 h if not logged) and the 512-d embedding. Embeddings of corrected photos are kept for personalisation until the user deletes them (`DELETE /api/corrections`).

## Change log

- **2026-11-26 (`2026-11-26_main`):** the label embeddings are now built with the full-precision text
  tower. The 8-bit one used before had only 0.74 cosine similarity to fp32
  ([COMPARISON.md](COMPARISON.md)). All val-fitted settings were re-fitted. Test top-1 changed from 79.4% to
  77.3%, Commons top-1 from 63.3% to 86.1%, and ECE after scaling from 0.062 to 0.046. The same work
  found 49 test photos that duplicate train/val photos in the Indian source data; removing them
  changes accuracy by under 0.5 points ([DATASET.md](DATASET.md)).
- **2026-09-16 (`2026-09-16_main`):** first calibrated hybrid model.
