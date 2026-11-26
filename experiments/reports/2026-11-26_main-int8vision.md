# Run `2026-11-26_main-int8vision`: faster vision model (current app model)

**What changed:** only the vision model. Previously the app used Xenova's 8-bit export
(`vision_model_quantized.onnx`). Now `scripts.download_models` downloads the full-precision export and
quantises it with onnxruntime's `quantize_dynamic` (int8 weights, per-channel), producing
`vision_model_int8.onnx` (SHA-256 prefix `c7ab00a1e7235ca6`). Everything else is identical to
`2026-11-26_main`: the fp32 text tower, prompts, data, splits, seed, and the config
(`configs/main_int8vision.yaml` differs only in `run_name`). Every image was re-embedded, and
everything fitted on val was re-fitted.

**Why:** the ONNX study (`2026-11-26_onnx`) measured this model as 2.2× faster on CPU and closer to fp32
(embedding cosine 0.973 vs 0.932).

## Speed and memory (in-process `FoodRecognizer.analyse`, 2 threads, median of 20; same VM, back to back)

| | Xenova INT8 | **Per-channel INT8** |
|---|---|---|
| Whole image | 47.1 ms | **22.4 ms** (2.1× faster) |
| With the 5 multi-item crops | 277.3 ms | **117.9 ms** (2.4× faster) |
| Peak RSS (load + requests) | 243 MB | 243 MB |
| File | 89 MB | 89 MB |

## Accuracy (test split, 2,461 photos; Commons, 79 photos)

| Model | Classes | `2026-11-26_main` top-1 / top-5 | **This run** | Indian top-1 |
|---|---|---|---|---|
| Zero-shot, single prompt | 188 | 74.8% / 94.0% | **76.3% / 94.2%** | 72.0% → 72.8% |
| Zero-shot, 7-prompt ensemble | 188 | 77.2% / 94.2% | **78.6% / 94.6%** | 75.9% → 76.6% |
| Zero-shot ensemble, closed 45 | 45 | 86.4% / 98.0% | **87.5% / 98.4%** | 85.7% → 87.5% |
| Linear probe | 45 | 92.3% / 99.2% | **93.2% / 99.5%** | 93.6% → 95.1% |
| **Hybrid (app)** | 188 | 77.3% / 95.2% | **79.1% / 96.3%** | 76.1% → 77.0% |

| App model | `2026-11-26_main` | **This run** |
|---|---|---|
| Hybrid weight / temperature (val) | 0.25 / 0.821 | 0.5 / 0.758 |
| ECE test before → after | 0.118 → 0.046 | 0.155 → **0.044** |
| Abstention threshold | 0.52 | 0.48 |
| Test coverage / accuracy when answering | 75.8% / 87.5% | **82.2%** / 86.4% |
| Declined photos: right dish in top-5 | 84.7% | **87.4%** |
| Commons top-1 / top-5 | **86.1%** / 100% | 79.7% / 98.7% |
| Commons, dishes the probe never saw (23) | **82.6%** (19) | 69.6% (16) |
| Not-food threshold (val) | 0.95 | 0.70 |
| Non-food caught / food wrongly flagged | 94.4% (17/18) / 0.24% | 88.9% (16/18) / **0.0%** (ROC-AUC 0.992) |
| Personalisation, ≤ 3 corrections per class | 81.2% | **84.3%** |

## Is the Commons drop real?

No. It's within noise, and the old model was above its own reference. Zero-shot (188-way, fp32 text)
by vision model ([vision_precision_zero_shot.json](../results/2026-11-26_onnx/vision_precision_zero_shot.json)):

| Vision model | Test (2,461) | Commons (79) |
|---|---|---|
| fp32 (reference) | 79.4% | 82.3% (65) |
| Xenova INT8 | 77.2% (−2.2) | 84.8% (67, *above* fp32) |
| **Per-channel INT8** | **78.6% (−0.8)** | 79.7% (63) |

On the large split, the new model loses a third as much accuracy to quantisation as the old one. On 79
photos, both INT8 models are within ±2 photos of the fp32 reference. The old model's Commons score was
luck, not a better model. The same caution applies to the not-food guard: 16 vs 17 of 18 photos.

## Behaviour checks

- `pizza.jpg` → pizza 0.98 (confident). `dosa.jpg` → plain dosa 0.65 (confident).
- One correction to masala dosa flips the next prediction (personalisation works).
- The e2e "not sure" photo (`butter-chicken_2.jpg`) is still uncertain (0.28).

Top confusions are near-synonyms and look-alikes, as before: club sandwich → sandwich (32%), chicken
curry → butter chicken (30%), kathi roll → shawarma (22%), paneer → butter chicken (27%), chocolate cake →
brownie (24%).
