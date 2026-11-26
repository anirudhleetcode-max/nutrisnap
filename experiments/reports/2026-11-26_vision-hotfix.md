# 2026-11-26 hotfix: the int8 vision model gave wrong answers on Render

## What happened

`2026-11-26_main-int8vision` switched the app to a vision model quantised by `scripts.download_models`
with onnxruntime's `quantize_dynamic(weight_type=QInt8, per_channel=True)`. It was evaluated on this
project's build machine and passed CI. After deployment to Render (20:12 UTC), a production smoke test
through the Vercel frontend found that recognition returned nonsense: `pizza.jpg` → "almonds" (0.21,
`not_food`), `dosa.jpg` → "sushi". `/api/health` still said `ok`, because the model had loaded.

## Cause

Dynamic quantisation multiplies uint8 activations by int8 weights. On x86 CPUs **without VNNI** (AVX2
or AVX-512 without VNNI), onnxruntime's kernels add pairs of these products into 16-bit integers, which
can **saturate** (a documented onnxruntime limitation). The build machine and the CI runners have
AVX-512 VNNI, so they never hit it. Render's instances don't.

Reproduced here by running the same Python and model under `qemu-x86_64 -cpu Haswell` (AVX2, no VNNI):

| Vision model | This machine (VNNI) | Emulated Haswell (no VNNI) |
|---|---|---|
| int8, per-channel (deployed, **broken**) | pizza 0.991 | **almonds 0.134** |
| Xenova's 8-bit export (previous) | pizza 0.995 | pizza 0.996 |
| uint8 weights, per-channel | pizza 0.994 | pizza 0.994 |
| int8, per-channel, **reduce_range** (fix) | pizza 0.992 | pizza 0.993 |

## Fix

1. **Quantise with `reduce_range=True`** (7-bit weights, so the 16-bit sums can't saturate). Speed and
   accuracy vs the alternatives (this machine, 2 threads, val 900 photos / test 2,461 photos):

   | Vision model | Cosine to fp32 (mean / min) | Latency | Zero-shot test top-1 (188) | Safe without VNNI |
   |---|---|---|---|---|
   | fp32 (reference) | 1 / 1 | 56 ms | 79.4% | yes |
   | Xenova 8-bit (before) | 0.932 / 0.787 | 44.5 ms | 77.2% | yes |
   | int8 per-channel (broken) | 0.973 / 0.849 | 19.7 ms | 78.6% | **no** |
   | uint8 weights, per-channel | **0.975 / 0.893** | 43.6 ms | **78.3%** | yes |
   | **int8 per-channel, reduce_range (chosen)** | 0.928 / 0.770 | **19.0 ms** | 77.4% | yes |

   reduce_range keeps the 2.3× speed-up and matches the old model's accuracy. The uint8 variant is
   ~1 point more accurate but not faster. It is a one-line change in `scripts/download_models.quantize`
   if accuracy matters more than latency. The speed on Render's own CPU has not been measured.

2. **Startup self-check.** `FoodRecognizer` now embeds a reference photo (`artifacts/selfcheck.jpg`) at
   startup and compares it with the embedding recorded by `experiments/run.py`
   (`artifacts/selfcheck.json`). If the cosine is below 0.95, loading fails, `/api/health` reports
   `degraded` with the reason, and `/api/recognize` returns 503 instead of confident wrong answers.
   Verified on emulated Haswell: the fixed model passes (cosine 0.994), and the broken model is refused
   (cosine 0.333).

3. **Embedding cache key.** The re-quantised file has the same name *and byte size* as the broken one,
   and `ml/embeddings.py` keyed its cache on name and size, so the first re-fit silently reused the
   broken model's embeddings (its metrics were identical to the earlier run, which is how it was
   caught). The cache is now keyed on the file's SHA-256.

## Re-fitted model (run `2026-11-26_main-int8rr`)

| | broken `2026-11-26_main-int8vision` (measured on VNNI) | **`2026-11-26_main-int8rr`** |
|---|---|---|
| Hybrid test top-1 / top-5 | 79.1% / 96.3% | 78.0% / 95.7% |
| Zero-shot ensemble test top-1 (188) | 78.6% | 77.4% |
| Linear probe (45) | 93.2% | 92.1% |
| ECE after temperature | 0.044 | 0.052 |
| Abstention threshold / test coverage / accuracy when answering | 0.48 / 82.2% / 86.4% | 0.52 / 76.7% / 87.3% |
| Commons top-1 (79) / dishes the probe never saw (23) | 79.7% / 69.6% | 86.1% / 82.6% |
| Not-food caught / food wrongly flagged | 88.9% / 0% | 88.9% / 0% (threshold 0.55, ROC-AUC 0.993) |
| Vision latency (this machine) | 22 ms | 22 ms |

## Lessons

- Quantised models must be tested on the CPU family they will run on. CI and the build machine both
  had VNNI, so every test passed while production was wrong.
- A health check that only says "model loaded" can't catch wrong numbers. The self-check now checks
  that the model computes what it computed when it was fitted.
- Cache keys must use content hashes, not file sizes.
