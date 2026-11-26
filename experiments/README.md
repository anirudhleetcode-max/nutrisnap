# Experiments

Each run is described by a config, executed by one script, and writes everything it measured to
`results/<run_id>/`. Only runs that actually happened are committed.

```bash
cd backend
pip install -r requirements-ml.txt
python -m scripts.download_models          # CLIP ONNX weights
python -m ml.datasets                      # Food-101 + Indian food samples (see data/README.md)
python -m ml.fetch_testset                 # Commons OOD / non-food images
python ../experiments/run.py --config ../experiments/configs/main.yaml
python ../experiments/bench_latency.py     # API latency
```

Each `results/<run_id>/` folder contains:

| File | Content |
|---|---|
| `metrics.json` | Every model × split × dataset: top-1, top-5, macro top-1; probe/hybrid search; calibration (ECE, Brier, NLL, reliability tables); abstention curves; not-food guard; simulated personalisation; latency; data manifest |
| `errors.json`, `confusions.csv` | Top confusion pairs (chosen model and zero-shot), the 30 most confident mistakes (row ids), per-class top-1 |
| `calibration.png` | Reliability diagram and coverage–accuracy curve (test) |
| `config.yaml` | Config snapshot |
| `env.json` | Python and library versions, git commit, dirty flag |

`run.py` also writes `backend/artifacts/recognizer.json` (+ `probe_v1.npz/.json`) when `export_artifacts: true`. The API loads these at startup.

**Adding an experiment:** copy `configs/main.yaml`, change `run_name` and the parameters, and run it. Anything that is *chosen* (hyper-parameters, thresholds, model selection) must be chosen on `val`; `test` is only scored.

See [compare/README.md](compare/README.md) for the comparison, ONNX and CLIP-analysis scripts.

## Runs

| Run | What | Data | Headline (test) | Report |
|---|---|---|---|---|
| `2026-09-16_commons-phase1` | Phase-1 check: single prompt vs ensemble, no split | 79 Commons photos, 20 classes | top-1 36.7% → 50.6% | (numbers only, `metrics.json`) |
| `2026-09-16_main` | Baselines, linear probe, hybrid, calibration, abstention, not-food guard | Food-101 (30 cl.) + Indian Food Images (15 cl.); 1800 / 900 / 2461 | hybrid top-1 79.4%, top-5 97.3%; ECE 0.443 → 0.062 | [reports/2026-09-16_main.md](reports/2026-09-16_main.md) |
| `2026-09-16_latency` | `/api/recognize` wall-clock, 20 requests each | test fixtures | 70.6 ms median (whole image), 352 ms with crops | `latency.json` |
| `2026-11-26_dataset` | Datasheet statistics: balance, sizes, brightness, duplicates | same as main | 49 test photos duplicate train/val | [docs/DATASET.md](../docs/DATASET.md) |
| `2026-11-26_compare` | CNN from scratch vs ResNet-50 transfer (frozen, fine-tuned) vs CLIP zero-shot vs CLIP + probe / Tip-Adapter; calibration, AURC, few-shot curve, unseen photos | same as main + Commons | probe 92.3%, zero-shot fp32-text 86.4%, ResNet-50 fine-tuned 82.4%, CNN 34.9% | [docs/COMPARISON.md](../docs/COMPARISON.md) |
| `2026-11-26_onnx` | CLIP fp32/fp16/INT8 latency, size, accuracy; graph optimisation, threads, batching; ResNet-50 → ONNX → static INT8; request stage breakdown | val | INT8 text tower costs 22 points of 188-way zero-shot; INT8 vision quantised here is 2.2× faster | [docs/COMPARISON.md §6](../docs/COMPARISON.md) |
| `2026-11-26_clip_explain` | Architecture from ONNX weights, modality gap, logit-scale sweep, prompt sensitivity | val | ensemble 70.1% vs best single prompt 69.1% | [docs/CLIP.md](../docs/CLIP.md) |
| – | Real phone photos from users | – | Dataset required before evaluation. | – |

Note: `2026-09-16_main/env.json` records commit `2d9d82e` with `dirty: true`. The run used the experiment code from the working tree, which was committed right after in `bedbcd0` without changes.
