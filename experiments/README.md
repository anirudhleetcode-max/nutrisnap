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

## Runs

| Run | What | Data | Headline (test) | Report |
|---|---|---|---|---|
| `2026-09-16_commons-phase1` | Phase-1 check: single prompt vs ensemble, no split | 79 Commons photos, 20 classes | top-1 36.7% → 50.6% | (numbers only, `metrics.json`) |
| `2026-09-16_main` | Baselines, linear probe, hybrid, calibration, abstention, not-food guard | Food-101 (30 cl.) + Indian Food Images (15 cl.); 1800 / 900 / 2461 | hybrid top-1 79.4%, top-5 97.3%; ECE 0.443 → 0.062 | [reports/2026-09-16_main.md](reports/2026-09-16_main.md) |
| `2026-09-16_latency` | `/api/recognize` wall-clock, 20 requests each | test fixtures | 70.6 ms median (whole image), 352 ms with crops | `latency.json` |
| – | Real phone photos from users | – | Dataset required before evaluation. | – |

Note: `2026-09-16_main/env.json` records commit `2d9d82e` with `dirty: true`. The run used the experiment code from the working tree, which was committed right after in `bedbcd0` without changes.
