# Model comparison, ONNX optimisation and CLIP analysis

These scripts back [docs/COMPARISON.md](../../docs/COMPARISON.md), [docs/CLIP.md](../../docs/CLIP.md) and
[docs/DATASET.md](../../docs/DATASET.md). They use the same data, splits and accept-set rule as
`experiments/run.py`, so the numbers can be compared directly.

```bash
cd backend
pip install -r requirements-ml.txt -r ../experiments/compare/requirements.txt   # + torch/torchvision (CPU is fine)
python -m scripts.download_models && python -m ml.datasets && python -m ml.fetch_testset

python ../experiments/compare/dataset_stats.py                      # ~1 min   -> results/<date>_dataset/
python ../experiments/compare/train_cnn.py --model resnet50_frozen  # ~4 min   -> results/<date>_compare/cnn/
python ../experiments/compare/train_cnn.py --model resnet50_finetune# ~40 min on 4 CPU cores
python ../experiments/compare/train_cnn.py --model scratch          # ~30 min on 4 CPU cores
python ../experiments/compare/compare.py                            # ~8 min (CLIP embeddings cached after the first run)
python ../experiments/compare/onnx_bench.py                         # ~15 min, downloads fp32/fp16 CLIP (~600 MB)
python ../experiments/compare/clip_explain.py                       # ~1 min
```

| Script | What it does | Output |
|---|---|---|
| `data.py` | Loads the splits, maps labels to the 45 classes, caches 256×256 uint8 pixels for the CNNs | `data/processed/pix256_*.npy` |
| `train_cnn.py` | CNN from scratch; ResNet-50 frozen + logistic regression; ResNet-50 fine-tuned (layer4 + head). Epoch and C chosen on val | `cnn/<model>.npz/.json`, `backend/models/cnn/*.pt` |
| `compare.py` | CLIP zero-shot (1 and 7 prompts), CLIP + linear probe, CLIP + Tip-Adapter, plus the CNN outputs. Accuracy, deduplicated test, calibration, uncertainty (AURC), unseen photos, open-vocabulary new classes, few-shot curve | `metrics.json`, `accuracy.png`, `few_shot.png`, `uncertainty.png` |
| `onnx_bench.py` | CLIP vision fp32 / fp16 / int8 (download) / int8 (quantised here): size, latency (1/2/4 threads, graph optimisation levels, batch 8), accuracy impact; text tower; ResNet-50 PyTorch → ONNX → static INT8; per-stage request latency | `<date>_onnx/onnx.json`, `onnx.png` |
| `clip_explain.py` | Architecture read from the ONNX weights; image-text cosine statistics; logit-scale sweep; prompt sensitivity; closest labels in text space | `<date>_clip_explain/clip.json`, `clip_similarity.png` |
| `dataset_stats.py` | Class balance, image sizes, brightness, duplicate check | `<date>_dataset/dataset_stats.json`, `class_balance.png` |

Rule kept everywhere: anything chosen (C, epoch, α/β, temperature, threshold) is chosen on **val**.
**Test** and the **unseen** Commons photos are only scored.

Note: the committed `2026-11-26_compare` and `2026-11-26_clip_explain` results were measured with Xenova's 8-bit vision
model (the app's at the time). Re-running them now uses the per-channel int8 model that `download_models`
builds, so CLIP numbers will shift by about a point (see `experiments/reports/2026-11-26_main-int8vision.md`).
