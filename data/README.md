# Data

Nothing in this folder except this file is committed. Everything is re-created by scripts.

| Set | Used for | Source | Licence / terms | How to get it |
|---|---|---|---|---|
| Food-101 (30 of 101 classes) | train / val / test | Bossard et al., ECCV 2014, via [ethz/food101](https://huggingface.co/datasets/ethz/food101) | Research dataset; images come from foodspotting.com. The HF card lists the licence as unknown, so images are used locally for evaluation only and never redistributed | `cd backend && python -m ml.datasets food101` |
| Indian Food Images (15 classes) | train / val / test | [bharat-raghunathan/indian-foods-dataset](https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset), originally Kaggle "The Massive Indian Food Dataset" | The HF card says CC0-1.0, but the images look web-scraped (recipe sites, watermarks), so they are also not redistributed | `cd backend && python -m ml.datasets indianfoods` |
| Wikimedia Commons set (79 food + 36 non-food) | out-of-distribution check, not-food guard | Hand-picked from Commons categories | Per image (mostly CC BY / CC BY-SA); author and licence are in `backend/ml/testset/manifest.json` | `cd backend && python -m ml.fetch_testset` |
| USDA FoodData Central (FNDDS 2021-2023, SR Legacy) | nutrition verification | [fdc.nal.usda.gov](https://fdc.nal.usda.gov/download-datasets) | Public domain (U.S. Government work) | `cd backend && python -m scripts.usda_verify` |

Considered but not used: `rajistics/indian_food_images` has no licence on its card and its images were scraped from Google.

## Sampling (seed 42)

- **Food-101:** the parquet shards are sorted by class. `ml/datasets.py` reads each shard's footer, uses the per-row-group label statistics, and fetches only the row groups it needs over HTTP range requests (about 1 GB instead of 5 GB). From the official *train* split it takes 40 train and 20 val images per class (disjoint). From the official *validation* split, which is Food-101's test set, it takes 50 test images per class.
- **Indian Food Images:** 40 train and 20 val images per class from its train split, plus all 961 images of its test split.
- Images are re-encoded as JPEG with the longest side at most 512 px.
- The class → food-id mapping is in `backend/ml/class_mapping.json`. The first id in each list is the training label, and any id in the list counts as correct. Classes that aren't mapped are not used.

## Snapshot used for the committed results

From `data/processed/manifest.json`, written by the script:

| Set | Rows | Train | Val | Test | Classes | Row-id hash | File hash |
|---|---|---|---|---|---|---|---|
| food101 | 3300 | 1200 | 600 | 1500 | 30 | 7d1fdca089516c59 | f91745cfbf0c3e66 |
| indianfoods | 1861 | 600 | 300 | 961 | 15 | 56d709d246578e58 | 890a750d5fbd24e1 |

## Label quality notes (from looking at samples)

- `indianfoods/paneer` is a mix of paneer curries (butter masala, kadai, …), so it accepts any of six paneer dishes.
- `indianfoods/kofta` is mostly malai kofta, but some images look like egg curry or nargisi kofta.
- `indianfoods/cholebhature` shows the plate (chole + bhature), so it accepts either food.
- The Indian photos are styled recipe-blog shots, often with text overlays. They are easier than real phone photos of a home thali.
