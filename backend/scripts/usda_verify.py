"""Verify foods.json against USDA FoodData Central bulk downloads.

    python -m scripts.usda_verify            # download (cached), extract, update data/foods.json
    python -m scripts.usda_verify --check    # only report differences, don't write

Source: https://fdc.nal.usda.gov/download-datasets (public domain, U.S. Government work).
  * FNDDS / Survey foods, release 2024-10-31 (includes many Indian dishes: dal, idli, dosa, biryani...)
  * SR Legacy, release 2018-04

The per-item mapping lives in data/usda_mapping.json and was made by hand (only same-dish
matches). For each mapped food this script copies energy, protein, carbohydrate, fat and fibre per
100 g from the USDA record, sets `verified: true` and records the fdcId and URL. The previous
hand-written estimate is kept in `estimate_per100g` so the difference is visible. Unmapped foods
are marked `verified: false` with an honest source string. Serving-size gram weights are NOT
verified (they stay estimates) - see `servings_source`.

The DEMO_KEY REST API was rate-limited to 10 requests/hour here, so the bulk CSVs are used
instead; they contain the same fdcIds as the API.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
CACHE = DATA / "usda_cache"
DATASETS = {
    "fndds": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip",
    "sr": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
}
# FNDDS files use legacy nutrient numbers, SR Legacy uses FDC nutrient ids
NUTRIENTS = {"1008": "kcal", "208": "kcal", "1003": "protein", "203": "protein", "1005": "carbs", "205": "carbs",
             "1004": "fat", "204": "fat", "1079": "fibre", "291": "fibre"}
UNVERIFIED_SOURCE = "Estimate - typical recipe, written by the project author; not checked against a food database"


def download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / Path(DATASETS[name]).name
    if not dest.exists():
        print(f"downloading {DATASETS[name]}")
        req = urllib.request.Request(DATASETS[name], headers={"User-Agent": "NutriSnap/1.0 (student project)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def extract(zip_path: Path, wanted: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        with z.open(names["food.csv"]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                if row["fdc_id"] in wanted:
                    out[row["fdc_id"]] = {"description": row["description"], "per100g": {}}
        with z.open(names["food_nutrient.csv"]) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                if row["fdc_id"] in out and row["nutrient_id"] in NUTRIENTS:
                    out[row["fdc_id"]]["per100g"][NUTRIENTS[row["nutrient_id"]]] = round(float(row["amount"]), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mapping = json.loads((DATA / "usda_mapping.json").read_text(encoding="utf-8"))["foods"]
    wanted = {str(m["fdc_id"]) for m in mapping.values()}
    records: dict[str, dict] = {}
    for name in DATASETS:
        records.update(extract(download(name), wanted))
    missing = wanted - records.keys()
    if missing:
        print("fdcIds not found:", sorted(missing))
        return 1

    doc = json.loads((DATA / "foods.json").read_text(encoding="utf-8"))
    diffs = []
    for food in doc["foods"]:
        m = mapping.get(food["id"])
        est = food.get("estimate_per100g") or food["per100g"]
        if m is None:
            food.update({"per100g": est, "verified": False, "source": UNVERIFIED_SOURCE,
                         "source_id": None, "source_url": None})
            food.pop("estimate_per100g", None)
        else:
            rec = records[str(m["fdc_id"])]
            if rec["description"] != m["usda_description"]:
                print(f"description mismatch for {food['id']}: {rec['description']!r}")
                return 1
            p = rec["per100g"]
            usda = {k: round(p.get(k, 0.0), 1) for k in ("kcal", "protein", "carbs", "fat", "fibre")}
            diffs.append((food["id"], est["kcal"], usda["kcal"]))
            food.update({
                "per100g": usda, "estimate_per100g": est, "verified": True,
                "source": f"USDA FoodData Central - {m['dataset']}: \"{rec['description']}\"",
                "source_id": f"fdc:{m['fdc_id']}",
                "source_url": f"https://fdc.nal.usda.gov/food-details/{m['fdc_id']}/nutrients",
            })
            if m.get("note"):
                food["source_note"] = m["note"]
        food["servings_source"] = "estimate"
    n_ver = sum(f["verified"] for f in doc["foods"])
    doc["note"] = (f"{n_ver} of {len(doc['foods'])} foods have per-100 g values copied from USDA FoodData Central "
                   "(verified: true, see source_id). The rest are unverified estimates. Serving gram weights are "
                   "estimates for all foods.")
    diffs.sort(key=lambda d: -abs(d[2] - d[1]))
    rel = [abs(u - e) / max(u, 1) for _, e, u in diffs]
    print(f"verified {n_ver}/{len(doc['foods'])}; median |estimate-USDA| kcal error "
          f"{sorted(rel)[len(rel) // 2] * 100:.0f}%")
    for fid, e, u in diffs[:12]:
        print(f"  {fid:22s} estimate {e:6.0f}  USDA {u:6.0f}")
    report = {"verified": n_ver, "total": len(doc["foods"]),
              "median_relative_kcal_error_of_old_estimates": round(sorted(rel)[len(rel) // 2], 3),
              "largest_differences": [{"food": f, "estimate_kcal": e, "usda_kcal": u} for f, e, u in diffs[:15]]}
    if not args.check:
        (DATA / "foods.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (DATA / "usda_verification.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print("wrote data/foods.json and data/usda_verification.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
