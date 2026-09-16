"""Download the evaluation images listed in ml/testset/manifest.json from Wikimedia Commons.

    python -m ml.fetch_testset

Images go to ml/testset/images/ (git-ignored). Each image keeps its Commons license; see the
manifest for author, license and the description page. Wikimedia asks for a descriptive
User-Agent and polite request rates, so this downloads sequentially with a short pause and backs
off on HTTP 429.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "testset"
MANIFEST = ROOT / "manifest.json"
IMAGES = ROOT / "images"
UA = "NutriSnapEval/1.0 (student project; evaluation of zero-shot food recognition) python-urllib"


def fetch(url: str, dest: Path, retries: int = 5) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    raise ValueError(f"not an image: {r.headers.get('Content-Type')}")
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                print(f"  HTTP {e.code} for {url}")
                return False
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"  {e} (attempt {attempt + 1})")
        time.sleep(5 * (attempt + 1))
    return False


def main() -> int:
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    ok = skipped = 0
    for it in items:
        dest = IMAGES / it["file"]
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        print("GET", it["file"])
        if fetch(it["url"], dest):
            ok += 1
        time.sleep(1.0)
    missing = [it["file"] for it in items if not (IMAGES / it["file"]).exists()]
    print(f"downloaded {ok}, already present {skipped}, missing {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
