import os
import io
import json
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
import requests
from PIL import Image

# Builds the RQ1 image set from VLSU (apple/ml-vlsu, data/VLSU.csv). Network-bound only, no model:
# meant to run on the login node, since compute nodes may have no internet.
#
#   unsafe        image_grade == unsafe, random draw
#   safe_matched  image_grade == safe with a harm category, per-category quotas = unsafe counts
#   safe_generic  image_grade == safe with no category
#
# Category of an image = image_category, else combined_category, else text_category (code only,
# e.g. "C3"). For safe images this is the harm context VLSU paired them with. Dead links are skipped,
# so each group is drawn from a shuffled pool until its quota downloads. Splits (pilot/fit/test) are
# assigned after download, stratified on (label, group, category). See docs/analysis_plan.md.

EXCLUDE = ("C6:", "C7:", "C13:", "C15:")
CATEGORY_COLS = ["image_category", "combined_category", "text_category"]
SPLITS = {"pilot": 0.2, "fit": 0.4, "test": 0.4}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36"}


def category(row):
    for col in CATEGORY_COLS:
        v = row[col]
        if isinstance(v, str) and v:
            return v.split(":")[0]
    return ""


def fetch(row, img_dir):
    path = os.path.join(img_dir, f"{row['uuid']}.png")
    if os.path.exists(path):
        return path
    try:
        r = requests.get(row["web_path"], headers=HEADERS, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 64:
            return None
        img.save(path)
        return path
    except Exception:
        return None


def draw(pool, n, img_dir, workers):
    """Download rows from a shuffled pool until n succeed; returns (kept rows, tried)."""
    kept, tried = [], 0
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        while len(kept) < n and tried < len(pool):
            chunk = pool[tried:tried + workers * 2]
            tried += len(chunk)
            for row, path in zip(chunk, ex.map(lambda r: fetch(r, img_dir), chunk)):
                if path and len(kept) < n:
                    kept.append({**row, "path": path})
    return kept, tried


def assign_splits(items, seed):
    rng = np.random.default_rng(seed)
    strata = {}
    for it in items:
        strata.setdefault((it["label"], it["group"], it["category"]), []).append(it)
    # Deal shuffled strata into a repeating pilot/fit/fit/test/test cycle; the cycle position carries
    # over between strata, so small categories don't all land in the same split
    cycle = [s for s in SPLITS for _ in range(round(SPLITS[s] * 5))]
    k = 0
    for key in sorted(strata):
        members = strata[key]
        rng.shuffle(members)
        for it in members:
            it["split"] = cycle[k % len(cycle)]
            k += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/VLSU.csv")
    parser.add_argument("--out", default="data")
    parser.add_argument("--n_unsafe", type=int, default=300)
    parser.add_argument("--n_generic", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    excluded = df[CATEGORY_COLS].fillna("").apply(lambda r: any(v.startswith(EXCLUDE) for v in r), axis=1)
    df = df[~excluded].drop_duplicates("web_path").copy()
    df["category"] = df.apply(category, axis=1)
    cols = ["uuid", "web_path", "image_grade", "category"]
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    def pool(mask):
        return df[mask][cols].sample(frac=1, random_state=args.seed).to_dict("records")

    report = {}
    unsafe, tried = draw(pool(df["image_grade"] == "unsafe"), args.n_unsafe, img_dir, args.workers)
    report["unsafe"] = {"kept": len(unsafe), "tried": tried}
    quotas = pd.Series([r["category"] for r in unsafe]).value_counts().to_dict()

    matched, shortfall = [], {}
    for cat, q in sorted(quotas.items()):
        rows, tried = draw(pool((df["image_grade"] == "safe") & (df["category"] == cat)), q, img_dir, args.workers)
        matched += rows
        if len(rows) < q:
            shortfall[cat] = q - len(rows)
    report["safe_matched"] = {"kept": len(matched), "quotas": quotas, "shortfall": shortfall}

    generic, tried = draw(pool((df["image_grade"] == "safe") & (df["category"] == "")), args.n_generic, img_dir, args.workers)
    report["safe_generic"] = {"kept": len(generic), "tried": tried}

    items = []
    for group, rows, label in [("unsafe", unsafe, 1), ("safe_matched", matched, 0), ("safe_generic", generic, 0)]:
        for r in rows:
            items.append({"uuid": r["uuid"], "image": os.path.relpath(r["path"], args.out), "label": label,
                          "group": group, "category": r["category"], "web_path": r["web_path"]})
    assign_splits(items, args.seed)
    items.sort(key=lambda it: it["uuid"])

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    report["splits"] = pd.DataFrame(items).groupby(["split", "group"]).size().unstack(fill_value=0).to_dict()
    report.update({"source": "apple/ml-vlsu data/VLSU.csv", "seed": args.seed, "excluded_categories": list(EXCLUDE)})
    with open(os.path.join(args.out, "dataset_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
