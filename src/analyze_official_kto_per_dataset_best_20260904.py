#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

CSV_FILE = Path(
    "outputs/reproduce_paper/"
    "official_kto_infer_only_seed42_strict_20260904/"
    "summary/per_dataset_results.csv"
)

rows = []

with CSV_FILE.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for r in reader:
        rows.append({
            "ratio": float(r["ratio"]),
            "dataset": r["dataset"],
            "n": int(r["n"]),
            "ConR": float(r["ConR"]),
            "MemR": float(r["MemR"]),
            "MR": float(r["MR"]),
            "EM": float(r["EM"]),
        })

by_dataset = defaultdict(list)

for r in rows:
    by_dataset[r["dataset"]].append(r)

dataset_order = [
    "NaturalQuestionsShort",
    "NewsQA",
    "SQuAD",
    "SearchQA",
    "TriviaQA-web",
    "hotpotq",
]

print()
print("==========================================================================")
print("PER-DATASET FULL RATIO SWEEP")
print("==========================================================================")

for dataset in dataset_order:

    ds_rows = sorted(
        by_dataset[dataset],
        key=lambda x: x["ratio"]
    )

    print()
    print(f"================ {dataset} ================")

    print(
        f"{'ratio':>7}"
        f"{'ConR↑':>12}"
        f"{'MemR↓':>12}"
        f"{'MR↓':>12}"
        f"{'EM↑':>12}"
    )

    for r in ds_rows:
        print(
            f"{r['ratio']:>7.2f}"
            f"{r['ConR']:>12.4f}"
            f"{r['MemR']:>12.4f}"
            f"{r['MR']:>12.4f}"
            f"{r['EM']:>12.4f}"
        )

print()
print()
print("==========================================================================")
print("BEST RATIO BY SINGLE METRIC FOR EACH DATASET")
print("==========================================================================")

print(
    f"{'Dataset':<24}"
    f"{'Best ConR':<20}"
    f"{'Best MemR':<20}"
    f"{'Best MR':<20}"
    f"{'Best EM':<20}"
)

for dataset in dataset_order:

    ds_rows = by_dataset[dataset]

    best_conr = max(ds_rows, key=lambda x: x["ConR"])
    best_memr = min(ds_rows, key=lambda x: x["MemR"])
    best_mr = min(ds_rows, key=lambda x: x["MR"])
    best_em = max(ds_rows, key=lambda x: x["EM"])

    conr_str = f"{best_conr['ratio']:.2f} ({best_conr['ConR']:.2f})"
    memr_str = f"{best_memr['ratio']:.2f} ({best_memr['MemR']:.2f})"
    mr_str = f"{best_mr['ratio']:.2f} ({best_mr['MR']:.2f})"
    em_str = f"{best_em['ratio']:.2f} ({best_em['EM']:.2f})"

    print(
        f"{dataset:<24}"
        f"{conr_str:<20}"
        f"{memr_str:<20}"
        f"{mr_str:<20}"
        f"{em_str:<20}"
    )

print()
print("==========================================================================")
print("DELTA OF EACH RATIO VS ratio=1.0")
print("==========================================================================")

for dataset in dataset_order:

    ds_rows = sorted(
        by_dataset[dataset],
        key=lambda x: x["ratio"]
    )

    base = next(
        r for r in ds_rows
        if abs(r["ratio"] - 1.0) < 1e-12
    )

    print()
    print(f"================ {dataset} ================")

    print(
        f"{'ratio':>7}"
        f"{'ΔConR':>12}"
        f"{'ΔMemR':>12}"
        f"{'ΔMR':>12}"
        f"{'ΔEM':>12}"
    )

    for r in ds_rows:

        print(
            f"{r['ratio']:>7.2f}"
            f"{r['ConR'] - base['ConR']:>+12.4f}"
            f"{r['MemR'] - base['MemR']:>+12.4f}"
            f"{r['MR'] - base['MR']:>+12.4f}"
            f"{r['EM'] - base['EM']:>+12.4f}"
        )

