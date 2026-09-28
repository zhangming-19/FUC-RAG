#!/usr/bin/env python3

from pathlib import Path
import csv
import re
import statistics

RGDU_ROOT = Path(
    "outputs/dkpo/unified_eval/generation/"
    "rgdu_infer_ratio_sweep_seed42_strict_20260905"
)

SUMMARY_DIR = RGDU_ROOT / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_CSV = Path(
    "outputs/reproduce_paper/"
    "official_kto_infer_only_seed42_strict_20260904/"
    "summary/per_dataset_results.csv"
)

RATIOS = [
    0.0, 0.1, 0.2, 0.25,
    0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0,
]

DATASETS = [
    "NaturalQuestionsShort",
    "NewsQA",
    "SQuAD",
    "SearchQA",
    "TriviaQA-web",
    "hotpotq",
]

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

PATTERN = re.compile(
    rf"Step:\s*(\d+):\s*"
    rf"pc\s*({NUM}),\s*"
    rf"po\s*({NUM}),\s*"
    rf"mr\s*({NUM}),\s*"
    rf"em\s*({NUM})"
)

def ratio_tag(r):
    s = str(r)
    return s.replace(".", "p")

# ============================================================
# 1. Parse RGDU logs
# ============================================================

rows = []

for ratio in RATIOS:
    tag = ratio_tag(ratio)

    for dataset in DATASETS:
        log_file = (
            RGDU_ROOT /
            f"ratio_{tag}" /
            f"{dataset}.log"
        )

        if not log_file.exists():
            print(
                f"[MISSING] ratio={ratio:.2f} "
                f"dataset={dataset} "
                f"path={log_file}"
            )
            continue

        text = log_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        matches = PATTERN.findall(text)

        if not matches:
            print(
                f"[PARSE_ERROR] ratio={ratio:.2f} "
                f"dataset={dataset}"
            )
            continue

        step, conr, memr, mr, em = matches[-1]

        rows.append({
            "ratio": float(ratio),
            "dataset": dataset,
            "step": int(step),
            "ConR": float(conr),
            "MemR": float(memr),
            "MR": float(mr),
            "EM": float(em),
        })

print()
print("============================================================")
print(f"RGDU PARSED = {len(rows)} / 72")
print("============================================================")

if len(rows) != 72:
    raise SystemExit(
        "ERROR: RGDU results incomplete. "
        "Do not select final lambdas."
    )

# ============================================================
# 2. Save complete per-dataset results
# ============================================================

per_dataset_csv = SUMMARY_DIR / "rgdu_per_dataset_results.csv"

with per_dataset_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "ratio",
            "step",
            "ConR",
            "MemR",
            "MR",
            "EM",
        ],
    )

    writer.writeheader()

    for dataset in DATASETS:
        for ratio in RATIOS:
            r = next(
                x for x in rows
                if x["dataset"] == dataset
                and abs(x["ratio"] - ratio) < 1e-12
            )
            writer.writerow({
                "dataset": r["dataset"],
                "ratio": r["ratio"],
                "step": r["step"],
                "ConR": r["ConR"],
                "MemR": r["MemR"],
                "MR": r["MR"],
                "EM": r["EM"],
            })

# ============================================================
# 3. Print full curves
# ============================================================

print()
print("============================================================")
print("RGDU FULL PER-DATASET RATIO SWEEP")
print("============================================================")

for dataset in DATASETS:

    ds_rows = sorted(
        [r for r in rows if r["dataset"] == dataset],
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

# ============================================================
# 4. RGDU best ratio by MR
# ============================================================

best_rgdu = {}

for dataset in DATASETS:
    ds_rows = [
        r for r in rows
        if r["dataset"] == dataset
    ]

    # Primary criterion: MR minimum.
    # Tie-break:
    # 1. lower MemR
    # 2. higher ConR
    # 3. higher EM
    best = min(
        ds_rows,
        key=lambda x: (
            x["MR"],
            x["MemR"],
            -x["ConR"],
            -x["EM"],
        )
    )

    best_rgdu[dataset] = best

best_csv = SUMMARY_DIR / "rgdu_best_by_mr.csv"

with best_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "best_ratio",
            "ConR",
            "MemR",
            "MR",
            "EM",
        ],
    )

    writer.writeheader()

    for dataset in DATASETS:
        r = best_rgdu[dataset]
        writer.writerow({
            "dataset": dataset,
            "best_ratio": r["ratio"],
            "ConR": r["ConR"],
            "MemR": r["MemR"],
            "MR": r["MR"],
            "EM": r["EM"],
        })

print()
print("============================================================")
print("RGDU BEST λ_infer BY MR")
print("============================================================")

print(
    f"{'Dataset':<24}"
    f"{'λ*':>7}"
    f"{'ConR↑':>12}"
    f"{'MemR↓':>12}"
    f"{'MR↓':>12}"
    f"{'EM↑':>12}"
)

for dataset in DATASETS:
    r = best_rgdu[dataset]

    print(
        f"{dataset:<24}"
        f"{r['ratio']:>7.2f}"
        f"{r['ConR']:>12.4f}"
        f"{r['MemR']:>12.4f}"
        f"{r['MR']:>12.4f}"
        f"{r['EM']:>12.4f}"
    )

# ============================================================
# 5. Macro by fixed ratio
# ============================================================

macro_rows = []

for ratio in RATIOS:

    subset = [
        r for r in rows
        if abs(r["ratio"] - ratio) < 1e-12
    ]

    macro_rows.append({
        "ratio": ratio,
        "ConR": statistics.mean(r["ConR"] for r in subset),
        "MemR": statistics.mean(r["MemR"] for r in subset),
        "MR": statistics.mean(r["MR"] for r in subset),
        "EM": statistics.mean(r["EM"] for r in subset),
    })

macro_csv = SUMMARY_DIR / "rgdu_macro_by_ratio.csv"

with macro_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "ratio",
            "ConR",
            "MemR",
            "MR",
            "EM",
        ],
    )

    writer.writeheader()
    writer.writerows(macro_rows)

print()
print("============================================================")
print("RGDU MACRO BY FIXED RATIO")
print("============================================================")

print(
    f"{'ratio':>7}"
    f"{'ConR↑':>12}"
    f"{'MemR↓':>12}"
    f"{'MR↓':>12}"
    f"{'EM↑':>12}"
)

for r in macro_rows:
    print(
        f"{r['ratio']:>7.2f}"
        f"{r['ConR']:>12.4f}"
        f"{r['MemR']:>12.4f}"
        f"{r['MR']:>12.4f}"
        f"{r['EM']:>12.4f}"
    )

# ============================================================
# 6. Dataset-specific RGDU macro
# ============================================================

rgdu_adaptive_macro = {
    "ConR": statistics.mean(
        best_rgdu[d]["ConR"] for d in DATASETS
    ),
    "MemR": statistics.mean(
        best_rgdu[d]["MemR"] for d in DATASETS
    ),
    "MR": statistics.mean(
        best_rgdu[d]["MR"] for d in DATASETS
    ),
    "EM": statistics.mean(
        best_rgdu[d]["EM"] for d in DATASETS
    ),
}

ratio0 = next(
    r for r in macro_rows
    if abs(r["ratio"] - 0.0) < 1e-12
)

print()
print("============================================================")
print("RGDU FIXED λ=0 vs DATASET-SPECIFIC λ*")
print("============================================================")

print(
    f"{'Method':<26}"
    f"{'ConR↑':>12}"
    f"{'MemR↓':>12}"
    f"{'MR↓':>12}"
    f"{'EM↑':>12}"
)

print(
    f"{'RGDU fixed λ=0':<26}"
    f"{ratio0['ConR']:>12.4f}"
    f"{ratio0['MemR']:>12.4f}"
    f"{ratio0['MR']:>12.4f}"
    f"{ratio0['EM']:>12.4f}"
)

print(
    f"{'RGDU dataset-specific':<26}"
    f"{rgdu_adaptive_macro['ConR']:>12.4f}"
    f"{rgdu_adaptive_macro['MemR']:>12.4f}"
    f"{rgdu_adaptive_macro['MR']:>12.4f}"
    f"{rgdu_adaptive_macro['EM']:>12.4f}"
)

print()
print("Delta adaptive - fixed0:")

print(
    "ΔConR = "
    f"{rgdu_adaptive_macro['ConR'] - ratio0['ConR']:+.4f}"
)

print(
    "ΔMemR = "
    f"{rgdu_adaptive_macro['MemR'] - ratio0['MemR']:+.4f}"
)

print(
    "ΔMR   = "
    f"{rgdu_adaptive_macro['MR'] - ratio0['MR']:+.4f}"
)

print(
    "ΔEM   = "
    f"{rgdu_adaptive_macro['EM'] - ratio0['EM']:+.4f}"
)

# ============================================================
# 7. Load Official KTO scan and compare best MR ratios
# ============================================================

if not OFFICIAL_CSV.exists():
    print()
    print(
        "[WARNING] Official KTO CSV not found:"
    )
    print(OFFICIAL_CSV)
else:

    official_rows = []

    with OFFICIAL_CSV.open(
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for r in reader:
            official_rows.append({
                "dataset": r["dataset"],
                "ratio": float(r["ratio"]),
                "ConR": float(r["ConR"]),
                "MemR": float(r["MemR"]),
                "MR": float(r["MR"]),
                "EM": float(r["EM"]),
            })

    official_best = {}

    for dataset in DATASETS:

        ds = [
            r for r in official_rows
            if r["dataset"] == dataset
        ]

        official_best[dataset] = min(
            ds,
            key=lambda x: (
                x["MR"],
                x["MemR"],
                -x["ConR"],
                -x["EM"],
            )
        )

    compare_csv = (
        SUMMARY_DIR /
        "official_vs_rgdu_best_mr_ratio.csv"
    )

    with compare_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "official_best_ratio",
                "rgdu_best_ratio",
                "ratio_shift_rgdu_minus_official",
                "official_best_MR",
                "rgdu_best_MR",
            ],
        )

        writer.writeheader()

        for dataset in DATASETS:

            o = official_best[dataset]
            g = best_rgdu[dataset]

            writer.writerow({
                "dataset": dataset,
                "official_best_ratio": o["ratio"],
                "rgdu_best_ratio": g["ratio"],
                "ratio_shift_rgdu_minus_official":
                    g["ratio"] - o["ratio"],
                "official_best_MR": o["MR"],
                "rgdu_best_MR": g["MR"],
            })

    print()
    print("============================================================")
    print("OFFICIAL KTO vs RGDU: BEST MR λ_infer")
    print("============================================================")

    print(
        f"{'Dataset':<24}"
        f"{'Official λ*':>12}"
        f"{'RGDU λ*':>12}"
        f"{'Shift':>12}"
        f"{'Official MR':>14}"
        f"{'RGDU MR':>12}"
    )

    for dataset in DATASETS:

        o = official_best[dataset]
        g = best_rgdu[dataset]

        print(
            f"{dataset:<24}"
            f"{o['ratio']:>12.2f}"
            f"{g['ratio']:>12.2f}"
            f"{g['ratio'] - o['ratio']:>+12.2f}"
            f"{o['MR']:>14.4f}"
            f"{g['MR']:>12.4f}"
        )

print()
print("============================================================")
print("OUTPUT FILES")
print("============================================================")
print(per_dataset_csv)
print(best_csv)
print(macro_csv)

if OFFICIAL_CSV.exists():
    print(compare_csv)

print()
print("SUMMARY COMPLETE")
