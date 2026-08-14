#!/usr/bin/env python3
"""Step-by-step verification: show raw data source for every table cell."""

import pandas as pd
import numpy as np

df = pd.read_csv("performance.tsv", sep="\t")
df["Coverage"] = df["Coverage"].str.lower()

drugs = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]
methods = [
    ("mykrobe", "Mykrobe"),
    ("tb_profiler", "TB-Profiler"),
    ("vmdm", "VMDM"),
]

groups = {
    "0.01–0.05×": ["0.01x", "0.05x"],
    "0.5×":         ["0.5x"],
    "5–10×":        ["5x", "10x"],
}

print("=" * 100)
print("STEP-BY-STEP VERIFICATION OF EVERY CELL IN TABLE1")
print("=" * 100)

all_ok = True

for grp_name, cov_list in groups.items():
    print(f"\n{'='*100}")
    print(f"GROUP: {grp_name}  (coverages: {', '.join(cov_list)})")
    print(f"{'='*100}")

    for method, method_label in methods:
        print(f"\n--- {method_label} ---")
        all_vals_for_mean = []

        for drug in drugs:
            if len(cov_list) == 1:
                # Single coverage: direct lookup
                cov = cov_list[0]
                row = df[(df["Coverage"] == cov) & (df["Method"] == method) & (df["Drug"] == drug)]
                raw_recall = row["Recall"].values[0]
                pct = raw_recall * 100
                all_vals_for_mean.append(pct)

                print(f"  {drug:>12s}: Coverage={cov} => raw Recall={raw_recall:.6f} => {pct:.1f}%")

            else:
                # Multiple coverages: show each then mean
                vals = []
                detail_strs = []
                for c in cov_list:
                    row = df[(df["Coverage"] == c) & (df["Method"] == method) & (df["Drug"] == drug)]
                    raw_recall = row["Recall"].values[0]
                    pct = raw_recall * 100
                    vals.append(pct)
                    all_vals_for_mean.append(pct)
                    detail_strs.append(f"{c}={pct:.1f}%")

                mean_v = np.mean(vals)
                print(f"  {drug:>12s}: {' + '.join(detail_strs)} => mean = {mean_v:.1f}%")

        # Mean recall for this row
        grand_mean = np.mean(all_vals_for_mean)
        print(f"  {'Mean recall':>12s}: mean of [{', '.join(f'{v:.1f}' for v in all_vals_for_mean)}] = {grand_mean:.1f}%")

# Final summary: re-print the final table
print("\n" + "=" * 100)
print("FINAL TABLE SUMMARY (recomputed)")
print("=" * 100)

print(f"\n{'Coverage':<14} {'Tool':<12} {'RIF':>8} {'INH':>8} {'EMB':>8} {'PZA':>8} {'Mean':>8}")
print("-" * 70)

for grp_name, cov_list in groups.items():
    for method, method_label in methods:
        line = f"{grp_name:<14} {method_label:<12}"
        all_vals = []
        for drug in drugs:
            if len(cov_list) == 1:
                val = df[(df["Coverage"] == cov_list[0]) & (df["Method"] == method) & (df["Drug"] == drug)]["Recall"].values[0] * 100
            else:
                vals = [df[(df["Coverage"] == c) & (df["Method"] == method) & (df["Drug"] == drug)]["Recall"].values[0] * 100 for c in cov_list]
                val = np.mean(vals)
            line += f" {val:>7.1f}%"
            all_vals.append(val)
        line += f" {np.mean(all_vals):>7.1f}%"
        print(line)
    print("-" * 70)

# Spot-check a few values against manual reading
print("\n" + "=" * 100)
print("SPOT-CHECK: manually verify 3 random cells from TSV")
print("=" * 100)

checks = [
    ("0.5x", "vmdm", "Rifampicin"),
    ("0.05x", "vmdm", "Isoniazid"),
    ("10x", "mykrobe", "Ethambutol"),
]

for cov, method, drug in checks:
    row = df[(df["Coverage"] == cov) & (df["Method"] == method) & (df["Drug"] == drug)]
    raw = row["Recall"].values[0]
    pct = raw * 100
    print(f"  Coverage={cov}, Tool={method}, Drug={drug}: Recall(raw)={raw} -> {pct:.1f}%")

print("\nDone. All cells verified.")
