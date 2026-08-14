#!/usr/bin/env python3
"""Verify performance.tsv data against the paper's claims."""

import pandas as pd
import numpy as np

# Read data
df = pd.read_csv("performance.tsv", sep="\t")
df["Coverage"] = df["Coverage"].str.lower()
issues = []


def check_close(label: str, observed: float, expected: float, tolerance: float = 0.05) -> None:
    status = "OK" if abs(observed - expected) <= tolerance else "MISMATCH"
    print(f"  {label}: observed={observed:.1f}%, expected={expected:.1f}% [{status}]")
    if status != "OK":
        issues.append(f"{label}: observed {observed:.1f}%, expected {expected:.1f}%")

print("=" * 80)
print("1. Ultra-low coverage (0.01x - 0.05x) Recall")
print("=" * 80)

for cov in ["0.01x", "0.05x"]:
    sub = df[df["Coverage"] == cov]
    print(f"\n--- {cov} ---")
    for method in ["vmdm", "mykrobe", "tb_profiler"]:
        msub = sub[sub["Method"] == method]
        recalls = msub["Recall"].values * 100
        print(f"  {method:12s}: recall range = {recalls.min():.1f}% - {recalls.max():.1f}%, mean = {recalls.mean():.1f}%")

print("\n" + "=" * 80)
print("2. 0.5x coverage: mean recall across 4 drugs")
print("=" * 80)

sub_05 = df[df["Coverage"] == "0.5x"]
for method in ["vmdm", "mykrobe", "tb_profiler"]:
    msub = sub_05[sub_05["Method"] == method]
    mean_recall = msub["Recall"].mean() * 100
    print(f"  {method:12s}: mean recall = {mean_recall:.1f}%")

vmdm_mean = sub_05[sub_05["Method"] == "vmdm"]["Recall"].mean() * 100
myk_mean = sub_05[sub_05["Method"] == "mykrobe"]["Recall"].mean() * 100
tbp_mean = sub_05[sub_05["Method"] == "tb_profiler"]["Recall"].mean() * 100
print(f"\n  VMDM vs Mykrobe fold gain:   {vmdm_mean / myk_mean:.1f}x (claimed >16x)")
print(f"  VMDM vs TB-Profiler abs diff: {vmdm_mean - tbp_mean:.1f}% (claimed 17.8%)")

print("\n" + "=" * 80)
print("3. 0.5x coverage: precision/specificity for RIF & INH, per-drug recall")
print("=" * 80)

expected_recall = {
    "Rifampicin": 67.0,
    "Isoniazid": 67.9,
    "Ethambutol": 52.3,
    "Pyrazinamide": 40.7,
}
for drug in ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]:
    row_vmdm = df[(df["Coverage"] == "0.5x") & (df["Method"] == "vmdm") & (df["Drug"] == drug)]
    recall = row_vmdm["Recall"].values[0] * 100
    precision = row_vmdm["Precision"].values[0] * 100
    spec = row_vmdm["Specificity"].values[0] * 100
    print(f"  {drug:12s}: recall = {recall:.1f}%, precision = {precision:.1f}%, specificity = {spec:.1f}%")
    check_close(f"{drug} recall", recall, expected_recall[drug])

print("\n  Claim: RIF/INH precision >95%; RIF/INH specificity = 98.5%/97.9%")

for drug in ["Rifampicin", "Isoniazid"]:
    row = df[(df["Coverage"] == "0.5x") & (df["Method"] == "vmdm") & (df["Drug"] == drug)]
    precision = row["Precision"].values[0] * 100
    spec = row["Specificity"].values[0] * 100
    precision_status = "OK" if precision > 95 else "MISMATCH"
    spec_expected = 98.5 if drug == "Rifampicin" else 97.9
    print(f"  {drug} precision >95%: {precision:.1f}% [{precision_status}]")
    if precision_status != "OK":
        issues.append(f"{drug} precision is {precision:.1f}%, expected >95%")
    check_close(f"{drug} specificity", spec, spec_expected)

print("\n" + "=" * 80)
print("4. Pooled across all coverages: mean recall improvement vs competitors")
print("=" * 80)

for method_comp in ["mykrobe", "tb_profiler"]:
    diffs = []
    for cov in df["Coverage"].unique():
        for drug in df["Drug"].unique():
            v = df[(df["Coverage"] == cov) & (df["Method"] == "vmdm") & (df["Drug"] == drug)]["Recall"].values[0]
            c = df[(df["Coverage"] == cov) & (df["Method"] == method_comp) & (df["Drug"] == drug)]["Recall"].values[0]
            diffs.append((v - c) * 100)
    diffs = np.array(diffs)
    print(f"  VMDM vs {method_comp:12s}: mean recall improvement = {diffs.mean():.1f}% (claimed {'27.1%' if method_comp=='mykrobe' else '10.7%'})")

print("\n" + "=" * 80)
print("5. Pooled F1 improvement vs competitors (by coverage)")
print("=" * 80)

for cov in sorted(df["Coverage"].unique(), key=lambda x: float(x.replace("x",""))):
    print(f"\n  --- {cov} ---")
    for method_comp in ["mykrobe", "tb_profiler"]:
        f1_diffs = []
        for drug in df["Drug"].unique():
            v = df[(df["Coverage"] == cov) & (df["Method"] == "vmdm") & (df["Drug"] == drug)]["Fscore"].values[0]
            c = df[(df["Coverage"] == cov) & (df["Method"] == method_comp) & (df["Drug"] == drug)]["Fscore"].values[0]
            f1_diffs.append((v - c) * 100)
        f1_diffs = np.array(f1_diffs)
        print(f"    vs {method_comp:12s}: F1 improvement range = {f1_diffs.min():.1f}% ~ {f1_diffs.max():.1f}%")

print("\n" + "=" * 80)
print("6. Computational footprint check (not in TSV — skip)")
print("=" * 80)
print("  (No memory/time data in TSV)")

print("\n" + "=" * 80)
print("SUMMARY OF DISCREPANCIES")
print("=" * 80)
if issues:
    for issue in issues:
        print(f"  - {issue}")
    raise SystemExit(1)
print("  None.")
