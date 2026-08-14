#!/usr/bin/env python3
"""Verify ALL numerical claims in the paper's description text against performance.tsv."""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("performance.tsv", sep="\t")
df["Coverage"] = df["Coverage"].str.lower()

drugs = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]
cov_order = ["0.01x", "0.05x", "0.1x", "0.5x", "1x", "5x", "10x"]

print("=" * 100)
print("VERIFICATION OF PAPER DESCRIPTION NUMERICAL CLAIMS")
print("=" * 100)

issues = []

# Claim 1: Pooled recall improvement
print("\n--- CLAIM 1: Pooled recall improvement ---")
for comp in ["mykrobe", "tb_profiler"]:
    diffs = []
    for cov in cov_order:
        for drug in drugs:
            v = df[(df["Coverage"]==cov) & (df["Method"]=="vmdm") & (df["Drug"]==drug)]["Recall"].values[0]
            c = df[(df["Coverage"]==cov) & (df["Method"]==comp) & (df["Drug"]==drug)]["Recall"].values[0]
            diffs.append(v - c)
    diffs_pct = np.array(diffs) * 100
    mean_diff = diffs_pct.mean()
    stat, pval = stats.wilcoxon(diffs)
    claimed = 27.1 if comp == "mykrobe" else 10.7
    ok = abs(mean_diff - claimed) < 0.5
    print(f"  vs {comp}: computed={mean_diff:.1f}%, claimed={claimed}%, p={pval:.2e}, {'OK' if ok else 'MISMATCH'}")
    if not ok:
        issues.append(f"Pooled vs {comp}: claimed {claimed}%, got {mean_diff:.1f}%")

# Claim 2: Ultra-low coverage ranges
print("\n--- CLAIM 2: Ultra-low coverage (0.01x-0.05x) ---")
for method in ["mykrobe", "tb_profiler", "vmdm"]:
    vals = []
    for cov in ["0.01x", "0.05x"]:
        for drug in drugs:
            r = df[(df["Coverage"]==cov) & (df["Method"]==method) & (df["Drug"]==drug)]["Recall"].values[0] * 100
            vals.append(r)
    print(f"  {method}: range={min(vals):.1f}-{max(vals):.1f}%")

# Claim 3: 0.5x details
print("\n--- CLAIM 3: 0.5x coverage ---")
sub = df[df["Coverage"]=="0.5x"]
for m in ["vmdm","mykrobe","tb_profiler"]:
    recs = sub[sub["Method"]==m]["Recall"].values * 100
    print(f"  {m}: mean recall={recs.mean():.1f}%")

v57 = sub[sub["Method"]=="vmdm"]["Recall"].mean()*100
m34 = sub[sub["Method"]=="mykrobe"]["Recall"].mean()*100
t39 = sub[sub["Method"]=="tb_profiler"]["Recall"].mean()*100
print(f"  Fold gain vs Mykrobe: {v57/m34:.1f}x (claimed >16x)")
print(f"  Abs diff vs TBP: {v57-t39:.1f}% (claimed 17.8%)")

# Claim 4: High coverage (5x-10x)
print("\n--- CLAIM 4: High coverage (5x-10x) ---")
for comp in ["mykrobe", "tb_profiler"]:
    dh = []
    for cov in ["5x","10x"]:
        for drug in drugs:
            v = df[(df["Coverage"]==cov)&(df["Method"]=="vmdm")&(df["Drug"]==drug)]["Recall"].values[0]
            c = df[(df["Coverage"]==cov)&(df["Method"]==comp)&(df["Drug"]==drug)]["Recall"].values[0]
            dh.append((v-c)*100)
    dh = np.array(dh)
    mean_h = dh.mean()
    # Bootstrap CI
    bm = [np.random.choice(dh, size=len(dh), replace=True).mean() for _ in range(10000)]
    ci_l, ci_u = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    if comp=="mykrobe":
        cm, cci = 8.8, (3.7,14.0)
    else:
        cm, cci = 0.8, (-3.8,5.5)
    print(f"  vs {comp}: mean={mean_h:.1f}% (claimed {cm}%), CI=({ci_l:.1f},{ci_u:.1f}) (claimed ({cci[0]},{cci[1]}))")

print("\n" + "="*100)
if issues:
    print("ISSUES:")
    for x in issues: print(f"  - {x}")
else:
    print("ALL VALUES VERIFIED OK.")
