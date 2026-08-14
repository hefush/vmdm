#!/usr/bin/env python3
"""Verify the final Table 1 workbook against performance.tsv."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


PERFORMANCE_TSV = "performance.tsv"
TABLE_XLSX = "table1.xlsx"
TOLERANCE = 0.05

DRUGS = [
    ("Rifampicin", "Rifampicin recall"),
    ("Isoniazid", "Isoniazid recall"),
    ("Ethambutol", "Ethambutol recall"),
    ("Pyrazinamide", "Pyrazinamide recall"),
]
GROUPS = [
    ("0.01–0.05×", ["0.01x", "0.05x"]),
    ("0.5×", ["0.5x"]),
    ("5–10×", ["5x", "10x"]),
]
METHODS = [
    ("mykrobe", "Mykrobe"),
    ("tb_profiler", "TB-Profiler"),
    ("vmdm", "VMDM"),
]


def recompute_table() -> pd.DataFrame:
    df = pd.read_csv(PERFORMANCE_TSV, sep="\t")
    df["Coverage"] = df["Coverage"].astype(str).str.lower()

    rows = []
    for coverage_label, coverages in GROUPS:
        for method, method_label in METHODS:
            row = {"Coverage": coverage_label, "Tool": method_label}
            drug_values = []
            for drug, column in DRUGS:
                values = df.loc[
                    df["Coverage"].isin(coverages)
                    & (df["Method"] == method)
                    & (df["Drug"] == drug),
                    "Recall",
                ].to_numpy()
                if len(values) != len(coverages):
                    raise ValueError(f"Missing rows for {coverage_label}, {method}, {drug}")
                value = float(np.mean(values) * 100)
                row[column] = round(value, 1)
                drug_values.append(value)
            row["Mean recall"] = round(float(np.mean(drug_values)), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def load_workbook_table() -> pd.DataFrame:
    table = pd.read_excel(TABLE_XLSX, sheet_name="Table1")
    expected_columns = ["Coverage", "Tool", *(column for _, column in DRUGS), "Mean recall"]
    missing = [column for column in expected_columns if column not in table.columns]
    if missing:
        raise ValueError(f"{TABLE_XLSX} is missing columns: {', '.join(missing)}")
    return table[expected_columns].copy()


def main() -> None:
    expected = recompute_table()
    observed = load_workbook_table()

    print("=" * 100)
    print("FINAL TABLE 1 VERIFICATION")
    print("=" * 100)
    print("\nRecomputed from performance.tsv:")
    print(expected.to_string(index=False))

    all_ok = True
    if len(observed) != len(expected):
        print(f"\nRow count mismatch: workbook={len(observed)}, recomputed={len(expected)}")
        all_ok = False

    comparable_rows = min(len(observed), len(expected))
    numeric_columns = [column for _, column in DRUGS] + ["Mean recall"]
    for i in range(comparable_rows):
        for column in ("Coverage", "Tool"):
            if str(observed.loc[i, column]) != str(expected.loc[i, column]):
                print(
                    f"Row {i + 1} {column} mismatch: "
                    f"workbook={observed.loc[i, column]!r}, recomputed={expected.loc[i, column]!r}"
                )
                all_ok = False
        for column in numeric_columns:
            obs = float(observed.loc[i, column])
            exp = float(expected.loc[i, column])
            if abs(obs - exp) > TOLERANCE:
                print(
                    f"Row {i + 1} {column} mismatch: "
                    f"workbook={obs:.1f}, recomputed={exp:.1f}"
                )
                all_ok = False

    print("\n" + "=" * 100)
    if all_ok:
        print(f"ALL TABLE 1 VALUES MATCH {TABLE_XLSX} within +/-{TOLERANCE:.2f} percentage points.")
    else:
        print("TABLE 1 VERIFICATION FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
