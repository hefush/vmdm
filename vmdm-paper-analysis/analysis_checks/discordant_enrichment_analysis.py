#!/usr/bin/env python3
"""Enrichment analysis for structured genotypic/pDST discordance.

Inputs:
  - evidence.tsv: training-set phenotype and mutation evidence
  - subFig4_matrix_full.tsv: validation discordance matrix

The main test treats pDST-susceptible, row_group=5 discordant cases as the
current structured discordance set. Canonical mutation sets are defined before
looking at the current labels.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__file__).resolve().parent
EVIDENCE_TSV = WORK / "evidence.tsv"
DISCORDANCE_TSV = ROOT / "figures" / "figure5_and_supplementary_figure4" / "subFig4_matrix_full.tsv"
OUT_SUMMARY = WORK / "discordant_enrichment_summary.tsv"
OUT_CASES = WORK / "discordant_cases_with_labels.tsv"
OUT_TOP = WORK / "training_top_resistance_mutations.tsv"

DRUGS = ("Rifampicin", "Isoniazid", "Ethambutol")

CANONICAL = {
    "Rifampicin": {"rpoB_p.Ser450Leu"},
    "Isoniazid": {"katG_p.Ser315Thr", "fabG1_c.-15C>T"},
    "Ethambutol": {
        "embB_p.Met306Val",
        "embB_p.Gly406Ser",
        "embB_p.Gln497Arg",
    },
}

SHORT_TO_TRAINING_LABEL = {
    "S450L": "rpoB_p.Ser450Leu",
    "D435Y": "rpoB_p.Asp435Tyr",
    "S315T": "katG_p.Ser315Thr",
    "-15C>T": "fabG1_c.-15C>T",
    # Tool-specific notation for the inhA/fabG1 promoter variant.
    "inhA-777C>T": "fabG1_c.-15C>T",
    "M306V": "embB_p.Met306Val",
    "G406S": "embB_p.Gly406Ser",
    "Q497R": "embB_p.Gln497Arg",
}

MUT_WITH_VALUE = re.compile(r"^(.*)\(([^()]*)\)$")


def split_evidence(evidence: str) -> list[str]:
    if not evidence or evidence == "NULL":
        return []
    labels = []
    for item in evidence.split(","):
        item = item.strip()
        match = MUT_WITH_VALUE.match(item)
        labels.append(match.group(1) if match else item)
    return labels


def normalize_short_label(label: str) -> str:
    return SHORT_TO_TRAINING_LABEL.get(label, label)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def binomial_upper_tail(n: int, k: int, p: float) -> float:
    return sum(comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test for enrichment in row 1.

    Table layout:
      row 1: current canonical, current non-canonical
      row 2: background canonical, background non-canonical
    """
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    denom = comb(total, row1)
    return sum(comb(col1, x) * comb(total - col1, row1 - x) / denom for x in range(max(a, lo), hi + 1))


def poisson_binomial_upper_tail(probs: list[float], observed: int) -> float:
    distribution = [1.0]
    for p in probs:
        updated = [0.0] * (len(distribution) + 1)
        for i, value in enumerate(distribution):
            updated[i] += value * (1 - p)
            updated[i + 1] += value * p
        distribution = updated
    return sum(distribution[observed:])


def read_training_rows() -> list[dict[str, str]]:
    with EVIDENCE_TSV.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_current_cases() -> list[dict[str, object]]:
    cases = []
    with DISCORDANCE_TSV.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not (
                row["pDST"] == "S"
                and row["discordant"] == "True"
                and row["row_group"] == "5"
            ):
                continue

            labels = []
            for column in (
                "Mykrobe_evidence_short",
                "TBProfiler_evidence_short",
                "VMDM_evidence_short",
            ):
                value = (row[column] or "").strip()
                if not value or value.startswith("mdl"):
                    continue
                for part in re.split(r"[,;]", value):
                    part = part.strip()
                    if part:
                        labels.append(normalize_short_label(part))

            unique_labels = list(dict.fromkeys(labels))
            cases.append(
                {
                    "drug": row["Drug"],
                    "name": row["Name"],
                    "labels": unique_labels,
                    "canonical": any(label in CANONICAL[row["Drug"]] for label in unique_labels),
                    "raw_mykrobe": row["Mykrobe_evidence_short"],
                    "raw_tbprofiler": row["TBProfiler_evidence_short"],
                    "raw_vmdm": row["VMDM_evidence_short"],
                }
            )
    return cases


def write_current_cases(cases: list[dict[str, object]]) -> None:
    with OUT_CASES.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=[
                "Drug",
                "Name",
                "canonical_case",
                "normalized_labels",
                "Mykrobe_evidence_short",
                "TBProfiler_evidence_short",
                "VMDM_evidence_short",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "Drug": case["drug"],
                    "Name": case["name"],
                    "canonical_case": int(bool(case["canonical"])),
                    "normalized_labels": ",".join(case["labels"]),
                    "Mykrobe_evidence_short": case["raw_mykrobe"],
                    "TBProfiler_evidence_short": case["raw_tbprofiler"],
                    "VMDM_evidence_short": case["raw_vmdm"],
                }
            )


def write_top_mutations(rows: list[dict[str, str]]) -> None:
    with OUT_TOP.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=[
                "Drug",
                "rank",
                "mutation_label",
                "resistant_occurrences",
                "fraction_of_resistant_occurrences",
                "canonical_label",
            ],
        )
        writer.writeheader()
        for drug in DRUGS:
            counter = Counter(
                mutation
                for row in rows
                if row["Drug"] == drug
                and row["laboratory_method"] == "Resistant"
                and row["Evidence"] != "NULL"
                for mutation in split_evidence(row["Evidence"])
            )
            total = sum(counter.values())
            for rank, (mutation, count) in enumerate(counter.most_common(20), start=1):
                writer.writerow(
                    {
                        "Drug": drug,
                        "rank": rank,
                        "mutation_label": mutation,
                        "resistant_occurrences": count,
                        "fraction_of_resistant_occurrences": count / total if total else "",
                        "canonical_label": int(mutation in CANONICAL[drug]),
                    }
                )


def main() -> None:
    rows = read_training_rows()
    cases = read_current_cases()
    write_current_cases(cases)
    write_top_mutations(rows)

    summary_rows = []
    uniform_probs = []
    susceptible_probs = []
    total_observed_canonical = 0
    total_current_cases = 0

    for drug in DRUGS:
        current = [case for case in cases if case["drug"] == drug]
        current_n = len(current)
        current_canonical = sum(bool(case["canonical"]) for case in current)

        resistant_evidence = [
            row
            for row in rows
            if row["Drug"] == drug
            and row["laboratory_method"] == "Resistant"
            and row["Evidence"] != "NULL"
        ]
        susceptible_evidence = [
            row
            for row in rows
            if row["Drug"] == drug
            and row["laboratory_method"] == "Susceptible"
            and row["Evidence"] != "NULL"
        ]

        resistant_labels = Counter(
            mutation
            for row in resistant_evidence
            for mutation in split_evidence(row["Evidence"])
        )
        distinct_resistant_labels = len(resistant_labels)

        resistant_canonical = sum(
            any(mutation in CANONICAL[drug] for mutation in split_evidence(row["Evidence"]))
            for row in resistant_evidence
        )
        susceptible_canonical = sum(
            any(mutation in CANONICAL[drug] for mutation in split_evidence(row["Evidence"]))
            for row in susceptible_evidence
        )

        uniform_p = len(CANONICAL[drug]) / distinct_resistant_labels
        susceptible_p = susceptible_canonical / len(susceptible_evidence)

        p_uniform = binomial_upper_tail(current_n, current_canonical, uniform_p)
        p_susceptible_binom = binomial_upper_tail(current_n, current_canonical, susceptible_p)
        p_susceptible_fisher = fisher_greater(
            current_canonical,
            current_n - current_canonical,
            susceptible_canonical,
            len(susceptible_evidence) - susceptible_canonical,
        )

        uniform_probs.extend([uniform_p] * current_n)
        susceptible_probs.extend([susceptible_p] * current_n)
        total_observed_canonical += current_canonical
        total_current_cases += current_n

        summary_rows.append(
            {
                "Drug": drug,
                "current_cases": current_n,
                "current_canonical_cases": current_canonical,
                "current_canonical_fraction": current_canonical / current_n,
                "training_resistant_evidence_rows": len(resistant_evidence),
                "training_resistant_canonical_rows": resistant_canonical,
                "training_resistant_canonical_fraction": resistant_canonical / len(resistant_evidence),
                "training_susceptible_evidence_rows": len(susceptible_evidence),
                "training_susceptible_canonical_rows": susceptible_canonical,
                "training_susceptible_canonical_fraction": susceptible_p,
                "distinct_resistant_mutation_labels": distinct_resistant_labels,
                "predefined_canonical_labels": len(CANONICAL[drug]),
                "p_uniform_distinct_resistant_labels": p_uniform,
                "p_binomial_vs_susceptible_evidence": p_susceptible_binom,
                "p_fisher_vs_susceptible_evidence": p_susceptible_fisher,
            }
        )

    summary_rows.append(
        {
            "Drug": "Combined",
            "current_cases": total_current_cases,
            "current_canonical_cases": total_observed_canonical,
            "current_canonical_fraction": total_observed_canonical / total_current_cases,
            "training_resistant_evidence_rows": "",
            "training_resistant_canonical_rows": "",
            "training_resistant_canonical_fraction": "",
            "training_susceptible_evidence_rows": "",
            "training_susceptible_canonical_rows": "",
            "training_susceptible_canonical_fraction": "",
            "distinct_resistant_mutation_labels": "",
            "predefined_canonical_labels": "",
            "p_uniform_distinct_resistant_labels": poisson_binomial_upper_tail(
                uniform_probs, total_observed_canonical
            ),
            "p_binomial_vs_susceptible_evidence": poisson_binomial_upper_tail(
                susceptible_probs, total_observed_canonical
            ),
            "p_fisher_vs_susceptible_evidence": "",
        }
    )

    with OUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=list(summary_rows[0]),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {OUT_CASES.relative_to(ROOT)}")
    print(f"Wrote {OUT_TOP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
