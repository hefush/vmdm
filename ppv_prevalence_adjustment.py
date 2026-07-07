#!/usr/bin/env python3
"""Calculate prevalence-adjusted PPV thresholds for VMDM.

VMDM's model scripts select a probability threshold by requiring a minimum
positive predictive value (PPV) on the training split. When the target
population has a different resistance prevalence from the training data, the
PPV requirement can be adjusted by assuming the threshold's positive likelihood
ratio is stable and only prevalence changes.

The script can either:
  * calculate a single manual adjustment from three proportions, or
  * read drug-specific training prevalence from MTBdb/*.data.xls(.gz).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


DEFAULT_DRUGS = ("Rifampicin", "Isoniazid", "Pyrazinamide", "Ethambutol")
DATA_SUFFIXES = (".data.xls.gz", ".data.tsv.gz", ".data.xls", ".data.tsv")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINDB = SCRIPT_DIR / "MTBdb"


@dataclass(frozen=True)
class TrainingPrevalence:
    drug: str
    positives: int
    total: int

    @property
    def prevalence(self) -> float:
        return self.positives / self.total


def _check_prevalence(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, excluding endpoints; got {value}")


def _check_ppv(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1; got {value}")


def required_training_ppv(
    training_prevalence: float,
    target_prevalence: float,
    target_ppv: float,
) -> float:
    """Return the training PPV required to achieve ``target_ppv``.

    Parameters are proportions, not percentages. For example, use 0.8 for 80%.
    """

    _check_prevalence("training_prevalence", training_prevalence)
    _check_prevalence("target_prevalence", target_prevalence)
    _check_ppv("target_ppv", target_ppv)

    numerator = target_ppv * training_prevalence * (1.0 - target_prevalence)
    denominator = numerator + (1.0 - target_ppv) * target_prevalence * (
        1.0 - training_prevalence
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_float(value: str, option_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be numeric; got {value!r}") from exc
    return parsed


def _parse_drug_list(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_DRUGS)
    drugs = [item.strip() for item in value.split(",") if item.strip()]
    if not drugs:
        raise ValueError("--drugs must contain at least one drug name")
    return drugs


def _parse_value_or_map(value: str, option_name: str) -> float | dict[str, float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option_name} must not be empty")

    has_map_items = any("=" in item for item in items)
    if not has_map_items:
        if len(items) != 1:
            raise ValueError(
                f"{option_name} accepts one shared value or drug=value pairs; got {value!r}"
            )
        return _parse_float(items[0], option_name)

    parsed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"{option_name} mixes bare values and drug=value entries: {value!r}"
            )
        drug, raw_value = item.split("=", 1)
        drug = drug.strip()
        if not drug:
            raise ValueError(f"{option_name} contains an empty drug name: {item!r}")
        if drug in parsed:
            raise ValueError(f"{option_name} contains {drug!r} more than once")
        parsed[drug] = _parse_float(raw_value.strip(), f"{option_name} for {drug}")
    return parsed


def _value_for_drug(value: float | dict[str, float], option_name: str, drug: str | None) -> float:
    if isinstance(value, float):
        return value
    if drug is None:
        raise ValueError(f"{option_name} cannot use drug=value pairs in manual mode")
    try:
        return value[drug]
    except KeyError as exc:
        raise ValueError(f"{option_name} is missing a value for {drug}") from exc


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _find_training_data_file(traindb: Path, drug: str) -> Path:
    for suffix in DATA_SUFFIXES:
        path = traindb / f"{drug}{suffix}"
        if path.exists():
            return path
    suffixes = ", ".join(f"{drug}{suffix}" for suffix in DATA_SUFFIXES)
    raise FileNotFoundError(f"Could not find training data for {drug}; tried {suffixes}")


def _label_to_binary(raw_value: str, path: Path, line_number: int) -> int:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"1", "1.0", "true", "t", "r", "resistant", "resistance"}:
        return 1
    if lowered in {"0", "0.0", "false", "f", "s", "susceptible", "sensitive"}:
        return 0
    raise ValueError(
        f"Unsupported label {raw_value!r} in {path} line {line_number}; expected 0/1"
    )


def read_training_prevalence(
    traindb: Path,
    drug: str,
    label_column: str = "Drug",
) -> TrainingPrevalence:
    """Read positive-label prevalence for one drug from the training database."""

    data_file = _find_training_data_file(traindb, drug)
    positives = 0
    total = 0
    with _open_text(data_file) as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"{data_file} is empty")
        columns = header_line.rstrip("\r\n").split("\t")
        if label_column not in columns:
            column_names = ", ".join(columns)
            raise ValueError(
                f"{data_file} does not contain label column {label_column!r}; "
                f"available columns: {column_names}"
            )
        label_index = columns.index(label_column)

        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split("\t", label_index + 1)
            if len(fields) <= label_index:
                raise ValueError(
                    f"{data_file} line {line_number} has fewer columns than the header"
                )
            total += 1
            positives += _label_to_binary(fields[label_index], data_file, line_number)

    if total == 0:
        raise ValueError(f"{data_file} does not contain any training rows")
    _check_prevalence(f"{drug} training prevalence", positives / total)
    return TrainingPrevalence(drug=drug, positives=positives, total=total)


def _format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _write_rows(rows: list[dict[str, str]], output: TextIO, delimiter: str) -> None:
    fieldnames = [
        "drug",
        "training_positive",
        "training_total",
        "training_prevalence",
        "target_prevalence",
        "target_ppv",
        "required_min_ppv",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def _build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    target_prevalence = _parse_value_or_map(args.target_prevalence, "--target-prevalence")
    target_ppv = _parse_value_or_map(args.target_ppv, "--target-ppv")

    if args.training_prevalence is not None:
        training_prevalence = _parse_float(args.training_prevalence, "--training-prevalence")
        target_prev = _value_for_drug(target_prevalence, "--target-prevalence", None)
        target_ppv_value = _value_for_drug(target_ppv, "--target-ppv", None)
        required = required_training_ppv(training_prevalence, target_prev, target_ppv_value)
        return [
            {
                "drug": "manual",
                "training_positive": "",
                "training_total": "",
                "training_prevalence": _format_float(training_prevalence, args.digits),
                "target_prevalence": _format_float(target_prev, args.digits),
                "target_ppv": _format_float(target_ppv_value, args.digits),
                "required_min_ppv": _format_float(required, args.digits),
            }
        ]

    traindb = Path(args.traindb)
    drugs = _parse_drug_list(args.drugs)
    rows: list[dict[str, str]] = []
    for drug in drugs:
        prevalence = read_training_prevalence(traindb, drug, args.label_column)
        target_prev = _value_for_drug(target_prevalence, "--target-prevalence", drug)
        target_ppv_value = _value_for_drug(target_ppv, "--target-ppv", drug)
        required = required_training_ppv(
            prevalence.prevalence,
            target_prev,
            target_ppv_value,
        )
        rows.append(
            {
                "drug": drug,
                "training_positive": str(prevalence.positives),
                "training_total": str(prevalence.total),
                "training_prevalence": _format_float(prevalence.prevalence, args.digits),
                "target_prevalence": _format_float(target_prev, args.digits),
                "target_ppv": _format_float(target_ppv_value, args.digits),
                "required_min_ppv": _format_float(required, args.digits),
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate VMDM --min_ppv values after adjusting for different "
            "training and target prevalences."
        ),
        epilog=(
            "Examples:\n"
            "  ppv_prevalence_adjustment.py --training-prevalence 0.30 "
            "--target-prevalence 0.10 --target-ppv 0.90\n"
            "  ppv_prevalence_adjustment.py --traindb MTBdb --target-prevalence 0.10 "
            "--target-ppv 0.90\n"
            "  ppv_prevalence_adjustment.py --traindb MTBdb --drugs Rifampicin,Isoniazid "
            "--target-prevalence Rifampicin=0.10,Isoniazid=0.08 --target-ppv 0.90"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "legacy_values",
        nargs="*",
        metavar="legacy",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--training-prevalence",
        help="Manual training prevalence. If omitted, read drug-specific values from --traindb.",
    )
    parser.add_argument(
        "--target-prevalence",
        help="Target validation/population prevalence, or comma-separated drug=value entries.",
    )
    parser.add_argument(
        "--target-ppv",
        help="Target validation/population PPV, or comma-separated drug=value entries.",
    )
    parser.add_argument(
        "--traindb",
        default=str(DEFAULT_TRAINDB),
        help=(
            "Training database directory containing <Drug>.data.xls(.gz) files. "
            "Default: MTBdb next to this script"
        ),
    )
    parser.add_argument(
        "--drugs",
        help=(
            "Comma-separated drugs to read from --traindb. Default: "
            + ",".join(DEFAULT_DRUGS)
        ),
    )
    parser.add_argument(
        "--label-column",
        default="Drug",
        help="Label column in training data. Default: Drug",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "csv"),
        default="tsv",
        help="Output table format. Default: tsv",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Number of decimal places for proportions. Default: 6",
    )
    parser.add_argument(
        "--output",
        help="Optional output file. Default: stdout",
    )
    return parser


def _normalize_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.legacy_values:
        return args
    if len(args.legacy_values) != 3:
        raise ValueError(
            "Legacy positional mode expects exactly: "
            "training_prevalence target_prevalence target_ppv"
        )
    forbidden = {
        "--training-prevalence": args.training_prevalence,
        "--target-prevalence": args.target_prevalence,
        "--target-ppv": args.target_ppv,
        "--drugs": args.drugs,
    }
    used = [name for name, value in forbidden.items() if value is not None]
    if used:
        raise ValueError(f"Do not combine legacy positional values with {', '.join(used)}")
    args.training_prevalence, args.target_prevalence, args.target_ppv = args.legacy_values
    return args


def main() -> int:
    parser = _build_parser()
    try:
        args = _normalize_legacy_args(parser.parse_args())
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    if args.digits < 0:
        parser.error("--digits must be non-negative")
    if args.target_prevalence is None:
        parser.error("--target-prevalence is required")
    if args.target_ppv is None:
        parser.error("--target-ppv is required")

    try:
        rows = _build_rows(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    delimiter = "\t" if args.format == "tsv" else ","
    if args.output:
        with Path(args.output).open("w", newline="") as handle:
            _write_rows(rows, handle, delimiter)
    else:
        _write_rows(rows, sys.stdout, delimiter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
