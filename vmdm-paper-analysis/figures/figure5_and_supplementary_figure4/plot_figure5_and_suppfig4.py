from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

WORK = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORK / ".cache"))

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Patch
import pandas as pd

DATA = WORK / "subFig4_data.tsv"
FIG_DIR = WORK / "figs_png"

_FONT_CANDIDATE_PATHS = (
    Path(os.environ.get("ARIAL_FONT_PATH", "")),
    WORK / "fonts" / "Arial.ttf",
    WORK / "fonts" / "arial.ttf",
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
)


def _font_family() -> list[str]:
    for path in _FONT_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except (OSError, ValueError):
            continue
        name = font_manager.FontProperties(fname=str(path)).get_name()
        return [name, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    return ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": _font_family(),
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)

DRUG_ORDER = ["Rifampicin", "Isoniazid", "Ethambutol"]
METHOD_ORDER = ["Mykrobe", "TB-Profiler", "VMDM"]

PDST_RED = "#c43d4b"
PDST_BLUE = "#4c78a8"
PRED_RED = "#f6d8dd"
PRED_SUS = "#fbfdff"
GRID = "#d8dee9"
SEPARATOR = "#cbd5e1"
DISCORD_EDGE = "#4b5563"
TXT = "#1f2937"


@dataclass
class MatrixRow:
    name: str
    lab: str
    mykrobe_pred: str
    mykrobe_evi: str
    tb_pred: str
    tb_evi: str
    vmdm_pred: str
    vmdm_evi: str
    discordant: bool


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t")
    df["drug"] = df["Drug"].astype(str)
    df["method"] = (
        df["Method"]
        .astype(str)
        .replace({"mykrobe": "Mykrobe", "tbprofiler": "TB-Profiler", "vmdm": "VMDM"})
    )
    df["name"] = df["Name"].astype(str)
    df["lab"] = (
        df["laboratory_method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"resistant": "R", "susceptible": "S"})
    )
    df["pred"] = df["y_pred"].astype(str).map({"1": "R", "0": "S"})
    return df


def abbreviate_evidence(value: object, drug: str) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.upper() == "NULL":
        return ""

    tokens: list[str] = []
    aa3 = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V",
    }

    protein_changes = re.findall(r"_p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", s)
    for ref, pos, alt in protein_changes:
        if ref in aa3 and alt in aa3:
            tokens.append(f"{aa3[ref]}{pos}{aa3[alt]}")

    emba_promoter = re.search(r"embA_c\.\-(\d+)([ACGT])>([ACGT])", s, flags=re.I)
    if emba_promoter:
        pos, ref, alt = emba_promoter.groups()
        tokens.append(f"embA-{pos}{ref}>{alt}")

    inh_promoters = re.findall(r"(inhA|fabG1)_c\.\-(\d+)([ACGT])>([ACGT])", s, flags=re.I)
    for gene, pos, ref, alt in inh_promoters:
        gene_label = "inhA" if gene.lower() == "inha" else "fabG1"
        tokens.append(f"{gene_label}-{pos}{ref}>{alt}")

    canonical_patterns = [
        ("S450L", r"S450L"),
        ("D435Y", r"D435Y"),
        ("H445D", r"H445D"),
        ("H445Y", r"H445Y"),
        ("H445R", r"H445R"),
        ("L430P", r"L430P"),
        ("S315T", r"S315T"),
        ("-15C>T", r"C-15|[- ]15C>T|[- ]15CT"),
        ("M306V", r"M306V"),
        ("M306I", r"M306I"),
        ("G406S", r"G406S"),
        ("Q497R", r"Q497R"),
    ]
    for label, pat in canonical_patterns:
        if re.search(pat, s, flags=re.I):
            tokens.append(label)

    # Keep only the dominant biological signal for the panel/drug.
    priority_by_drug = {
        "Rifampicin": ["S450L", "D435Y", "H445D", "H445Y", "H445R", "L430P"],
        "Isoniazid": ["S315T", "-15C>T"],
        "Ethambutol": ["M306V", "M306I", "G406S", "Q497R"],
    }
    unique_tokens = list(dict.fromkeys(tokens))
    for token in priority_by_drug.get(drug, []):
        if token in unique_tokens:
            return token

    if drug == "Ethambutol":
        for token in unique_tokens:
            if token.startswith("embA-"):
                return token

    if drug == "Isoniazid":
        for token in unique_tokens:
            if token.startswith("inhA-") or token.startswith("fabG1-"):
                return token

    model = re.search(r"model\(([^)]+)\)", s)
    if model and not unique_tokens:
        unique_tokens.append(f"mdl {model.group(1)}")

    if not unique_tokens:
        generic = re.search(r"([A-Z]\d+[A-Z])", s)
        if generic:
            unique_tokens.append(generic.group(1))

    if not unique_tokens and drug == "Isoniazid":
        if re.search(r"inhA|fabG1|katG", s, flags=re.I):
            unique_tokens.append("non-can")

    if not unique_tokens:
        return ""
    return unique_tokens[0][:18]


def row_group(row: MatrixRow) -> int:
    preds = [row.mykrobe_pred, row.tb_pred, row.vmdm_pred]
    if row.lab == "R" and row.vmdm_pred == "R" and row.mykrobe_pred != "R" and row.tb_pred != "R":
        return 0
    if row.lab == "R" and row.vmdm_pred == "R" and (row.mykrobe_pred != "R" or row.tb_pred != "R"):
        return 1
    if row.lab == "R" and all(p == "R" for p in preds):
        return 2
    if row.lab == "R" and all(p == "S" for p in preds):
        return 3
    if row.lab == "S" and row.vmdm_pred == "R" and row.mykrobe_pred == "S" and row.tb_pred == "S":
        return 4
    if row.lab == "S" and row.discordant:
        return 5
    return 6


def build_matrix(df: pd.DataFrame, include_all: bool) -> dict[str, list[MatrixRow]]:
    out: dict[str, list[MatrixRow]] = {}
    for drug in DRUG_ORDER:
        subset = df[df["drug"] == drug].copy()
        rows: list[MatrixRow] = []
        for name in sorted(subset["name"].unique()):
            s = subset[subset["name"] == name]
            by_method = {m: s[s["method"] == m] for m in METHOD_ORDER}
            row = MatrixRow(
                name=name,
                lab=s["lab"].iloc[0],
                mykrobe_pred=by_method["Mykrobe"]["pred"].iloc[0],
                mykrobe_evi=abbreviate_evidence(by_method["Mykrobe"]["Evidence"].iloc[0], drug),
                tb_pred=by_method["TB-Profiler"]["pred"].iloc[0],
                tb_evi=abbreviate_evidence(by_method["TB-Profiler"]["Evidence"].iloc[0], drug),
                vmdm_pred=by_method["VMDM"]["pred"].iloc[0],
                vmdm_evi=abbreviate_evidence(by_method["VMDM"]["Evidence"].iloc[0], drug),
                discordant=False,
            )
            preds = [row.mykrobe_pred, row.tb_pred, row.vmdm_pred]
            row.discordant = len(set(preds + [row.lab])) > 1
            if include_all or row.lab == "R" or row.discordant:
                rows.append(row)
        rows.sort(key=lambda r: (row_group(r), r.name))
        out[drug] = rows
    return out


def export_matrix_tsv(matrix: dict[str, list[MatrixRow]], path: Path) -> None:
    records = []
    for drug, rows in matrix.items():
        for row in rows:
            records.append(
                {
                    "Drug": drug,
                    "Name": row.name,
                    "pDST": row.lab,
                    "Mykrobe_pred": row.mykrobe_pred,
                    "Mykrobe_evidence_short": row.mykrobe_evi,
                    "TBProfiler_pred": row.tb_pred,
                    "TBProfiler_evidence_short": row.tb_evi,
                    "VMDM_pred": row.vmdm_pred,
                    "VMDM_evidence_short": row.vmdm_evi,
                    "discordant": row.discordant,
                    "row_group": row_group(row),
                }
            )
    pd.DataFrame.from_records(records).to_csv(path, sep="\t", index=False)


def export_stable4(df: pd.DataFrame, path_tsv: Path, path_xlsx: Path) -> None:
    """Complete supplementary table with raw and simplified evidence."""
    wide_pred = (
        df.pivot_table(index=["name", "drug", "lab"], columns="method", values="pred", aggfunc="first")
        .reset_index()
    )
    wide_evi = (
        df.pivot_table(index=["name", "drug", "lab"], columns="method", values="Evidence", aggfunc="first")
        .reset_index()
    )
    merged = wide_pred.merge(wide_evi, on=["name", "drug", "lab"], suffixes=("_pred", "_evidence_raw"))
    records = []
    for _, r in merged.iterrows():
        row = {
            "Drug": r["drug"],
            "Name": r["name"],
            "pDST": r["lab"],
        }
        for method in METHOD_ORDER:
            row[f"{method}_pred"] = r.get(f"{method}_pred", "")
            row[f"{method}_evidence_raw"] = r.get(f"{method}_evidence_raw", "")
            row[f"{method}_evidence_short"] = abbreviate_evidence(r.get(f"{method}_evidence_raw", ""), r["drug"])
        records.append(row)
    out = pd.DataFrame.from_records(records).sort_values(["Drug", "Name"]).reset_index(drop=True)
    out.to_csv(path_tsv, sep="\t", index=False)
    try:
        out.to_excel(path_xlsx, index=False)
    except ModuleNotFoundError as exc:
        if exc.name not in {"openpyxl", "xlsxwriter"}:
            raise
        print(f"Skipped Excel export because {exc.name} is not installed: {path_xlsx}")


def draw_panel(ax, drug: str, rows: list[MatrixRow]) -> None:
    ax.set_axis_off()

    sample_right = 1.40
    strip_x = 1.56
    cell_xs = [1.90, 3.48, 5.06]
    cell_w = 1.35
    strip_w = 0.11
    row_h = 0.82
    top = len(rows) + 1.48
    header_y = top - 0.52

    panel_center_x = (0.0 + 6.55) / 2
    ax.text(
        panel_center_x,
        top + 0.08,
        f"{drug} (n={len(rows)})",
        fontsize=11.6,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=TXT,
    )
    ax.text(0.44, header_y, "Sample", fontsize=8.4, fontweight="bold", ha="left", va="bottom", color=TXT)
    ax.text(
        strip_x + strip_w / 2,
        header_y,
        "pDST",
        fontsize=8.2,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=TXT,
    )
    for x, label in zip(cell_xs, METHOD_ORDER):
        ax.text(
            x + cell_w / 2,
            header_y,
            label,
            fontsize=8.4,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=TXT,
        )
    ax.plot([0.0, 6.55], [header_y - 0.17, header_y - 0.17], color=GRID, lw=0.8, solid_capstyle="butt")

    for i, row in enumerate(rows):
        y = len(rows) - i
        ax.text(sample_right, y, row.name, fontsize=7.6, ha="right", va="center", color=TXT)

        ax.add_patch(
            Rectangle(
                (strip_x, y - row_h / 2),
                strip_w,
                row_h,
                facecolor=PDST_RED if row.lab == "R" else PDST_BLUE,
                edgecolor="none",
            )
        )

        cell_specs = [
            (cell_xs[0], row.mykrobe_pred, row.mykrobe_evi),
            (cell_xs[1], row.tb_pred, row.tb_evi),
            (cell_xs[2], row.vmdm_pred, row.vmdm_evi),
        ]
        for x, pred, text in cell_specs:
            discordant_call = pred != row.lab
            ax.add_patch(
                Rectangle(
                    (x, y - row_h / 2),
                    cell_w,
                    row_h,
                    facecolor=PRED_RED if pred == "R" else PRED_SUS,
                    edgecolor=DISCORD_EDGE if discordant_call else GRID,
                    linewidth=0.75 if discordant_call else 0.55,
                )
            )
            if text:
                ax.text(
                    x + cell_w / 2,
                    y,
                    text,
                    fontsize=7.2,
                    ha="center",
                    va="center",
                    color=TXT,
                )
        if i < len(rows) - 1 and row_group(row) != row_group(rows[i + 1]):
            ax.plot([0.36, 6.48], [y - 0.5, y - 0.5], color=SEPARATOR, lw=0.65, solid_capstyle="butt")

    ax.set_xlim(-0.05, 6.6)
    ax.set_ylim(0.2, len(rows) + 1.9)


def render(matrix: dict[str, list[MatrixRow]], out_png: Path, out_pdf: Path, title_suffix: str) -> None:
    max_rows = max(len(v) for v in matrix.values())
    fig_h = max(5.2, 1.45 + 0.34 * max_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, fig_h))
    fig.patch.set_facecolor("white")

    for ax, drug in zip(axes, DRUG_ORDER):
        draw_panel(ax, drug, matrix[drug])

    handles = [
        Patch(facecolor=PDST_RED, edgecolor="none", label="pDST resistant"),
        Patch(facecolor=PDST_BLUE, edgecolor="none", label="pDST susceptible"),
        Patch(facecolor=PRED_RED, edgecolor=GRID, label="Prediction resistant"),
        Patch(facecolor=PRED_SUS, edgecolor=GRID, label="Prediction susceptible"),
        Patch(facecolor=PRED_SUS, edgecolor=DISCORD_EDGE, linewidth=0.75, label="Prediction differs from pDST"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.3,
        handlelength=1.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=[0.01, 0.06, 0.99, 0.98], w_pad=1.45)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_tiff = out_png.with_suffix(".tiff")
    fig.savefig(out_png, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tiff, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_data()
    main_matrix = build_matrix(df, include_all=False)
    full_matrix = build_matrix(df, include_all=True)

    export_matrix_tsv(main_matrix, WORK / "Fig5_matrix_main.tsv")
    export_matrix_tsv(full_matrix, WORK / "subFig4_matrix_full.tsv")

    render(
        main_matrix,
        FIG_DIR / "Fig5-1.png",
        WORK / "Fig5-1.pdf",
        "main figure: direct sputum validation",
    )
    render(
        full_matrix,
        FIG_DIR / "subFig4-1.png",
        WORK / "subFig4-1.pdf",
        "supplementary full matrix",
    )
    export_stable4(
        df,
        WORK / "sTable4_subFig4_direct_sputum_predictions.tsv",
        WORK / "sTable4_subFig4_direct_sputum_predictions.xlsx",
    )


if __name__ == "__main__":
    main()
