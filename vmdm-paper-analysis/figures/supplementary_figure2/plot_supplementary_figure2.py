from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import auc, roc_curve

WORK = Path(__file__).resolve().parent
FIG_DIR = WORK / "figs_png"
DATA = WORK / "subFig2_data.tsv"

os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))

FIG_W_IN = 7.2
FIG_H_IN = 3.2
WORD_EMBED_W_IN = 6.0
_FONT_SCALE = FIG_W_IN / WORD_EMBED_W_IN
NATURE_PANEL_PT = 8
NATURE_TEXT_PT = 7

PANEL_FS = NATURE_PANEL_PT * _FONT_SCALE
AXIS_FS = NATURE_TEXT_PT * _FONT_SCALE
TICK_FS = NATURE_TEXT_PT * _FONT_SCALE
LEGEND_FS = NATURE_TEXT_PT * _FONT_SCALE
ANNOT_FS = 6.5 * _FONT_SCALE
_FONT_CANDIDATE_PATHS = (
    Path(os.environ.get("ARIAL_FONT_PATH", "")),
    WORK / "fonts" / "Arial.ttf",
    WORK / "fonts" / "arial.ttf",
    WORK / "fonts" / "Helvetica.ttf",
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
)

DRUG_ORDER = ["Rifampicin", "Isoniazid", "Pyrazinamide", "Ethambutol"]
DRUG_COLORS = {
    "Rifampicin": "#d73027",
    "Isoniazid": "#377eb8",
    "Pyrazinamide": "#984EA3",
    "Ethambutol": "#4daf4a",
}
SIZE_KEYS = ("t1000", "t10000")
FEATURE_LABELS = ("1,000", "10,000")
LINE_STYLES = ("-", "--")

OUT_PDF = WORK / "subFig2.pdf"
OUT_PNG = FIG_DIR / "subFig2-1.png"
OUT_TIFF = FIG_DIR / "subFig2-1.tiff"


def _setup_publication_font() -> tuple[str, str]:
    from matplotlib import font_manager

    for path in _FONT_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except (OSError, ValueError):
            continue
        name = font_manager.FontProperties(fname=str(path)).get_name()
        if "arial" in path.name.lower():
            note = "bundled Arial"
        elif "nimbus" in name.lower():
            note = "Nimbus Sans (Helvetica-compatible; Arial not installed)"
        else:
            note = f"{name} (Arial not installed)"
        mpl.rcParams["font.sans-serif"] = [name, "Arial", "Helvetica", "Nimbus Sans", "Liberation Sans"]
        mpl.rcParams["font.family"] = "sans-serif"
        return name, note

    installed = {f.name.lower() for f in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"):
        if any(candidate.lower() in name for name in installed):
            mpl.rcParams["font.sans-serif"] = [candidate, "Arial", "Helvetica", "Nimbus Sans", "Liberation Sans"]
            note = (
                "Nimbus Sans (Helvetica-compatible; Arial not installed)"
                if candidate == "Nimbus Sans"
                else f"{candidate} (Arial not installed)"
            )
            return candidate, note
    return "DejaVu Sans", "DejaVu Sans fallback"


_ACTIVE_FONT, _FONT_NOTE = _setup_publication_font()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": AXIS_FS,
        "axes.labelsize": AXIS_FS,
        "axes.titlesize": AXIS_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#333333",
        "axes.labelweight": "normal",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.6)


def _plot_roc_panel(ax: plt.Axes, df: pd.DataFrame) -> list[tuple[str, float, float]]:
    auc_rows: list[tuple[str, float, float]] = []
    for drug in DRUG_ORDER:
        color = DRUG_COLORS[drug]
        aucs: list[float] = []
        for size_key, linestyle in zip(SIZE_KEYS, LINE_STYLES):
            subset = df[(df["Drug"] == drug) & (df["Size"] == size_key)]
            if subset.empty or subset["Phen"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(subset["Phen"], subset["y_pred_prob"])
            roc_auc = float(auc(fpr, tpr))
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=1.6, zorder=3)
        if len(aucs) == 2:
            auc_rows.append((drug, aucs[0], aucs[1]))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#bbbbbb", linewidth=0.75, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate", labelpad=4)
    ax.set_ylabel("True positive rate", labelpad=4)
    _style_axes(ax)
    return auc_rows


def _add_auc_table(ax: plt.Axes, auc_rows: list[tuple[str, float, float]]) -> mpl.table.Table | None:
    if not auc_rows:
        return None

    row_labels = [drug for drug, _, _ in auc_rows]
    cell_text = [[f"{a1:.3f}", f"{a2:.3f}"] for _, a1, a2 in auc_rows]
    nrows = len(auc_rows)
    table_h = 0.050 * (nrows + 1.15)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=["", ""],
        cellLoc="center",
        rowLoc="left",
        loc="lower right",
        bbox=[0.56, 0.05, 0.43, table_h],
        colWidths=[0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(ANNOT_FS)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#e8e8e8")
        cell.set_text_props(fontfamily=_ACTIVE_FONT, fontsize=ANNOT_FS, color="#333333")

        if row == 0:
            cell.get_text().set_text("")
            cell.set_facecolor("#f6f6f6")
        elif col == -1:
            drug = row_labels[row - 1]
            cell.set_facecolor(mpl.colors.to_rgba(DRUG_COLORS[drug], 0.08))
            cell.get_text().set_color(DRUG_COLORS[drug])
        else:
            cell.set_facecolor("#ffffff")

        if row == nrows and col in (-1, 0, 1):
            cell.set_edgecolor("#d8d8d8")
            cell.set_linewidth(0.45)

    return table


def _finalize_auc_table(fig: plt.Figure, table: mpl.table.Table) -> None:
    """Draw AUC title and linestyle headers after the full layout is fixed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    header_cell = table[(0, 0)].get_window_extent(renderer).transformed(to_fig)
    row_label_cell = table[(1, -1)].get_window_extent(renderer).transformed(to_fig)
    fig.text(
        0.5 * (row_label_cell.x0 + row_label_cell.x1),
        0.5 * (header_cell.y0 + header_cell.y1),
        "AUC",
        transform=fig.transFigure,
        ha="center",
        va="center",
        fontsize=ANNOT_FS,
        fontweight="bold",
        fontfamily=_ACTIVE_FONT,
        color="#333333",
        zorder=101,
    )

    for col, linestyle in ((0, "-"), (1, "--")):
        bbox = table[(0, col)].get_window_extent(renderer).transformed(to_fig)
        pad = 0.20 * bbox.width
        fig.add_artist(
            Line2D(
                [bbox.x0 + pad, bbox.x1 - pad],
                [0.5 * (bbox.y0 + bbox.y1)] * 2,
                transform=fig.transFigure,
                color="#333333",
                linewidth=1.4,
                linestyle=linestyle,
                solid_capstyle="round",
                clip_on=False,
                zorder=101,
            )
        )


def _plot_time_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    group_centers = np.arange(len(FEATURE_LABELS), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(DRUG_ORDER))
    box_w = 0.13

    for i, size_key in enumerate(SIZE_KEYS):
        for j, drug in enumerate(DRUG_ORDER):
            vals = df[(df["Size"] == size_key) & (df["Drug"] == drug)]["time"].to_numpy()
            if vals.size == 0:
                continue
            pos = group_centers[i] + offsets[j]
            ax.boxplot(
                vals,
                positions=[pos],
                widths=box_w,
                patch_artist=True,
                showfliers=True,
                boxprops={
                    "facecolor": DRUG_COLORS[drug],
                    "alpha": 0.58,
                    "linewidth": 0.75,
                    "edgecolor": "#333333",
                },
                medianprops={"color": "#1a1a1a", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.75, "color": "#444444"},
                capprops={"linewidth": 0.75, "color": "#444444"},
                flierprops={
                    "marker": "o",
                    "markersize": 1.8,
                    "alpha": 0.18,
                    "markerfacecolor": DRUG_COLORS[drug],
                    "markeredgecolor": "none",
                },
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(FEATURE_LABELS)
    ax.set_xlabel("PRESS features retained", labelpad=4)
    ax.set_ylabel("Analysis time (s)", labelpad=4)
    ax.set_yscale("log")
    ymin = max(df["time"].min() * 0.85, 1.0)
    ax.set_ylim(ymin, df["time"].max() * 1.35)
    _style_axes(ax)


def _add_panel_labels(fig: plt.Figure, axes: list[plt.Axes], labels: tuple[str, str]) -> None:
    fig.canvas.draw()
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.014,
            bbox.y1 + 0.010,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="right",
        )


def render() -> None:
    print(
        f"subFig2 font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt"
    )

    df = pd.read_csv(DATA, sep="\t")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, left=0.11, right=0.98, top=0.90, bottom=0.30, wspace=0.38)

    ax_roc = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])

    auc_rows = _plot_roc_panel(ax_roc, df)
    _plot_time_panel(ax_time, df)
    auc_table = _add_auc_table(ax_roc, auc_rows)
    fig.align_ylabels([ax_roc, ax_time])
    _add_panel_labels(fig, [ax_roc, ax_time], ("A", "B"))

    drug_handles = [Line2D([0], [0], color=DRUG_COLORS[d], linewidth=2.0, label=d) for d in DRUG_ORDER]
    style_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle="-", label="1,000 features"),
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle="--", label="10,000 features"),
    ]
    fig.legend(
        handles=drug_handles + style_handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.008),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    if auc_table is not None:
        _finalize_auc_table(fig, auc_table)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(WORK / "subFig2.png", dpi=500, facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
