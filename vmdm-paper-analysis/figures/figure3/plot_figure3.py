from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

WORK = Path(__file__).resolve().parent
FIG_DIR = WORK / "figs_png"
DATA = WORK / "Fig3_data.tsv"

os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))

FIG_W_IN = 8.2
FIG_H_IN = 7.0
# Manuscript embeds Fig. 3 at 6.3 in (build_manuscript_final.py).
WORD_EMBED_W_IN = 6.3
_FONT_SCALE = FIG_W_IN / WORD_EMBED_W_IN
NATURE_PANEL_PT = 8
NATURE_TEXT_PT = 7

PANEL_FS = NATURE_PANEL_PT * _FONT_SCALE
AXIS_FS = NATURE_TEXT_PT * _FONT_SCALE
TICK_FS = NATURE_TEXT_PT * _FONT_SCALE
LEGEND_FS = NATURE_TEXT_PT * _FONT_SCALE

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

METHOD_ORDER = ["mykrobe", "tb_profiler", "vmdm"]
LEGEND_ORDER = ["vmdm", "tb_profiler", "mykrobe"]  # matches manuscript caption
LINE_LW = 1.35
MARKER_SIZE = 3.8
METHOD_ZORDER = {"mykrobe": 3, "tb_profiler": 4, "vmdm": 5}
METHOD_STYLES = {
    "mykrobe": {"color": "#377eb8", "linestyle": "-", "marker": "o", "label": "Mykrobe"},
    "tb_profiler": {"color": "#4daf4a", "linestyle": "--", "marker": "s", "label": "TB-Profiler"},
    "vmdm": {"color": "#d73027", "linestyle": "-.", "marker": "^", "label": "VMDM"},
}

SIZE_KEYS = ["r1000", "r5000", "r10000", "r50000", "r100000", "r500000", "r1000000"]
COVERAGE_LABELS = ["0.01×", "0.05×", "0.1×", "0.5×", "1×", "5×", "10×"]

METRIC_ROWS = [
    ("Recall", "A", "Recall"),
    ("Precision", "B", "Precision"),
    ("Fscore", "C", "F1 score"),
]

OUT_PDF = WORK / "Fig3-1.pdf"
OUT_PNG = FIG_DIR / "Fig3-1.png"
OUT_TIFF = FIG_DIR / "Fig3-1.tiff"


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


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t")
    df = df[df["Drug"].isin(DRUG_ORDER)].copy()
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    df["Drug"] = pd.Categorical(df["Drug"], categories=DRUG_ORDER, ordered=True)
    return df


def _plot_metric_panel(ax: plt.Axes, df: pd.DataFrame, drug: str, metric: str) -> None:
    drug_df = df[df["Drug"] == drug]
    x = np.arange(len(SIZE_KEYS), dtype=float)

    for method in METHOD_ORDER:
        style = METHOD_STYLES[method]
        method_df = drug_df[drug_df["Method"] == method].set_index("Size")
        y = [float(method_df.loc[key, metric]) if key in method_df.index else np.nan for key in SIZE_KEYS]
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=LINE_LW,
            markersize=MARKER_SIZE,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=METHOD_ZORDER[method],
        )

    ax.set_xlim(-0.25, len(SIZE_KEYS) - 0.75)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(x)
    ax.set_xticklabels(COVERAGE_LABELS, rotation=45, ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", linewidth=0.3, color="#d8d8d8", alpha=0.9, zorder=0)
    _style_axes(ax)


def _add_panel_labels(fig: plt.Figure, axes: list[plt.Axes], labels: tuple[str, str, str]) -> None:
    fig.canvas.draw()
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.010,
            bbox.y1 + 0.006,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="right",
        )


def _bottom_row_center_x(bottom_axes: list[plt.Axes]) -> float:
    left = bottom_axes[0].get_position()
    right = bottom_axes[-1].get_position()
    return 0.5 * (left.x0 + right.x1)


def _add_shared_xlabel(fig: plt.Figure, bottom_axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    y_bottom = bottom_axes[0].get_position().y0
    fig.text(
        _bottom_row_center_x(bottom_axes),
        y_bottom - 0.060,
        "Genome coverage",
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=AXIS_FS,
        fontfamily=_ACTIVE_FONT,
        color="#333333",
    )


def _add_column_titles(fig: plt.Figure, top_axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for ax, drug in zip(top_axes, DRUG_ORDER):
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 + 0.014,
            drug,
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            fontsize=AXIS_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            color=DRUG_COLORS[drug],
        )


def render() -> None:
    print(
        f"Fig3 font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt"
    )

    df = _load_data()

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(
        3,
        4,
        figure=fig,
        left=0.08,
        right=0.99,
        top=0.90,
        bottom=0.19,
        hspace=0.38,
        wspace=0.28,
    )

    row_axes: list[list[plt.Axes]] = []
    for row_idx, (metric, _panel, ylabel) in enumerate(METRIC_ROWS):
        axes_row: list[plt.Axes] = []
        for col_idx, drug in enumerate(DRUG_ORDER):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            _plot_metric_panel(ax, df, drug, metric)
            if col_idx == 0:
                ax.set_ylabel(ylabel, labelpad=4)
            else:
                ax.set_ylabel("")
            if row_idx < len(METRIC_ROWS) - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("")
            axes_row.append(ax)
        row_axes.append(axes_row)

    fig.align_ylabels([ax for row in row_axes for ax in row])
    _add_column_titles(fig, row_axes[0])
    _add_panel_labels(fig, [row[0] for row in row_axes], ("A", "B", "C"))
    _add_shared_xlabel(fig, row_axes[-1])

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[m]["color"],
            linestyle=METHOD_STYLES[m]["linestyle"],
            marker=METHOD_STYLES[m]["marker"],
            linewidth=LINE_LW,
            markersize=MARKER_SIZE + 0.4,
            markerfacecolor=METHOD_STYLES[m]["color"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=METHOD_STYLES[m]["label"],
        )
        for m in LEGEND_ORDER
    ]
    fig.canvas.draw()
    bottom_row = row_axes[-1]
    bottom_y = bottom_row[0].get_position().y0
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(_bottom_row_center_x(bottom_row), bottom_y - 0.095),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(WORK / "Fig3.pdf", format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
