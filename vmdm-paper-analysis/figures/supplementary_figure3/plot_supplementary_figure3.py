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
DATA = WORK / "subFig3_data.tsv"

os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))

FIG_W_IN = 7.2
FIG_H_IN = 5.8
WORD_EMBED_W_IN = 6.0
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

SIZE_MAP = {
    "r1000": "0.01x",
    "r5000": "0.05x",
    "r10000": "0.1x",
    "r50000": "0.5x",
    "r100000": "1x",
    "r500000": "5x",
    "r1000000": "10x",
}
ORDER_SIZES = ["0.01x", "0.05x", "0.1x", "0.5x", "1x", "5x", "10x"]
SIZE_LABELS = ["0.01×", "0.05×", "0.1×", "0.5×", "1×", "5×", "10×"]

THEORY_RECALL = {
    "0.01x": 0.009950166,
    "0.05x": 0.04877058,
    "0.1x": 0.09516258,
    "0.5x": 0.3934693,
    "1x": 0.6321206,
}
PRECISION_TARGETS = {
    "Rifampicin": 0.95,
    "Isoniazid": 0.97,
    "Ethambutol": 0.75,
    "Pyrazinamide": 0.85,
}

OUT_PDF = WORK / "subFig3.pdf"
OUT_PNG = FIG_DIR / "subFig3-1.png"
OUT_TIFF = FIG_DIR / "subFig3-1.tiff"

_BOX_WIDTH = 0.54
# Match subFig2_panel B boxplot styling (DRUG_COLORS + alpha/edge/whisker).
_BOX_STYLE = {
    "alpha": 0.58,
    "linewidth": 0.75,
    "edgecolor": "#333333",
    "median_color": "#1a1a1a",
    "median_lw": 1.0,
    "whisker_color": "#444444",
    "whisker_lw": 0.75,
}


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
    df = df.dropna(subset=["Recall", "Precision"]).copy()
    df["Size"] = df["Size"].map(SIZE_MAP)
    df["Size"] = pd.Categorical(df["Size"], categories=ORDER_SIZES, ordered=True)
    df["Drug"] = pd.Categorical(df["Drug"], categories=DRUG_ORDER, ordered=True)
    return df


def _plot_drug_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    drug: str,
    metric: str,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    ylabel: str,
    add_theory: bool = False,
    add_precision_target: bool = False,
) -> None:
    color = DRUG_COLORS[drug]
    positions = np.arange(len(ORDER_SIZES), dtype=float)

    for i, size in enumerate(ORDER_SIZES):
        vals = subset.loc[subset["Size"] == size, metric].to_numpy(dtype=float)
        if vals.size == 0:
            continue

        ax.boxplot(
            vals,
            positions=[positions[i]],
            widths=_BOX_WIDTH,
            patch_artist=True,
            showfliers=True,
            zorder=3,
            boxprops={
                "facecolor": color,
                "alpha": _BOX_STYLE["alpha"],
                "linewidth": _BOX_STYLE["linewidth"],
                "edgecolor": _BOX_STYLE["edgecolor"],
            },
            medianprops={"color": _BOX_STYLE["median_color"], "linewidth": _BOX_STYLE["median_lw"]},
            whiskerprops={"linewidth": _BOX_STYLE["whisker_lw"], "color": _BOX_STYLE["whisker_color"]},
            capprops={"linewidth": _BOX_STYLE["whisker_lw"], "color": _BOX_STYLE["whisker_color"]},
            flierprops={
                "marker": "o",
                "markersize": 1.8,
                "alpha": 0.18,
                "markerfacecolor": color,
                "markeredgecolor": "none",
            },
        )

    if add_theory:
        theory_x = [i for i, size in enumerate(ORDER_SIZES) if size in THEORY_RECALL]
        theory_y = [THEORY_RECALL[ORDER_SIZES[i]] for i in theory_x]
        ax.scatter(
            theory_x,
            theory_y,
            color="#c41e1e",
            s=38,
            zorder=6,
            edgecolors="white",
            linewidths=0.55,
        )

    if add_precision_target:
        target = PRECISION_TARGETS[drug]
        ax.axhline(target, color="#d62728", linestyle="--", linewidth=0.9, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(SIZE_LABELS, rotation=45, ha="right")
    ax.set_xlim(-0.55, len(ORDER_SIZES) - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    if show_xlabel:
        ax.set_xlabel("Genome coverage", labelpad=4)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    else:
        ax.set_ylabel("")
    _style_axes(ax)


def _add_panel_labels(fig: plt.Figure, axes: list[plt.Axes], labels: tuple[str, str]) -> None:
    fig.canvas.draw()
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.010,
            bbox.y1 + 0.008,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="right",
        )


def _add_column_titles(fig: plt.Figure, top_axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    for ax, drug in zip(top_axes, DRUG_ORDER):
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 + 0.018,
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
        f"subFig3 font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt"
    )

    df = _load_data()

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(2, 4, figure=fig, left=0.07, right=0.99, top=0.86, bottom=0.24, hspace=0.50, wspace=0.30)

    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []
    for j, drug in enumerate(DRUG_ORDER):
        subset = df[df["Drug"] == drug]
        ax_top = fig.add_subplot(gs[0, j])
        ax_bottom = fig.add_subplot(gs[1, j])
        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)

        _plot_drug_panel(
            ax_top,
            subset,
            drug,
            "Recall",
            show_xlabel=False,
            show_ylabel=j == 0,
            ylabel="Recall",
            add_theory=True,
        )
        _plot_drug_panel(
            ax_bottom,
            subset,
            drug,
            "Precision",
            show_xlabel=True,
            show_ylabel=j == 0,
            ylabel="Precision",
            add_precision_target=True,
        )

    fig.align_ylabels(top_axes + bottom_axes)
    _add_column_titles(fig, top_axes)
    _add_panel_labels(fig, [top_axes[0], bottom_axes[0]], ("A", "B"))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#c41e1e",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label="Theoretical recall (Poisson)",
        ),
        Line2D(
            [0],
            [0],
            color="#c41e1e",
            linestyle="--",
            linewidth=1.0,
            label="Drug-specific precision target",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.05),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(FIG_DIR / "subFig3.png", dpi=500, facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
