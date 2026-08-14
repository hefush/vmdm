from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

WORK = Path(__file__).resolve().parent
FIG_DIR = WORK / "figs_png"
DATA = WORK / "Fig4_data.tsv"

os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))

FIG_W_IN = 7.2
FIG_H_IN = 3.5
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

METHOD_ORDER = ["mykrobe", "tb_profiler", "vmdm"]
LEGEND_ORDER = ["mykrobe", "tb_profiler", "vmdm"]
METHOD_COLORS = {
    "mykrobe": "#377eb8",
    "tb_profiler": "#4daf4a",
    "vmdm": "#d73027",
}
METHOD_LABELS = {
    "mykrobe": "Mykrobe",
    "tb_profiler": "TB-Profiler",
    "vmdm": "VMDM",
}

SIZE_KEYS = ["r1000", "r5000", "r10000", "r50000", "r100000", "r500000", "r1000000"]
COVERAGE_LABELS = ["0.01×", "0.05×", "0.1×", "0.5×", "1×", "5×", "10×"]

METRIC_PANELS = [
    ("CPUs", "A", "Total CPU time (s)"),
    ("Time", "B", "Wall-clock time (s)"),
    ("Maxmem", "C", "Peak memory (MB)"),
]

OUTLIER_QUANTILE = 0.95
_BOX_STYLE = {
    "alpha": 0.7,
    "linewidth": 0.5,
    "edgecolor": "#666666",
    "median_color": "#1a1a1a",
    "median_lw": 1.0,
    "whisker_color": "#444444",
    "whisker_lw": 0.75,
}

OUT_PDF = WORK / "Fig4-1.pdf"
OUT_PNG = FIG_DIR / "Fig4-1.png"
OUT_TIFF = FIG_DIR / "Fig4-1.tiff"
OUT_MEDIAN_CSV = WORK / "Fig4_median_stats.csv"


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


def _remove_outliers(group: pd.DataFrame, cols: list[str], quantile: float) -> pd.DataFrame:
    out = group
    for col in cols:
        threshold = out[col].quantile(quantile)
        out = out[out[col] <= threshold]
    return out


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t")
    df = df[df["Size"] != "r500"].copy()
    df["Maxmem_MB"] = df["Maxmem"] / 1024.0
    # Avoid pandas-version-dependent handling of grouping columns in
    # DataFrameGroupBy.apply.
    df = pd.concat(
        [
            _remove_outliers(group, cols=["CPUs", "Time", "Maxmem"], quantile=OUTLIER_QUANTILE)
            for _keys, group in df.groupby(["Method", "Size"], sort=False)
        ],
        ignore_index=True,
    )
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    return df


def _export_median_stats(df: pd.DataFrame) -> None:
    coverage_map = dict(zip(SIZE_KEYS, COVERAGE_LABELS))
    stats = (
        df.assign(Coverage=df["Size"].map(coverage_map))
        .groupby(["Method", "Coverage"], observed=True)[["CPUs", "Time", "Maxmem_MB"]]
        .median()
        .reset_index()
    )
    stats.to_csv(OUT_MEDIAN_CSV, index=False, float_format="%.2f")


def _plot_metric_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    plot_metric = "Maxmem_MB" if metric == "Maxmem" else metric
    group_centers = np.arange(len(SIZE_KEYS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHOD_ORDER))
    box_w = 0.15

    for i, size_key in enumerate(SIZE_KEYS):
        for j, method in enumerate(METHOD_ORDER):
            vals = df[(df["Size"] == size_key) & (df["Method"] == method)][plot_metric].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            color = METHOD_COLORS[method]
            ax.boxplot(
                vals,
                positions=[group_centers[i] + offsets[j]],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": color,
                    "alpha": _BOX_STYLE["alpha"],
                    "linewidth": _BOX_STYLE["linewidth"],
                    "edgecolor": color,
                },
                medianprops={"color": _BOX_STYLE["median_color"], "linewidth": _BOX_STYLE["median_lw"]},
                whiskerprops={"linewidth": _BOX_STYLE["whisker_lw"], "color": color},
                capprops={"linewidth": _BOX_STYLE["whisker_lw"], "color": color},
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(COVERAGE_LABELS, rotation=45, ha="right")
    ax.set_xlim(-0.45, len(SIZE_KEYS) - 0.55)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", linewidth=0.3, color="#d8d8d8", alpha=0.9, zorder=0)
    _style_axes(ax)


def _add_panel_labels(fig: plt.Figure, axes: list[plt.Axes], labels: tuple[str, str, str]) -> None:
    fig.canvas.draw()
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.012,
            bbox.y1 + 0.010,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="right",
        )


def _panel_row_center_x(axes: list[plt.Axes]) -> float:
    left = axes[0].get_position()
    right = axes[-1].get_position()
    return 0.5 * (left.x0 + right.x1)


def _add_shared_xlabel(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    y_bottom = axes[0].get_position().y0
    fig.text(
        _panel_row_center_x(axes),
        y_bottom - 0.085,
        "Genome coverage",
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=AXIS_FS,
        fontfamily=_ACTIVE_FONT,
        color="#333333",
    )


def render() -> None:
    print(
        f"Fig4 font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt"
    )

    df = _load_data()
    _export_median_stats(df)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, left=0.08, right=0.99, top=0.88, bottom=0.22, wspace=0.36)

    axes: list[plt.Axes] = []
    panel_labels: list[str] = []
    for col_idx, (metric, panel, ylabel) in enumerate(METRIC_PANELS):
        ax = fig.add_subplot(gs[0, col_idx])
        _plot_metric_panel(ax, df, metric, ylabel)
        axes.append(ax)
        panel_labels.append(panel)

    fig.align_ylabels(axes)
    _add_panel_labels(fig, axes, tuple(panel_labels))
    _add_shared_xlabel(fig, axes)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=METHOD_COLORS[m],
            linewidth=_BOX_STYLE["linewidth"],
            alpha=_BOX_STYLE["alpha"],
            label=METHOD_LABELS[m],
        )
        for m in LEGEND_ORDER
    ]
    fig.canvas.draw()
    bottom_y = axes[0].get_position().y0
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(_panel_row_center_x(axes), bottom_y - 0.135),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(WORK / "Fig4.pdf", format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
