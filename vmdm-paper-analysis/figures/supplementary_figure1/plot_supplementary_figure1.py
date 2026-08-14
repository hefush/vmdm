from __future__ import annotations

import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
_MPLCONFIG_DIR = WORK / ".mplconfig"
_XDG_CACHE_HOME = WORK / ".cache"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FIG_DIR = WORK / "figs_png"
DATA = WORK / "subFig1_data"

FIG_W_IN = 7.2
FIG_H_IN = 6.4
# Nature: panel labels 8 pt bold; other text 5–7 pt at final print size.
# Manuscript embeds supp figures at 6.0 in (build_manuscript_final.py).
WORD_EMBED_W_IN = 6.0
_FONT_SCALE = FIG_W_IN / WORD_EMBED_W_IN
NATURE_PANEL_PT = 8
NATURE_TEXT_PT = 7
NATURE_ANNOT_PT = 6.5

PANEL_FS = NATURE_PANEL_PT * _FONT_SCALE
AXIS_FS = NATURE_TEXT_PT * _FONT_SCALE
TICK_FS = NATURE_TEXT_PT * _FONT_SCALE
LEGEND_FS = NATURE_TEXT_PT * _FONT_SCALE
ANNOT_FS = NATURE_ANNOT_PT * _FONT_SCALE

_ARIAL_FONT_ENV = "ARIAL_FONT_PATH"
_ARIAL_CANDIDATE_PATHS = (
    WORK / "fonts" / "Arial.ttf",
    WORK / "fonts" / "arial.ttf",
    WORK / "fonts" / "ArialMT.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/arial.ttf"),
    Path("/usr/local/share/fonts/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)
_BUNDLED_ARIAL_PATHS = (
    WORK / "fonts" / "Arial.ttf",
    WORK / "fonts" / "Arialbd.ttf",
    WORK / "fonts" / "Ariali.ttf",
    WORK / "fonts" / "Arialbi.ttf",
)

DEPTH_ORDER = ["1x", "2x~3x", ">=4x"]
DEPTH_LABELS = ["1×", "2–3×", "≥4×"]
FREQ_ORDER = ["<0.25", "0.25~0.50", "0.50~1.00", "1.00~5.00", ">5.00"]
FREQ_LABELS = ["<0.25", "0.25–0.50", "0.50–1.00", "1.00–5.00", ">5.00"]
SIZE_ORDER = ["0.05x", "0.1x", "0.5x", "1x", "5x", "10x"]
SIZE_MAPPING = {
    "r5000": "0.05x",
    "r10000": "0.1x",
    "r50000": "0.5x",
    "r100000": "1x",
    "r500000": "5x",
    "r1000000": "10x",
}

COLORS = {
    "before": "#4E79A7",
    "after": "#E15759",
}
PPV_CMAP = LinearSegmentedColormap.from_list(
    "ppv_reds", ["#fff5eb", "#fee6ce", "#fcae91", "#fb6a4a", "#cb181d"], N=256
)
HOM_PPV_THRESH = 0.95
HET_PPV_THRESH = 0.94

OUT_PDF = WORK / "subFig1.pdf"
OUT_PNG = FIG_DIR / "subFig1-1.png"
OUT_TIFF = FIG_DIR / "subFig1-1.tiff"


def _is_arial_font(font_manager, path: Path) -> bool:
    try:
        return font_manager.FontProperties(fname=str(path)).get_name().lower() == "arial"
    except Exception:
        return False


def _find_arial_font(font_manager) -> Path | None:
    env_path = os.environ.get(_ARIAL_FONT_ENV)
    candidates = [Path(env_path).expanduser()] if env_path else []
    candidates.extend(_ARIAL_CANDIDATE_PATHS)

    seen = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file() and _is_arial_font(font_manager, path):
            return path

    for font_path in font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf"):
        path = Path(font_path)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_arial_font(font_manager, path):
            return path
    return None


def _setup_publication_font() -> tuple[str, str]:
    """Register and require true Arial for publication output."""
    from matplotlib import font_manager

    path = _find_arial_font(font_manager)
    if path is None:
        raise RuntimeError(
            "Arial font was not found. Put a real Arial regular font at "
            f"{WORK / 'fonts' / 'Arial.ttf'} or run with "
            f"{_ARIAL_FONT_ENV}=/path/to/Arial.ttf. "
            "Nimbus Sans/Liberation Sans are not used because this figure requires Arial."
        )

    for font_path in (*_BUNDLED_ARIAL_PATHS, path):
        if font_path.is_file() and _is_arial_font(font_manager, font_path):
            font_manager.fontManager.addfont(str(font_path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [name]
    return name, f"required Arial from {path}"


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
        "axes.titleweight": "normal",
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


def _ppv_pivot(data: pd.DataFrame) -> pd.DataFrame:
    pivot = data.pivot(index="Freq", columns="Depth", values="PPV")
    return pivot.reindex(index=FREQ_ORDER, columns=DEPTH_ORDER)


def _hom_retained(freq: str, depth: str) -> bool:
    if depth in ("2x~3x", ">=4x"):
        return True
    return depth == "1x" and freq in ("0.25~0.50", "0.50~1.00", "1.00~5.00", ">5.00")


def _het_retained(freq: str, depth: str) -> bool:
    return depth == ">=4x" and freq in ("1.00~5.00", ">5.00")


def _plot_ppv_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    *,
    retained_fn,
    ppv_thresh: float,
    show_ylabel: bool,
) -> mpl.cm.ScalarMappable:
    values = pivot.to_numpy(dtype=float)
    cmap = PPV_CMAP.copy()
    cmap.set_bad("#ececec")

    im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="equal", origin="upper")
    ax.set_xticks(range(len(DEPTH_ORDER)))
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_yticks(range(len(FREQ_ORDER)))
    ax.set_yticklabels(FREQ_LABELS)
    if show_ylabel:
        ax.set_ylabel("Population frequency bin (%)", labelpad=5)

    for i, freq in enumerate(FREQ_ORDER):
        for j, depth in enumerate(DEPTH_ORDER):
            val = values[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=ANNOT_FS, color="#666666")
                continue
            text_color = "white" if val >= 0.75 else "#222222"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=ANNOT_FS,
                color=text_color,
                fontweight="normal",
            )
            if retained_fn(freq, depth) and val >= ppv_thresh:
                ax.add_patch(
                    Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#1a1a1a",
                        linewidth=1.0,
                        zorder=5,
                    )
                )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


def _fit_square_heatmap_axes(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Resize heatmap axes so cells are square and row tops align."""
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    max_width = max(pos.width for pos in positions)
    target_h = max_width * (len(FREQ_ORDER) / len(DEPTH_ORDER))
    top = max(pos.y1 for pos in positions)
    for ax, pos in zip(axes, positions):
        ax.set_position([pos.x0, top - target_h, pos.width, target_h])


def _place_heatmap_row_labels(
    fig: plt.Figure,
    ax_het: plt.Axes,
    ax_hom: plt.Axes,
) -> None:
    fig.canvas.draw()
    for ax, label in ((ax_het, "Heterozygous"), (ax_hom, "Homozygous")):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.014,
            label,
            ha="center",
            va="bottom",
            fontsize=AXIS_FS,
            fontfamily=_ACTIVE_FONT,
        )


def _place_shared_xlabel(
    fig: plt.Figure,
    axes: list[plt.Axes],
    text: str,
    y_offset: float = 0.028,
) -> None:
    fig.canvas.draw()
    bboxes = [ax.get_position() for ax in axes]
    x_center = (min(bb.x0 for bb in bboxes) + max(bb.x1 for bb in bboxes)) / 2
    y_bottom = min(bb.y0 for bb in bboxes)
    fig.text(
        x_center,
        y_bottom - y_offset,
        text,
        ha="center",
        va="top",
        fontsize=AXIS_FS,
        fontfamily=_ACTIVE_FONT,
    )


def _place_colorbar(
    fig: plt.Figure,
    ax_ref: plt.Axes,
    im: mpl.cm.ScalarMappable,
) -> mpl.colorbar.Colorbar:
    """Anchor PPV colorbar to the heatmap matrix (same top/bottom as reference axes)."""
    fig.canvas.draw()
    pos = ax_ref.get_position()
    cbar_w = 0.014
    pad = 0.018
    cbar_ax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_w, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("PPV", rotation=270, labelpad=12, fontsize=AXIS_FS)
    cbar.ax.tick_params(labelsize=TICK_FS, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return cbar


def _plot_metric_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    *,
    show_xlabel: bool = True,
    ylim: tuple[float, float] = (0.0, 1.05),
) -> None:
    x = np.arange(len(SIZE_ORDER))
    for method, color, label in (
        ("Initial", COLORS["before"], "Before QC"),
        ("AfterQC", COLORS["after"], "After QC"),
    ):
        subset = df[df["Method"] == method].set_index("Size").reindex(SIZE_ORDER)
        ax.plot(
            x,
            subset[metric_col].to_numpy(),
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=label,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    if show_xlabel:
        ax.set_xlabel("Genome coverage", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, pad=6, fontsize=AXIS_FS, fontweight="normal")
    ax.set_ylim(*ylim)
    ax.axhline(0.9, color="#aaaaaa", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(axis="y", linestyle=":", alpha=0.22, linewidth=0.5, color="#cccccc")
    _style_axes(ax)


def _align_figure_columns(
    fig: plt.Figure,
    gs: GridSpec,
    left_col: list[plt.Axes],
    right_col: list[plt.Axes],
) -> None:
    for ax, (row, col) in zip(left_col, [(0, 0), (1, 0)]):
        ax.set_position(gs[row, col].get_position(fig))
    for ax, (row, col) in zip(right_col, [(0, 1), (1, 1)]):
        ax.set_position(gs[row, col].get_position(fig))
    fig.canvas.draw()
    fig.align_ylabels(left_col)
    fig.align_xlabels([left_col[0], right_col[0]])
    fig.align_xlabels(left_col)


def _add_row_panel_labels(
    fig: plt.Figure,
    left_axes: list[plt.Axes],
    labels: tuple[str, str],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fallback_x = min(ax.get_position().x0 for ax in left_axes) - 0.014
    for ax, label in zip(left_axes, labels):
        ax_bbox = ax.get_position()
        ylabel_bbox = ax.yaxis.get_label().get_window_extent(renderer=renderer)
        ylabel_bbox = ylabel_bbox.transformed(fig.transFigure.inverted())
        label_x = (ylabel_bbox.x0 + ylabel_bbox.x1) / 2 if ax.yaxis.get_label_text() else fallback_x
        fig.text(
            label_x,
            ax_bbox.y1 + 0.008,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="center",
        )


def render() -> None:
    print(
        f"subFig1 font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: "
        f"panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt, annot {NATURE_ANNOT_PT} pt"
    )

    stat = pd.read_csv(DATA / "stat.tsv", sep="\t")
    het_pivot = _ppv_pivot(stat[stat["Type"] == "het"])
    hom_pivot = _ppv_pivot(stat[stat["Type"] == "hom"])

    scores = pd.read_csv(DATA / "method_score.tsv", sep="\t")
    scores = scores[scores["Label"] == "ALL"].copy()
    scores["Size"] = scores["Size"].map(SIZE_MAPPING)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.12,
        right=0.90,
        top=0.93,
        bottom=0.15,
        wspace=0.30,
        hspace=0.48,
        height_ratios=[0.92, 1.0],
    )

    ax_het = fig.add_subplot(gs[0, 0])
    ax_hom = fig.add_subplot(gs[0, 1])
    ax_recall = fig.add_subplot(gs[1, 0])
    ax_prec = fig.add_subplot(gs[1, 1])

    im = _plot_ppv_heatmap(
        ax_het,
        het_pivot,
        retained_fn=_het_retained,
        ppv_thresh=HET_PPV_THRESH,
        show_ylabel=True,
    )
    _plot_ppv_heatmap(
        ax_hom,
        hom_pivot,
        retained_fn=_hom_retained,
        ppv_thresh=HOM_PPV_THRESH,
        show_ylabel=False,
    )

    left_col, right_col = [ax_het, ax_recall], [ax_hom, ax_prec]
    _align_figure_columns(fig, gs, left_col, right_col)
    _fit_square_heatmap_axes(fig, [ax_het, ax_hom])
    _place_heatmap_row_labels(fig, ax_het, ax_hom)
    _place_shared_xlabel(fig, [ax_het, ax_hom], "Sequencing depth", y_offset=0.032)
    _place_colorbar(fig, ax_hom, im)

    _plot_metric_lines(
        ax_recall, scores, "Genotype Recall", "Genotype recall", "Recall", show_xlabel=False
    )
    _plot_metric_lines(
        ax_prec,
        scores,
        "Genotype Precision",
        "Genotype precision",
        "Precision",
        show_xlabel=False,
        ylim=(0.8, 1.0),
    )
    ax_prec.axhline(0.95, color="#cccccc", linestyle=":", linewidth=0.8, zorder=1)
    _place_shared_xlabel(fig, [ax_recall, ax_prec], "Genome coverage", y_offset=0.038)
    _add_row_panel_labels(fig, [ax_het, ax_recall], ("A", "B"))

    legend_handles = [
        Line2D([0], [0], color=COLORS["before"], linewidth=1.6, marker="o", markersize=4.5, label="Before QC"),
        Line2D([0], [0], color=COLORS["after"], linewidth=1.6, marker="o", markersize=4.5, label="After QC"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(WORK / "subFig1.png", dpi=500, facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
