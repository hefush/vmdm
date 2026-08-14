from __future__ import annotations

import os
from pathlib import Path

WORK = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORK / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

DATA = WORK / "subFig5_data.tsv"
FIG_DIR = WORK / "figs_png"
TABLE_OUT = WORK / "subFig5_performance_by_QC.tsv"
DELTA_OUT = WORK / "subFig5_delta_bootstrap.tsv"
POOLED_OUT = WORK / "subFig5_pooled_performance_by_QC.tsv"

# Nature-style figure sizing and font scaling
FIG_W_IN = 9.2
FIG_H_IN = 6.4
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

DRUG_ORDER = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]
DRUG_SHORT = {
    "Rifampicin": "RIF",
    "Isoniazid": "INH",
    "Ethambutol": "EMB",
    "Pyrazinamide": "PZA",
}
METHOD_ORDER = ["TB-Profiler", "VMDM"]
QC_ORDER = ["pass", "partial", "fail"]
QC_SEQTREAT_OUTCOME = {
    "pass": "Complete sequence",
    "partial": "Partial sequence",
    "fail": "Sequence failure",
    "not_run": "Not run",
}
QC_AXIS_LABELS = {
    "pass": "Complete",
    "partial": "Partial",
    "fail": "Failure",
}
METHOD_COLORS = {"TB-Profiler": "#4daf4a", "VMDM": "#d73027"}
QC_COLORS = {
    "pass": "#6f6f6f",
    "partial": "#b07d20",
    "fail": "#d73027",
}
QC_BACKGROUNDS = {
    "pass": "#f7f7f7",
    "partial": "#fff8e8",
    "fail": "#fff0f0",
}
DELTA_METRICS = [
    ("Recall", "Recall"),
    ("Precision", "Precision"),
    ("F1", "F1-score"),
]
GRID_COLOR = "#d8d8d8"
GRID_LW = 0.3
N_BOOT = 2000


def load_tngs() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep="\t")
    df["drug"] = df["Drug"].astype(str)
    df["method"] = (
        df["Method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"tbprofiler": "TB-Profiler", "vmdm": "VMDM"})
    )
    df["lab"] = (
        df["laboratory_method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"resistant": "R", "susceptible": "S"})
    )
    df["pred"] = df["y_pred"].astype(str).map({"1": "R", "0": "S", "1.0": "R", "0.0": "S"})
    df["qc"] = (
        df["QC"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"patrial": "partial", "not_run": "not_run"})
    )
    return df


def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (drug, method, qc), sub in df.groupby(["drug", "method", "qc"], sort=False):
        tp = int(((sub["lab"] == "R") & (sub["pred"] == "R")).sum())
        tn = int(((sub["lab"] == "S") & (sub["pred"] == "S")).sum())
        fp = int(((sub["lab"] == "S") & (sub["pred"] == "R")).sum())
        fn = int(((sub["lab"] == "R") & (sub["pred"] == "S")).sum())
        n_lab_r = tp + fn
        n_pred_r = tp + fp
        recall = tp / n_lab_r if n_lab_r else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        precision = tp / n_pred_r if n_pred_r else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        recall_lo, recall_hi = wilson_ci(tp, n_lab_r)
        prec_lo, prec_hi = wilson_ci(tp, n_pred_r)
        records.append(
            {
                "Drug": drug,
                "Method": method,
                "QC": qc,
                "SeqTreat_sequence_outcome": QC_SEQTREAT_OUTCOME.get(qc, qc),
                "n_records": len(sub),
                "n_lab_R": n_lab_r,
                "n_pred_R": n_pred_r,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Recall": recall,
                "Recall_CI_low": recall_lo,
                "Recall_CI_high": recall_hi,
                "Specificity": spec,
                "Precision": precision,
                "Precision_CI_low": prec_lo,
                "Precision_CI_high": prec_hi,
                "F1": f1,
                "NPV": npv,
            }
        )
    out = pd.DataFrame.from_records(records)
    out["Drug"] = pd.Categorical(out["Drug"], categories=DRUG_ORDER, ordered=True)
    out["Method"] = pd.Categorical(out["Method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["Drug", "QC", "Method"]).reset_index(drop=True)


def _array_metrics(lab: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((lab == "R") & (pred == "R")).sum())
    tn = int(((lab == "S") & (pred == "S")).sum())
    fp = int(((lab == "S") & (pred == "R")).sum())
    fn = int(((lab == "R") & (pred == "S")).sum())
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {"Recall": recall, "Precision": precision, "F1": f1, "Specificity": specificity}


def compute_paired_deltas(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 20260701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for qc in QC_ORDER:
        for drug in DRUG_ORDER:
            sub = df[(df["drug"] == drug) & (df["qc"] == qc) & (df["method"].isin(METHOD_ORDER))]
            pred = sub.pivot_table(index="Name", columns="method", values="pred", aggfunc="first")
            lab = sub.groupby("Name", sort=False)["lab"].first()
            pred = pred.dropna(subset=METHOD_ORDER)
            common = pred.index.intersection(lab.index)
            if len(common) == 0:
                continue

            lab_arr = lab.loc[common].to_numpy(dtype=str)
            tb_arr = pred.loc[common, "TB-Profiler"].to_numpy(dtype=str)
            vmdm_arr = pred.loc[common, "VMDM"].to_numpy(dtype=str)
            tb_metrics = _array_metrics(lab_arr, tb_arr)
            vmdm_metrics = _array_metrics(lab_arr, vmdm_arr)

            boot_idx = rng.integers(0, len(common), size=(n_boot, len(common)))
            boot_delta: dict[str, list[float]] = {metric: [] for metric, _ in DELTA_METRICS}
            for idx in boot_idx:
                tb_boot = _array_metrics(lab_arr[idx], tb_arr[idx])
                vmdm_boot = _array_metrics(lab_arr[idx], vmdm_arr[idx])
                for metric, _ in DELTA_METRICS:
                    boot_delta[metric].append(vmdm_boot[metric] - tb_boot[metric])

            for metric, metric_label in DELTA_METRICS:
                vals = np.array(boot_delta[metric], dtype=float)
                vals = vals[~np.isnan(vals)]
                lo, hi = (np.nan, np.nan) if vals.size == 0 else np.percentile(vals, [2.5, 97.5])
                records.append(
                    {
                        "Drug": drug,
                        "Drug_short": DRUG_SHORT[drug],
                        "QC": qc,
                        "SeqTreat_sequence_outcome": QC_SEQTREAT_OUTCOME[qc],
                        "Metric": metric,
                        "Metric_label": metric_label,
                        "n_specimens": len(common),
                        "TB_Profiler": tb_metrics[metric],
                        "VMDM": vmdm_metrics[metric],
                        "Delta": vmdm_metrics[metric] - tb_metrics[metric],
                        "Delta_CI_low": lo,
                        "Delta_CI_high": hi,
                        "Delta_pp": 100 * (vmdm_metrics[metric] - tb_metrics[metric]),
                        "Delta_CI_low_pp": 100 * lo,
                        "Delta_CI_high_pp": 100 * hi,
                    }
                )
    return pd.DataFrame.from_records(records)


def compute_pooled_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for qc in QC_ORDER:
        for method in METHOD_ORDER:
            sub = df[(df["qc"] == qc) & (df["method"] == method)]
            if sub.empty:
                continue
            vals = _array_metrics(sub["lab"].to_numpy(dtype=str), sub["pred"].to_numpy(dtype=str))
            records.append(
                {
                    "QC": qc,
                    "Method": method,
                    "n_records": len(sub),
                    "Recall": vals["Recall"],
                    "Precision": vals["Precision"],
                    "F1": vals["F1"],
                    "Specificity": vals["Specificity"],
                }
            )
    out = pd.DataFrame.from_records(records)
    out["QC"] = pd.Categorical(out["QC"], categories=QC_ORDER, ordered=True)
    out["Method"] = pd.Categorical(out["Method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["QC", "Method"]).reset_index(drop=True)


def _delta_row_layout() -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    group_mids: dict[str, float] = {}
    y = 0.0
    for qc in QC_ORDER:
        group_ys = []
        for drug in DRUG_ORDER:
            rows.append({"QC": qc, "Drug": drug, "Drug_short": DRUG_SHORT[drug], "y": y})
            group_ys.append(y)
            y += 1.0
        group_mids[qc] = float(np.mean(group_ys))
        y += 0.85
    return pd.DataFrame(rows), group_mids


def _style_delta_axis(ax: plt.Axes, *, show_y: bool, row_df: pd.DataFrame, group_mids: dict[str, float]) -> None:
    xmin, xmax = -42, 72
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(row_df["y"].max() + 0.75, -0.75)
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xticks([-40, -20, 0, 20, 40, 60])
    ax.set_xticklabels(["-40", "-20", "0", "+20", "+40", "+60"])
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=GRID_LW, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if show_y:
        ax.set_yticks(row_df["y"])
        ax.set_yticklabels(row_df["Drug_short"], fontsize=TICK_FS)
        for qc, mid in group_mids.items():
            ax.text(
                -0.26,
                mid,
                QC_AXIS_LABELS[qc],
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=TICK_FS,
                fontweight="bold" if qc == "fail" else "normal",
                color=QC_COLORS[qc],
            )
    else:
        ax.set_yticks(row_df["y"])
        ax.set_yticklabels([])
    ax.text(
        0.98,
        1.03,
        "VMDM higher",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=TICK_FS,
        color=METHOD_COLORS["VMDM"],
    )
    ax.text(
        0.02,
        1.03,
        "TB-Profiler higher",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_FS,
        color=METHOD_COLORS["TB-Profiler"],
    )


def _add_qc_backgrounds(ax: plt.Axes, row_df: pd.DataFrame) -> None:
    for qc in QC_ORDER:
        ys = row_df.loc[row_df["QC"] == qc, "y"]
        ax.axhspan(ys.min() - 0.48, ys.max() + 0.48, color=QC_BACKGROUNDS[qc], zorder=-2)


def _plot_delta_panel(ax: plt.Axes, delta_df: pd.DataFrame, row_df: pd.DataFrame, metric: str) -> None:
    _add_qc_backgrounds(ax, row_df)
    for row in row_df.itertuples(index=False):
        hit = delta_df[(delta_df["Drug"] == row.Drug) & (delta_df["QC"] == row.QC) & (delta_df["Metric"] == metric)]
        if hit.empty:
            continue
        hit = hit.iloc[0]
        color = QC_COLORS[row.QC]
        ax.hlines(
            row.y,
            hit["Delta_CI_low_pp"],
            hit["Delta_CI_high_pp"],
            color=color,
            linewidth=0.8 if row.QC != "fail" else 1.1,
            alpha=0.62 if row.QC != "fail" else 0.88,
            zorder=3,
        )
        ax.plot(
            hit["Delta_pp"],
            row.y,
            marker="o",
            markersize=4.0 if row.QC != "fail" else 5.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.95,
            zorder=4,
        )


def _plot_absolute_panel(ax: plt.Axes, pooled_df: pd.DataFrame, metric: str, title: str) -> None:
    x = np.arange(len(QC_ORDER), dtype=float)
    offsets = {"TB-Profiler": -0.035, "VMDM": 0.035}
    marker_style = {"TB-Profiler": "o", "VMDM": "s"}
    for method in METHOD_ORDER:
        vals = []
        for qc in QC_ORDER:
            hit = pooled_df[(pooled_df["QC"] == qc) & (pooled_df["Method"] == method)]
            vals.append(float(hit.iloc[0][metric]) * 100 if not hit.empty else np.nan)
        ax.plot(
            x + offsets[method],
            vals,
            color=METHOD_COLORS[method],
            linewidth=1.15,
            marker=marker_style[method],
            markersize=4.6,
            markerfacecolor="white" if method == "TB-Profiler" else METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            markeredgewidth=1.0,
            zorder=3 if method == "VMDM" else 2,
            label=method,
        )
        for xi, yi in zip(x + offsets[method], vals):
            dy = 2.1 if method == "VMDM" else -2.4
            va = "bottom" if method == "VMDM" else "top"
            ax.text(
                xi,
                yi + dy,
                f"{yi:.1f}",
                color=METHOD_COLORS[method],
                fontsize=TICK_FS * 0.78,
                ha="center",
                va=va,
                clip_on=False,
            )

    ax.set_title(title, fontsize=AXIS_FS, fontweight="bold", pad=7, fontfamily=_ACTIVE_FONT)
    ax.set_ylim(30, 104)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels([QC_AXIS_LABELS[q] for q in QC_ORDER])
    for tick, qc in zip(ax.get_xticklabels(), QC_ORDER):
        if qc == "fail":
            tick.set_color(QC_COLORS[qc])
            tick.set_fontweight("bold")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=GRID_LW, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("")


def _plot_failure_tradeoff(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    fail = metrics[metrics["QC"] == "fail"].copy()
    precision_grid = np.linspace(0.50, 1.02, 240)
    for f1 in (0.50, 0.60, 0.70, 0.80):
        recall_curve = f1 * precision_grid / (2 * precision_grid - f1)
        mask = (recall_curve >= 0.25) & (recall_curve <= 0.92)
        ax.plot(
            precision_grid[mask],
            recall_curve[mask],
            color="#cfcfcf",
            linewidth=0.45,
            linestyle="--",
            zorder=0,
        )
        if mask.any():
            ax.text(
                precision_grid[mask][-1] - 0.002,
                recall_curve[mask][-1],
                f"F1={f1:.1f}",
                ha="right",
                va="bottom",
                fontsize=TICK_FS * 0.75,
                color="#999999",
            )

    label_offsets = {
        "Rifampicin": (0.010, 0.020),
        "Isoniazid": (0.010, -0.005),
        "Ethambutol": (-0.055, 0.018),
        "Pyrazinamide": (-0.060, -0.012),
    }
    for drug in DRUG_ORDER:
        tb = fail[(fail["Drug"] == drug) & (fail["Method"] == "TB-Profiler")].iloc[0]
        vm = fail[(fail["Drug"] == drug) & (fail["Method"] == "VMDM")].iloc[0]
        ax.annotate(
            "",
            xy=(vm["Precision"], vm["Recall"]),
            xytext=(tb["Precision"], tb["Recall"]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "linewidth": 0.9,
                "shrinkA": 5,
                "shrinkB": 5,
                "mutation_scale": 8,
            },
            zorder=2,
        )
        ax.scatter(
            tb["Precision"],
            tb["Recall"],
            s=28,
            marker="o",
            facecolor="white",
            edgecolor=METHOD_COLORS["TB-Profiler"],
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            vm["Precision"],
            vm["Recall"],
            s=34,
            marker="s",
            facecolor=METHOD_COLORS["VMDM"],
            edgecolor=METHOD_COLORS["VMDM"],
            linewidth=0.8,
            zorder=4,
        )
        dx, dy = label_offsets[drug]
        ax.text(
            vm["Precision"] + dx,
            vm["Recall"] + dy,
            DRUG_SHORT[drug],
            fontsize=TICK_FS,
            fontweight="bold",
            color="#333333",
            ha="left",
            va="center",
        )

    ax.set_xlim(0.50, 1.02)
    ax.set_ylim(0.25, 0.92)
    ax.set_xlabel("Precision in sequence failures")
    ax.set_ylabel("Recall in sequence failures")
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LW, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=METHOD_COLORS["TB-Profiler"],
            markeredgewidth=1.0,
            markersize=5,
            label="TB-Profiler",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=METHOD_COLORS["VMDM"],
            markeredgecolor=METHOD_COLORS["VMDM"],
            markersize=5,
            label="VMDM",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.1)


def render_validation_figure(
    metrics: pd.DataFrame,
    delta_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    panel_label: str | None = None,
) -> None:
    print(
        f"tNGS figure font: {_ACTIVE_FONT} ({_FONT_NOTE}); "
        f"print targets at {WORD_EMBED_W_IN}\" embed: panel {NATURE_PANEL_PT} pt, text {NATURE_TEXT_PT} pt"
    )

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.17,
        right=0.985,
        top=0.935,
        bottom=0.10,
        height_ratios=[1.45, 2.85],
        hspace=0.50,
        wspace=0.16,
    )

    absolute_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (metric, label) in enumerate(DELTA_METRICS):
        ax = absolute_axes[i]
        _plot_absolute_panel(ax, pooled_df, metric, label)
        if i == 0:
            ax.set_ylabel("Pooled performance (%)")
        else:
            ax.set_yticklabels([])

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=METHOD_COLORS["TB-Profiler"],
            markerfacecolor="white",
            markeredgecolor=METHOD_COLORS["TB-Profiler"],
            markeredgewidth=1.0,
            markersize=5,
            linewidth=1.0,
            label="TB-Profiler",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=METHOD_COLORS["VMDM"],
            markerfacecolor=METHOD_COLORS["VMDM"],
            markeredgecolor=METHOD_COLORS["VMDM"],
            markersize=5,
            linewidth=1.0,
            label="VMDM",
        ),
    ]
    absolute_axes[2].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=False,
        ncol=1,
        handletextpad=0.4,
        borderaxespad=0,
    )

    row_df, group_mids = _delta_row_layout()
    delta_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, (metric, label) in enumerate(DELTA_METRICS):
        ax = delta_axes[i]
        _plot_delta_panel(ax, delta_df, row_df, metric)
        _style_delta_axis(ax, show_y=(i == 0), row_df=row_df, group_mids=group_mids)
        ax.set_title(label, fontsize=AXIS_FS, fontweight="bold", pad=18, fontfamily=_ACTIVE_FONT)
        if i == 1:
            ax.set_xlabel("VMDM - TB-Profiler (percentage points)", labelpad=6)

    if panel_label:
        fig.text(
            0.01,
            0.99,
            panel_label,
            ha="left",
            va="top",
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
        )
    else:
        fig.text(0.012, 0.985, "A", ha="left", va="top", fontsize=PANEL_FS, fontweight="bold", fontfamily=_ACTIVE_FONT)
        fig.text(0.012, 0.560, "B", ha="left", va="top", fontsize=PANEL_FS, fontweight="bold", fontfamily=_ACTIVE_FONT)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=500, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_png.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_tngs()
    metrics = compute_metrics(df)
    metrics = metrics[metrics["QC"].isin(QC_ORDER)].copy()
    plot_df = df[df["qc"].isin(QC_ORDER)].copy()
    delta_df = compute_paired_deltas(plot_df)
    pooled_df = compute_pooled_metrics(plot_df)
    metrics.to_csv(TABLE_OUT, sep="\t", index=False)
    delta_df.to_csv(DELTA_OUT, sep="\t", index=False, float_format="%.6f")
    pooled_df.to_csv(POOLED_OUT, sep="\t", index=False, float_format="%.6f")
    render_validation_figure(
        metrics,
        delta_df=delta_df,
        pooled_df=pooled_df,
        out_png=FIG_DIR / "subFig5.png",
        out_pdf=WORK / "subFig5.pdf",
    )


if __name__ == "__main__":
    main()
