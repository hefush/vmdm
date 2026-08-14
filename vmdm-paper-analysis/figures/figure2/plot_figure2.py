from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
WORK = Path(__file__).resolve().parent
FIG_DIR = WORK / "figs_png"
DATA = WORK

os.environ.setdefault("MPLCONFIGDIR", str(WORK / ".mplconfig"))

FIG_W_IN = 11.2
FIG_H_IN = 9.0
# Nature double-column width ≈ 183 mm; scale pt sizes so text is 5–7 pt after print resize
NATURE_PRINT_W_IN = 183 / 25.4
_FONT_SCALE = FIG_W_IN / NATURE_PRINT_W_IN
NATURE_PANEL_PT = 8   # Nature: panel labels 8 pt bold at final print size
NATURE_TEXT_PT = 7    # Nature: axis / legend / ticks 5–7 pt (use 7 pt)

PANEL_FS = NATURE_PANEL_PT * _FONT_SCALE
AXIS_FS = NATURE_TEXT_PT * _FONT_SCALE
TICK_FS = NATURE_TEXT_PT * _FONT_SCALE
LEGEND_FS = NATURE_TEXT_PT * _FONT_SCALE
ANNOT_FS = NATURE_TEXT_PT * _FONT_SCALE
NETWORK_LABEL_FS = NATURE_TEXT_PT * _FONT_SCALE

_FONT_CANDIDATE_PATHS = (
    Path(os.environ.get("ARIAL_FONT_PATH", "")),
    WORK / "fonts" / "Arial.ttf",
    WORK / "fonts" / "arial.ttf",
    WORK / "fonts" / "Helvetica.ttf",
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
)


def _setup_publication_font() -> tuple[str, str]:
    """Register Arial if bundled; else Nimbus Sans (Helvetica clone) or Liberation Sans."""
    from matplotlib import font_manager

    note = ""
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

COLORS = {
    "roc": "#d73027",
    "bar": "#4C72B0",
    "line": "#DD8452",
    "rif": "#d73027",
    "inh": "#377eb8",
    "emb": "#4daf4a",
    "bg": "#bdbdbd",
}

OUT_PDF = WORK / "Fig2-1.pdf"
OUT_PNG = FIG_DIR / "Fig2-1.png"
OUT_TIFF = FIG_DIR / "Fig2-1.tiff"

# Pairwise LD edges in edge.tsv are from the full PLINK export
# (--r2 inter-chr --ld-window-r2 0.2), prefiltered to rows needed for Fig. 2D.
LD_DISPLAY_MIN = 0.10
TOP_IMPORTANCE_N = 150
LABEL_TOP_K = 10  # max labels; only loci with entries in SHORT_LABELS are annotated
LAYOUT_SEED = 42
PRIMARY_HUB_LABEL = "rpoB_p.Ser450Leu"
# One-hop LD expansion for top-importance loci whose only partner lies outside top-N.
ONE_HOP_LABELS = ("rpoB_p.His445Asp",)
SHORT_LABELS = {
    "rpoB_p.Ser450Leu": "S450L",
    "katG_p.Ser315Thr": "S315T",
    "embB_p.Met306Val": "M306V",
    "rpoB_p.Asp435Val": "D435V",
    "embB_p.Asp354Ala": "D354A",
    "rpoB_p.Leu452Pro": "L452P",
    "rpoC_p.Gly594Glu": "G594E",
    "embB_p.Met306Ile": "M306I",
    "rpoB_p.His445Tyr": "H445Y",
    "rpoB_p.His445Asp": "H445D",
}
def node_class(label: str) -> str:
    low = label.lower()
    if low.startswith(("rpob", "rpoc", "rpoa")):
        return "rif"
    if low.startswith(("katg", "inha", "fabg1")):
        return "inh"
    if low.startswith(("emba", "embb", "embc")):
        return "emb"
    return "bg"


def _top_importance_ids(nodes: pd.DataFrame, n: int) -> set[str]:
    """Top-N RIF PRESS loci ranked by cumulative PRESS importance."""
    return set(nodes.nlargest(n, "Importance")["ID"].astype(str))


def _expand_one_hop_ld_neighbors(edges: pd.DataFrame, seed_ids: set[str]) -> set[str]:
    """Include direct LD partners so high-importance loci with peripheral links are retained."""
    expanded = set(seed_ids)
    for node_id in seed_ids:
        mask = (edges["ID_A"].astype(str) == node_id) | (edges["ID_B"].astype(str) == node_id)
        for _, row in edges.loc[mask].iterrows():
            expanded.add(str(row["ID_A"]))
            expanded.add(str(row["ID_B"]))
    return expanded


def _add_orphan_top_nodes(g: nx.Graph, nodes: pd.DataFrame, top_ids: set[str]) -> None:
    """Retain top-importance loci that lack LD edges in the LCC."""
    for node_id in top_ids:
        if node_id in g:
            continue
        row = nodes.loc[nodes["ID"].astype(str) == node_id]
        if row.empty:
            continue
        row = row.iloc[0]
        label = row["Label"] if pd.notna(row.get("Label")) else str(row["ID"])
        g.add_node(
            str(row["ID"]),
            importance=float(row["Importance"]),
            cover=float(row["Cover"]),
            label=str(label),
            drug=node_class(str(label)),
        )


def _top_label_nodes(g: nx.Graph, top_k: int = LABEL_TOP_K) -> list[str]:
    """Label up to K nodes with a pre-defined short name (SHORT_LABELS), by importance."""
    ranked = sorted(g.nodes, key=lambda n: g.nodes[n]["importance"], reverse=True)
    labelled: list[str] = []
    seen_short: set[str] = set()
    for node in ranked:
        full_label = g.nodes[node]["label"]
        if full_label not in SHORT_LABELS:
            continue
        short = SHORT_LABELS[full_label]
        if short in seen_short:
            continue
        seen_short.add(short)
        labelled.append(node)
        if len(labelled) >= top_k:
            break
    return labelled


def _build_ld_graph(lcc_only: bool = True) -> tuple[nx.Graph, pd.DataFrame, int]:
    """Build the Fig. 2D LD subgraph.

    The distributed node/edge tables are a prefiltered subset of the original
    full-network export. They retain all rows needed by the manuscript panel:
    top PRESS loci, required one-hop LD neighbours, and edges above the display
    threshold. The plotting logic below is unchanged from the full-data path.
    """
    nodes = pd.read_csv(DATA / "network/node.tsv", delimiter="\t")
    edges = pd.read_csv(DATA / "network/edge.tsv", delimiter="\t")
    edges = edges[edges["LD"] >= LD_DISPLAY_MIN].copy()

    top_ids = _top_importance_ids(nodes, TOP_IMPORTANCE_N)
    candidate_ids = set(top_ids)
    one_hop_ids = set(nodes.loc[nodes["Label"].isin(ONE_HOP_LABELS), "ID"].astype(str))
    candidate_ids |= _expand_one_hop_ld_neighbors(edges, one_hop_ids)
    edge_df = edges[
        edges["ID_A"].astype(str).isin(candidate_ids) & edges["ID_B"].astype(str).isin(candidate_ids)
    ].copy()
    active_ids = set(edge_df["ID_A"].astype(str)).union(edge_df["ID_B"].astype(str))
    active_ids |= top_ids
    node_df = nodes[nodes["ID"].astype(str).isin(active_ids)].copy()

    g = nx.Graph()
    for _, row in node_df.iterrows():
        label = row["Label"] if pd.notna(row.get("Label")) else str(row["ID"])
        g.add_node(
            str(row["ID"]),
            importance=float(row["Importance"]),
            cover=float(row["Cover"]),
            label=str(label),
            drug=node_class(str(label)),
        )
    for _, row in edge_df.iterrows():
        id_a, id_b = str(row["ID_A"]), str(row["ID_B"])
        if id_a in g.nodes and id_b in g.nodes:
            g.add_edge(id_a, id_b, weight=float(row["LD"]))

    n_other_components = 0
    if lcc_only and g.number_of_nodes() > 0:
        comps = sorted(nx.connected_components(g), key=len, reverse=True)
        lcc = comps[0]
        n_other_components = len(comps) - 1
        g = g.subgraph(lcc).copy()
        edge_df = edge_df[
            edge_df["ID_A"].astype(str).isin(lcc) & edge_df["ID_B"].astype(str).isin(lcc)
        ].copy()

    _add_orphan_top_nodes(g, nodes, top_ids)
    return g, edge_df, n_other_components


def _node_by_label(g: nx.Graph, label: str) -> str | None:
    for node in g.nodes:
        if g.nodes[node]["label"] == label:
            return node
    return None


def _normalize_positions(
    pos: dict[str, np.ndarray], frame_nodes: set[str], margin: float = 0.07
) -> dict[str, np.ndarray]:
    coords = np.array([pos[n] for n in frame_nodes if n in pos])
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    target = 1.0 - 2 * margin
    scale = target / span
    return {node: np.array([(p[0] - cx) * scale, (p[1] - cy) * scale]) for node, p in pos.items()}


def _place_on_ring(
    centre: np.ndarray, angle: float, radius: float, jitter: float, rng: np.random.Generator
) -> np.ndarray:
    rad = radius + rng.uniform(-jitter, jitter)
    ang = angle + rng.uniform(-jitter / max(radius, 0.1), jitter / max(radius, 0.1))
    return centre + rad * np.array([np.cos(ang), np.sin(ang)])


def _aesthetic_layout(g: nx.Graph) -> dict[str, np.ndarray]:
    """Visual layout only (does not alter nodes/edges).

    Labelled loci are placed at fixed hub coordinates; background loci are distributed on
    concentric rings around S450L for readability. LAYOUT_SEED controls ring jitter only.
    """
    rng = np.random.default_rng(LAYOUT_SEED)
    pos: dict[str, np.ndarray] = {}
    labelled = set(_top_label_nodes(g))
    primary = _node_by_label(g, PRIMARY_HUB_LABEL)
    hub = np.array([0.0, 0.0])

    anchor_specs = {
        PRIMARY_HUB_LABEL: np.array([0.00, 0.00]),
        "katG_p.Ser315Thr": np.array([-0.78, 0.50]),
        "embB_p.Met306Ile": np.array([-0.48, 0.66]),
        "embB_p.Met306Val": np.array([0.72, 0.52]),
        "embB_p.Asp354Ala": np.array([-0.58, -0.62]),
        "rpoB_p.Asp435Val": np.array([-0.46, 0.18]),
        "rpoB_p.His445Tyr": np.array([-0.24, 0.14]),
        "rpoB_p.His445Asp": np.array([0.24, 0.14]),
        "rpoB_p.Leu452Pro": np.array([0.46, 0.18]),
        "rpoC_p.Gly594Glu": np.array([0.68, -0.58]),
    }
    for label, coord in anchor_specs.items():
        if (node := _node_by_label(g, label)) and node in g:
            pos[node] = coord.copy()
    if primary and primary in pos:
        hub = pos[primary]

    bg_nodes = sorted(n for n in g.nodes if g.nodes[n]["drug"] == "bg" and n not in pos)
    ring_specs = ((0.34, 0.03), (0.50, 0.04), (0.66, 0.05))
    counts = [len(bg_nodes) // 3, len(bg_nodes) // 3, len(bg_nodes) - 2 * (len(bg_nodes) // 3)]
    bg_idx = 0
    for (radius, jitter), count in zip(ring_specs, counts):
        for j in range(count):
            if bg_idx >= len(bg_nodes):
                break
            angle = 2 * np.pi * j / max(count, 1) + 0.17
            pos[bg_nodes[bg_idx]] = _place_on_ring(hub, angle, radius, jitter, rng)
            bg_idx += 1

    for node in g.nodes:
        if node in pos:
            continue
        neighbours = [n for n in g.neighbors(node) if n in pos]
        if neighbours:
            anchor = max(neighbours, key=lambda n: g.nodes[n]["importance"])
            direction = pos[anchor] - hub
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = rng.uniform(-1.0, 1.0, 2)
                norm = np.linalg.norm(direction)
            direction /= norm
            radial = np.linalg.norm(pos[anchor] - hub)
            pos[node] = pos[anchor] + 0.14 * direction + rng.uniform(-0.05, 0.05, 2)
            if radial < 0.25:
                angle = np.arctan2(direction[1], direction[0])
                pos[node] = _place_on_ring(hub, angle, 0.28, 0.04, rng)
        else:
            pos[node] = _place_on_ring(hub, rng.uniform(0, 2 * np.pi), 0.40, 0.05, rng)

    return _normalize_positions(pos, frame_nodes=set(labelled))


def _node_draw_size(importance: float, imp_max: float) -> float:
    """Node area scales with PRESS importance; identical rule for every node."""
    norm = (importance / max(imp_max, 1e-9)) ** 0.55
    return 10 + norm * 320


def _label_above_below(label_nodes: list[str], pos: dict[str, np.ndarray]) -> dict[str, bool]:
    """Upper half of labelled nodes (by y) → label above; lower half → label below."""
    ordered = sorted(label_nodes, key=lambda n: pos[n][1])
    mid = len(ordered) // 2
    above = set(ordered[mid:])
    return {n: n in above for n in label_nodes}


def _set_network_view(
    ax: plt.Axes,
    g: nx.Graph,
    pos: dict[str, np.ndarray],
    texts: list,
    *,
    pad_left: float = 0.025,
    pad_top: float = 0.025,
    pad_right: float = 0.07,
    pad_bottom: float = 0.05,
) -> None:
    """Square view anchored top-left: minimal left/top padding, extra space to right/bottom."""
    coords = np.array([pos[n] for n in g.nodes])
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for text in texts:
        bb = text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        xmin = min(xmin, bb.x0)
        xmax = max(xmax, bb.x1)
        ymin = min(ymin, bb.y0)
        ymax = max(ymax, bb.y1)

    span_x = (xmax - xmin) + pad_left + pad_right
    span_y = (ymax - ymin) + pad_top + pad_bottom
    side = max(span_x, span_y)
    x0 = xmin - pad_left
    y1 = ymax + pad_top
    ax.set_xlim(x0, x0 + side)
    ax.set_ylim(y1 - side, y1)


def _vertical_label_anchor(
    pos_xy: np.ndarray,
    importance: float,
    imp_max: float,
    place_above: bool,
) -> tuple[float, float, str, str]:
    """Place label directly above or below the marker centre (ha = centre)."""
    norm = (importance / max(imp_max, 1e-9)) ** 0.55
    gap = 0.036 + norm * 0.030
    x, y = float(pos_xy[0]), float(pos_xy[1])
    if place_above:
        return x, y + gap, "center", "bottom"
    return x, y - gap, "center", "top"


def _lock_axes_to_grid(
    fig: plt.Figure,
    gs: GridSpec,
    axes: list[plt.Axes],
    cells: list[tuple[int, int]],
    twins: list[tuple[plt.Axes, plt.Axes]] | None = None,
) -> None:
    """Pin every axes box to its GridSpec cell (set_aspect must not resize the cell)."""
    fig.canvas.draw()
    for ax, (row, col) in zip(axes, cells):
        ax.set_position(gs[row, col].get_position(fig))
    if twins:
        for parent, twin in twins:
            twin.set_position(parent.get_position())


def _add_panel_labels(
    fig: plt.Figure,
    axes: list[plt.Axes],
    labels: list[str],
    col_groups: tuple[list[plt.Axes], list[plt.Axes]],
) -> None:
    """Uppercase A/B/C/D anchored to aligned column spines (A–C, B–D)."""
    fig.canvas.draw()
    left_col, right_col = col_groups
    left_x = min(ax.get_position().x0 for ax in left_col)
    right_x = min(ax.get_position().x0 for ax in right_col)
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        col = 0 if ax in left_col else 1
        fig.text(
            (left_x if col == 0 else right_x) - 0.012,
            bbox.y1 + 0.006,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            fontfamily=_ACTIVE_FONT,
            va="bottom",
            ha="right",
        )


def _align_figure_columns(
    fig: plt.Figure,
    gs: GridSpec,
    left_col: list[plt.Axes],
    right_col: list[plt.Axes],
    twins: list[tuple[plt.Axes, plt.Axes]] | None = None,
) -> None:
    """Lock column boxes to GridSpec, then align y-labels so left spines line up."""
    cells_left = [(0, 0), (1, 0)]
    cells_right = [(0, 1), (1, 1)]
    for axes, cells in ((left_col, cells_left), (right_col, cells_right)):
        for ax, (row, col) in zip(axes, cells):
            ax.set_position(gs[row, col].get_position(fig))
    fig.canvas.draw()
    fig.align_ylabels(left_col)
    fig.align_ylabels(right_col)
    fig.align_xlabels([left_col[0], right_col[0]])
    fig.align_xlabels(left_col)
    fig.canvas.draw()
    if twins:
        for parent, twin in twins:
            twin.set_position(parent.get_position())


def plot_ld_network(ax: plt.Axes) -> None:
    g, edge_df, _ = _build_ld_graph(lcc_only=True)
    pos = _aesthetic_layout(g)
    label_nodes = _top_label_nodes(g, LABEL_TOP_K)
    label_set = set(label_nodes)
    imp_max = max(g.nodes[n]["importance"] for n in g.nodes)
    primary = _node_by_label(g, PRIMARY_HUB_LABEL)
    focal_hubs = {n for n in label_nodes if g.nodes[n]["drug"] != "bg"}

    ld_max = max(float(edge_df["LD"].max()), LD_DISPLAY_MIN + 1e-6)
    bg_segments: list[np.ndarray] = []
    bg_colors: list[tuple[float, float, float, float]] = []
    bg_widths: list[float] = []
    key_segments: list[np.ndarray] = []
    key_colors: list[tuple[float, float, float, float]] = []
    key_widths: list[float] = []
    focal_segments: list[np.ndarray] = []
    focal_colors: list[tuple[float, float, float, float]] = []
    focal_widths: list[float] = []
    cross_segments: list[np.ndarray] = []
    cross_colors: list[tuple[float, float, float, float]] = []
    cross_widths: list[float] = []

    def _edge_anchor(u: str, v: str) -> str | None:
        if primary and (u == primary or v == primary):
            return primary
        if u in label_set and v in label_set:
            return max((u, v), key=lambda n: g.nodes[n]["importance"])
        if u in label_set:
            return u
        if v in label_set:
            return v
        return None

    def _is_cross_resistance(u: str, v: str) -> bool:
        du, dv = g.nodes[u]["drug"], g.nodes[v]["drug"]
        return du != dv and "bg" not in (du, dv) and (u in focal_hubs or v in focal_hubs)

    for _, row in edge_df.iterrows():
        id_a, id_b = str(row["ID_A"]), str(row["ID_B"])
        if id_a not in pos or id_b not in pos:
            continue
        ld = float(row["LD"])
        norm_ld = (ld - LD_DISPLAY_MIN) / max(ld_max - LD_DISPLAY_MIN, 1e-6)
        seg = np.array([pos[id_a], pos[id_b]])
        anchor = _edge_anchor(id_a, id_b)
        if anchor is not None and _is_cross_resistance(id_a, id_b):
            edge_rgb = mpl.colors.to_rgb("#4a4a4a")
            cross_segments.append(seg)
            cross_colors.append((*edge_rgb, 0.55 + norm_ld * 0.35))
            cross_widths.append(1.2 + norm_ld * 1.8)
        elif anchor is not None:
            drug = g.nodes[anchor]["drug"]
            edge_rgb = mpl.colors.to_rgb(COLORS[drug])
            if anchor in focal_hubs:
                focal_segments.append(seg)
                focal_colors.append((*edge_rgb, 0.35 + norm_ld * 0.45))
                focal_widths.append(0.7 + norm_ld * 1.4)
            else:
                key_segments.append(seg)
                key_colors.append((*edge_rgb, 0.20 + norm_ld * 0.28))
                key_widths.append(0.35 + norm_ld * 0.75)
        else:
            bg_segments.append(seg)
            if g.nodes[id_a]["drug"] == "bg" and g.nodes[id_b]["drug"] == "bg":
                bg_colors.append((0.78, 0.78, 0.78, 0.04 + norm_ld * 0.04))
                bg_widths.append(0.08 + norm_ld * 0.10)
            else:
                bg_colors.append((0.78, 0.78, 0.78, 0.06 + norm_ld * 0.06))
                bg_widths.append(0.12 + norm_ld * 0.15)

    for segments, colors, widths, z in [
        (bg_segments, bg_colors, bg_widths, 1),
        (key_segments, key_colors, key_widths, 2),
        (focal_segments, focal_colors, focal_widths, 3),
        (cross_segments, cross_colors, cross_widths, 4),
    ]:
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", zorder=z)
            )

    draw_order = sorted(
        g.nodes,
        key=lambda n: _node_draw_size(g.nodes[n]["importance"], imp_max),
    )
    for node in draw_order:
        x, y = pos[node]
        is_labelled = node in label_set
        size = _node_draw_size(g.nodes[node]["importance"], imp_max)
        drug = g.nodes[node]["drug"]
        ax.scatter(
            [x],
            [y],
            s=size,
            c=COLORS[drug],
            edgecolors="#2f2f2f" if is_labelled else "#888888",
            linewidths=0.6 if is_labelled else 0.25,
            alpha=0.98 if is_labelled else (0.38 if drug == "bg" else 0.72),
            zorder=4 + size / 600,
        )

    label_side = _label_above_below(label_nodes, pos)
    texts = []
    for node in label_nodes:
        full_label = g.nodes[node]["label"]
        text = SHORT_LABELS[full_label]
        tx, ty, ha, va = _vertical_label_anchor(
            pos[node], g.nodes[node]["importance"], imp_max, label_side[node]
        )
        texts.append(
            ax.text(
                tx,
                ty,
                text,
                fontsize=NETWORK_LABEL_FS,
                ha=ha,
                va=va,
                fontweight="normal",
                color="#222222",
                zorder=10,
            )
        )
    adjust_text(
        texts,
        x=[pos[n][0] for n in label_nodes],
        y=[pos[n][1] for n in label_nodes],
        ax=ax,
        only_move={"text": "x"},
        force_text=(0.45, 0.0),
        force_points=(0.0, 0.0),
        expand_text=(1.04, 1.0),
        expand_points=(1.0, 1.0),
        lim=30,
    )

    _set_network_view(ax, g, pos, texts)
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Rifampicin", markerfacecolor=COLORS["rif"], markeredgecolor="#666", markersize=4.5, markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Isoniazid", markerfacecolor=COLORS["inh"], markeredgecolor="#666", markersize=4.5, markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Ethambutol", markerfacecolor=COLORS["emb"], markeredgecolor="#666", markersize=4.5, markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Background", markerfacecolor=COLORS["bg"], markeredgecolor="#666", markersize=4.0, markeredgewidth=0.4),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FS,
        handletextpad=0.35,
        columnspacing=1.0,
        borderpad=0.1,
    )


def plot_roc_panel(ax: plt.Axes, roc_path: Path, auc_text: str) -> None:
    roc_df = pd.read_csv(roc_path, delimiter="\t")
    ax.fill_between(
        roc_df["mean_fpr"],
        roc_df["tprs_lower"],
        roc_df["tprs_upper"],
        color=COLORS["roc"],
        alpha=0.10,
        zorder=2,
    )
    ax.plot(roc_df["mean_fpr"], roc_df["mean_tpr"], color=COLORS["roc"], lw=1.8, label=auc_text, zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", fontsize=AXIS_FS)
    ax.set_ylabel("True positive rate", fontsize=AXIS_FS)
    ax.legend(loc="lower right", fontsize=LEGEND_FS, frameon=False, handlelength=1.5)
    ax.tick_params(length=3, labelsize=TICK_FS)


def plot_feature_stability(ax: plt.Axes) -> plt.Axes:
    stats = pd.read_csv(DATA / "feature_stability_summary.tsv", delimiter="\t")

    ax_line = ax.twinx()

    ax.bar(
        stats["selection_frequency"],
        stats["feature_count"],
        color=COLORS["bar"],
        alpha=0.88,
        width=0.72,
        zorder=1,
        edgecolor="none",
    )
    ax_line.plot(
        stats["selection_frequency"],
        stats["normalised_importance_percent"],
        color=COLORS["line"],
        lw=1.6,
        marker="o",
        markersize=3.8,
        markerfacecolor=COLORS["line"],
        markeredgecolor="white",
        markeredgewidth=0.3,
        zorder=3,
    )

    ax.set_ylabel("Feature count", fontsize=AXIS_FS, color=COLORS["bar"])
    ax_line.set_ylabel("Normalised importance (%)", fontsize=AXIS_FS, color=COLORS["line"])
    ax.set_xlabel("Selection frequency (50 rounds)", fontsize=AXIS_FS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}"))
    ax.set_xlim(-1, 51)
    ax.set_ylim(0, stats["feature_count"].max() * 1.08)
    ax_line.set_ylim(0, stats["normalised_importance_percent"].max() * 1.12)
    ax.tick_params(axis="y", labelcolor=COLORS["bar"], labelsize=TICK_FS)
    ax_line.tick_params(axis="y", labelcolor=COLORS["line"], labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.grid(True, linestyle=":", alpha=0.22, linewidth=0.5, zorder=0)
    ax_line.grid(False)
    ax.axvline(5, linestyle="--", color="#aaaaaa", linewidth=0.8, zorder=2)
    ax.text(
        6,
        ax.get_ylim()[1] * 0.94,
        "≥5/50",
        fontsize=ANNOT_FS,
        color="#555555",
        va="top",
    )
    return ax_line


def render() -> None:
    print(f"Fig2 font: {_ACTIVE_FONT} — {_FONT_NOTE}")
    print(
        f"Fig2 type (mpl pt → ~{NATURE_PRINT_W_IN:.1f} in print): "
        f"panel {PANEL_FS:.1f}→{NATURE_PANEL_PT} pt, text {AXIS_FS:.1f}→{NATURE_TEXT_PT} pt"
    )
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        left=0.11,
        right=0.98,
        top=0.97,
        bottom=0.11,
        wspace=0.34,
        hspace=0.34,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_roc_panel(ax_a, DATA / "S450L_model_roc_values.txt", "AUC = 0.975")
    plot_roc_panel(ax_b, DATA / "noS450L_model_roc_values.txt", "AUC = 0.97")
    ax_c_line = plot_feature_stability(ax_c)
    plot_ld_network(ax_d)

    left_col, right_col = [ax_a, ax_c], [ax_b, ax_d]
    _align_figure_columns(fig, gs, left_col, right_col, twins=[(ax_c, ax_c_line)])
    _add_panel_labels(
        fig,
        [ax_a, ax_b, ax_c, ax_d],
        ["A", "B", "C", "D"],
        (left_col, right_col),
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=500, facecolor="white")
    fig.savefig(OUT_TIFF, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
