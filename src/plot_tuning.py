"""
Generate hyperparameter tuning charts from tuning_results.csv.
Produces a single multi-panel figure saved to results/tuning_charts.png.

Usage:
    python src/plot_tuning.py

Requirements:
    pip install matplotlib pandas
"""

import os
import sys
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CSV_PATH    = os.path.join(RESULTS_DIR, "tuning_results.csv")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "tuning_charts.png")

BG        = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT      = "#e8eaf0"
MUTED     = "#6b7280"
ACCENT    = "#f59e0b" 
BLUE      = "#3b82f6"
GREEN     = "#10b981"
RED       = "#ef4444"
PURPLE    = "#8b5cf6"
BASELINE  = "#64748b"

PARAM_COLORS = {
    "softmax_lambda": ACCENT,
    "alpha":          BLUE,
    "gamma":          GREEN,
    "combo":          PURPLE,
    "baseline":       BASELINE,
}

FONT_TITLE  = {"fontsize": 11, "fontweight": "bold", "color": TEXT}
FONT_LABEL  = {"fontsize": 9,  "color": MUTED}
FONT_TICK   = {"labelsize": 8, "colors": MUTED}
FONT_ANNOT  = {"fontsize": 7.5, "color": TEXT}


def load_csv() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run tune_hyperparams.py first.")
        sys.exit(1)
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["alpha"]          = float(r["alpha"])
        r["gamma"]          = float(r["gamma"])
        r["softmax_lambda"] = float(r["softmax_lambda"])
        r["train_win_rate"] = float(r["train_win_rate"])
        r["eval_win_rate"]  = float(r["eval_win_rate"])
        r["q_states"]       = int(r["q_states"])
        r["train_time_s"]   = float(r["train_time_s"])
    return rows


def row_color(row: dict) -> str:
    rid = row["id"]
    if rid == "baseline":
        return PARAM_COLORS["baseline"]
    if rid.startswith("lam"):
        return PARAM_COLORS["softmax_lambda"]
    if rid.startswith("alpha"):
        return PARAM_COLORS["alpha"]
    if rid.startswith("gamma"):
        return PARAM_COLORS["gamma"]
    return PARAM_COLORS["combo"]


def apply_panel_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(**FONT_TICK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.7)


def add_value_labels(ax, bars, fmt="{:.1f}%", offset=0.3):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            **FONT_ANNOT,
        )


# Panel builders

def panel_bar_sweep(ax, rows: list[dict], param: str, param_label: str, color: str):
    """Bar chart for a single-parameter sweep (eval win rate)."""
    baseline = next((r for r in rows if r["id"] == "baseline"), None)
    sweep = [r for r in rows if r["id"].startswith(param.split("_")[0][:5])]
    if baseline and baseline not in sweep:
        sweep = [baseline] + sweep
    sweep.sort(key=lambda r: r[param])

    labels = [str(r[param]) for r in sweep]
    evals  = [r["eval_win_rate"] for r in sweep]
    trains = [r["train_win_rate"] for r in sweep]
    colors = [BASELINE if r["id"] == "baseline" else color for r in sweep]

    x = np.arange(len(sweep))
    w = 0.38

    b1 = ax.bar(x - w/2, trains, w, color=[c + "99" for c in colors], label="Train")
    b2 = ax.bar(x + w/2, evals,  w, color=colors,                      label="Eval")

    add_value_labels(ax, b2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=TEXT)
    ax.set_xlabel(param_label, **FONT_LABEL)
    ax.set_ylabel("Win Rate (%)", **FONT_LABEL)
    ax.set_title(f"{param_label} Sweep", **FONT_TITLE)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID_COL)
    apply_panel_style(ax)

    # Highlight best
    best_idx = int(np.argmax(evals))
    ax.patches[len(sweep) + best_idx].set_edgecolor(ACCENT)
    ax.patches[len(sweep) + best_idx].set_linewidth(2)


def panel_ranking(ax, rows: list[dict]):
    """Horizontal bar chart ranking all configs by eval win rate."""
    ranked = sorted(rows, key=lambda r: r["eval_win_rate"])
    labels = [r["id"] for r in ranked]
    evals  = [r["eval_win_rate"] for r in ranked]
    colors = [row_color(r) for r in ranked]

    y = np.arange(len(ranked))
    bars = ax.barh(y, evals, color=colors, edgecolor="none", height=0.65)

    for bar, val in zip(bars, evals):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", **FONT_ANNOT,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.set_xlabel("Eval Win Rate (%)", **FONT_LABEL)
    ax.set_title("All Configs Ranked by Eval Win Rate", **FONT_TITLE)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    apply_panel_style(ax)

    # Legend for param groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BASELINE,                    label="Baseline"),
        Patch(facecolor=PARAM_COLORS["softmax_lambda"], label="λ sweep"),
        Patch(facecolor=PARAM_COLORS["alpha"],          label="α sweep"),
        Patch(facecolor=PARAM_COLORS["gamma"],          label="γ sweep"),
        Patch(facecolor=PARAM_COLORS["combo"],          label="Combo"),
    ]
    ax.legend(
        handles=legend_elements, fontsize=7,
        facecolor=PANEL_BG, labelcolor=TEXT, edgecolor=GRID_COL,
        loc="lower right",
    )


def panel_training_curves(ax, rows: list[dict]):
    """Training win-rate curves for all configs (loaded from curve_*.json)."""
    plotted = 0
    for row in rows:
        curve_path = os.path.join(RESULTS_DIR, f"curve_{row['id']}.json")
        if not os.path.exists(curve_path):
            continue
        with open(curve_path) as f:
            curve = json.load(f)
        xs = [pt["battle"] for pt in curve]
        ys = [pt["win_rate"] for pt in curve]
        color = row_color(row)
        lw    = 2.0 if row["id"] == "baseline" else 1.0
        alpha = 1.0 if row["id"] == "baseline" else 0.7
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, label=row["id"])
        plotted += 1

    ax.set_xlabel("Training Battles", **FONT_LABEL)
    ax.set_ylabel("Win Rate (%)", **FONT_LABEL)
    ax.set_title("Training Win Rate Curves", **FONT_TITLE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    if plotted > 0:
        ax.legend(
            fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT,
            edgecolor=GRID_COL, ncol=2, loc="lower right",
        )
    else:
        ax.text(
            0.5, 0.5, "curve_*.json files not found\n(run tune_hyperparams.py first)",
            ha="center", va="center", transform=ax.transAxes,
            color=MUTED, fontsize=9,
        )
    apply_panel_style(ax)


def panel_scatter(ax, rows: list[dict]):
    """Scatter: Q-states discovered vs eval win rate, sized by training time."""
    times  = np.array([r["train_time_s"] for r in rows])
    sizes  = 40 + 160 * (times - times.min()) / ((times.max() - times.min()) + 1e-9)

    for row, sz in zip(rows, sizes):
        ax.scatter(
            row["q_states"], row["eval_win_rate"],
            color=row_color(row), s=sz,
            edgecolors=BG, linewidths=0.8, zorder=3,
        )
        ax.annotate(
            row["id"],
            (row["q_states"], row["eval_win_rate"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=6.5, color=MUTED,
        )

    ax.set_xlabel("Q-table States Discovered", **FONT_LABEL)
    ax.set_ylabel("Eval Win Rate (%)", **FONT_LABEL)
    ax.set_title("State Coverage vs Performance\n(bubble size = training time)", **FONT_TITLE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    apply_panel_style(ax)


# Main

def main():
    rows = load_csv()
    print(f"Loaded {len(rows)} configs from {CSV_PATH}")

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.07, right=0.97, top=0.93, bottom=0.06)

    ax_lam    = fig.add_subplot(gs[0, 0])
    ax_alpha  = fig.add_subplot(gs[0, 1])
    ax_gamma  = fig.add_subplot(gs[0, 2])
    ax_rank   = fig.add_subplot(gs[1, :2])
    ax_curve  = fig.add_subplot(gs[1, 2])
    ax_scatter= fig.add_subplot(gs[2, :])

    # Lambda sweep 
    lam_rows = [r for r in rows if r["id"] == "baseline" or r["id"].startswith("lam")]
    if lam_rows:
        panel_bar_sweep(ax_lam, lam_rows, "softmax_lambda", "Softmax λ", ACCENT)

    # Alpha sweep 
    alpha_rows = [r for r in rows if r["id"] == "baseline" or r["id"].startswith("alpha")]
    if alpha_rows:
        panel_bar_sweep(ax_alpha, alpha_rows, "alpha", "Learning Rate α", BLUE)

    # Gamma sweep 
    gamma_rows = [r for r in rows if r["id"] == "baseline" or r["id"].startswith("gamma")]
    if gamma_rows:
        panel_bar_sweep(ax_gamma, gamma_rows, "gamma", "Discount Factor γ", GREEN)

    # Ranking 
    panel_ranking(ax_rank, rows)

    # Training curves 
    panel_training_curves(ax_curve, rows)

    # Scatter 
    panel_scatter(ax_scatter, rows)

    # Title 
    best = max(rows, key=lambda r: r["eval_win_rate"])
    fig.suptitle(
        f"Hyperparameter Tuning — Q-Learning Pokémon Battle Agent\n"
        f"Best: [{best['id']}]  α={best['alpha']}  γ={best['gamma']}  "
        f"λ={best['softmax_lambda']}  →  {best['eval_win_rate']:.1f}% eval win rate",
        fontsize=12, fontweight="bold", color=TEXT, y=0.97,
    )

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved to: {OUTPUT_PATH}")

    # Print quick table 
    print(f"\n{'ID':<16} {'α':>5} {'γ':>5} {'λ':>5}  {'Train%':>8} {'Eval%':>8}  States")
    print("-" * 65)
    for r in sorted(rows, key=lambda x: -x["eval_win_rate"]):
        print(
            f"{r['id']:<16} {r['alpha']:>5} {r['gamma']:>5} "
            f"{r['softmax_lambda']:>5}  "
            f"{r['train_win_rate']:>7.1f}%  {r['eval_win_rate']:>7.1f}%  {r['q_states']}"
        )


if __name__ == "__main__":
    main()