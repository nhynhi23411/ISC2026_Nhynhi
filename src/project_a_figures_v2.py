"""Redesigned Project A figures — consistent, cleaner visual style.

Improves on project_a_figures.py: unified color palette, no chartjunk (top/
right spines removed), subtle gridlines, consistent typography, better
labels. Also adds two new figures for the "tech upgrade" round: the hazard
curve (item C) and the propensity-feature vs IPW-weighting comparison
(item A). Figure 1 (pipeline) is intentionally left to an external
image-generation tool (see the ChatGPT prompt delivered separately); this
script does not touch it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS, DATA_DIR

FIG_DIR = Path(r"D:\Thầy Khánh\ISC\project_a_figures")
RESULTS = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
EXTENDED = Path(r"D:\Thầy Khánh\ISC\project_a_extended_modal_results")

# ---------------------------------------------------------------- house style
BLUE = "#2563EB"
BLUE_SOFT = "#93C5FD"
RED = "#DC2626"
RED_SOFT = "#FCA5A5"
AMBER = "#D97706"
GREEN = "#059669"
PURPLE = "#7C3AED"
GRID = "#E5E7EB"
INK = "#111827"
SUBINK = "#6B7280"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10.5,
    "figure.dpi": 170,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "text.color": INK,
    "xtick.color": SUBINK,
    "ytick.color": SUBINK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def clean_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(grid_axis != "y")
    ax.grid(axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def suptitle_block(fig, title, subtitle=None, y=0.985):
    # Single-line, safely-inside-canvas title only. The descriptive subtitle
    # lives in the Word-doc figure caption instead of stacking a second text
    # object in figure coordinates (which is fragile across different
    # subplot-grid heights and was overlapping the title).
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=y, color=INK)


# ============================================================ FIGURE 2
def fig2_reporting_timeline():
    presence = pd.read_csv(DATA_DIR / "condition_presence_matrix.csv").set_index("condition")
    presence_bool = presence.astype(str).eq("Y")
    weeks = list(presence_bool.columns)
    t = np.arange(len(weeks))

    complete = [c for c in COMPLETE_CONDITIONS if c in presence_bool.index]
    rotating = [c for c in presence_bool.index if c not in complete]
    rotating_sorted = presence_bool.loc[rotating].sum(axis=1).sort_values(ascending=False).index.tolist()
    ordered = complete + rotating_sorted
    mat = presence_bool.loc[ordered].astype(int).values
    n_present = presence_bool.sum(axis=0).values
    transition_idx = weeks.index("2023w32") if "2023w32" in weeks else None

    fig, (ax_top, ax_heat) = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.7], "hspace": 0.06}
    )
    suptitle_block(fig, "Reporting completeness over time", "which of the 16 conditions appear in each weekly bulletin, and the 2023w32 format change")

    ax_top.plot(t, n_present, color=BLUE, lw=1.6, zorder=3)
    ax_top.fill_between(t, 0, n_present, color=BLUE, alpha=0.06, zorder=1)
    ax_top.axhline(6, color=SUBINK, ls=(0, (4, 3)), lw=1, zorder=2)
    ax_top.text(len(weeks) - 2, 6.5, "6 always-complete", ha="right", fontsize=8, color=SUBINK)
    if transition_idx is not None:
        ax_top.axvline(transition_idx, color=RED, ls=(0, (4, 3)), lw=1.4, zorder=2)
        ax_top.text(transition_idx + 2, 15.4, "2023w32: rotating\ntop-10 layout begins", fontsize=8, color=RED)
    ax_top.set_ylabel("Conditions\nin bulletin")
    ax_top.set_ylim(0, 17)
    clean_axes(ax_top)

    im = ax_heat.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1.35, interpolation="nearest",
                         extent=[0, len(weeks), len(ordered), 0])
    ax_heat.axhline(len(complete), color=RED, lw=1.3)
    if transition_idx is not None:
        ax_heat.axvline(transition_idx, color=RED, ls=(0, (4, 3)), lw=1.4)
    ax_heat.set_yticks(np.arange(len(ordered)) + 0.5)
    ax_heat.set_yticklabels(ordered, fontsize=8)
    ax_heat.set_xlabel("Bulletin week index (t = 0..163)")
    ax_heat.tick_params(left=False, bottom=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    label_box = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2.5, boxstyle="round,pad=0.3")
    ax_heat.text(3, len(complete) / 2, "ALWAYS COMPLETE", fontsize=8, va="center", color=BLUE, fontweight="bold", bbox=label_box)
    ax_heat.text(3, len(complete) + 1.2, "ROTATING (top-10 by rank)", fontsize=8, va="center", color=BLUE, fontweight="bold", bbox=label_box)

    tick_idx = list(range(0, len(weeks), 20))
    ax_heat.set_xticks([i + 0.5 for i in tick_idx])
    ax_heat.set_xticklabels([weeks[i] for i in tick_idx], rotation=40, ha="right", fontsize=8)

    fig.savefig(FIG_DIR / "figure2_reporting_timeline.png")
    plt.close(fig)


# ============================================================ FIGURE 3
def fig3_paired_errors():
    m = pd.read_csv(RESULTS / "local_model_metrics.csv")
    m = m[(m.model == "lightgbm_mask") & (m.condition.isin(COMPLETE_CONDITIONS))]
    piv = m.pivot_table(index=["condition", "horizon"], columns="mode", values="MAE").reset_index()
    piv = piv.sort_values(["horizon", "condition"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=False)
    suptitle_block(fig, "Paired forecast error: zero-fill vs. reporting-aware", "six always-complete conditions, LightGBM + mask, locked final test block")
    for ax, h in zip(axes, (1, 2, 4)):
        sub = piv[piv.horizon == h]
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub.zero_fill, width, label="Zero-filled", color=RED_SOFT, edgecolor=RED, linewidth=0.8, zorder=3)
        ax.bar(x + width / 2, sub.reporting_aware, width, label="Reporting-aware", color=BLUE_SOFT, edgecolor=BLUE, linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.condition, rotation=38, ha="right", fontsize=8)
        ax.set_title(f"Horizon = {h} week(s)", fontsize=10.5)
        if ax is axes[0]:
            ax.set_ylabel("MAE")
        clean_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG_DIR / "figure3_paired_errors.png")
    plt.close(fig)


# ============================================================ FIGURE 4
def fig4_error_and_coverage():
    m = pd.read_csv(RESULTS / "local_model_metrics.csv")
    m = m[(m.model == "lightgbm_mask") & (m["mode"] == "reporting_aware")]
    heat = m.pivot_table(index="condition", columns="horizon", values="MAE")
    heat = heat.loc[heat.mean(axis=1).sort_values().index]

    q = pd.read_csv(RESULTS / "quantile_uncertainty_metrics.csv")
    q = q[q["mode"] == "reporting_aware"]
    cov80 = q.groupby("condition").coverage80.mean().reindex(heat.index)
    cov90 = q.groupby("condition").coverage90.mean().reindex(heat.index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    suptitle_block(fig, "Error and interval coverage by disease and horizon", "LightGBM + mask, reporting-aware; quantile-regression intervals")

    from matplotlib.colors import PowerNorm
    im = axes[0].imshow(heat.values, aspect="auto", cmap="YlOrRd", norm=PowerNorm(gamma=0.45))
    axes[0].set_yticks(range(len(heat.index)))
    axes[0].set_yticklabels(heat.index, fontsize=8.5)
    axes[0].set_xticks(range(len(heat.columns)))
    axes[0].set_xticklabels([f"h = {c}" for c in heat.columns], fontsize=9)
    axes[0].set_title("MAE by disease and horizon", fontsize=10.5)
    axes[0].tick_params(left=False, bottom=False)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)

    y = np.arange(len(heat.index))
    axes[1].barh(y - 0.19, cov80.values, height=0.36, label="80% interval", color=BLUE, zorder=3)
    axes[1].barh(y + 0.19, cov90.values, height=0.36, label="90% interval", color=PURPLE, zorder=3)
    axes[1].axvline(0.80, color=BLUE, ls=(0, (3, 2)), lw=1.2, zorder=2)
    axes[1].axvline(0.90, color=PURPLE, ls=(0, (3, 2)), lw=1.2, zorder=2)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(heat.index, fontsize=8.5)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("Quantile-interval coverage\nvs. nominal 80% / 90%", fontsize=10.5)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.28, 1.16), ncol=2)
    axes[1].set_xlim(0, 1.02)
    clean_axes(axes[1], grid_axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "figure4_error_coverage.png")
    plt.close(fig)


# ============================================================ FIGURE 5
def fig5_trajectory_distortion():
    from project_a_common import load_source, build_panel
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    t = np.arange(n)
    CASE_CONDITIONS = ("Measles", "TB")

    def provincial_reconstruction(cond):
        g = panel[panel.condition == cond]
        total = g.groupby("t").apply(
            lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
            include_groups=False,
        ).reindex(range(n)).to_numpy(dtype=float)
        observed_mask = np.isfinite(total)
        zero_fill = np.where(observed_mask, total, 0.0)
        aware = np.empty(n)
        last = 0.0
        for i in range(n):
            if observed_mask[i]:
                last = total[i]
            aware[i] = last
        return zero_fill, aware, observed_mask

    fig, axes = plt.subplots(len(CASE_CONDITIONS), 1, figsize=(11.5, 4.6 * len(CASE_CONDITIONS)), sharex=True)
    suptitle_block(fig, "Trajectory distortion: zero-fill vs. reporting-aware reconstruction", "provincial weekly case counts for two rotating conditions")

    for ax, cond in zip(axes, CASE_CONDITIONS):
        zero_fill, aware, observed_mask = provincial_reconstruction(cond)
        ymax = max(zero_fill.max(), aware.max()) * 1.08
        ax.fill_between(t, 0, ymax, where=~observed_mask, color=SUBINK, alpha=0.08, step="mid", zorder=0,
                         label="Structurally unreported week")
        ax.plot(t, zero_fill, color=RED, lw=1.2, alpha=0.9, zorder=2, label="Zero-filled reconstruction")
        ax.plot(t, aware, color=BLUE, lw=1.7, zorder=3, label="Reporting-aware (carried forward)")
        obs_t = t[observed_mask]
        ax.scatter(obs_t, zero_fill[observed_mask], s=9, color=INK, zorder=4, label="Actually observed")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Provincial cases")
        ax.set_title(cond, fontsize=11, loc="left")
        clean_axes(ax)
        if cond == CASE_CONDITIONS[0]:
            ax.legend(fontsize=8, loc="upper left", ncol=2)

    axes[-1].set_xlabel("Bulletin week index (t = 0..163)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "figure5_trajectory_distortion.png")
    plt.close(fig)


# ============================================================ FIGURE 6
def fig6_district_gain():
    df = pd.read_csv(RESULTS / "propensity_gain_by_district.csv")
    d = df[df.condition_group == "rotating_stress"].sort_values("mean_delta", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, 10.5))
    suptitle_block(fig, "Propensity-feature gain by district", "rotating-stress conditions, 95% bootstrap CI (mask_MAE − propensity_MAE)")
    colors = [RED if v < 0 else BLUE for v in d.mean_delta]
    edge = [RED for _ in d.mean_delta]
    y = np.arange(len(d))
    xerr = np.abs(np.c_[d.mean_delta - d.ci_low, d.ci_high - d.mean_delta].T)
    ax.barh(y, d.mean_delta, xerr=xerr, color=colors, alpha=0.85,
            ecolor="#9CA3AF", capsize=2, height=0.68, error_kw={"elinewidth": 1})
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d.district, fontsize=8)
    ax.set_xlabel("Mean delta  (positive = propensity feature helps)")
    clean_axes(ax, grid_axis="x")
    legend_handles = [mpatches.Patch(color=BLUE, label="Positive gain"), mpatches.Patch(color=RED, label="Negative gain")]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "figure6_propensity_gain_by_district.png")
    plt.close(fig)


# ============================================================ FIGURE 7 (new) hazard curve
def fig7_hazard_curve():
    grid = pd.read_csv(RESULTS / "hazard_shape_by_duration.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    suptitle_block(fig, "Hazard of reappearing after censorship", "discrete-time hazard model, evaluated at the median pre-spell volume")
    ax.plot(grid.duration, grid.hazard_at_median_volume * 100, color=BLUE, lw=2.2, marker="o", markersize=4, zorder=3)
    ax.fill_between(grid.duration, 0, grid.hazard_at_median_volume * 100, color=BLUE, alpha=0.08, zorder=1)
    ax.set_xlabel("Consecutive weeks already absent (duration)")
    ax.set_ylabel("P(reappears next week)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(1, grid.duration.max())
    ax.set_ylim(0, None)
    clean_axes(ax)
    ax.annotate("Longer absence -> steadily\nlower chance of returning soon",
                xy=(12, grid.hazard_at_median_volume.iloc[11] * 100), xytext=(11, 15),
                fontsize=9, color=SUBINK,
                arrowprops=dict(arrowstyle="->", color=SUBINK, lw=1))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "figure7_hazard_curve.png")
    plt.close(fig)


# ============================================================ FIGURE 8 (new) IPW vs propensity-feature
def fig8_ipw_vs_feature():
    prop = pd.read_csv(RESULTS / "propensity_vs_mask_ensemble_bootstrap.csv")
    ipw = pd.read_csv(RESULTS / "ipw_vs_mask_ensemble_bootstrap.csv")
    prop["approach"] = "Propensity as feature"
    ipw["approach"] = "Propensity as IPW weight"
    ipw = ipw.rename(columns={"delta": "mean_delta"}) if "delta" in ipw.columns else ipw
    both = pd.concat([
        prop[["condition_group", "horizon", "mean_delta", "ci_low", "ci_high", "approach"]],
        ipw[["condition_group", "horizon", "mean_delta", "ci_low", "ci_high", "approach"]],
    ], ignore_index=True)
    both["label"] = both.condition_group.map({"complete_six": "6 complete", "rotating_stress": "10 rotating"}) + " · h=" + both.horizon.astype(str)
    both = both.sort_values(["condition_group", "horizon"])

    fig, ax = plt.subplots(figsize=(10, 6))
    suptitle_block(fig, "Same causal quantity, opposite outcome", "propensity used as a model feature vs. as an inverse-propensity training weight")
    labels = both.label.unique()
    y_base = np.arange(len(labels))
    width = 0.32
    for i, approach in enumerate(["Propensity as feature", "Propensity as IPW weight"]):
        sub = both[both.approach == approach].set_index("label").reindex(labels)
        color = BLUE if approach.endswith("feature") else AMBER
        offset = -width / 2 if i == 0 else width / 2
        xerr = np.abs(np.c_[sub.mean_delta - sub.ci_low, sub.ci_high - sub.mean_delta].T)
        ax.barh(y_base + offset, sub.mean_delta, height=width, xerr=xerr, color=color, alpha=0.85,
                label=approach, ecolor="#9CA3AF", capsize=2, error_kw={"elinewidth": 1}, zorder=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean delta vs. mask-only LightGBM (positive = helps)")
    clean_axes(ax, grid_axis="x")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "figure8_ipw_vs_feature.png")
    plt.close(fig)


if __name__ == "__main__":
    fig2_reporting_timeline()
    fig3_paired_errors()
    fig4_error_and_coverage()
    fig5_trajectory_distortion()
    fig6_district_gain()
    fig7_hazard_curve()
    fig8_ipw_vs_feature()
    print("done")
