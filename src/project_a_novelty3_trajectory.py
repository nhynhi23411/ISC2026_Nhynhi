"""Novelty #3: trajectory-distortion case study.

Directly illustrates the paper's title claim ("Missing Does Not Mean Zero")
by reconstructing the real provincial-level trajectory of two rotating
conditions -- Measles (outbreak-sensitive) and TB (chronic, should not show
spurious "outbreaks" or "eliminations") -- under zero-fill vs
reporting-aware (last-observation-carried-forward) preprocessing, and
quantifying how often and how much the two reconstructions disagree about
the apparent trend.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_a_common import load_source, build_panel

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
FIG_DIR = Path(r"D:\Thầy Khánh\ISC\project_a_figures")
FIG_DIR.mkdir(exist_ok=True)
CASE_CONDITIONS = ("Measles", "TB")
MA_WINDOW = 8

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})


def provincial_reconstruction(panel, cond, n):
    g = panel[panel.condition == cond]
    present = g.groupby("t").condition_present.first().reindex(range(n)).astype(bool).to_numpy()
    total = g.groupby("t").apply(
        lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
        include_groups=False,
    ).reindex(range(n)).to_numpy(dtype=float)

    zero_fill = np.where(np.isfinite(total), total, 0.0)
    aware = np.empty(n)
    last = 0.0
    observed_mask = np.isfinite(total)
    for t in range(n):
        if observed_mask[t]:
            last = total[t]
        aware[t] = last
    return zero_fill, aware, observed_mask, present


def moving_average(x, w=MA_WINDOW):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.mean(x[lo:i + 1])
    return out


def trend_direction(ma):
    d = np.diff(ma)
    return np.sign(d)


def main():
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    t = np.arange(n)

    summary = {}
    fig, axes = plt.subplots(len(CASE_CONDITIONS), 1, figsize=(11, 4.6 * len(CASE_CONDITIONS)), sharex=True)
    if len(CASE_CONDITIONS) == 1:
        axes = [axes]

    for ax, cond in zip(axes, CASE_CONDITIONS):
        zero_fill, aware, observed_mask, present = provincial_reconstruction(panel, cond, n)

        ma_zero = moving_average(zero_fill)
        ma_aware = moving_average(aware)
        dir_zero = trend_direction(ma_zero)
        dir_aware = trend_direction(ma_aware)
        disagree = dir_zero != dir_aware
        pct_disagree = float(np.mean(disagree))

        rel_diff = np.abs(zero_fill - aware) / np.maximum(aware, 1.0)
        material_diff = rel_diff > 0.25
        pct_material = float(np.mean(material_diff[~observed_mask]))  # only meaningful where censored

        summary[cond] = {
            "weeks_present_of_164": int(present.sum()),
            "pct_8wk_trend_direction_disagreement": round(pct_disagree, 4),
            "pct_censored_weeks_with_over25pct_relative_diff": round(pct_material, 4),
            "n_censored_weeks": int((~observed_mask).sum()),
        }

        ax.plot(t, zero_fill, color="#c53030", lw=1.1, label="Zero-filled reconstruction", alpha=0.85)
        ax.plot(t, aware, color="#2b6cb0", lw=1.3, label="Reporting-aware (carried forward)")
        obs_t = t[observed_mask]
        ax.scatter(obs_t, zero_fill[observed_mask], s=10, color="#1a365d", zorder=5, label="Actually observed")
        # Shade weeks where the condition was censored (absent from that week's table).
        censored = ~observed_mask
        ax.fill_between(t, 0, ax.get_ylim()[1] if False else max(zero_fill.max(), aware.max()) * 1.05,
                         where=censored, color="grey", alpha=0.12, step="mid", label="Structurally unreported week")
        ax.set_title(f"{cond}: provincial weekly cases -- zero-fill vs reporting-aware reconstruction")
        ax.set_ylabel("Provincial cases")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Bulletin week index (t = 0..163)")
    fig.suptitle("Figure 5. Trajectory distortion under zero-fill vs reporting-aware reconstruction", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_trajectory_distortion.png")
    plt.close(fig)

    with open(OUT / "trajectory_distortion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
