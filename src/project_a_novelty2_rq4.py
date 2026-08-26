"""Novelty #2 -- brief RQ4: "Can prediction uncertainty identify unreliable
district-week forecasts?"

Attaches the pre-forecast missing-run-length (n_missing_8: number of the 8
weeks before the target week where that district-condition series was NOT
observed) to every existing quantile-interval prediction, then checks
whether recent non-reporting predicts (a) larger point-forecast error and
(b) worse interval calibration. No retraining: reuses
quantile_interval_predictions.csv from project_a_local_uncertainty.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS, load_source, build_panel, build_series

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def n_missing_8_for_row(seen, target_t):
    return int(np.sum(~seen[max(0, target_t - 8):target_t]))


def main():
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)

    df = pd.read_csv(OUT / "quantile_interval_predictions.csv")
    df["target_t"] = df.origin_t + df.horizon - 1
    df["abs_error"] = (df.target - df.median_prediction).abs()

    cache = {}
    n_missing = np.empty(len(df), dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        key = (row.condition, row.district)
        if key not in cache:
            cache[key] = series[key][1]  # seen array
        n_missing[i] = n_missing_8_for_row(cache[key], int(row.target_t))
    df["n_missing_8"] = n_missing

    # complete_six conditions have n_missing_8 == 0 by construction (they are
    # never absent), and they include the highest-volume series in the
    # dataset. Pooling them with rotating conditions confounds "how long has
    # this series been unreported" with "which disease is this" (a pure scale
    # effect). The causal question -- does recent non-reporting predict worse
    # forecasts -- only has genuine within-series variation among rotating
    # conditions, so that is the primary analysis; complete_six is reported
    # separately as a labeled scale-confound control, not pooled in.
    df["condition_group"] = np.where(df.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")
    # Scale-normalized error, robust to the same confound: divide by each
    # series' own median observed level (Hyndman-Koehler-style scaling).
    series_scale = {}
    for key, (vals, seen) in series.items():
        obs = vals[seen]
        series_scale[key] = float(np.median(obs)) if len(obs) and np.median(obs) > 0 else 1.0
    df["series_scale"] = [series_scale[(c, d)] for c, d in zip(df.condition, df.district)]
    df["scaled_abs_error"] = df.abs_error / df.series_scale
    df["scaled_width80"] = df.width80 / df.series_scale

    bins = [-1, 0, 1, 2, 4, 8]
    labels = ["0", "1", "2", "3-4", "5-8"]
    df["missing_bin"] = pd.cut(df.n_missing_8, bins=bins, labels=labels)

    by_bin = df.groupby(["condition_group", "mode", "missing_bin"], observed=True).agg(
        n=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        mean_scaled_abs_error=("scaled_abs_error", "mean"),
        mean_width80=("width80", "mean"),
        mean_scaled_width80=("scaled_width80", "mean"),
        coverage80=("in80", "mean"),
        mean_width90=("width90", "mean"),
        coverage90=("in90", "mean"),
    ).reset_index()
    by_bin.to_csv(OUT / "rq4_uncertainty_by_missing_run.csv", index=False)

    rot = df[df.condition_group == "rotating_stress"]
    corr_error = float(rot.n_missing_8.corr(rot.scaled_abs_error, method="spearman"))
    corr_width80 = float(rot.n_missing_8.corr(rot.scaled_width80, method="spearman"))
    corr_width90 = float(rot.n_missing_8.corr(rot.width90 / rot.series_scale, method="spearman"))

    summary = {
        "n_predictions": int(len(df)),
        "n_rotating_only": int(len(rot)),
        "scope": "rotating_stress conditions only; complete_six excluded because n_missing_8=0 for them by construction (confounds run-length with disease scale)",
        "spearman_missing_run_vs_scaled_abs_error": corr_error,
        "spearman_missing_run_vs_scaled_width80": corr_width80,
        "spearman_missing_run_vs_scaled_width90": corr_width90,
        "interpretation": (
            "Positive correlations (computed on rotating conditions only, using "
            "series-scale-normalized error/width to remove the disease-scale "
            "confound) mean forecasts made after a longer run of non-reporting "
            "have both larger relative errors and wider relative quantile "
            "intervals -- i.e. interval width is a usable signal of which "
            "district-week forecasts are less reliable (brief RQ4). A naive, "
            "unscaled, all-condition version of this correlation is confounded "
            "by scale and gives the opposite sign; see rq4_uncertainty_by_missing_run.csv "
            "condition_group='complete_six' vs 'rotating_stress' rows."
        ),
    }
    with open(OUT / "rq4_uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(by_bin.to_string(index=False))


if __name__ == "__main__":
    main()
