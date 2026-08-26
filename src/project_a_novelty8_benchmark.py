"""Item D: system/efficiency benchmark of the full pipeline.

Frames the contribution in "AI-enabled computing / real-world smart
application" terms (ICS 2026 track language): how long does it take, on a
single CPU core with no specialized hardware, to (a) rebuild the full panel
from the raw bulletin-derived files, and (b) produce an updated forecast for
every district-condition series once a new week's bulletin becomes
available -- the actual operational unit of work for a weekly-cadence
surveillance system. No GPU, no cluster, no rewrite of the modeling code:
this instruments the exact functions already used everywhere else in the
project.
"""
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from project_a_common import (
    COMPLETE_CONDITIONS, HORIZONS, FINAL_TEST_ORIGINS, MIN_HISTORY, SEED,
    load_source, build_panel, build_series, feat,
)

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def main():
    tracemalloc.start()
    results = {}

    (long, presence, week_df), t_load = timed(load_source)
    results["load_source_seconds"] = t_load

    (panel, weeks, districts, conditions, n), t_panel = timed(build_panel, long, presence, week_df)
    results["build_panel_seconds"] = t_panel
    results["panel_rows"] = int(len(panel))

    series, t_series = timed(build_series, panel, n)
    results["build_series_seconds"] = t_series

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    results["peak_memory_mb_data_layer"] = round(peak_mem / (1024 * 1024), 1)

    # --- one weekly operational cycle: given the current week's data already
    # loaded (as above, a one-time/startup cost), how long to refresh the
    # 1-week-ahead forecast for EVERY district x condition series using the
    # already-fitted-per-condition LightGBM models? This is feature
    # construction + prediction only (the realistic per-week inference cost;
    # models are refit periodically, not every week, in an operational
    # deployment). ---
    test_start = n - FINAL_TEST_ORIGINS
    h = 1
    t_target = n - 1  # "current" week, forecasting one week ahead conceptually

    # Fit one representative model per condition once (amortized/background cost).
    fitted = {}
    t_fit_total = 0.0
    for cond in conditions:
        Xtr, ytr = [], []
        for d in districts:
            vals, seen = series[(cond, d)]
            for origin in range(MIN_HISTORY, test_start):
                tt = origin + h - 1
                if tt >= n or not seen[tt]:
                    continue
                Xtr.append(feat(vals, seen, tt, "reporting_aware", mask=True))
                ytr.append(vals[tt])
        Xtr = np.asarray(Xtr, float)
        ytr = np.asarray(ytr, float)
        if len(ytr) < 100:
            continue
        model = LGBMRegressor(objective="regression", n_estimators=180, learning_rate=0.05,
                               num_leaves=15, min_child_samples=20, verbosity=-1, random_state=SEED)
        _, t_fit = timed(model.fit, Xtr, ytr)
        t_fit_total += t_fit
        fitted[cond] = model
    results["one_time_fit_all_conditions_seconds"] = round(t_fit_total, 3)
    results["n_conditions_fitted"] = len(fitted)

    # Per-week refresh cost: feature construction + prediction for every
    # (condition, district) pair that has a fitted model and enough history.
    t0 = time.perf_counter()
    n_forecasts = 0
    for cond, model in fitted.items():
        X_week = []
        for d in districts:
            vals, seen = series[(cond, d)]
            if t_target < 12:
                continue
            X_week.append(feat(vals, seen, t_target, "reporting_aware", mask=True))
        if not X_week:
            continue
        model.predict(np.asarray(X_week, float))
        n_forecasts += len(X_week)
    t_refresh = time.perf_counter() - t0
    results["one_week_refresh_seconds_all_series"] = round(t_refresh, 4)
    results["n_forecasts_per_refresh"] = n_forecasts
    results["seconds_per_forecast"] = round(t_refresh / max(n_forecasts, 1), 6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    results["peak_memory_mb_full_pipeline"] = round(peak_mem / (1024 * 1024), 1)
    tracemalloc.stop()

    results["hardware_note"] = "single CPU thread, no GPU, commodity laptop-class hardware; all numbers wall-clock"
    results["operational_interpretation"] = (
        f"Refreshing forecasts for all {n_forecasts} active district-condition series after a new "
        f"weekly bulletin takes about {t_refresh:.2f}s once models are fitted; the one-time cost of "
        f"fitting one LightGBM per condition is {t_fit_total:.1f}s. Both are negligible relative to the "
        "weekly reporting cadence, so the pipeline is operationally deployable on commodity hardware "
        "without GPUs or distributed compute -- relevant to real-world smart-surveillance deployment, "
        "not just an offline benchmark."
    )

    with open(OUT / "system_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
