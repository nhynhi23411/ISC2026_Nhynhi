"""Temporal generalization and feature-component ablation for Project A.

CPU-only, leakage-safe LightGBM experiments on the six complete conditions.
Temporal blocks are evaluated at several historical cut points; ablation uses
the locked final 20-origin block and compares base lags, seen-lag indicators,
missing-run length, and the full observation mask.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from project_a_common import COMPLETE_CONDITIONS, FINAL_TEST_ORIGINS, HORIZONS, MIN_HISTORY, SEED, build_panel, build_series, feat, load_source

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_a_local_results"


def feature_vector(vals, seen, t, mode, component):
    base = feat(vals, seen, t, mode, mask=False)
    if component == "base":
        return base
    mask = feat(vals, seen, t, mode, mask=True)
    if component == "seen_lags":
        return np.r_[base, mask[len(base):-1]]
    if component == "missing_run":
        return np.r_[base, mask[-1]]
    if component == "mask_full":
        return mask
    raise ValueError(component)


def fit_predict(X, y, Xq):
    model = LGBMRegressor(objective="regression", n_estimators=180, learning_rate=.05,
                          num_leaves=15, min_child_samples=20, subsample=.85,
                          colsample_bytree=.85, verbosity=-1, random_state=SEED)
    model.fit(X, y)
    pred = np.asarray(model.predict(Xq), float)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite prediction")
    return pred


def rows_for_block(series, conditions, districts, n, train_end, test_start, test_end, component, horizon, mode):
    Xtr, ytr, Xte, meta = [], [], [], []
    for cond in conditions:
        for district in districts:
            vals, seen = series[(cond, district)]
            for origin in range(MIN_HISTORY, train_end):
                target_t = origin + horizon - 1
                if target_t >= train_end or target_t >= n:
                    continue
                if mode == "reporting_aware" and not seen[target_t]:
                    continue
                target = vals[target_t] if seen[target_t] and np.isfinite(vals[target_t]) else 0.0
                Xtr.append(feature_vector(vals, seen, target_t, mode, component)); ytr.append(target)
            for origin in range(test_start, min(test_end, n - horizon)):
                target_t = origin + horizon - 1
                if not seen[target_t] or not np.isfinite(vals[target_t]):
                    continue
                Xte.append(feature_vector(vals, seen, target_t, mode, component))
                meta.append((cond, district, origin, float(vals[target_t])))
    return np.asarray(Xtr, float), np.asarray(ytr, float), np.asarray(Xte, float), meta


def run(smoke=False, out_dir=OUT):
    started = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, _, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    conditions = list(COMPLETE_CONDITIONS)
    if smoke:
        conditions, districts = conditions[:1], districts[:4]
    temporal_starts = [80, 100, 120, 140]
    if smoke:
        temporal_starts = [100, 140]
    temporal_rows, ablation_rows = [], []

    # Historical temporal blocks: train strictly before each block.
    for block_start in temporal_starts:
        block_end = min(block_start + 12, n - 1)
        for horizon in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                X, y, Xq, meta = rows_for_block(series, conditions, districts, n, block_start, block_start, block_end, "mask_full", horizon, mode)
                if len(y) < 100 or not len(meta):
                    continue
                pred = fit_predict(X, y, Xq)
                for (cond, district, origin, target), p in zip(meta, pred):
                    err = target - float(p)
                    temporal_rows.append({"block_start": block_start, "block_end": block_end, "horizon": horizon, "mode": mode, "condition": cond, "district": district, "origin_t": origin, "target": target, "prediction": float(p), "abs_error": abs(err), "sq_error": err * err})

    # Final locked block ablation: same rows, same model family, different feature components.
    test_start = n - FINAL_TEST_ORIGINS
    components = ("base", "seen_lags", "missing_run", "mask_full")
    for horizon in HORIZONS:
        for mode in ("zero_fill", "reporting_aware"):
            for component in components:
                X, y, Xq, meta = rows_for_block(series, conditions, districts, n, test_start, test_start, n, component, horizon, mode)
                if len(y) < 100 or not len(meta):
                    continue
                pred = fit_predict(X, y, Xq)
                for (cond, district, origin, target), p in zip(meta, pred):
                    err = target - float(p)
                    ablation_rows.append({"component": component, "horizon": horizon, "mode": mode, "condition": cond, "district": district, "origin_t": origin, "target": target, "prediction": float(p), "abs_error": abs(err), "sq_error": err * err})

    temporal = pd.DataFrame(temporal_rows)
    ablation = pd.DataFrame(ablation_rows)
    if temporal.empty or ablation.empty:
        raise RuntimeError("generalization/ablation output is empty")
    temporal_metrics = temporal.groupby(["block_start", "horizon", "mode"]).agg(n=("abs_error", "size"), MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s))))).reset_index()
    ablation_metrics = ablation.groupby(["component", "horizon", "mode"]).agg(n=("abs_error", "size"), MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s))))).reset_index()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(out_dir / "temporal_generalization_predictions.csv", index=False)
    temporal_metrics.to_csv(out_dir / "temporal_generalization_metrics.csv", index=False)
    ablation.to_csv(out_dir / "feature_ablation_predictions.csv", index=False)
    ablation_metrics.to_csv(out_dir / "feature_ablation_metrics.csv", index=False)
    manifest = {"status":"ok", "conditions":conditions, "temporal_block_starts":temporal_starts, "horizons":list(HORIZONS), "components":list(components), "test_origins":FINAL_TEST_ORIGINS, "temporal_prediction_rows":len(temporal), "ablation_prediction_rows":len(ablation), "wall_time_seconds":round(time.time()-started,2), "modal_used":False, "leakage_control":"train targets and features are strictly before each test block"}
    (out_dir / "generalization_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2)); print("TEMPORAL"); print(temporal_metrics.to_string(index=False)); print("ABLATION"); print(ablation_metrics.to_string(index=False))
    return temporal_metrics, ablation_metrics, manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); p.add_argument("--out-dir", type=Path, default=OUT); a = p.parse_args(); run(a.smoke, a.out_dir)
