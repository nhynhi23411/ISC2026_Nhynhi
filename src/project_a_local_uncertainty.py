"""Split/quantile forecast intervals for Project A, run locally.

The Modal extended run already showed that naive empirical residual intervals
are under-covered (see EXTENDED_MODAL_RUN_REPORT.md, section 3). This script
adds quantile-regression LightGBM intervals (pinball loss at the 10/90 and
5/95 percentiles) as the more defensible uncertainty method the brief asks
for ("Uncertainty | Quantile LightGBM or split conformal intervals"), fit and
evaluated with the same rolling-origin discipline as every other Project A
model. No Modal job is involved.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from project_a_common import HORIZONS, FINAL_TEST_ORIGINS, MIN_HISTORY, SEED, load_source, build_panel, build_series, feat

warnings.filterwarnings("ignore")

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
OUT.mkdir(exist_ok=True)


def fit_quantile(Xtr, ytr, alpha):
    model = LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=180, learning_rate=0.05,
        num_leaves=15, min_child_samples=20, verbosity=-1, random_state=SEED,
    )
    model.fit(Xtr, ytr)
    return model


def main():
    t0 = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS

    rows = []
    for cond in conditions:
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                mask = True
                Xtr, ytr = [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(MIN_HISTORY, test_start):
                        tt = origin + h - 1
                        if tt >= n:
                            continue
                        if mode == "reporting_aware" and not seen[tt]:
                            continue
                        target = vals[tt] if (seen[tt] and np.isfinite(vals[tt])) else 0.0
                        Xtr.append(feat(vals, seen, tt, mode, mask))
                        ytr.append(target)
                Xtr = np.asarray(Xtr, float)
                ytr = np.asarray(ytr, float)
                if len(ytr) < 100:
                    continue

                Xte, meta = [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(test_start, n - h):
                        tt = origin + h - 1
                        if tt >= n or not seen[tt]:
                            continue
                        Xte.append(feat(vals, seen, tt, mode, mask))
                        meta.append((district, origin, float(vals[tt])))
                if not meta:
                    continue
                Xte = np.asarray(Xte, float)

                try:
                    lo80 = fit_quantile(Xtr, ytr, 0.10).predict(Xte)
                    hi80 = fit_quantile(Xtr, ytr, 0.90).predict(Xte)
                    lo90 = fit_quantile(Xtr, ytr, 0.05).predict(Xte)
                    hi90 = fit_quantile(Xtr, ytr, 0.95).predict(Xte)
                    med = fit_quantile(Xtr, ytr, 0.50).predict(Xte)
                except Exception as exc:
                    print(f"QUANTILE FIT FAILED cond={cond} h={h} mode={mode}: {exc}")
                    continue

                lo80 = np.clip(lo80, 0, None)
                lo90 = np.clip(lo90, 0, None)
                hi80 = np.maximum(hi80, lo80)
                hi90 = np.maximum(hi90, lo90)

                for i, (district, origin, target) in enumerate(meta):
                    rows.append({
                        "condition": cond, "district": district, "origin_t": origin,
                        "horizon": h, "mode": mode, "model": "lightgbm_quantile_mask",
                        "target": target, "median_prediction": float(med[i]),
                        "lo80": float(lo80[i]), "hi80": float(hi80[i]),
                        "lo90": float(lo90[i]), "hi90": float(hi90[i]),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty quantile interval output")
    df["in80"] = (df.target >= df.lo80) & (df.target <= df.hi80)
    df["width80"] = df.hi80 - df.lo80
    df["in90"] = (df.target >= df.lo90) & (df.target <= df.hi90)
    df["width90"] = df.hi90 - df.lo90
    df.to_csv(OUT / "quantile_interval_predictions.csv", index=False)

    summary = df.groupby(["condition", "horizon", "mode"]).agg(
        n=("target", "size"),
        coverage80=("in80", "mean"), width80=("width80", "mean"),
        coverage90=("in90", "mean"), width90=("width90", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "quantile_uncertainty_metrics.csv", index=False)

    overall = df.groupby(["mode"]).agg(
        n=("target", "size"),
        coverage80=("in80", "mean"), width80=("width80", "mean"),
        coverage90=("in90", "mean"), width90=("width90", "mean"),
    ).reset_index()
    overall.to_csv(OUT / "quantile_uncertainty_overall.csv", index=False)

    manifest = {
        "run_id": f"local_quantile_{int(time.time())}",
        "prediction_rows": len(df),
        "wall_time_seconds": round(time.time() - t0, 1),
        "method": "LightGBM pinball-loss quantile regression, alpha in {0.05,0.10,0.50,0.90,0.95}",
        "note": "Compare against project_a_extended_modal_results/uncertainty_metrics.csv (empirical residual method).",
    }
    with open(OUT / "quantile_uncertainty_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({**manifest, "overall": overall.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
