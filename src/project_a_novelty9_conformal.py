"""Item B: split-conformal quantile regression (CQR, Romano et al. 2019).

Plain LightGBM quantile regression (project_a_local_uncertainty.py) already
beat the empirical-residual interval, but its coverage guarantee is only
whatever the pinball-loss fit happens to achieve -- no finite-sample
guarantee. CQR adds a calibration step: reserve a block of origins strictly
after the quantile models' own training window and strictly before the
locked final test block, compute conformity scores there, and widen/shrink
the raw quantile interval by the empirical (1-alpha) quantile of those
scores. This gives a distribution-free, finite-sample marginal coverage
guarantee under exchangeability -- the calibration set here is a
contiguous block of the most recent pre-test origins, which is the standard
adaptation of split conformal to a rolling-origin time series setting.
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
CALIB_SIZE = 20  # same size as the locked test block, a block immediately before it


def fit_quantile(Xtr, ytr, alpha):
    model = LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=180, learning_rate=0.05,
        num_leaves=15, min_child_samples=20, verbosity=-1, random_state=SEED,
    )
    model.fit(Xtr, ytr)
    return model


def conformal_quantile(scores, alpha, n):
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def main():
    t0 = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS
    calib_start = test_start - CALIB_SIZE

    rows = []
    skipped = []
    for cond in conditions:
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                mask = True

                def gather(origin_lo, origin_hi, need_target=True):
                    X, y, meta = [], [], []
                    for district in districts:
                        vals, seen = series[(cond, district)]
                        for origin in range(origin_lo, origin_hi):
                            tt = origin + h - 1
                            if tt >= n:
                                continue
                            if mode == "reporting_aware" and not seen[tt]:
                                continue
                            if need_target and not seen[tt]:
                                continue
                            target = vals[tt] if (seen[tt] and np.isfinite(vals[tt])) else 0.0
                            X.append(feat(vals, seen, tt, mode, mask))
                            y.append(target)
                            meta.append((district, origin, float(vals[tt]) if seen[tt] else None))
                    return np.asarray(X, float), np.asarray(y, float), meta

                Xtr, ytr, _ = gather(MIN_HISTORY, calib_start, need_target=False)
                if len(ytr) < 100:
                    skipped.append((cond, h, mode, "too_few_train"))
                    continue
                Xcal, ycal, _ = gather(calib_start, test_start, need_target=True)
                if len(ycal) < 15:
                    skipped.append((cond, h, mode, "too_few_calib"))
                    continue
                Xte, yte, meta_te = gather(test_start, n - h, need_target=True)
                if len(yte) == 0:
                    skipped.append((cond, h, mode, "no_test"))
                    continue

                try:
                    lo80_m = fit_quantile(Xtr, ytr, 0.10)
                    hi80_m = fit_quantile(Xtr, ytr, 0.90)
                    lo90_m = fit_quantile(Xtr, ytr, 0.05)
                    hi90_m = fit_quantile(Xtr, ytr, 0.95)
                except Exception as exc:
                    skipped.append((cond, h, mode, f"fit_failed:{exc}"))
                    continue

                lo80_cal = np.clip(lo80_m.predict(Xcal), 0, None)
                hi80_cal = np.maximum(hi80_m.predict(Xcal), lo80_cal)
                lo90_cal = np.clip(lo90_m.predict(Xcal), 0, None)
                hi90_cal = np.maximum(hi90_m.predict(Xcal), lo90_cal)

                scores80 = np.maximum(lo80_cal - ycal, ycal - hi80_cal)
                scores90 = np.maximum(lo90_cal - ycal, ycal - hi90_cal)
                Q80 = conformal_quantile(scores80, 0.20, len(ycal))
                Q90 = conformal_quantile(scores90, 0.10, len(ycal))

                lo80_te = np.clip(lo80_m.predict(Xte), 0, None)
                hi80_te = np.maximum(hi80_m.predict(Xte), lo80_te)
                lo90_te = np.clip(lo90_m.predict(Xte), 0, None)
                hi90_te = np.maximum(hi90_m.predict(Xte), lo90_te)

                lo80_conf = np.clip(lo80_te - Q80, 0, None)
                hi80_conf = hi80_te + Q80
                lo90_conf = np.clip(lo90_te - Q90, 0, None)
                hi90_conf = hi90_te + Q90

                for (district, origin, target), l80, h80, l80c, h80c, l90, h90, l90c, h90c in zip(
                    meta_te, lo80_te, hi80_te, lo80_conf, hi80_conf, lo90_te, hi90_te, lo90_conf, hi90_conf
                ):
                    rows.append({
                        "condition": cond, "district": district, "origin_t": origin, "horizon": h, "mode": mode,
                        "target": target,
                        "lo80_raw": l80, "hi80_raw": h80, "lo80_conf": l80c, "hi80_conf": h80c,
                        "lo90_raw": l90, "hi90_raw": h90, "lo90_conf": l90c, "hi90_conf": h90c,
                        "Q80": Q80, "Q90": Q90, "n_calib": len(ycal),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty conformal output")
    df["in80_raw"] = (df.target >= df.lo80_raw) & (df.target <= df.hi80_raw)
    df["width80_raw"] = df.hi80_raw - df.lo80_raw
    df["in80_conf"] = (df.target >= df.lo80_conf) & (df.target <= df.hi80_conf)
    df["width80_conf"] = df.hi80_conf - df.lo80_conf
    df["in90_raw"] = (df.target >= df.lo90_raw) & (df.target <= df.hi90_raw)
    df["width90_raw"] = df.hi90_raw - df.lo90_raw
    df["in90_conf"] = (df.target >= df.lo90_conf) & (df.target <= df.hi90_conf)
    df["width90_conf"] = df.hi90_conf - df.lo90_conf
    df.to_csv(OUT / "conformal_predictions.csv", index=False)

    overall = df.groupby("mode").agg(
        n=("target", "size"),
        coverage80_raw=("in80_raw", "mean"), width80_raw=("width80_raw", "mean"),
        coverage80_conf=("in80_conf", "mean"), width80_conf=("width80_conf", "mean"),
        coverage90_raw=("in90_raw", "mean"), width90_raw=("width90_raw", "mean"),
        coverage90_conf=("in90_conf", "mean"), width90_conf=("width90_conf", "mean"),
    ).reset_index()
    overall.to_csv(OUT / "conformal_overall.csv", index=False)

    by_cond = df.groupby(["condition", "mode"]).agg(
        n=("target", "size"),
        coverage80_conf=("in80_conf", "mean"), coverage90_conf=("in90_conf", "mean"),
    ).reset_index()
    by_cond.to_csv(OUT / "conformal_by_condition.csv", index=False)

    manifest = {
        "run_id": f"conformal_{int(time.time())}",
        "calib_size_origins": CALIB_SIZE,
        "n_skipped_condition_horizon_mode": len(skipped),
        "skipped_examples": skipped[:10],
        "prediction_rows": len(df),
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "conformal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
