"""Item C: does district-level scale heterogeneity explain why pooled
Poisson/negative-binomial/ridge regression badly underperform LightGBM/RF on
AD_noncholera specifically? Diagnostic confirmed the hypothesis (Poisson MAE
448.6 -> 191.0 on AD_noncholera after fixing it). This script applies the fix
-- per-district scale-normalized features plus a log(scale) GLM offset (the
standard technique for count regression without a population denominator,
using each district's own historical median case count as the exposure
proxy) -- to Poisson, negative-binomial and ridge, across all 16 conditions,
and pairs the result against the existing (unscaled) local_model_predictions.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialP

from project_a_common import (
    COMPLETE_CONDITIONS, HORIZONS, FINAL_TEST_ORIGINS, MIN_HISTORY, SEED,
    load_source, build_panel, build_series, feat,
)

warnings.filterwarnings("ignore")

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def district_scales(series, cond, districts, test_start):
    scale = {}
    for d in districts:
        vals, seen = series[(cond, d)]
        obs = vals[:test_start][seen[:test_start]]
        scale[d] = max(float(np.median(obs)), 1.0) if len(obs) else 1.0
    return scale


def poisson_scaled_fit_predict(Xtr, ytr, offtr, Xte, offte):
    mu, sd = standardize_fit(Xtr)
    Xs = sm.add_constant((Xtr - mu) / sd, has_constant="add")
    Xqs = sm.add_constant((Xte - mu) / sd, has_constant="add")
    naive_mae = float(np.mean(np.abs(ytr - ytr.mean())))
    model = sm.GLM(ytr, Xs, family=sm.families.Poisson(), offset=offtr).fit(maxiter=100)
    train_pred = model.predict(Xs, offset=offtr)
    if np.mean(np.abs(ytr - train_pred)) >= naive_mae:
        return np.full(len(Xte), ytr.mean())
    return np.asarray(model.predict(Xqs, offset=offte))


def negbin_scaled_fit_predict(Xtr, ytr, offtr, Xte, offte):
    mu, sd = standardize_fit(Xtr)
    Xs = sm.add_constant((Xtr - mu) / sd, has_constant="add")
    Xqs = sm.add_constant((Xte - mu) / sd, has_constant="add")
    naive_mae = float(np.mean(np.abs(ytr - ytr.mean())))
    try:
        model = NegativeBinomialP(ytr, Xs, offset=offtr, p=2).fit(disp=0, maxiter=200, method="bfgs")
        train_pred = np.asarray(model.predict(Xs, offset=offtr))
        if np.all(np.isfinite(train_pred)) and np.mean(np.abs(ytr - train_pred)) < naive_mae:
            pred = np.asarray(model.predict(Xqs, offset=offte))
            if np.all(np.isfinite(pred)):
                return pred
    except Exception:
        pass
    return poisson_scaled_fit_predict(Xtr, ytr, offtr, Xte, offte)


def ridge_scaled_fit_predict(Xtr_scaled, ytr_scaled, Xte_scaled, scale_te, alpha=10.0):
    mu, sd = standardize_fit(Xtr_scaled)
    Xs = (Xtr_scaled - mu) / sd
    Xqs = (Xte_scaled - mu) / sd
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ ytr_scaled)
    return (Xqs @ beta) * scale_te


def main():
    t0 = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS

    rows = []
    fit_count = skip_count = fail_count = 0
    for cond in conditions:
        scale = district_scales(series, cond, districts, test_start)
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                Xtr, Xtr_scaled, ytr, ytr_scaled, offtr = [], [], [], [], []
                for d in districts:
                    vals, seen = series[(cond, d)]
                    sc = scale[d]
                    v_scaled = vals / sc
                    for origin in range(MIN_HISTORY, test_start):
                        tt = origin + h - 1
                        if tt >= n:
                            continue
                        if mode == "reporting_aware" and not seen[tt]:
                            continue
                        target = vals[tt] if (seen[tt] and np.isfinite(vals[tt])) else 0.0
                        Xtr.append(feat(vals, seen, tt, mode, mask=True))
                        Xtr_scaled.append(feat(v_scaled, seen, tt, mode, mask=True))
                        ytr.append(target)
                        ytr_scaled.append(target / sc)
                        offtr.append(np.log(sc))
                Xtr = np.asarray(Xtr, float); Xtr_scaled = np.asarray(Xtr_scaled, float)
                ytr = np.asarray(ytr, float); ytr_scaled = np.asarray(ytr_scaled, float)
                offtr = np.asarray(offtr, float)
                if len(ytr) < 100:
                    skip_count += 1
                    continue

                Xte, Xte_scaled, offte, meta = [], [], [], []
                for d in districts:
                    vals, seen = series[(cond, d)]
                    sc = scale[d]
                    v_scaled = vals / sc
                    for origin in range(test_start, n - h):
                        tt = origin + h - 1
                        if tt >= n or not seen[tt]:
                            continue
                        Xte.append(feat(vals, seen, tt, mode, mask=True))
                        Xte_scaled.append(feat(v_scaled, seen, tt, mode, mask=True))
                        offte.append(np.log(sc))
                        meta.append((d, origin, float(vals[tt]), sc))
                if not meta:
                    skip_count += 1
                    continue
                Xte = np.asarray(Xte, float); Xte_scaled = np.asarray(Xte_scaled, float)
                offte = np.asarray(offte, float)
                scale_te = np.array([m[3] for m in meta], float)

                try:
                    pred_poisson = poisson_scaled_fit_predict(Xtr_scaled, ytr, offtr, Xte_scaled, offte)
                    pred_negbin = negbin_scaled_fit_predict(Xtr_scaled, ytr, offtr, Xte_scaled, offte)
                    pred_ridge = ridge_scaled_fit_predict(Xtr_scaled, ytr_scaled, Xte_scaled, scale_te)
                except Exception as exc:
                    fail_count += 1
                    print(f"FIT FAILED cond={cond} h={h} mode={mode}: {exc}")
                    continue
                fit_count += 1

                for model_name, preds in (
                    ("poisson_scaled", pred_poisson),
                    ("negbin_scaled", pred_negbin),
                    ("ridge_scaled", pred_ridge),
                ):
                    for (d, origin, target, sc), pred in zip(meta, preds):
                        err = target - float(pred)
                        rows.append({
                            "condition": cond, "district": d, "origin_t": origin,
                            "horizon": h, "mode": mode, "model": model_name,
                            "target": target, "prediction": float(pred),
                            "abs_error": abs(err), "sq_error": err * err,
                        })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty or not np.isfinite(pred_df.abs_error).all():
        raise RuntimeError("empty or non-finite district-scaled predictions")
    pred_df.to_csv(OUT / "district_scaled_predictions.csv", index=False)

    metrics = pred_df.groupby(["condition", "horizon", "mode", "model"]).agg(
        n=("abs_error", "size"), MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
    ).reset_index()
    metrics.to_csv(OUT / "district_scaled_metrics.csv", index=False)

    # Compare against the existing unscaled versions, condition by condition.
    baseline = pd.read_csv(OUT / "local_model_metrics.csv")
    name_map = {"poisson_scaled": "poisson_mask", "negbin_scaled": "negbin_mask", "ridge_scaled": "ridge_mask"}
    compare_rows = []
    for scaled_name, unscaled_name in name_map.items():
        a = metrics[metrics.model == scaled_name][["condition", "horizon", "mode", "MAE"]].rename(columns={"MAE": "MAE_scaled"})
        b = baseline[baseline.model == unscaled_name][["condition", "horizon", "mode", "MAE"]].rename(columns={"MAE": "MAE_unscaled"})
        c = a.merge(b, on=["condition", "horizon", "mode"], how="inner")
        c["model_family"] = unscaled_name
        compare_rows.append(c)
    comparison = pd.concat(compare_rows, ignore_index=True)
    comparison["improvement"] = comparison.MAE_unscaled - comparison.MAE_scaled
    comparison.to_csv(OUT / "district_scaled_vs_unscaled_comparison.csv", index=False)

    ad_summary = comparison[comparison.condition == "AD_noncholera"]
    manifest = {
        "run_id": f"district_scaled_{int(time.time())}",
        "fit_count": fit_count, "skip_count": skip_count, "fail_count": fail_count,
        "prediction_rows": len(pred_df),
        "wall_time_seconds": round(time.time() - t0, 1),
        "ad_noncholera_headline": ad_summary.to_dict(orient="records"),
    }
    with open(OUT / "district_scaled_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(comparison.groupby("model_family")[["MAE_unscaled", "MAE_scaled", "improvement"]].mean().to_string())


if __name__ == "__main__":
    main()
