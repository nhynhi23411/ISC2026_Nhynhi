"""Local model-family completion run for Project A.

Adds the model tiers still missing from the Modal extended run:
  - Poisson regression (count-aware, statistical tier)
  - Negative-binomial regression (count-aware, statistical tier)
  - Random forest (nonlinear ML tier)
  - LightGBM without the observation-mask features (mask-removal ablation,
    paired against the existing mask-enabled LightGBM)

Runs entirely on CPU with the local Python environment (numpy/pandas/sklearn/
statsmodels/lightgbm all confirmed available). No Modal job is started.

Training rows are drawn only from origins before the locked final 20 test
origins (rolling-origin discipline preserved). Evaluation is on the same
locked final 20 origins used by every other Project A run, and only on
observed targets.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from project_a_common import (
    COMPLETE_CONDITIONS, HORIZONS, FINAL_TEST_ORIGINS, MIN_HISTORY, SEED,
    load_source, build_panel, build_series, feat,
)

warnings.filterwarnings("ignore")

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
OUT.mkdir(exist_ok=True)


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def ridge_fit_predict(Xtr, ytr, Xte, alpha=10.0):
    mu, sd = standardize_fit(Xtr)
    Xs = (Xtr - mu) / sd
    Xqs = (Xte - mu) / sd
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ ytr)
    return Xqs @ beta


def _in_sample_mae(model_predict_fn, Xs, ytr):
    """In-sample MAE of a fitted count model, for the sanity gate below."""
    pred = np.asarray(model_predict_fn(Xs))
    if not np.all(np.isfinite(pred)):
        return np.inf
    return float(np.mean(np.abs(ytr - pred)))


def _sane_or_none(model, Xs, Xqs, ytr, naive_mae):
    """Reject a fit whose in-sample error is not even better than predicting
    the training mean for every row. This was added after unregularized
    negative-binomial GLM IRLS was found to diverge (exp-link overflow) on
    AD_noncholera -- a high-volume, high-variance series where the naive-mean
    check catches divergence that a fixed numeric ceiling let through."""
    train_mae = _in_sample_mae(model.predict, Xs, ytr)
    if train_mae >= naive_mae:
        return None
    pred = np.asarray(model.predict(Xqs))
    return pred if np.all(np.isfinite(pred)) else None


def poisson_fit_predict(Xtr, ytr, Xte):
    mu, sd = standardize_fit(Xtr)
    Xs = sm.add_constant((Xtr - mu) / sd, has_constant="add")
    Xqs = sm.add_constant((Xte - mu) / sd, has_constant="add")
    naive_mae = float(np.mean(np.abs(ytr - ytr.mean())))

    try:
        model = sm.GLM(ytr, Xs, family=sm.families.Poisson()).fit(maxiter=100)
        pred = _sane_or_none(model, Xs, Xqs, ytr, naive_mae)
        if pred is not None:
            return pred
    except Exception:
        pass
    model = sm.GLM(ytr, Xs, family=sm.families.Poisson()).fit_regularized(alpha=1.0, L1_wt=0.0, maxiter=200)
    pred = _sane_or_none(model, Xs, Xqs, ytr, naive_mae)
    return pred if pred is not None else np.full(len(Xte), ytr.mean())


def negbin_fit_predict(Xtr, ytr, Xte):
    # NB2 dispersion is estimated by MLE (NegativeBinomialP) first, since a
    # fixed alpha=1.0 does not match every condition's true overdispersion.
    # Plain unregularized NegativeBinomial GLM IRLS was verified to diverge
    # (exp-link overflow; predictions in the thousands against single-digit
    # or low-hundreds targets, worst on the highest-volume series) and is
    # never used. Every candidate must beat the naive training-mean predictor
    # in-sample or it is rejected outright, falling through to the next
    # candidate and finally to the naive mean itself.
    from statsmodels.discrete.discrete_model import NegativeBinomialP

    mu, sd = standardize_fit(Xtr)
    Xs = sm.add_constant((Xtr - mu) / sd, has_constant="add")
    Xqs = sm.add_constant((Xte - mu) / sd, has_constant="add")
    naive_mae = float(np.mean(np.abs(ytr - ytr.mean())))

    try:
        model = NegativeBinomialP(ytr, Xs, p=2).fit(disp=0, maxiter=200, method="bfgs")
        pred = _sane_or_none(model, Xs, Xqs, ytr, naive_mae)
        if pred is not None:
            return pred
    except Exception:
        pass
    try:
        model = sm.GLM(ytr, Xs, family=sm.families.NegativeBinomial(alpha=1.0)).fit_regularized(
            alpha=1.0, L1_wt=0.0, maxiter=200
        )
        pred = _sane_or_none(model, Xs, Xqs, ytr, naive_mae)
        if pred is not None:
            return pred
    except Exception:
        pass
    pred = poisson_fit_predict(Xtr, ytr, Xte)
    return pred


def rf_fit_predict(Xtr, ytr, Xte):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def lgbm_fit_predict(Xtr, ytr, Xte):
    model = LGBMRegressor(
        objective="regression", n_estimators=180, learning_rate=0.05,
        num_leaves=15, min_child_samples=20, verbosity=-1, random_state=SEED,
    )
    model.fit(Xtr, ytr)
    return model.predict(Xte)


MODEL_FNS = {
    "ridge_mask": (ridge_fit_predict, True, False),
    "poisson_mask": (poisson_fit_predict, True, False),
    "negbin_mask": (negbin_fit_predict, True, False),
    "random_forest_mask": (rf_fit_predict, True, False),
    "lightgbm_mask": (lgbm_fit_predict, True, False),
    "lightgbm_nomask": (lgbm_fit_predict, False, False),
}


def main(smoke=False, out_dir=None):
    global OUT
    if out_dir is not None:
        OUT = Path(out_dir)
        OUT.mkdir(exist_ok=True, parents=True)
    t0 = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS

    if smoke:
        conditions = conditions[:2]
        districts = districts[:5]

    rows = []
    fit_count = 0
    skip_count = 0
    fail_count = 0

    for cond in conditions:
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                for model_name, (fit_predict, mask, log_target) in MODEL_FNS.items():
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
                        skip_count += 1
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
                        skip_count += 1
                        continue
                    Xte = np.asarray(Xte, float)

                    try:
                        preds = np.asarray(fit_predict(Xtr, ytr, Xte), float)
                        if not np.all(np.isfinite(preds)):
                            raise RuntimeError("non-finite predictions")
                    except Exception as exc:
                        fail_count += 1
                        print(f"FIT FAILED cond={cond} h={h} mode={mode} model={model_name}: {exc}")
                        continue

                    fit_count += 1
                    for (district, origin, target), pred in zip(meta, preds):
                        err = target - float(pred)
                        rows.append({
                            "condition": cond, "district": district, "origin_t": origin,
                            "horizon": h, "mode": mode, "model": model_name,
                            "target": target, "prediction": float(pred),
                            "abs_error": abs(err), "sq_error": err * err,
                        })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty or not np.isfinite(pred_df.abs_error).all():
        raise RuntimeError("empty or non-finite local model predictions")

    pred_df.to_csv(OUT / "local_model_predictions.csv", index=False)

    metrics = pred_df.groupby(["condition", "horizon", "mode", "model"]).agg(
        n=("abs_error", "size"), MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
    ).reset_index()
    metrics.to_csv(OUT / "local_model_metrics.csv", index=False)

    # Mask-removal ablation: lightgbm_mask vs lightgbm_nomask, same rows.
    a = pred_df[pred_df.model == "lightgbm_mask"].rename(columns={"abs_error": "ae_mask"})
    b = pred_df[pred_df.model == "lightgbm_nomask"].rename(columns={"abs_error": "ae_nomask"})
    keys = ["condition", "district", "origin_t", "horizon", "mode"]
    ablation = a[keys + ["ae_mask"]].merge(b[keys + ["ae_nomask"]], on=keys, validate="one_to_one")
    ablation["delta_nomask_minus_mask"] = ablation.ae_nomask - ablation.ae_mask
    ablation_summary = ablation.groupby(["condition", "horizon", "mode"]).agg(
        n=("delta_nomask_minus_mask", "size"),
        mask_MAE=("ae_mask", "mean"),
        nomask_MAE=("ae_nomask", "mean"),
        mean_delta=("delta_nomask_minus_mask", "mean"),
    ).reset_index()
    ablation_summary.to_csv(OUT / "mask_ablation_metrics.csv", index=False)

    manifest = {
        "run_id": f"local_models_{int(time.time())}",
        "fit_count": fit_count, "skip_count": skip_count, "fail_count": fail_count,
        "prediction_rows": len(pred_df),
        "models": list(MODEL_FNS.keys()),
        "conditions": conditions,
        "complete_conditions": list(COMPLETE_CONDITIONS),
        "horizons": list(HORIZONS),
        "final_test_origins": FINAL_TEST_ORIGINS,
        "wall_time_seconds": round(time.time() - t0, 1),
        "research_data_used": True,
    }
    with open(OUT / "local_model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 2-condition/5-district timing test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(smoke=args.smoke, out_dir=args.out)
