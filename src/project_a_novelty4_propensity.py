"""Novelty #4: reporting-propensity-augmented LightGBM.

Adds one new causal feature to the existing lag/seasonal/mask feature set:
propensity_t = P(this condition is ranked/observed at week t), estimated by
the pooled logistic model fit in project_a_novelty1_mnar.py from each
condition's own causal last-observed provincial volume. This directly
implements the brief's framing ("explicit modeling of the observation
process improves ... validity") as a real additional signal rather than
just a binary mask -- the point is that presence is not random (novelty #1
showed AUROC 0.805 predicting it from volume alone), so which observed
values are "lucky/high" survivors of that selection is informative.

Only the new propensity-augmented model is fit here; it is paired against
the already-computed lightgbm_mask predictions from
project_a_local_models.py (same rolling-origin protocol, same features plus
one extra column), so no duplicate compute.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from project_a_common import (
    COMPLETE_CONDITIONS, HORIZONS, FINAL_TEST_ORIGINS, MIN_HISTORY, SEED,
    load_source, build_panel, build_series, feat,
)

PROPENSITY_MODEL_FILE = "reporting_propensity_model_trainonly.json"
SEEDS = [SEED, 1, 2, 3, 4]

warnings.filterwarnings("ignore")

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def provincial_totals(panel, cond, n):
    g = panel[panel.condition == cond]
    total = g.groupby("t").apply(
        lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
        include_groups=False,
    ).reindex(range(n)).to_numpy(dtype=float)
    return total


def causal_last_observed(totals):
    last = np.nan
    out = np.full(len(totals), np.nan)
    for i, v in enumerate(totals):
        out[i] = last
        if np.isfinite(v):
            last = v
    return out


def propensity_series_for_condition(panel, cond, n, const, coef):
    totals = provincial_totals(panel, cond, n)
    last_obs = causal_last_observed(totals)
    x = np.where(np.isfinite(last_obs), np.log1p(np.maximum(last_obs, 0.0)), np.log1p(0.0))
    z = const + coef * x
    prop = 1.0 / (1.0 + np.exp(-z))
    prop[~np.isfinite(last_obs)] = 0.5  # no history yet: maximally uncertain
    return prop


def feat_with_propensity(vals, seen, t, mode, propensity):
    base = feat(vals, seen, t, mode, mask=True)
    return np.concatenate([base, [propensity[t]]])


def boot_ci(x, seed, B=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    t0 = time.time()
    with open(OUT / PROPENSITY_MODEL_FILE) as f:
        prop_model = json.load(f)
    if "trainonly" not in PROPENSITY_MODEL_FILE:
        raise RuntimeError("refusing to build forecasting features from a non-train-only propensity model")
    const = prop_model["const"]
    coef = prop_model["coef_log1p_last_provincial_total"]

    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS
    assert test_start == prop_model["test_start_t"], "propensity model test_start mismatch"

    propensity_by_cond = {c: propensity_series_for_condition(panel, c, n, const, coef) for c in conditions}

    # Precompute BOTH feature variants once per (condition, horizon, mode) --
    # mask-only (no propensity) and mask+propensity -- shared across all seeds,
    # so the seed-robustness check refits both models under the identical
    # seed/subsample draw for a fair paired comparison (reusing the single
    # fixed-seed lightgbm_mask from local_model_predictions.csv would not test
    # anything: only one of the two arms would vary across "seeds").
    cache = {}
    for cond in conditions:
        propensity = propensity_by_cond[cond]
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                Xtr_mask, Xtr_prop, ytr = [], [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(MIN_HISTORY, test_start):
                        tt = origin + h - 1
                        if tt >= n:
                            continue
                        if mode == "reporting_aware" and not seen[tt]:
                            continue
                        target = vals[tt] if (seen[tt] and np.isfinite(vals[tt])) else 0.0
                        Xtr_mask.append(feat(vals, seen, tt, mode, mask=True))
                        Xtr_prop.append(feat_with_propensity(vals, seen, tt, mode, propensity))
                        ytr.append(target)
                Xtr_mask = np.asarray(Xtr_mask, float)
                Xtr_prop = np.asarray(Xtr_prop, float)
                ytr = np.asarray(ytr, float)
                if len(ytr) < 100:
                    continue

                Xte_mask, Xte_prop, meta = [], [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(test_start, n - h):
                        tt = origin + h - 1
                        if tt >= n or not seen[tt]:
                            continue
                        Xte_mask.append(feat(vals, seen, tt, mode, mask=True))
                        Xte_prop.append(feat_with_propensity(vals, seen, tt, mode, propensity))
                        meta.append((district, origin, float(vals[tt])))
                if not meta:
                    continue
                cache[(cond, h, mode)] = (Xtr_mask, Xtr_prop, ytr, np.asarray(Xte_mask, float), np.asarray(Xte_prop, float), meta)

    def fit_lgbm(Xtr, ytr, Xte, seed):
        model = LGBMRegressor(
            objective="regression", n_estimators=180, learning_rate=0.05,
            num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbosity=-1, random_state=seed,
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        if not np.all(np.isfinite(preds)):
            raise RuntimeError("non-finite predictions")
        return preds

    rows = []
    fit_count = skip_count = fail_count = 0
    for seed in SEEDS:
        for (cond, h, mode), (Xtr_mask, Xtr_prop, ytr, Xte_mask, Xte_prop, meta) in cache.items():
            try:
                preds_mask = fit_lgbm(Xtr_mask, ytr, Xte_mask, seed)
                preds_prop = fit_lgbm(Xtr_prop, ytr, Xte_prop, seed)
            except Exception as exc:
                fail_count += 1
                print(f"FIT FAILED seed={seed} cond={cond} h={h} mode={mode}: {exc}")
                continue
            fit_count += 1
            for (district, origin, target), pred_m, pred_p in zip(meta, preds_mask, preds_prop):
                for model_name, pred in (("lightgbm_mask_seedcheck", pred_m), ("lightgbm_propensity_mask", pred_p)):
                    err = target - float(pred)
                    rows.append({
                        "seed": seed, "condition": cond, "district": district, "origin_t": origin,
                        "horizon": h, "mode": mode, "model": model_name,
                        "target": target, "prediction": float(pred),
                        "abs_error": abs(err), "sq_error": err * err,
                    })
    skip_count = len(conditions) * len(HORIZONS) * 2 - len(cache)

    pred_df = pd.DataFrame(rows)
    if pred_df.empty or not np.isfinite(pred_df.abs_error).all():
        raise RuntimeError("empty or non-finite propensity-model predictions")
    pred_df.to_csv(OUT / "propensity_model_predictions.csv", index=False)

    metrics = pred_df.groupby(["seed", "model", "condition", "horizon", "mode"]).agg(
        n=("abs_error", "size"), MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
    ).reset_index()
    metrics.to_csv(OUT / "propensity_model_metrics.csv", index=False)

    # Pair mask vs propensity WITHIN the same seed (both refit under that seed's
    # subsample/colsample draw), so the comparison is a true paired same-seed test.
    keys = ["seed", "condition", "district", "origin_t", "horizon", "mode"]
    mask_arm = pred_df[pred_df.model == "lightgbm_mask_seedcheck"][keys + ["abs_error"]].rename(columns={"abs_error": "ae_mask"})
    prop_arm = pred_df[pred_df.model == "lightgbm_propensity_mask"][keys + ["abs_error"]].rename(columns={"abs_error": "ae_propensity"})
    paired = mask_arm.merge(prop_arm, on=keys, validate="one_to_one")
    paired["delta_mask_minus_propensity"] = paired.ae_mask - paired.ae_propensity
    paired["condition_group"] = np.where(paired.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")

    summary_rows = []
    for cols, g in paired.groupby(["condition_group", "horizon", "seed"]):
        m, lo, hi = boot_ci(g.delta_mask_minus_propensity, seed=SEED)
        summary_rows.append({
            "condition_group": cols[0], "horizon": cols[1], "seed": cols[2], "n": len(g),
            "mean_delta_mask_minus_propensity": m, "ci_low": lo, "ci_high": hi,
            "mask_MAE": g.ae_mask.mean(), "propensity_MAE": g.ae_propensity.mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "propensity_vs_mask_bootstrap_by_seed.csv", index=False)

    # Cross-seed consistency: mean delta and its seed-to-seed spread, per condition_group/horizon.
    consistency = summary.groupby(["condition_group", "horizon"]).agg(
        n_seeds=("seed", "nunique"),
        mean_delta_across_seeds=("mean_delta_mask_minus_propensity", "mean"),
        std_delta_across_seeds=("mean_delta_mask_minus_propensity", "std"),
        min_delta=("mean_delta_mask_minus_propensity", "min"),
        max_delta=("mean_delta_mask_minus_propensity", "max"),
        all_ci_exclude_zero=("ci_low", lambda s: bool((s > 0).all())),
    ).reset_index()
    consistency.to_csv(OUT / "propensity_vs_mask_seed_consistency.csv", index=False)

    manifest = {
        "run_id": f"propensity_{int(time.time())}",
        "propensity_model_file_used": PROPENSITY_MODEL_FILE,
        "seeds": SEEDS,
        "fit_count": fit_count, "skip_count": skip_count, "fail_count": fail_count,
        "prediction_rows": len(pred_df),
        "propensity_model": prop_model,
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "propensity_model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(consistency.to_string(index=False))


if __name__ == "__main__":
    main()
