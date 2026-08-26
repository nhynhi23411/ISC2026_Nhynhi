"""Item A: inverse-propensity-weighted (IPW) LightGBM.

Novelty #4 (project_a_novelty4_propensity.py) added the propensity score as
an input FEATURE. This script instead uses it the way selection-correction
methods (Heckman-style, IPW estimators) actually use a propensity score: as
a per-example SAMPLE WEIGHT in the training loss, weight_i = 1 / P(observed
at that condition-week). The logic: only condition-weeks that were "lucky"
enough to be ranked/observed appear in the training data at all, and
novelty #1 showed that being ranked is itself correlated with having a high
value (AUROC 0.805 from volume alone) -- so the training sample is a
biased-high sample of the true conditional distribution. Upweighting the
rare, low-propensity-but-still-observed examples (which are underrepresented
relative to how often "not observed" actually happens at that propensity
level) is the standard correction for this kind of selection bias.

Uses the SAME leakage-safe (train-only-fit) propensity model as novelty #4,
and the mask-only feature set (no propensity as a covariate here, so this is
a clean ablation of "weighting" vs "feature" as two different uses of the
same causal quantity). Fit under the same 5 seeds with real subsampling for
a fair, paired, seed-robust comparison against lightgbm_mask.
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

warnings.filterwarnings("ignore")

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
SEEDS = [SEED, 1, 2, 3, 4]
MAX_WEIGHT = 20.0  # cap 1/propensity so a handful of near-zero-propensity rows can't dominate the loss


def provincial_totals(panel, cond, n):
    g = panel[panel.condition == cond]
    return g.groupby("t").apply(
        lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
        include_groups=False,
    ).reindex(range(n)).to_numpy(dtype=float)


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
    x = np.where(np.isfinite(last_obs), np.log1p(np.maximum(last_obs, 0.0)), 0.0)
    z = const + coef * x
    prop = 1.0 / (1.0 + np.exp(-z))
    prop[~np.isfinite(last_obs)] = 0.5
    return np.clip(prop, 1.0 / MAX_WEIGHT, 1.0)


def main():
    t0 = time.time()
    with open(OUT / "reporting_propensity_model_trainonly.json") as f:
        prop_model = json.load(f)
    const = prop_model["const"]
    coef = prop_model["coef_log1p_last_provincial_total"]

    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    series = build_series(panel, n)
    test_start = n - FINAL_TEST_ORIGINS
    assert test_start == prop_model["test_start_t"]

    propensity_by_cond = {c: propensity_series_for_condition(panel, c, n, const, coef) for c in conditions}

    cache = {}
    for cond in conditions:
        propensity = propensity_by_cond[cond]
        for h in HORIZONS:
            for mode in ("zero_fill", "reporting_aware"):
                Xtr, ytr, wtr = [], [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(MIN_HISTORY, test_start):
                        tt = origin + h - 1
                        if tt >= n:
                            continue
                        if mode == "reporting_aware" and not seen[tt]:
                            continue
                        target = vals[tt] if (seen[tt] and np.isfinite(vals[tt])) else 0.0
                        Xtr.append(feat(vals, seen, tt, mode, mask=True))
                        ytr.append(target)
                        wtr.append(1.0 / propensity[tt])
                Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float); wtr = np.asarray(wtr, float)
                if len(ytr) < 100:
                    continue

                Xte, meta = [], []
                for district in districts:
                    vals, seen = series[(cond, district)]
                    for origin in range(test_start, n - h):
                        tt = origin + h - 1
                        if tt >= n or not seen[tt]:
                            continue
                        Xte.append(feat(vals, seen, tt, mode, mask=True))
                        meta.append((district, origin, float(vals[tt])))
                if not meta:
                    continue
                cache[(cond, h, mode)] = (Xtr, ytr, wtr, np.asarray(Xte, float), meta)

    def fit_lgbm(Xtr, ytr, Xte, seed, weight=None):
        model = LGBMRegressor(
            objective="regression", n_estimators=180, learning_rate=0.05,
            num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbosity=-1, random_state=seed,
        )
        model.fit(Xtr, ytr, sample_weight=weight)
        preds = model.predict(Xte)
        if not np.all(np.isfinite(preds)):
            raise RuntimeError("non-finite predictions")
        return preds

    rows = []
    fit_count = fail_count = 0
    for seed in SEEDS:
        for (cond, h, mode), (Xtr, ytr, wtr, Xte, meta) in cache.items():
            try:
                preds_mask = fit_lgbm(Xtr, ytr, Xte, seed, weight=None)
                preds_ipw = fit_lgbm(Xtr, ytr, Xte, seed, weight=wtr)
            except Exception as exc:
                fail_count += 1
                print(f"FIT FAILED seed={seed} cond={cond} h={h} mode={mode}: {exc}")
                continue
            fit_count += 1
            for (district, origin, target), pred_m, pred_i in zip(meta, preds_mask, preds_ipw):
                for model_name, pred in (("lightgbm_mask_seedcheck", pred_m), ("lightgbm_ipw", pred_i)):
                    err = target - float(pred)
                    rows.append({
                        "seed": seed, "condition": cond, "district": district, "origin_t": origin,
                        "horizon": h, "mode": mode, "model": model_name,
                        "target": target, "prediction": float(pred),
                        "abs_error": abs(err), "sq_error": err * err,
                    })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty or not np.isfinite(pred_df.abs_error).all():
        raise RuntimeError("empty or non-finite IPW predictions")
    pred_df.to_csv(OUT / "ipw_model_predictions.csv", index=False)

    keys = ["seed", "condition", "district", "origin_t", "horizon", "mode"]
    mask_arm = pred_df[pred_df.model == "lightgbm_mask_seedcheck"][keys + ["abs_error"]].rename(columns={"abs_error": "ae_mask"})
    ipw_arm = pred_df[pred_df.model == "lightgbm_ipw"][keys + ["abs_error"]].rename(columns={"abs_error": "ae_ipw"})
    paired = mask_arm.merge(ipw_arm, on=keys, validate="one_to_one")
    paired["delta_mask_minus_ipw"] = paired.ae_mask - paired.ae_ipw
    paired["condition_group"] = np.where(paired.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")

    def boot_ci(x, seed=SEED, B=2000):
        x = np.asarray(x, float)
        rng = np.random.default_rng(seed)
        means = np.empty(B)
        for i in range(B):
            means[i] = rng.choice(x, size=len(x), replace=True).mean()
        return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

    per_seed_rows = []
    for cols, g in paired.groupby(["condition_group", "horizon", "seed"]):
        m, lo, hi = boot_ci(g.delta_mask_minus_ipw)
        per_seed_rows.append({"condition_group": cols[0], "horizon": cols[1], "seed": cols[2], "n": len(g),
                               "mean_delta": m, "ci_low": lo, "ci_high": hi})
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(OUT / "ipw_vs_mask_bootstrap_by_seed.csv", index=False)

    # Ensemble across seeds (average prediction), then one final paired bootstrap.
    ens = pred_df.groupby(["condition", "district", "origin_t", "horizon", "mode", "model"]).agg(
        target=("target", "first"), prediction=("prediction", "mean")
    ).reset_index()
    ens["abs_error"] = (ens.target - ens.prediction).abs()
    ens_mask = ens[ens.model == "lightgbm_mask_seedcheck"].rename(columns={"abs_error": "ae_mask"})
    ens_ipw = ens[ens.model == "lightgbm_ipw"].rename(columns={"abs_error": "ae_ipw"})
    ekeys = ["condition", "district", "origin_t", "horizon", "mode"]
    ens_paired = ens_mask[ekeys + ["ae_mask"]].merge(ens_ipw[ekeys + ["ae_ipw"]], on=ekeys)
    ens_paired["delta"] = ens_paired.ae_mask - ens_paired.ae_ipw
    ens_paired["condition_group"] = np.where(ens_paired.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")

    ens_rows = []
    for cols, g in ens_paired.groupby(["condition_group", "horizon"]):
        m, lo, hi = boot_ci(g.delta)
        ens_rows.append({"condition_group": cols[0], "horizon": cols[1], "n": len(g),
                          "mean_delta": m, "ci_low": lo, "ci_high": hi,
                          "mask_MAE": g.ae_mask.mean(), "ipw_MAE": g.ae_ipw.mean()})
    ens_summary = pd.DataFrame(ens_rows)
    ens_summary.to_csv(OUT / "ipw_vs_mask_ensemble_bootstrap.csv", index=False)

    manifest = {
        "run_id": f"ipw_{int(time.time())}",
        "seeds": SEEDS, "max_weight_cap": MAX_WEIGHT,
        "fit_count": fit_count, "fail_count": fail_count,
        "prediction_rows": len(pred_df),
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    with open(OUT / "ipw_model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(ens_summary.to_string(index=False))


if __name__ == "__main__":
    main()
