"""Novelty #1: direct statistical evidence that rotating-condition absence is
rank-based right-censoring (MNAR-like), not missing-completely-at-random.

Logic: from 2023w32 onward, each week's table shows the 10 conditions with
the highest PROVINCIAL case totals that week (re-ranked weekly; README).
If that is true, a rotating condition's own recent provincial volume should
predict whether it is present (ranked) this week. This script tests exactly
that: logistic regression of weekly presence on the condition's own
last-observed provincial total (causal: only uses information strictly
before week t), pooled across the 10 rotating conditions with condition
fixed effects, restricted to the post-transition (rotating-regime) period.

This also produces the fitted propensity model artifact reused by novelty #4
(project_a_novelty4_propensity.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from project_a_common import COMPLETE_CONDITIONS, FINAL_TEST_ORIGINS, load_source, build_panel

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
OUT.mkdir(exist_ok=True)
TRANSITION_WEEK = "2023w32"


def provincial_series(panel, conditions, n):
    """For each condition: presence[t] (bool, week-level) and
    provincial_total[t] (sum of observed district cases that week, else NaN)."""
    out = {}
    for cond in conditions:
        g = panel[panel.condition == cond]
        presence = g.groupby("t").condition_present.first().reindex(range(n)).astype(bool).to_numpy()
        totals = g.groupby("t").apply(
            lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
            include_groups=False,
        ).reindex(range(n)).to_numpy(dtype=float)
        out[cond] = {"presence": presence, "provincial_total": totals}
    return out


def causal_last_observed(totals):
    """Carry the last known provincial total forward; NaN until first observation."""
    last = np.nan
    out = np.full(len(totals), np.nan)
    for i, v in enumerate(totals):
        out[i] = last
        if np.isfinite(v):
            last = v
    return out


def main():
    long, presence_df, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence_df, week_df)
    rotating = [c for c in conditions if c not in COMPLETE_CONDITIONS]
    transition_idx = weeks.index(TRANSITION_WEEK)

    series = provincial_series(panel, rotating, n)

    rows = []
    for cond in rotating:
        totals = series[cond]["provincial_total"]
        presence = series[cond]["presence"]
        last_obs = causal_last_observed(totals)
        for t in range(transition_idx, n):
            if not np.isfinite(last_obs[t]):
                continue
            rows.append({
                "condition": cond, "t": t, "week": weeks[t],
                "present": int(presence[t]),
                "log1p_last_provincial_total": float(np.log1p(last_obs[t])),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty MNAR test dataset")
    df.to_csv(OUT / "mnar_test_dataset.csv", index=False)

    dummies = pd.get_dummies(df.condition, prefix="cond", drop_first=True)
    X = sm.add_constant(pd.concat([df[["log1p_last_provincial_total"]], dummies], axis=1).astype(float))
    y = df.present.astype(float)
    model = sm.Logit(y, X).fit(disp=0, maxiter=200)

    auc_full = roc_auc_score(y, model.predict(X))
    # Volume-only AUROC (no condition fixed effects) shows the effect isn't just
    # "some conditions are always more present" -- it is driven by the recent value.
    X_volume_only = sm.add_constant(df[["log1p_last_provincial_total"]].astype(float))
    model_volume_only = sm.Logit(y, X_volume_only).fit(disp=0, maxiter=200)
    auc_volume_only = roc_auc_score(y, model_volume_only.predict(X_volume_only))

    coef = float(model.params["log1p_last_provincial_total"])
    pval = float(model.pvalues["log1p_last_provincial_total"])
    ci_low, ci_high = model.conf_int().loc["log1p_last_provincial_total"]

    per_condition = df.groupby("condition").agg(
        n=("present", "size"), presence_rate=("present", "mean"),
        mean_log1p_volume=("log1p_last_provincial_total", "mean"),
    ).reset_index()
    per_condition.to_csv(OUT / "mnar_test_per_condition.csv", index=False)

    summary = {
        "n_obs": int(len(df)),
        "n_rotating_conditions": len(rotating),
        "transition_week": TRANSITION_WEEK,
        "logit_coef_log1p_last_provincial_total": coef,
        "logit_coef_pvalue": pval,
        "logit_coef_95ci": [float(ci_low), float(ci_high)],
        "auroc_with_condition_fixed_effects": float(auc_full),
        "auroc_volume_only_no_fixed_effects": float(auc_volume_only),
        "interpretation": (
            "Positive, significant coefficient means a rotating condition's own recent "
            "provincial case volume predicts whether it is ranked (present) this week -- "
            "direct evidence of rank-based right-censoring (MNAR-like), not MCAR."
        ),
    }
    with open(OUT / "mnar_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Save the full-data model for the standalone MNAR-mechanism claim above --
    # this is a descriptive/inferential claim about a static, documented
    # institutional ranking policy (README: "ten highest provincial totals,
    # re-ranked weekly"), not an out-of-sample forecast, so using every week
    # gives the correct, most-powered estimate of that mechanism.
    prop_model_full = {
        "const": float(model_volume_only.params["const"]),
        "coef_log1p_last_provincial_total": float(model_volume_only.params["log1p_last_provincial_total"]),
        "transition_week": TRANSITION_WEEK,
        "fit_scope": "ALL 164 weeks (post-transition) -- for the MNAR-mechanism claim ONLY.",
        "warning": "DO NOT use this file to build a feature for any model evaluated on the locked final test block -- it was fit using presence/volume data from that block. Use reporting_propensity_model_trainonly.json instead.",
    }
    with open(OUT / "reporting_propensity_model.json", "w") as f:
        json.dump(prop_model_full, f, indent=2)

    # Train-only propensity model: identical specification, but fit using only
    # weeks strictly before the locked final test block, so it is safe to use
    # as a forecasting feature under the project's rolling-origin protocol
    # (this is what project_a_novelty4_propensity.py must load).
    test_start = n - FINAL_TEST_ORIGINS
    df_train = df[df.t < test_start]
    X_train_only = sm.add_constant(df_train[["log1p_last_provincial_total"]].astype(float))
    y_train = df_train.present.astype(float)
    model_train_only = sm.Logit(y_train, X_train_only).fit(disp=0, maxiter=200)
    auc_train_only = roc_auc_score(y_train, model_train_only.predict(X_train_only))

    prop_model_train_only = {
        "const": float(model_train_only.params["const"]),
        "coef_log1p_last_provincial_total": float(model_train_only.params["log1p_last_provincial_total"]),
        "transition_week": TRANSITION_WEEK,
        "test_start_t": int(test_start),
        "n_train_obs": int(len(df_train)),
        "auroc_in_sample_train_only": float(auc_train_only),
        "fit_scope": f"weeks t < {test_start} only (strictly before the locked final {FINAL_TEST_ORIGINS} test origins).",
        "note": "Safe to use as a causal forecasting feature: never saw presence/volume data from the locked test block.",
    }
    with open(OUT / "reporting_propensity_model_trainonly.json", "w") as f:
        json.dump(prop_model_train_only, f, indent=2)
    print("Train-only propensity model (leakage-safe):")
    print(json.dumps(prop_model_train_only, indent=2))


if __name__ == "__main__":
    main()
