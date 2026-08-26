"""Item D -- brief RQ3: "Are gains consistent across diseases and districts?
... Benefits will be largest where non-reporting is frequent or incidence is
intermittent."

Uses the already-computed 5-seed ensemble predictions from
project_a_novelty4_propensity.py (lightgbm_mask_seedcheck vs
lightgbm_propensity_mask, leakage-safe, seed-robust) and breaks the
mask-vs-propensity gain down by district, then correlates each district's
gain (on rotating-stress conditions, where the propensity feature actually
carries information) against two district-level intermittency/missingness
characteristics computed from the audited panel: the district's own
non-reporting rate and its share of low/zero-count weeks. No retraining.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS, load_source, build_panel

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def boot_ci(x, seed=20260826, B=2000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return float(x.mean()), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    df = pd.read_csv(OUT / "propensity_model_predictions.csv")
    keys = ["condition", "district", "origin_t", "horizon", "mode"]
    ens = df.groupby(keys + ["model"]).agg(
        target=("target", "first"), prediction=("prediction", "mean")
    ).reset_index()
    ens["abs_error"] = (ens.target - ens.prediction).abs()

    mask = ens[ens.model == "lightgbm_mask_seedcheck"].rename(columns={"abs_error": "ae_mask"})
    prop = ens[ens.model == "lightgbm_propensity_mask"].rename(columns={"abs_error": "ae_prop"})
    paired = mask[keys + ["ae_mask"]].merge(prop[keys + ["ae_prop"]], on=keys, validate="one_to_one")
    paired["delta"] = paired.ae_mask - paired.ae_prop
    paired["condition_group"] = np.where(paired.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")

    # --- per-district gain, each condition_group separately ---
    rows = []
    for (district, cgroup), g in paired.groupby(["district", "condition_group"]):
        m, lo, hi = boot_ci(g.delta)
        rows.append({
            "district": district, "condition_group": cgroup, "n": len(g),
            "mean_delta": m, "ci_low": lo, "ci_high": hi,
            "mask_MAE": g.ae_mask.mean(), "propensity_MAE": g.ae_prop.mean(),
        })
    by_district = pd.DataFrame(rows).sort_values(["condition_group", "mean_delta"], ascending=[True, False])
    by_district.to_csv(OUT / "propensity_gain_by_district.csv", index=False)

    # --- district-level intermittency/reporting characteristics, from the audited panel ---
    long, presence, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence, week_df)
    rotating = [c for c in conditions if c not in COMPLETE_CONDITIONS]
    sub = panel[panel.condition.isin(rotating)]

    char_rows = []
    for d in districts:
        g = sub[sub.district == d]
        ranked = g[g.condition_present]  # only weeks where the condition was nationally ranked
        n_ranked = len(ranked)
        own_unreported_rate = float((~ranked.row_present).mean()) if n_ranked else np.nan
        observed = ranked[ranked.observed_target]
        zero_share = float((observed.cases == 0).mean()) if len(observed) else np.nan
        char_rows.append({
            "district": d,
            "n_condition_weeks_ranked": n_ranked,
            "own_unreported_rate": own_unreported_rate,
            "zero_count_share_when_observed": zero_share,
        })
    district_chars = pd.DataFrame(char_rows)
    district_chars.to_csv(OUT / "district_intermittency_characteristics.csv", index=False)

    merged = by_district[by_district.condition_group == "rotating_stress"].merge(district_chars, on="district", how="left")
    merged.to_csv(OUT / "propensity_gain_vs_district_characteristics.csv", index=False)

    corr_unreported = float(merged.mean_delta.corr(merged.own_unreported_rate, method="spearman"))
    corr_zero = float(merged.mean_delta.corr(merged.zero_count_share_when_observed, method="spearman"))

    n_districts = len(by_district[by_district.condition_group == "rotating_stress"])
    n_positive = int((by_district[by_district.condition_group == "rotating_stress"].mean_delta > 0).sum())
    n_positive_complete = int((by_district[by_district.condition_group == "complete_six"].mean_delta > 0).sum())
    n_districts_complete = len(by_district[by_district.condition_group == "complete_six"])

    summary = {
        "n_districts": int(len(districts)),
        "rotating_stress": {
            "n_districts_with_estimate": n_districts,
            "n_districts_with_positive_gain": n_positive,
            "share_positive": round(n_positive / n_districts, 4) if n_districts else None,
        },
        "complete_six": {
            "n_districts_with_estimate": n_districts_complete,
            "n_districts_with_positive_gain": n_positive_complete,
            "share_positive": round(n_positive_complete / n_districts_complete, 4) if n_districts_complete else None,
        },
        "spearman_gain_vs_own_district_unreported_rate": corr_unreported,
        "spearman_gain_vs_zero_count_share": corr_zero,
        "interpretation": (
            "share_positive close to 1 means the propensity gain generalizes across "
            "nearly all districts rather than being driven by one or two outliers. "
            "A positive spearman correlation with own_unreported_rate/zero_count_share "
            "would support brief RQ3's hypothesis that benefits concentrate where "
            "reporting is patchiest or incidence is most intermittent."
        ),
    }
    with open(OUT / "propensity_gain_heterogeneity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print()
    print("Top 5 districts by rotating-stress gain:")
    print(by_district[by_district.condition_group == "rotating_stress"].head(5).to_string(index=False))
    print()
    print("Bottom 5 districts by rotating-stress gain:")
    print(by_district[by_district.condition_group == "rotating_stress"].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
