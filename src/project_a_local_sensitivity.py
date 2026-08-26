"""Quality-flag sensitivity check for Project A (brief section 2.7 / ablation
table row "Exclude flagged/inconsistent weeks").

This does NOT retrain any model. It reuses the already-computed, already
paired predictions from:
  - project_a_full_modal_results/forecast_predictions.csv (baselines)
  - project_a_local_results/local_model_predictions.csv (Poisson/NegBin/RF/
    LightGBM with and without the mask, produced by project_a_local_models.py)

and recomputes the primary metric table twice: once on the full locked test
block, and once after dropping every prediction whose target week carries a
data_quality_flags.csv flag. If the reporting-aware vs zero-fill ranking and
the approximate effect size survive, the main finding is robust to the known
source-quality issues; if not, that is itself a reportable limitation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS, load_source, load_flags

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
OUT.mkdir(exist_ok=True)


def add_target_week(df, weeks):
    df = df.copy()
    target_t = df.origin_t + df.horizon - 1
    df["target_week"] = [weeks[t] if 0 <= t < len(weeks) else None for t in target_t]
    return df


def summarize(df, label):
    g = df.groupby(["condition", "horizon", "mode"]).agg(
        n=("abs_error", "size"), MAE=("abs_error", "mean"),
    ).reset_index()
    g["dataset"] = label
    return g


def main():
    long, presence, week_df = load_source()
    weeks = list(week_df.week_key)
    flags = load_flags()
    flagged_weeks = set(flags.week_key.unique())
    print(f"Flagged weeks ({len(flagged_weeks)}): {sorted(flagged_weeks)}")

    baseline = pd.read_csv(r"D:\Thầy Khánh\ISC\project_a_full_modal_results\forecast_predictions.csv")
    baseline = baseline[baseline.split == "test"]
    baseline = add_target_week(baseline, weeks)
    baseline_complete = baseline[baseline.condition.isin(COMPLETE_CONDITIONS)]

    local_path = Path(r"D:\Thầy Khánh\ISC\project_a_local_results\local_model_predictions.csv")
    frames = []
    baseline_complete = baseline_complete.assign(model=baseline_complete.method)
    frames.append(baseline_complete[["condition", "district", "origin_t", "horizon", "mode", "model", "target", "prediction", "abs_error", "sq_error", "target_week"]])
    if local_path.exists():
        local_df = pd.read_csv(local_path)
        local_df = add_target_week(local_df, weeks)
        local_complete = local_df[local_df.condition.isin(COMPLETE_CONDITIONS)]
        frames.append(local_complete[["condition", "district", "origin_t", "horizon", "mode", "model", "target", "prediction", "abs_error", "sq_error", "target_week"]])
    else:
        print("NOTE: local_model_predictions.csv not found yet; sensitivity will only cover baselines.")

    all_pred = pd.concat(frames, ignore_index=True)
    all_pred["target_week_flagged"] = all_pred.target_week.isin(flagged_weeks)

    full_summary = summarize(all_pred, "full_test_block")
    clean = all_pred[~all_pred.target_week_flagged]
    clean_summary = summarize(clean, "flagged_weeks_excluded")

    combined = pd.concat([full_summary, clean_summary], ignore_index=True)
    combined.to_csv(OUT / "quality_flag_sensitivity_by_condition.csv", index=False)

    # Pooled six-complete-condition view per mode/model, the level the manuscript will quote.
    pooled_full = all_pred.groupby(["mode", "model"]).agg(n=("abs_error", "size"), MAE=("abs_error", "mean")).reset_index()
    pooled_full["dataset"] = "full_test_block"
    pooled_clean = clean.groupby(["mode", "model"]).agg(n=("abs_error", "size"), MAE=("abs_error", "mean")).reset_index()
    pooled_clean["dataset"] = "flagged_weeks_excluded"
    pooled = pd.concat([pooled_full, pooled_clean], ignore_index=True)
    pooled.to_csv(OUT / "quality_flag_sensitivity_pooled.csv", index=False)

    n_dropped = int(all_pred.target_week_flagged.sum())
    manifest = {
        "flagged_weeks": sorted(flagged_weeks),
        "n_flagged_weeks": len(flagged_weeks),
        "rows_total": int(len(all_pred)),
        "rows_dropped_as_flagged": n_dropped,
        "share_dropped": round(n_dropped / len(all_pred), 4),
    }
    with open(OUT / "quality_flag_sensitivity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
