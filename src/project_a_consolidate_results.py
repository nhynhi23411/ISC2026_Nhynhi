"""Pull every number the manuscript needs into one JSON file so the docx
generation script (manuscript/build.js) has a single source of truth instead
of re-deriving numbers from CSVs inside JS. Safe to re-run at any time; each
section degrades to null/"pending" if its upstream file does not exist yet.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS

ROOT = Path(r"D:\Thầy Khánh\ISC")
LOCAL = ROOT / "project_a_local_results"
FULL_MODAL = ROOT / "project_a_full_modal_results"
EXTENDED_MODAL = ROOT / "project_a_extended_modal_results"
OUT = ROOT / "manuscript" / "results_data.json"


def safe_read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def main():
    data = {}

    with open(FULL_MODAL / "audit_summary.json") as f:
        data["audit"] = json.load(f)

    # Baseline (last/mean4/seasonal) headline numbers, six complete conditions, test split.
    fp = safe_read_csv(FULL_MODAL / "forecast_predictions.csv")
    if fp is not None:
        test = fp[(fp.split == "test") & (fp.condition.isin(COMPLETE_CONDITIONS))]
        base_h1 = test[(test.horizon == 1) & (test.method == "mean4")]
        data["baseline_h1_mean4"] = {
            "zero_MAE": float(base_h1[base_h1["mode"] == "zero_fill"].abs_error.mean()),
            "aware_MAE": float(base_h1[base_h1["mode"] == "reporting_aware"].abs_error.mean()),
        }
        pooled = test.groupby(["horizon", "mode", "method"]).abs_error.mean().reset_index()
        data["baseline_pooled_by_horizon"] = pooled.to_dict(orient="records")
    else:
        data["baseline_h1_mean4"] = None
        data["baseline_pooled_by_horizon"] = None

    peff = safe_read_csv(ROOT / "project_a_results" / "paired_effects_bootstrap.csv")
    if peff is not None:
        rot = peff[(peff.split == "test") & (peff.condition_group == "rotating_stress") & (peff.horizon == 1)]
        data["rotating_stress_h1"] = rot.to_dict(orient="records")
    else:
        data["rotating_stress_h1"] = None

    # Local extended model family (Poisson/NegBin/RF/LightGBM +- mask).
    lm = safe_read_csv(LOCAL / "local_model_metrics.csv")
    if lm is not None:
        lm_complete = lm[lm.condition.isin(COMPLETE_CONDITIONS)]
        agg = lm_complete.groupby(["model", "mode", "horizon"]).apply(
            lambda g: pd.Series({"MAE": float(np.average(g.MAE, weights=g.n)), "n": int(g.n.sum())})
        ).reset_index()
        data["local_model_summary"] = agg.to_dict(orient="records")
        best = lm_complete.loc[lm_complete.groupby(["model"]).MAE.idxmin()]
        data["local_model_best_rows"] = best.to_dict(orient="records")
    else:
        data["local_model_summary"] = None

    boot = safe_read_csv(LOCAL / "local_model_paired_bootstrap.csv")
    data["local_model_bootstrap"] = boot.to_dict(orient="records") if boot is not None else None

    mask_abl = safe_read_csv(LOCAL / "mask_ablation_metrics.csv")
    if mask_abl is not None:
        mask_complete = mask_abl[mask_abl.condition.isin(COMPLETE_CONDITIONS)]
        data["mask_ablation_pooled"] = mask_complete.groupby(["mode", "horizon"]).apply(
            lambda g: pd.Series({
                "mask_MAE": float(np.average(g.mask_MAE, weights=g.n)),
                "nomask_MAE": float(np.average(g.nomask_MAE, weights=g.n)),
                "n": int(g.n.sum()),
            })
        ).reset_index().to_dict(orient="records")
    else:
        data["mask_ablation_pooled"] = None

    qflag = safe_read_csv(LOCAL / "quality_flag_sensitivity_pooled.csv")
    data["quality_flag_sensitivity"] = qflag.to_dict(orient="records") if qflag is not None else None

    qman = LOCAL / "quality_flag_sensitivity_manifest.json"
    if qman.exists():
        with open(qman) as f:
            data["quality_flag_manifest"] = json.load(f)
    else:
        data["quality_flag_manifest"] = None

    quant = safe_read_csv(LOCAL / "quantile_uncertainty_overall.csv")
    data["quantile_uncertainty_overall"] = quant.to_dict(orient="records") if quant is not None else None
    quant_by_cond = safe_read_csv(LOCAL / "quantile_uncertainty_metrics.csv")
    if quant_by_cond is not None:
        cc = quant_by_cond[quant_by_cond.condition.isin(COMPLETE_CONDITIONS)]
        data["quantile_uncertainty_complete_six"] = cc.groupby(["mode"]).agg(
            coverage80=("coverage80", "mean"), coverage90=("coverage90", "mean"),
        ).reset_index().to_dict(orient="records")
    else:
        data["quantile_uncertainty_complete_six"] = None

    emp_unc = safe_read_csv(EXTENDED_MODAL / "uncertainty_metrics.csv")
    if emp_unc is not None:
        lgbm_unc = emp_unc[emp_unc.model == "lightgbm"]
        data["empirical_residual_uncertainty_lightgbm"] = lgbm_unc.groupby("mode").agg(
            coverage80=("coverage80", "mean"), coverage90=("coverage90", "mean"),
        ).reset_index().to_dict(orient="records")
    else:
        data["empirical_residual_uncertainty_lightgbm"] = None

    sim = safe_read_csv(EXTENDED_MODAL / "simulation_metrics.csv")
    if sim is not None:
        # Keep only the estimand used in the manuscript.  Older runs also
        # contained a misleading legacy `delta` column equal to zero_MAE.
        if "delta_zero_minus_aware" in sim.columns:
            sim = sim.drop(columns=[c for c in ["delta"] if c in sim.columns])
            sim = sim.rename(columns={"delta_zero_minus_aware": "delta"})
        data["missingness_simulation"] = sim.to_dict(orient="records")
    else:
        data["missingness_simulation"] = None

    # New multi-mechanism stress test: higher missingness rates, outage blocks,
    # and value-dependent censoring. Keep the full summary auditable while the
    # manuscript can quote selected rows without re-reading large predictions.
    stress = safe_read_csv(LOCAL / "missingness_stress_summary.csv")
    data["missingness_stress_summary"] = stress.to_dict(orient="records") if stress is not None else None
    stress_man = LOCAL / "missingness_stress_manifest.json"
    data["missingness_stress_manifest"] = json.loads(stress_man.read_text(encoding="utf-8")) if stress_man.exists() else None

    tg = safe_read_csv(LOCAL / "temporal_generalization_metrics.csv")
    data["temporal_generalization_metrics"] = tg.to_dict(orient="records") if tg is not None else None
    fa = safe_read_csv(LOCAL / "feature_ablation_metrics.csv")
    data["feature_ablation_metrics"] = fa.to_dict(orient="records") if fa is not None else None
    ga_man = LOCAL / "generalization_ablation_manifest.json"
    data["generalization_ablation_manifest"] = json.loads(ga_man.read_text(encoding="utf-8")) if ga_man.exists() else None

    # Latest novelty outputs are maintained separately from the original
    # Modal bundle; include them so the DOCX and package have one auditable
    # source of truth rather than stale null placeholders.
    for key, filename in {
        "mnar_test_summary": "mnar_test_summary.json",
        "rq4_uncertainty_summary": "rq4_uncertainty_summary.json",
        "trajectory_distortion_summary": "trajectory_distortion_summary.json",
        "propensity_model_manifest": "propensity_model_manifest.json",
        "propensity_gain_heterogeneity_summary": "propensity_gain_heterogeneity_summary.json",
        "hazard_model_summary": "hazard_model_summary.json",
        "conformal_manifest": "conformal_manifest.json",
        "system_benchmark": "system_benchmark.json",
        "ipw_model_manifest": "ipw_model_manifest.json",
        "local_model_manifest": "local_model_manifest.json",
        "quality_flag_sensitivity_manifest": "quality_flag_sensitivity_manifest.json",
    }.items():
        p = LOCAL / filename
        data[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    het = safe_read_csv(EXTENDED_MODAL / "heterogeneity_metrics.csv")
    if het is not None:
        top = het[het.model == "lightgbm"].sort_values("MAE").head(5)
        data["heterogeneity_top5_lowest_mae"] = top.to_dict(orient="records")
    else:
        data["heterogeneity_top5_lowest_mae"] = None

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print("Wrote manuscript/results_data.json")
    missing = [k for k, v in data.items() if v is None]
    print("Still pending:", missing if missing else "none -- all sections ready")


if __name__ == "__main__":
    main()
