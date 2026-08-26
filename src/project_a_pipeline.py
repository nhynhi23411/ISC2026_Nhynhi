"""Project A: missingness-aware surveillance forecasting.

Runs a dependency-light audit and rolling-origin benchmark.  The script uses
only pandas/numpy from the bundled runtime, so it remains reproducible even
when optional ML packages are unavailable.
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

COMPLETE = ["AD_noncholera", "Malaria", "ILI", "ALRI_u5", "Bloody_diarrhoea", "Typhoid"]

def week_key(y, w): return f"{int(y)}w{int(w)}"

def load_inputs(data_dir: Path):
    long = pd.read_csv(data_dir / "idsr_kp_long.csv")
    pres = pd.read_csv(data_dir / "condition_presence_matrix.csv")
    pres = pres.set_index("condition").astype(str).eq("Y")
    weeks = [x for x in pres.columns]
    # Keep chronological order encoded by the source (2023w16, 2023w22, ...).
    week_df = pd.DataFrame({"week_key": weeks})
    week_df[["year", "epi_week"]] = week_df.week_key.str.extract(r"(\d{4})w(\d+)").astype(int)
    week_df["t"] = np.arange(len(week_df))
    districts = sorted(long.district.dropna().unique())
    return long, pres, week_df, districts

def make_panel(long, pres, week_df, districts):
    conditions = list(pres.index)
    idx = pd.MultiIndex.from_product([week_df.week_key, districts, conditions], names=["week_key", "district", "condition"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(week_df, on="week_key", how="left")
    obs = long.assign(week_key=long.apply(lambda r: week_key(r.year, r.epi_week), axis=1))
    obs = obs[["week_key", "district", "condition", "cases"]].copy()
    obs["source_row_present"] = True
    # Duplicate source rows are checked explicitly rather than silently averaged.
    dup = obs.duplicated(["week_key", "district", "condition"]).sum()
    if dup: raise ValueError(f"Duplicate week/district/condition rows: {dup}")
    panel = panel.merge(obs, on=["week_key", "district", "condition"], how="left")
    panel["condition_present"] = [bool(pres.loc[c, w]) for w, c in zip(panel.week_key, panel.condition)]
    panel["row_present"] = panel.source_row_present.fillna(False).astype(bool)
    panel["observed_target"] = panel.condition_present & panel.row_present & panel.cases.notna()
    panel["structurally_unreported"] = ~panel.condition_present
    panel["district_or_cell_unreported"] = panel.condition_present & ~panel.row_present
    panel["reported_but_NR"] = panel.condition_present & panel.row_present & panel.cases.isna()
    panel["cases_zero_filled"] = panel.cases.fillna(0.0)
    panel["cases_reporting_aware"] = panel.cases
    panel = panel.sort_values(["condition", "district", "t"]).reset_index(drop=True)
    return panel

def audit(panel, long, pres, week_df, data_dir, out):
    summary = {
        "n_long_rows": int(len(long)), "n_weeks": int(len(week_df)),
        "n_districts": int(panel.district.nunique()), "n_conditions": int(panel.condition.nunique()),
        "week_start": week_df.week_key.iloc[0], "week_end": week_df.week_key.iloc[-1],
        "complete_conditions": COMPLETE,
        "condition_presence_weeks": {c: int(pres.loc[c].sum()) for c in pres.index},
        "structural_unreported_cells": int(panel.structurally_unreported.sum()),
        "district_or_cell_unreported_cells": int(panel.district_or_cell_unreported.sum()),
        "reported_but_NR_cells": int(panel.reported_but_NR.sum()),
        "quality_flags": pd.read_csv(data_dir / "data_quality_flags.csv").flag_type.value_counts().to_dict(),
    }
    byc = panel.groupby("condition").agg(
        panel_cells=("cases", "size"), observed=("observed_target", "sum"),
        structural_unreported=("structurally_unreported", "sum"),
        district_or_cell_unreported=("district_or_cell_unreported", "sum"),
        mean_observed_cases=("cases", "mean"),
    ).reset_index()
    byc.to_csv(out / "audit_by_condition.csv", index=False)
    json.dump(summary, open(out / "audit_summary.json", "w", encoding="utf-8"), indent=2)
    return summary

def prev_value(vals, seen, t, mode, season=52):
    if mode == "zero_fill":
        prior = vals[:t]
        if len(prior) == 0: return 0.0
        if mode and t >= season and not np.isnan(vals[t-season]): return float(vals[t-season])
        return float(prior[-1]) if not np.isnan(prior[-1]) else 0.0
    # reporting-aware: skip unreported/NR values and use last available case.
    prior_seen = np.where(seen[:t] & ~np.isnan(vals[:t]))[0]
    if len(prior_seen) == 0: return 0.0
    if t >= season and seen[t-season] and not np.isnan(vals[t-season]): return float(vals[t-season])
    return float(vals[prior_seen[-1]])

def forecast_method(vals, seen, t, method, mode):
    vv = vals.copy()
    if mode == "zero_fill": vv[~seen | np.isnan(vv)] = 0.0
    if method == "last":
        if mode == "zero_fill": return float(vv[t-1]) if t else 0.0
        return prev_value(vals, seen, t, mode, season=10**6)
    if method == "seasonal":
        if t >= 52 and (mode == "zero_fill" or (seen[t-52] and not np.isnan(vals[t-52]))): return float(vv[t-52])
        return prev_value(vals, seen, t, mode, season=10**6)
    if method == "mean4":
        if mode == "zero_fill": return float(np.mean(vv[max(0,t-4):t])) if t else 0.0
        ix = np.where(seen[:t] & ~np.isnan(vals[:t]))[0][-4:]
        return float(np.mean(vals[ix])) if len(ix) else 0.0
    raise ValueError(method)

def rolling_benchmark(panel, week_df, out, horizons=(1,2,4)):
    methods = ["last", "seasonal", "mean4"]
    # Origins leave a final chronological test block untouched; report all origins too.
    n = len(week_df); test_start = max(0, n - 20)
    rows = []
    for condition in COMPLETE + [c for c in panel.condition.unique() if c not in COMPLETE]:
        sub = panel[panel.condition == condition]
        for district, g in sub.groupby("district", sort=False):
            g = g.sort_values("t"); vals = g.cases.to_numpy(float); seen = g.condition_present.to_numpy(bool) & g.row_present.to_numpy(bool)
            for h in horizons:
                for t in range(12, n-h):
                    target = vals[t+h-1]
                    if np.isnan(target) or not seen[t+h-1]: continue
                    split = "test" if t >= test_start else "validation"
                    for mode in ["zero_fill", "reporting_aware"]:
                        for method in methods:
                            pred = forecast_method(vals, seen, t+h-1, method, mode)
                            rows.append({"condition":condition,"district":district,"origin_t":t,"horizon":h,"split":split,"mode":mode,"method":method,"target":target,"prediction":pred,"abs_error":abs(target-pred),"sq_error":(target-pred)**2})
    res = pd.DataFrame(rows)
    res.to_csv(out / "forecast_predictions.csv", index=False)
    metrics = res.groupby(["split","condition","horizon","mode","method"], dropna=False).agg(
        n=("abs_error","size"), MAE=("abs_error","mean"), RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s))),),
        target_mean=("target","mean"), prediction_mean=("prediction","mean")
    ).reset_index()
    metrics.to_csv(out / "metrics_by_condition.csv", index=False)
    overall = res.groupby(["split","horizon","mode","method"], dropna=False).agg(n=("abs_error","size"), MAE=("abs_error","mean"), RMSE=("sq_error",lambda s: float(np.sqrt(np.mean(s))))).reset_index()
    overall.to_csv(out / "metrics_overall.csv", index=False)
    return res, overall

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); data_dir=Path(args.data_dir); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    long,pres,weeks,districts=load_inputs(data_dir); panel=make_panel(long,pres,weeks,districts)
    panel.to_csv(out/"panel_with_observation_status.csv", index=False)
    summary=audit(panel,long,pres,weeks,data_dir,out); res,overall=rolling_benchmark(panel,weeks,out)
    print(json.dumps({"audit":summary,"prediction_rows":len(res),"metrics_rows":len(overall)}, indent=2))

if __name__ == "__main__": main()
