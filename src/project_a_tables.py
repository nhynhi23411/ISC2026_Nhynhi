"""Manuscript-ready Table 1 / Table 2 / Table 3 for Project A.

Table 1 (dataset/cohort characteristics) needs only the audit outputs that
already exist from the full Modal baseline run. Tables 2-3 need the local
model/uncertainty/sensitivity outputs and are skipped with a clear message
until those files exist.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_tables")
OUT.mkdir(exist_ok=True)
RESULTS = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
FULL_MODAL = Path(r"D:\Thầy Khánh\ISC\project_a_full_modal_results")


def table1_dataset_characteristics():
    with open(FULL_MODAL / "audit_summary.json") as f:
        audit = json.load(f)
    rows = [
        ("Source rows (district-week-condition)", f"{audit['n_long_rows']:,}"),
        ("Bulletin weeks", f"{audit['n_weeks']}"),
        ("Reporting units (districts)", f"{audit['n_districts']}"),
        ("Notifiable conditions", f"{audit['n_conditions']}"),
        ("Always-complete conditions (confirmatory set)", f"{len(audit['complete_conditions'])}: " + ", ".join(audit["complete_conditions"])),
        ("Rotating conditions (exploratory stress test)", f"{audit['n_conditions'] - len(audit['complete_conditions'])}"),
        ("Full panel cells (weeks x districts x conditions)", f"{audit['panel_rows']:,}"),
        ("Structurally unreported cells (rotating, not ranked that week)", f"{audit['structurally_unreported_cells']:,}"),
        ("District/cell-unreported cells (condition ranked, district row absent)", f"{audit['district_or_cell_unreported_cells']:,}"),
        ("Reported-but-NR cells (row present, value blank)", f"{audit['reported_but_NR_cells']:,}"),
        ("Forecast horizons evaluated (weeks ahead)", ", ".join(str(h) for h in audit["horizons"])),
        ("Locked final test origins (chronological, untouched during tuning)", f"{audit['final_test_origins']}"),
        ("Known source-quality issues (data_quality_flags.csv)", "duplicate bulletin (2025w9), 4 dropped-row weeks, 4 corrected-cell rows, 1 reporting-unit redefinition (SWA, 2024w50), 1 boundary overlap (2024w49)"),
        ("Population denominators available", "No -- counts only, not incidence rates"),
        ("Dataset DOI / license / access", "10.17632/9yzvfgrhkt.1, CC BY 4.0, Mendeley Data, accessed 2026-08-26"),
    ]
    df = pd.DataFrame(rows, columns=["Characteristic", "Value"])
    df.to_csv(OUT / "table1_dataset_characteristics.csv", index=False)
    return df


def table2_model_comparison():
    p1 = RESULTS / "local_model_metrics.csv"
    p2 = RESULTS / "local_model_paired_bootstrap.csv"
    p3 = RESULTS / "quantile_uncertainty_overall.csv"
    if not (p1.exists() and p2.exists()):
        print("SKIP table2: local model metrics/bootstrap not ready yet")
        return None
    metrics = pd.read_csv(p1)
    metrics = metrics[metrics.condition.isin(COMPLETE_CONDITIONS)]
    summary = metrics.groupby(["model", "mode", "horizon"]).apply(
        lambda g: pd.Series({"MAE": np.average(g.MAE, weights=g.n), "n": g.n.sum()})
    ).reset_index()
    boot = pd.read_csv(p2)
    boot = boot[boot.condition_group == "complete_six"]
    merged = summary.merge(boot[["model", "horizon", "mean_delta", "ci_low", "ci_high"]], on=["model", "horizon"], how="left")
    merged.to_csv(OUT / "table2_model_comparison.csv", index=False)
    return merged


def table3_ablation_robustness():
    frames = {}
    mask_path = RESULTS / "mask_ablation_metrics.csv"
    if mask_path.exists():
        frames["mask_ablation"] = pd.read_csv(mask_path)
    flag_path = RESULTS / "quality_flag_sensitivity_pooled.csv"
    if flag_path.exists():
        frames["quality_flag_sensitivity"] = pd.read_csv(flag_path)
    quant_path = RESULTS / "quantile_uncertainty_overall.csv"
    if quant_path.exists():
        frames["interval_coverage"] = pd.read_csv(quant_path)
    sim_path = Path(r"D:\Thầy Khánh\ISC\project_a_extended_modal_results\simulation_metrics.csv")
    if sim_path.exists():
        frames["missingness_simulation"] = pd.read_csv(sim_path)
    if not frames:
        print("SKIP table3: no ablation/robustness inputs ready yet")
        return None
    with pd.ExcelWriter(OUT / "table3_ablation_robustness.xlsx") as writer:
        for name, df in frames.items():
            df.to_csv(OUT / f"table3_{name}.csv", index=False)
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return frames


if __name__ == "__main__":
    t1 = table1_dataset_characteristics()
    print(t1.to_string(index=False))
    table2_model_comparison()
    table3_ablation_robustness()
