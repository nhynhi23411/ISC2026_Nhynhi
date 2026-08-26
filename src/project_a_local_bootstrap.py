"""Paired per-origin bootstrap CIs for the local model family (Poisson,
negative-binomial, random forest, LightGBM with/without mask), the piece
Table 2 ("Overall model comparison with uncertainty") needs. Reuses the same
bootstrap-of-the-mean procedure as project_a_analysis.py, applied to
local_model_predictions.csv instead of the baseline predictions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import COMPLETE_CONDITIONS

RESULTS = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")


def boot_ci(x, seed=20260826, B=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return (float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def main():
    path = RESULTS / "local_model_predictions.csv"
    d = pd.read_csv(path)
    d["condition_group"] = np.where(d.condition.isin(COMPLETE_CONDITIONS), "complete_six", "rotating_stress")

    key = ["condition", "district", "origin_t", "horizon", "model"]
    z = d[d["mode"] == "zero_fill"][key + ["abs_error", "condition_group"]].rename(columns={"abs_error": "ae_zero"})
    r = d[d["mode"] == "reporting_aware"][key + ["abs_error"]].rename(columns={"abs_error": "ae_aware"})
    paired = z.merge(r, on=key, how="inner", validate="one_to_one")
    paired["delta_zero_minus_aware"] = paired.ae_zero - paired.ae_aware

    rows = []
    for cols, g in paired.groupby(["condition_group", "horizon", "model"]):
        m, lo, hi = boot_ci(g.delta_zero_minus_aware)
        rows.append(dict(zip(["condition_group", "horizon", "model"], cols),
                          n=len(g), mean_delta=m, ci_low=lo, ci_high=hi,
                          zero_MAE=g.ae_zero.mean(), aware_MAE=g.ae_aware.mean()))
    out = pd.DataFrame(rows).sort_values(["condition_group", "model", "horizon"])
    out.to_csv(RESULTS / "local_model_paired_bootstrap.csv", index=False)
    paired.to_csv(RESULTS / "local_model_paired_predictions.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
