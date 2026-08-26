"""Shared data/feature utilities for the local Project A completion scripts.

All scripts in this batch (project_a_local_models.py, project_a_local_uncertainty.py,
project_a_local_sensitivity.py, project_a_figures.py, project_a_tables.py) import
from here so the panel construction and feature definitions are identical everywhere.

Data source: D:\\Thầy Khánh\\ISC\\project_a_data\\... (Mendeley DOI 10.17632/9yzvfgrhkt.1).
No network access, no Modal job is triggered by anything in this module.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(r"D:\Thầy Khánh\ISC\project_a_data\District-week notifiable infectious disease survei")

COMPLETE_CONDITIONS = (
    "AD_noncholera", "Malaria", "ILI", "ALRI_u5", "Bloody_diarrhoea", "Typhoid"
)
LAGS = (1, 2, 4, 8, 12)
HORIZONS = (1, 2, 4)
FINAL_TEST_ORIGINS = 20
MIN_HISTORY = 12
SEED = 20260826


def load_source(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    long = pd.read_csv(data_dir / "idsr_kp_long.csv")
    presence = pd.read_csv(data_dir / "condition_presence_matrix.csv").set_index("condition")
    presence = presence.astype(str).eq("Y")
    weeks = list(presence.columns)
    week_df = pd.DataFrame({"week_key": weeks})
    week_df[["year", "epi_week"]] = week_df.week_key.str.extract(r"(\d{4})w(\d+)").astype(int)
    week_df["t"] = np.arange(len(week_df))
    return long, presence, week_df


def load_flags(data_dir=DATA_DIR):
    flags = pd.read_csv(Path(data_dir) / "data_quality_flags.csv")
    flags["week_key"] = [f"{int(y)}w{int(w)}" for y, w in zip(flags.year, flags.epi_week)]
    return flags


def build_panel(long, presence, week_df):
    weeks = list(presence.columns)
    districts = sorted(long.district.dropna().unique())
    conditions = list(presence.index)
    n = len(weeks)
    obs = long.copy()
    obs["week_key"] = [f"{int(y)}w{int(w)}" for y, w in zip(obs.year, obs.epi_week)]
    obs = obs[["week_key", "district", "condition", "cases"]].copy()
    obs["source_row_present"] = True
    if obs.duplicated(["week_key", "district", "condition"]).any():
        raise RuntimeError("duplicate source keys")
    idx = pd.MultiIndex.from_product([weeks, districts, conditions], names=["week_key", "district", "condition"])
    panel = pd.DataFrame(index=idx).reset_index().merge(week_df, on="week_key")
    panel = panel.merge(obs, on=["week_key", "district", "condition"], how="left")
    panel["condition_present"] = [bool(presence.loc[c, w]) for c, w in zip(panel.condition, panel.week_key)]
    panel["row_present"] = panel.source_row_present.eq(True)
    panel["observed_target"] = panel.condition_present & panel.row_present & panel.cases.notna()
    panel["reported_but_NR"] = panel.condition_present & panel.row_present & panel.cases.isna()
    panel["district_or_cell_unreported"] = panel.condition_present & ~panel.row_present
    panel["structurally_unreported"] = ~panel.condition_present
    panel["seen"] = panel.observed_target
    panel = panel.sort_values(["condition", "district", "t"]).reset_index(drop=True)
    return panel, weeks, districts, conditions, n


def build_series(panel, n):
    series = {}
    for (cond, district), g in panel.groupby(["condition", "district"], sort=False):
        g = g.sort_values("t")
        vals = np.full(n, np.nan)
        seen = np.zeros(n, bool)
        vals[g.t.astype(int)] = g.cases.to_numpy(float)
        seen[g.t.astype(int)] = g.seen.to_numpy(bool)
        series[(cond, district)] = (vals, seen)
    return series


def feat(vals, seen, t, mode, mask=False):
    arr = vals.copy()
    if mode == "zero_fill":
        arr[~seen | ~np.isfinite(arr)] = 0.0
    else:
        last = 0.0
        fill = arr.copy()
        for i in range(t):
            if seen[i] and np.isfinite(arr[i]):
                last = float(arr[i])
            fill[i] = last
        arr = fill
    x = [arr[t - k] if t >= k else 0.0 for k in LAGS]
    x.append(float(np.mean(arr[max(0, t - 4):t])) if t else 0.0)
    x.extend([np.sin(2 * np.pi * (t % 52) / 52), np.cos(2 * np.pi * (t % 52) / 52)])
    if mask:
        x.extend([float(seen[t - k]) if t >= k else 0.0 for k in LAGS])
        x.append(float(np.sum(~seen[max(0, t - 8):t])))
    return np.asarray(x, float)


FEATURE_NAMES_BASE = [f"lag{k}" for k in LAGS] + ["ma4", "sin52", "cos52"]
FEATURE_NAMES_MASK = [f"seen_lag{k}" for k in LAGS] + ["n_missing_8"]


def feature_names(mask):
    return FEATURE_NAMES_BASE + (FEATURE_NAMES_MASK if mask else [])


def week_index_map(weeks):
    return {w: i for i, w in enumerate(weeks)}
