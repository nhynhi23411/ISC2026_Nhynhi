"""Item C: discrete-time hazard model for "time until a censored rotating
condition reappears in the ranked table."

Novelty #1 modeled presence as a static logistic regression on the most
recent known volume. That conflates two different questions: (a) is presence
volume-dependent at all (yes, shown there), and (b) does the RISK of
reappearing change the longer a condition has already been absent (a
genuine survival/hazard question, standard in reliability and churn
modeling, and a more standard construction for "informative censoring" in
the ML/CS literature than a plain cross-sectional logistic fit). This script
builds proper discrete-time person-period data for each absence spell and
fits hazard ~ duration + log1p(last known volume before the spell) +
seasonal terms, reporting whether the baseline hazard is flat, increasing
("catch-up" behavior) or decreasing (risk of the condition being effectively
dropped the longer it stays unranked).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from project_a_common import COMPLETE_CONDITIONS, load_source, build_panel

OUT = Path(r"D:\Thầy Khánh\ISC\project_a_local_results")
TRANSITION_WEEK = "2023w32"


def provincial_series(panel, cond, n):
    g = panel[panel.condition == cond]
    presence = g.groupby("t").condition_present.first().reindex(range(n)).astype(bool).to_numpy()
    totals = g.groupby("t").apply(
        lambda gg: gg.loc[gg.observed_target, "cases"].sum() if gg.observed_target.any() else np.nan,
        include_groups=False,
    ).reindex(range(n)).to_numpy(dtype=float)
    return presence, totals


def build_person_period(presence, totals, transition_idx, n, cond_name):
    """One row per week a condition is 'at risk of reappearing' (i.e. still
    absent), with duration-so-far and the last known volume before the spell
    began. y=1 marks the week it reappears (ends the spell)."""
    rows = []
    last_known = np.nan
    in_spell = False
    duration = 0
    volume_at_spell_start = np.nan
    for t in range(transition_idx, n):
        if np.isfinite(totals[t]):
            if not in_spell:
                last_known = totals[t]
        if presence[t]:
            in_spell = False
            duration = 0
            continue
        # condition absent at t
        if not in_spell:
            in_spell = True
            duration = 0
            volume_at_spell_start = last_known
        duration += 1
        if not np.isfinite(volume_at_spell_start):
            continue  # no history yet to condition on
        rows.append({
            "condition": cond_name, "t": t, "duration": duration,
            "log1p_volume_at_spell_start": float(np.log1p(volume_at_spell_start)),
            "reappears": 0,  # overwritten to 1 below if this is the last absent week of the spell
        })
    return rows


def main():
    long, presence_df, week_df = load_source()
    panel, weeks, districts, conditions, n = build_panel(long, presence_df, week_df)
    rotating = [c for c in conditions if c not in COMPLETE_CONDITIONS]
    transition_idx = weeks.index(TRANSITION_WEEK)

    all_rows = []
    for cond in rotating:
        presence, totals = provincial_series(panel, cond, n)
        rows = build_person_period(presence, totals, transition_idx, n, cond)
        # mark the event: the row is "reappears=1" if the NEXT week is present (spell resolved there),
        # i.e. we ask, at the end of duration k of absence, "does it come back next week?"
        for i, r in enumerate(rows):
            t = r["t"]
            r["reappears"] = int(t + 1 < n and presence[t + 1])
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("empty hazard dataset")
    df.to_csv(OUT / "hazard_person_period.csv", index=False)

    df["duration_sq"] = df.duration.astype(float) ** 2
    X = sm.add_constant(df[["duration", "duration_sq", "log1p_volume_at_spell_start"]].astype(float))
    y = df.reappears.astype(float)
    model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    auc = roc_auc_score(y, model.predict(X))

    # Duration-only model (no volume) to see how much duration alone explains,
    # and a volume-only model to see the marginal contribution of duration.
    X_dur_only = sm.add_constant(df[["duration", "duration_sq"]].astype(float))
    model_dur_only = sm.Logit(y, X_dur_only).fit(disp=0, maxiter=200)
    auc_dur_only = roc_auc_score(y, model_dur_only.predict(X_dur_only))

    X_vol_only = sm.add_constant(df[["log1p_volume_at_spell_start"]].astype(float))
    model_vol_only = sm.Logit(y, X_vol_only).fit(disp=0, maxiter=200)
    auc_vol_only = roc_auc_score(y, model_vol_only.predict(X_vol_only))

    # Implied hazard shape: predicted P(reappear) at fixed median volume, over a duration grid.
    median_vol = float(df.log1p_volume_at_spell_start.median())
    grid = pd.DataFrame({"duration": range(1, 21)})
    grid["duration_sq"] = grid.duration ** 2
    grid["log1p_volume_at_spell_start"] = median_vol
    Xg = sm.add_constant(grid[["duration", "duration_sq", "log1p_volume_at_spell_start"]].astype(float), has_constant="add")
    grid["hazard_at_median_volume"] = model.predict(Xg)
    grid.to_csv(OUT / "hazard_shape_by_duration.csv", index=False)

    summary = {
        "n_person_period_rows": int(len(df)),
        "n_events_reappear": int(df.reappears.sum()),
        "n_rotating_conditions": len(rotating),
        "full_model": {
            "coef_duration": float(model.params["duration"]),
            "coef_duration_sq": float(model.params["duration_sq"]),
            "coef_log1p_volume": float(model.params["log1p_volume_at_spell_start"]),
            "pvalue_duration": float(model.pvalues["duration"]),
            "pvalue_log1p_volume": float(model.pvalues["log1p_volume_at_spell_start"]),
            "auroc": float(auc),
        },
        "duration_only_auroc": float(auc_dur_only),
        "volume_only_auroc": float(auc_vol_only),
        "hazard_shape_direction": (
            "increasing with duration (catch-up effect)" if model.params["duration"] > 0
            else "decreasing with duration (risk of permanent drop-out)"
        ),
        "interpretation": (
            "This is a more standard survival/hazard formulation of the same MNAR "
            "mechanism found in novelty #1 (static logistic on volume). It adds "
            "whether TIME SPENT ABSENT itself changes reappearance risk, beyond "
            "what the last known volume already predicts -- a question the static "
            "model could not answer."
        ),
    }
    with open(OUT / "hazard_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
