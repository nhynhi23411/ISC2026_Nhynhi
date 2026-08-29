"""Multi-mechanism missingness stress test for Project A.

This experiment extends the original 10/20/40% MCAR-only simulation by:
  * testing 10/20/40/60/80% injected historical missingness;
  * contrasting MCAR, contiguous outage blocks, and value-dependent censoring;
  * evaluating 1-, 2-, and 4-week horizons;
  * repeating every injection and reporting cluster-bootstrap confidence intervals.

Only pre-test history is modified. The locked final 20 origins and their targets
remain untouched. The script is CPU-only and never calls Modal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from project_a_common import (
    COMPLETE_CONDITIONS,
    FINAL_TEST_ORIGINS,
    HORIZONS,
    SEED,
    build_panel,
    build_series,
    load_source,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "project_a_local_results"
RATES = (0.10, 0.20, 0.40, 0.60, 0.80)
MECHANISMS = ("mcar", "block", "value_dependent")


def _target_count(n: int, rate: float) -> int:
    return min(n, max(1, int(round(n * rate))))


def inject_missingness(
    seen: np.ndarray,
    values: np.ndarray,
    eligible: np.ndarray,
    rate: float,
    mechanism: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copied observation mask with exactly the requested cells hidden."""
    injected = seen.copy()
    candidates = np.asarray(eligible, dtype=int)
    k = _target_count(len(candidates), rate)

    if mechanism == "mcar":
        hidden = rng.choice(candidates, size=k, replace=False)
    elif mechanism == "value_dependent":
        candidate_values = np.nan_to_num(values[candidates], nan=0.0)
        ranks = pd.Series(candidate_values).rank(method="average").to_numpy(float)
        weights = (0.05 + ranks / max(len(ranks), 1)) ** 2
        weights /= weights.sum()
        hidden = rng.choice(candidates, size=k, replace=False, p=weights)
    elif mechanism == "block":
        # Simulate reporting outages by growing contiguous blocks. Keep drawing
        # blocks until the exact target count is reached, then trim if needed.
        candidate_set = set(candidates.tolist())
        selected: set[int] = set()
        attempts = 0
        while len(selected) < k and attempts < max(100, 20 * k):
            start = int(rng.choice(candidates))
            length = int(min(12, 1 + rng.geometric(0.28)))
            for idx in range(start, start + length):
                if idx in candidate_set:
                    selected.add(idx)
                    if len(selected) == k:
                        break
            attempts += 1
        if len(selected) < k:
            remainder = np.array(sorted(candidate_set - selected), dtype=int)
            selected.update(rng.choice(remainder, size=k - len(selected), replace=False).tolist())
        hidden = np.array(sorted(selected), dtype=int)
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")

    injected[hidden] = False
    return injected


def last_forecasts(values: np.ndarray, injected_seen: np.ndarray, cutoff: int) -> tuple[float, float]:
    """Zero-fill last-week forecast and reporting-aware last-observed forecast."""
    zero = float(values[cutoff]) if injected_seen[cutoff] and np.isfinite(values[cutoff]) else 0.0
    prior = np.where(injected_seen[: cutoff + 1] & np.isfinite(values[: cutoff + 1]))[0]
    aware = float(values[prior[-1]]) if len(prior) else 0.0
    return zero, aware


def cluster_bootstrap_ci(cluster_delta: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(cluster_delta) < 2:
        return float("nan"), float("nan")
    draws = rng.integers(0, len(cluster_delta), size=(n_boot, len(cluster_delta)))
    boot = cluster_delta[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def run(repeats: int = 20, n_boot: int = 2000, smoke: bool = False, out_dir: Path = DEFAULT_OUT):
    started = time.time()
    long, presence, week_df = load_source()
    panel, weeks, districts, _, n_weeks = build_panel(long, presence, week_df)
    series = build_series(panel, n_weeks)
    test_start = n_weeks - FINAL_TEST_ORIGINS
    conditions = list(COMPLETE_CONDITIONS)
    rates = RATES
    mechanisms = MECHANISMS
    horizons = HORIZONS
    if smoke:
        conditions = conditions[:1]
        districts = districts[:3]
        rates = (0.20,)
        mechanisms = ("mcar", "block", "value_dependent")
        horizons = (1, 4)
        repeats = min(repeats, 2)
        n_boot = min(n_boot, 100)

    rows = []
    for mechanism in mechanisms:
        for rate in rates:
            for rep in range(repeats):
                for condition in conditions:
                    for district in districts:
                        values, seen = series[(condition, district)]
                        eligible = np.where(seen[:test_start] & np.isfinite(values[:test_start]))[0]
                        if len(eligible) < 20:
                            continue
                        key = f"{mechanism}|{rate:.2f}|{rep}|{condition}|{district}".encode("utf-8")
                        seed_offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
                        rng = np.random.default_rng((SEED + seed_offset) % (2**32 - 1))
                        injected_seen = inject_missingness(seen, values, eligible, rate, mechanism, rng)
                        hidden = int(np.sum(seen[:test_start] & ~injected_seen[:test_start]))

                        for horizon in horizons:
                            zero_errors, aware_errors = [], []
                            for target_t in range(test_start, n_weeks):
                                cutoff = target_t - horizon
                                if cutoff < 0 or not seen[target_t] or not np.isfinite(values[target_t]):
                                    continue
                                zero, aware = last_forecasts(values, injected_seen, cutoff)
                                target = float(values[target_t])
                                zero_errors.append(abs(target - zero))
                                aware_errors.append(abs(target - aware))
                            if not zero_errors:
                                continue
                            zero_mae = float(np.mean(zero_errors))
                            aware_mae = float(np.mean(aware_errors))
                            rows.append({
                                "mechanism": mechanism,
                                "rate": rate,
                                "rep": rep,
                                "condition": condition,
                                "district": district,
                                "horizon": horizon,
                                "n_targets": len(zero_errors),
                                "eligible_history": len(eligible),
                                "hidden_history": hidden,
                                "realized_rate": hidden / len(eligible),
                                "zero_MAE": zero_mae,
                                "aware_MAE": aware_mae,
                                "delta_zero_minus_aware": zero_mae - aware_mae,
                            })

    clusters = pd.DataFrame(rows)
    if clusters.empty or not np.isfinite(clusters[["zero_MAE", "aware_MAE"]].to_numpy()).all():
        raise RuntimeError("stress test produced empty or non-finite results")

    # Average repeated injections within each condition-district cluster before
    # bootstrapping, so repetitions do not masquerade as independent districts.
    cluster_means = clusters.groupby(
        ["mechanism", "rate", "condition", "district", "horizon"], as_index=False
    ).agg(
        n_targets=("n_targets", "sum"),
        realized_rate=("realized_rate", "mean"),
        zero_MAE=("zero_MAE", "mean"),
        aware_MAE=("aware_MAE", "mean"),
        delta_zero_minus_aware=("delta_zero_minus_aware", "mean"),
    )

    summary_rows = []
    boot_rng = np.random.default_rng(SEED + 991)
    for (mechanism, rate, horizon), group in cluster_means.groupby(["mechanism", "rate", "horizon"]):
        weights = group.n_targets.to_numpy(float)
        zero = float(np.average(group.zero_MAE, weights=weights))
        aware = float(np.average(group.aware_MAE, weights=weights))
        delta = group.delta_zero_minus_aware.to_numpy(float)
        lo, hi = cluster_bootstrap_ci(delta, boot_rng, n_boot)
        summary_rows.append({
            "mechanism": mechanism,
            "rate": rate,
            "horizon": horizon,
            "n_condition_district_clusters": len(group),
            "n_targets_across_repeats": int(group.n_targets.sum()),
            "zero_MAE": zero,
            "aware_MAE": aware,
            "delta_zero_minus_aware": zero - aware,
            "cluster_bootstrap_ci_low": lo,
            "cluster_bootstrap_ci_high": hi,
            "share_clusters_aware_better": float(np.mean(delta > 0)),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["mechanism", "horizon", "rate"])

    by_condition = cluster_means.groupby(
        ["mechanism", "rate", "condition", "horizon"], as_index=False
    ).agg(
        n_districts=("district", "nunique"),
        zero_MAE=("zero_MAE", "mean"),
        aware_MAE=("aware_MAE", "mean"),
        delta_zero_minus_aware=("delta_zero_minus_aware", "mean"),
        share_districts_aware_better=("delta_zero_minus_aware", lambda s: float(np.mean(s > 0))),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(out_dir / "missingness_stress_replicates.csv", index=False)
    cluster_means.to_csv(out_dir / "missingness_stress_clusters.csv", index=False)
    summary.to_csv(out_dir / "missingness_stress_summary.csv", index=False)
    by_condition.to_csv(out_dir / "missingness_stress_by_condition.csv", index=False)
    manifest = {
        "status": "ok",
        "simulation_scope": "historical observation masks only; locked test targets untouched",
        "mechanisms": list(mechanisms),
        "rates": list(rates),
        "horizons": list(horizons),
        "repeats": repeats,
        "bootstrap_draws": n_boot,
        "conditions": conditions,
        "districts": len(districts),
        "test_origins": FINAL_TEST_ORIGINS,
        "replicate_rows": len(clusters),
        "cluster_rows": len(cluster_means),
        "summary_rows": len(summary),
        "seed": SEED,
        "wall_time_seconds": round(time.time() - started, 2),
        "modal_used": False,
    }
    (out_dir / "missingness_stress_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    return summary, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.repeats, args.bootstrap, args.smoke, args.out_dir)


if __name__ == "__main__":
    main()
