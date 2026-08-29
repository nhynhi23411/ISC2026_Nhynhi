# Project A — execution status (updated 2026-08-29)

## What has been completed

- Downloaded the official Mendeley Data release (DOI `10.17632/9yzvfgrhkt.1`, CC BY 4.0).
- Verified the primary long file: 59,824 rows, 164 bulletin weeks, 44 canonical reporting units, 16 conditions.
- Built a complete week × district × condition panel with explicit observation-status fields.
- Preserved the distinction between:
  - `observed_target`: condition and district/cell present with a numeric case count;
  - `reported_but_NR`: source row present but case count is NR/blank;
  - `district_or_cell_unreported`: condition is in that week's columns but the district/cell is absent;
  - `structurally_unreported`: rotating condition is not included in that week's ranked columns.
- Ran rolling-origin forecasts at 1-, 2- and 4-week horizons.
- Compared `zero_fill` and `reporting_aware` variants for last-value, seasonal-naive and 4-observation moving-average baselines.
- Generated paired effect estimates with bootstrap 95% intervals.
- Added an optional pooled lag-ridge implementation (`project_a_ridge.py`). It is not part of the completed MVP run because the dependency-light implementation is CPU-expensive; the baseline evidence is complete and reproducible.
- Completed the extended local model family: Poisson, Negative-Binomial, Random Forest, LightGBM with/without observation-mask features, paired bootstrap, uncertainty, quality-flag sensitivity, propensity, hazard, IPW, and trajectory analyses.
- Added a multi-mechanism missingness stress test (`project_a_missingness_stress.py`) covering 10/20/40/60/80% injected historical missingness, MCAR, contiguous outage blocks, value-dependent censoring, horizons 1/2/4, 20 repetitions, and cluster-bootstrap 95% intervals.

## Audit findings

| Quantity | Value |
|---|---:|
| Long-format source rows | 59,824 |
| Bulletin weeks | 164 |
| Reporting units | 44 |
| Conditions | 16 |
| Complete six-condition panel cells | 43,296 |
| Rotating-condition panel cells | 72,160 |
| Structurally unreported rotating cells | 43,384 |
| District/cell unreported cells | 12,248 |
| Reported-but-NR cells | 291 |

The six confirmatory conditions are `AD_noncholera`, `Malaria`, `ILI`, `ALRI_u5`, `Bloody_diarrhoea`, and `Typhoid`. Structural non-reporting is absent for these six at the condition-column level, so the strongest zero-fill contrast appears in the rotating-condition stress test. This is expected and should be stated explicitly in the manuscript.

## MVP test results

On the untouched final chronological block (last 20 forecast origins), the best baseline was the 4-observation moving average. At horizon 1:

| Mode | MAE | RMSE |
|---|---:|---:|
| Zero-filled | 30.1934 | 115.6813 |
| Reporting-aware | 30.1703 | 115.6734 |

The paired analysis is more informative than the pooled mean. For the rotating-condition stress test, reporting-aware preprocessing reduced absolute error relative to zero-fill by:

- last-value, horizon 1: **0.167 MAE**, bootstrap 95% CI **[0.057, 0.299]**;
- seasonal-naive, horizon 1: **0.634 MAE**, bootstrap 95% CI **[0.421, 0.876]**;
- moving average, horizon 1: **0.058 MAE**, bootstrap 95% CI **[-0.009, 0.135]**.

For the complete six-condition confirmatory set, the last-value and moving-average contrasts are essentially zero, while the seasonal-naive contrast favors reporting-aware handling by about **0.31 MAE**. This supports a restrained claim: the current MVP demonstrates a measurable consequence of zero-filling mainly where structural reporting rotation exists; it does not yet establish a large improvement for the six complete conditions.

## Reproduction

```powershell
$py = 'C:\Users\pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$data = 'D:\Thầy Khánh\ISC\project_a_data\District-week notifiable infectious disease surveillance data for Khyber Pakhtunkhwa, Pakistan, 2023-2026'
& $py project_a_pipeline.py --data-dir $data --out project_a_results
& $py project_a_analysis.py --pred project_a_results\forecast_predictions.csv --out project_a_results
```

The main artifacts are:

- `project_a_results/audit_summary.json`
- `project_a_results/audit_by_condition.csv`
- `project_a_results/panel_with_observation_status.csv`
- `project_a_results/forecast_predictions.csv`
- `project_a_results/metrics_overall.csv`
- `project_a_results/metrics_by_condition.csv`
- `project_a_results/paired_effects_bootstrap.csv`
- `project_a_results/macro_series_metrics_test.csv`
- `project_a_pipeline.py`
- `project_a_analysis.py`
- `project_a_ridge.py` (optional pooled lag-ridge extension)

## Remaining work before a submission-quality paper

1. Render the final DOCX with Word/LibreOffice and verify pagination, figures, references, and metadata.
2. Copy the freestyle pipeline figure and newest stress-test artifacts into `ProjectA_Complete_Package` and push the synchronized package to GitHub.
3. Align all extended reports with the audited non-causal wording for controlled simulation.
4. Add external validation or a second surveillance dataset if time permits; this is the highest-value scientific extension but is not required for the current reproducibility package.
5. Freeze the protocol and submit after supervisor review.
