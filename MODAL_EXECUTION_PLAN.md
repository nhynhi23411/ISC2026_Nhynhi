# Project A — Modal execution plan and approval gates

## Operating agreement

This project will be executed as a sequence of separately approved tasks. Completing one task does **not** authorize the next task.

At the end of every task, Codex must report:

1. what was changed or executed;
2. exact validation evidence;
3. files and outputs created;
4. estimated cost/risk of the next task;
5. the exact next command or action proposed;
6. a request for explicit user approval before continuing.

No paid/full Modal experiment may be launched automatically.

## Mandatory Modal safety rules

1. **Never start with a full run.** The required order is static validation → local subset test → remote smoke test → limited pilot → full run.
2. **Remote execution requires explicit approval.** This includes `modal run`, `modal deploy`, scheduled functions, detached jobs, and volume-mutating remote functions.
3. **Local validation must pass first.** At minimum:
   - Python syntax/compile check;
   - import check;
   - unit tests for panel construction, observation status, lag boundaries and chronological splits;
   - execution on a tiny local subset;
   - output schema and metric sanity checks.
4. **Smoke jobs must be deliberately small.** Use one or two conditions, a small number of districts/origins, one horizon, minimal hyperparameter search, CPU only, short timeout, and a unique smoke output directory.
5. **Do not overwrite promoted results.** Every run receives a unique `run_id`; smoke, pilot and full outputs live in separate directories.
6. **Fail fast.** Assertions must stop the job on duplicate keys, leaked future data, invalid splits, missing required files, non-finite metrics, empty evaluation sets or unexpected schema changes.
7. **Bound cost explicitly.** Every proposed remote run must state CPU/GPU, memory, timeout, number of tasks/model fits and expected wall time. GPU is forbidden unless separately justified and approved.
8. **Do not use detached execution by default.** The job should remain observable and return a final status. Detached/background execution requires separate approval.
9. **Persist only required artifacts.** Save configurations, logs, predictions, metrics, split indices and figures; do not upload unnecessary raw or temporary files.
10. **Treat the final chronological test block as locked.** Smoke and pilot runs may validate code paths but must not be used for model/threshold selection on the final test block.
11. **Promotion is explicit.** A smoke or pilot configuration becomes a full configuration only after its validation report is reviewed and approved.
12. **Stop on ambiguous scientific choices.** Examples include redefining the six confirmatory conditions, changing the final test period, changing the primary metric or excluding flagged weeks.

## Proposed directory layout

```text
modal_project_a/
  app.py                       # Modal app and remote entry points
  config.py                    # frozen run configurations
  core/
    data.py                    # panel loading/audit
    features.py                # lag/mask/seasonality features
    splits.py                  # rolling-origin splits
    models.py                  # baselines and ML models
    evaluate.py                # metrics, paired comparisons, intervals
  tests/
    test_data.py
    test_features.py
    test_splits.py
    test_smoke.py
  requirements-modal.txt
  README.md

modal_outputs/
  smoke/<run_id>/
  pilot/<run_id>/
  full/<run_id>/
```

The original downloaded dataset remains under `project_a_data/`. Existing MVP outputs under `project_a_results/` will not be overwritten.

## Task sequence

### Task 0 — Planning and governance

**Status:** completed by creating this file.

Deliverable:

- `PROJECT_A_MODAL_EXECUTION_PLAN.md`

No Modal job is executed in Task 0.

---

### Task 1 — Local Modal scaffold and correctness tests

Purpose: prepare the Modal application without starting any remote compute.

Work:

- Refactor the current scripts into reusable modules.
- Create `modal_project_a/app.py` with smoke/pilot/full modes.
- Add frozen configuration objects and unique run IDs.
- Add local tests for:
  - exact 164-week ordering;
  - six prespecified confirmatory diseases;
  - four observation-status classes;
  - lag and rolling-window causality;
  - rolling-origin train/validation/test ordering;
  - preprocessing fitted only on historical/training data;
  - paired zero-fill/reporting-aware prediction keys;
  - expected output schemas.
- Run local syntax, imports, unit tests and a tiny end-to-end subset.

Modal cost: **zero remote compute**.

Pass criteria:

- all tests pass;
- tiny local run produces non-empty finite metrics;
- no target/future leakage assertion fires;
- existing MVP result files remain unchanged.

Approval gate: after Task 1, present the test report and request approval for Task 2.

---

### Task 2 — Modal environment/image build validation

Purpose: prove that the remote container can import dependencies and access staged data, without running the experiment.

Proposed environment:

- Python 3.11;
- pandas, numpy, scikit-learn;
- LightGBM;
- statsmodels or a validated count-model alternative;
- pyarrow and matplotlib only if required by artifact generation.

Proposed remote resources:

- CPU only;
- 1 CPU;
- 2–4 GB RAM;
- timeout 5 minutes;
- one function call that imports packages, reads a small staged fixture and exits.

Pass criteria:

- image builds successfully;
- exact package versions are logged;
- staged fixture checksum matches local checksum;
- no training is performed;
- estimated/observed runtime is reported.

Approval gate: show the exact `modal run` command and resource bounds before launching Task 2.

---

### Task 3 — Remote smoke experiment

Purpose: verify the complete remote data → feature → fit → predict → save path.

Smoke scope:

- 1 complete condition and 1 rotating condition;
- at most 3 districts;
- horizon 1 only;
- at most 2 rolling origins;
- seasonal naive, moving average, regularized linear/Poisson model and one small LightGBM fit;
- zero-fill and reporting-aware modes;
- no broad tuning;
- bootstrap limited to a small diagnostic count.

Proposed resources:

- CPU only;
- 2 CPUs;
- 4 GB RAM;
- timeout 10–15 minutes;
- unique `smoke/<run_id>` output.

Pass criteria:

- job exits successfully;
- every model/mode has predictions with matching paired keys;
- all metrics are finite;
- chronological split and leakage assertions pass;
- outputs can be downloaded and reopened locally;
- smoke output is clearly labeled non-publication.

Approval gate: present smoke logs, output manifest and failures/warnings before proposing Task 4.

---

### Task 4 — Limited pilot

Purpose: estimate runtime, memory and scientific signal before the full experiment.

Pilot scope:

- all six confirmatory conditions plus selected rotating conditions;
- representative subset of districts;
- horizons 1 and 4;
- several rolling origins, excluding use of the final block for tuning;
- baselines, Poisson/count-aware model and LightGBM;
- small fixed hyperparameter grid;
- limited paired/bootstrap uncertainty.

Proposed resources:

- CPU only unless evidence shows a GPU is beneficial;
- 4 CPUs;
- 8 GB RAM;
- timeout determined from Task 3, initially capped at 30–45 minutes;
- unique `pilot/<run_id>` output.

Pass criteria:

- runtime and memory remain within the declared bound;
- no task failures or missing model-condition combinations;
- predicted counts and error distributions pass sanity checks;
- paired effects can be reproduced from saved predictions;
- a full-run cost estimate is produced.

Approval gate: user reviews the pilot evidence and full-run estimate before Task 5.

---

### Task 5 — Frozen full primary experiment

Purpose: run the prespecified primary experiment once under a frozen configuration.

Scope:

- six confirmatory diseases;
- all eligible canonical reporting units with prespecified geographic exclusions;
- horizons 1, 2 and 4;
- rolling-origin evaluation;
- untouched final chronological block;
- zero-fill vs reporting-aware paired preprocessing;
- transparent baselines, count-aware model and one strong gradient-boosting model;
- saved split indices, predictions, metrics and configuration hash.

Before approval, present:

- exact frozen config;
- included/excluded weeks and reporting units;
- model list and hyperparameter search size;
- number of expected fits;
- CPUs, RAM, timeout and estimated cost/runtime;
- rollback/recovery behavior.

Pass criteria:

- all expected fits accounted for;
- final test block accessed only by the frozen run;
- complete metrics and denominators;
- results regenerate from stored predictions;
- no silent partial success.

Approval gate: full run requires a new explicit approval even if all prior tasks passed.

---

### Task 6 — Prespecified robustness and rotating-condition stress test

Possible analyses, each declared before running:

- exclude duplicate source week 2025w9;
- exclude quality-flagged/inconsistent weeks;
- South Waziristan ambiguity handling;
- rotating-condition stress test;
- remove observation-process features;
- negative-binomial vs alternative objectives;
- district/disease stratification;
- block-bootstrap/per-origin uncertainty.

These may be split into separate approved sub-tasks if the pilot indicates high cost.

---

### Task 7 — Result QA and manuscript artifacts

Work:

- download and checksum final outputs;
- reconcile all tables against prediction-level files;
- create publication-quality figures;
- create the frozen results summary and limitations log;
- identify which claims are supported, unsupported or exploratory;
- prepare manuscript-ready tables without rerunning models.

No new Modal compute is authorized by approval of this task unless separately stated.

## Required report template after every task

```text
Task completed:
Changes/files:
Commands executed:
Validation evidence:
Warnings or unresolved issues:
Remote resources and observed runtime/cost:

Proposed next task:
Exact proposed command/action:
Resource cap and estimated runtime/cost:
Decision requested from user:
```

## Current approval state

- Task 0: completed.
- Task 1: completed; see `TASK1_REPORT.md`.
- Task 2: completed; see `TASK2_REPORT.md`.
- Task 3: completed; see `TASK3_REPORT.md`.
- Tasks 4–5: completed as one approved full baseline run; see `FULL_MODAL_RUN_REPORT.md`.
- Task 6: extended model/uncertainty/simulation run completed; see `EXTENDED_MODAL_RUN_REPORT.md`.
- Approved environment, end-to-end smoke, full baseline and extended jobs have run.
