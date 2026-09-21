# Methodology

This document describes the experimental design actually implemented and run for this study. It is written from `protocol_v2.yaml` (the frozen, machine-readable configuration) and the project's frozen-protocol tracking notes, and it describes the final `PHASE_2`–`PHASE_8` pipeline in `code/` — not any earlier exploratory design.

## 1. Task and data

- **Outcome**: `label_mortality`, binary in-hospital mortality. N = 14,081 admissions, 358 positive (prevalence 2.54%).
- **Source**: MIMIC-IV v3.1 derived (see the "Data access" section of the top-level `README.md` for access instructions — the raw dataset is not distributed with this repository, per the PhysioNet Data Use Agreement).
- **Prediction landmark / lab observation window**: first 24 hours of admission. Every `lab_*` feature and every admission-context/demographic feature is treated as available at this landmark.
- **Split**: `StratifiedGroupKFold(subject_id)`, 5 outer folds, fixed across the whole study. `subject_id` is the group key so no patient appears in both the train and test side of a fold. A legacy alternative split (`StratifiedKFold`, no patient separation) is run once as an explicit sensitivity check (Phase 8, Part 4) — never used as a headline result.
- **Feature set**: see `feature_manifest.csv` for the full per-feature table (type, native missingness %, whether it is model input, whether it is eligible for synthetic masking, whether it is a MAR driver). In brief: 10 demographic/admission-context columns (never masked, always model input) + 30 `lab_*` columns (the only columns eligible for synthetic masking). `hadm_id` and `subject_id` are identifiers only. `discharge_location` is excluded from the model entirely — it is a near-1:1 post-index proxy for the label (353/358 positive cases have `discharge_location == "DIED"`) and is guarded by a permanent regression-guard unit test.

## 2. Missingness mechanisms

Three synthetic missingness mechanisms are injected on top of the real, already-present (native) missingness, using `code/PHASE_2_GENERATOR_V2/missingness_generator_v2.py`. Injection is restricted to originally-observed eligible cells (the 30 `lab_*` columns) — synthetic missingness never overwrites an already-missing value.

- **MCAR**: uniform random injection, no dependence on any covariate or the outcome.
- **MAR (admission-context-dependent)**: `P(S_ij=1) = sigmoid(alpha_j + theta * H_i)`, where `H_i` is a binary admission-context driver derived from `admission_type` (never the outcome). Main strength θ = ln(2) ≈ 0.693 (OR = 2).
- **MNAR-Y (outcome-dependent)**: `P(S_ij=1 | Y_i) = sigmoid(alpha_j + theta * Y_i)`. Main strength OR = 2.0 (same effect size as MAR, for direct comparability); sensitivity sweep at OR ∈ {1.5, 2.0, 4.0}.
- **MNAR-X (value-dependent, sensitivity only)**: `P(S_ij=1) = sigmoid(alpha_j + gamma * |z_ij|_clipped)`, i.e. injection probability depends on how extreme the lab value itself is (X-side only, no outcome dependence). `GAMMA_MAIN = ln(2)/2` (OR = 2.0 at the |z| = 2 reference point), with `z_clip = 5.0` to prevent underflow on a handful of real lab columns with very heavy tails.

For every mechanism, the per-column intercept `alpha_j` is calibrated by bisection search (fit on the train partition of each fold only, then applied unchanged to that fold's test partition) so that the realized injection rate matches the target rate to within ≈0.001–0.002 at every condition tested. Main rate grid: q ∈ {0, 0.10, 0.20, 0.30, 0.40}; a severe-rate sensitivity point at q = 0.50 is also run for a subset of conditions.

## 3. Model selection procedure

Five model families are evaluated: `lr_l2` (L2-regularized logistic regression), `svm_linear_cal` (calibrated linear SVM), `rf` (random forest), `extratrees`, `xgb` (XGBoost). For each outer fold:

1. An **inner, grouped cross-validation** on the outer-train partition only is used to tune each family's hyperparameters and to select the winning family by inner-CV AUROC.
2. The winning family is refit on the full outer-train partition.
3. The refit model is evaluated **once** on the outer-test partition.

The outer-test partition never influences which family is selected — this is enforced structurally (the selection function has no access to outer-test data), not just by convention. This closes a real bug in an earlier iteration of the codebase, where the winning family was chosen by best *mean outer-test* AUROC across folds — a form of test-set leakage into the model-selection decision. The fix, the located bug, and a fold-by-fold demonstration that it actually changes the selected family in at least one real fold are documented in the project's internal audit trail (not included in this repository; see the top-level README for how to request it).

## 4. Metrics

- **Primary selection criterion**: AUROC (inner-CV). Average precision (AP) is used as a selection-criterion sensitivity check.
- **Reported metrics**: AUROC, average precision, Brier score, calibration slope/intercept.
- **Stability/attribution metrics** (`code/PHASE_7_METRIC_LAYER/metrics_v2.py`): `baseline_selection_displacement_rate` (does the selected family at condition X differ from the family selected at the q=0 baseline, same fold?), `within_condition_selection_entropy` (how concentrated is the selected-family distribution across the 5 folds?), Kendall's τ_b rank-stability, and `selection_regret` (AUROC gap between the selected family and the fold's true outer-test oracle).

## 5. Research questions and which driver answers them

| RQ | Question | Driver | Results |
|---|---|---|---|
| RQ1 | Does the missingness mechanism change model-family selection stability, holding the injected rate constant? | `code/PHASE_4_RQ1_RQ2/run_phase4_rq1_rq2.py` | `docs/results_rq1_rq2.md` |
| RQ2 | How does model performance and selection stability change with MNAR-Y strength (OR)? | `code/PHASE_4_RQ1_RQ2/run_phase4_rq1_rq2.py` | `docs/results_rq1_rq2.md` |
| RQ3 | What happens when a model trained under one mechanism is deployed where the missingness mechanism differs? | `code/PHASE_5_RQ3/run_phase5_rq3.py` | `docs/results_rq3.md` |
| RQ4 | How much of the MNAR-Y performance gain is a genuine value-based signal vs. the model exploiting the missingness pattern itself as a shortcut? | `code/PHASE_6_RQ4/run_phase6_rq4.py`, `code/PHASE_6_SECTION_B_REPEATS/run_phase6_section_b_repeats.py` | `docs/results_rq4.md` |

Additional sensitivity analyses were run — MNAR-X (a fourth, value-dependent mechanism), a calibration-drift backfill across the full grid, an AP-based (vs. AUROC-based) selection-criterion check, and the legacy-split (no patient separation) sensitivity check — using `code/PHASE_8_CALIBRATION/`, `code/PHASE_8_MNAR_X/`, and `code/PHASE_8_S1_SENSITIVITY/`. These sensitivity results are not reproduced as standalone documents here; the drivers that produced them are included in full under `code/`.

## 6. Statistical layer

`code/PHASE_STATS_LAYER/phase_stats_layer.py` is a from-scratch, driver-independent analysis script that computes paired fold-level significance tests and confidence intervals for every headline claim across RQ1–RQ4, run directly on the already-produced result tables (no new experiments). Its central finding: with only n = 5 paired outer folds, the minimum achievable two-sided sign-test/Wilcoxon p-value is 0.0625, which cannot cross the conventional α = 0.05 threshold no matter how consistent an effect is — several headline comparisons are directionally consistent across all 5 folds but do not clear this floor on their own. `code/CASE_LEVEL_BOOTSTRAP_CI/case_level_bootstrap_ci.py` computes a complementary, non-substituting case-level (patient-level) bootstrap CI on headline AUROC/AP point estimates from already-saved per-case predictions — this characterizes finite-test-set sampling variance on a single point estimate, not cross-fold generalization, and does not get past the n=5 floor either. Both distinctions are stated explicitly wherever these numbers are reported (see `docs/results_rq1_rq2.md` §7).

## 7. Reproduction

Each `PHASE_*` folder under `code/` is self-contained: it carries its own copy of the shared modules (`missingness_generator_v2.py`, `selection_v2.py`, and, where needed, `metrics_v2.py`) as they actually existed when that phase was run, plus a vendored copy of `jcsse_audit_runner_tqdm_hardened.py` where the driver depends on it. This mirrors exactly how the experiments were executed and guarantees that running any driver reproduces the reported results without depending on a module version from a different phase. See the top-level `README.md` for exact run commands and the data-access prerequisite.
