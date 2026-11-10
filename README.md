# Decision Robustness under Missingness Shift in MIMIC-IV Mortality Prediction

Code for the study "Decision Robustness under Missingness Shift: Model-Selection Stability and Shortcut Dependence in MIMIC-IV Mortality Prediction". The study injects missing values into 30 laboratory variables of a MIMIC-IV in-hospital mortality cohort under controlled mechanisms and asks three questions: does the cross-validated choice of model family change with the mechanism, does a frozen model keep its performance when the mechanism changes at deployment, and how much of the discrimination under outcome-linked missingness is carried by the missingness pattern itself.

## Main results

All values below come from the result files written by the code in this repository (five patient-grouped outer folds unless stated; AUROC = area under the receiver operating characteristic curve).

- **Cohort.** 14,081 admissions of 6,179 patients, 358 in-hospital deaths; 39 model inputs (30 laboratory values, age, anchor year, 7 one-hot categorical features).
- **Selection.** XGBoost was selected in 68 of 70 injected condition-fold cells. The selected family differed from the q = 0 selection in 2/20 folds under MCAR and 0/20 under MAR and outcome-dependent missingness (MNAR-Y); over 25 folds (five split and mask seeds) the counts were 3/100, 0/100 and 0/100. Wilson upper bounds of 30.1% and 16.1% (five folds) mean the design detects only large mechanism effects on selection.
- **Discrimination.** Selected-family AUROC was 0.9581 ± 0.0082 at q = 0 and 0.9345, 0.9377 and 0.9432 under MCAR, MAR and MNAR-Y at q = 0.40. Under MNAR-Y at q = 0.30 it rose with the odds ratio: 0.9384 (OR 1.5), 0.9446 (OR 2), 0.9757 (OR 4).
- **Deployment shift.** A frozen model moved into or out of MNAR-Y at q = 0.30 lost 0.0164 to 0.0246 AUROC (negative in 5/5 folds; −0.0149 to −0.0264 and negative in at least 24 of 25 folds over the repeats); shifts between MCAR and MAR changed AUROC by −0.0013 and +0.0001. Over q = 0.10, 0.30 and 0.50, the four losses into or out of MNAR-Y grew with the rate (slopes −0.0733 to −0.0916 AUROC per unit rate, Benjamini–Hochberg adjusted p ≤ 0.00022 across six shifts).
- **Reliance on the mask.** Permuting the injected test mask across admissions changed the frozen MNAR-Y model's AUROC by +0.0093 (OR 1.5), −0.0099 (OR 2) and −0.0862 (OR 4) at q = 0.30 (means over 150 draws); at OR 4 the loss exceeds the whole OR 2 → OR 4 gain (+0.0311). The permutation change measures the frozen model's reliance on the pattern; it does not split AUROC into mask and laboratory parts.
- **Missingness handling.** At MNAR-Y OR 4 the permutation loss was −0.1592 with 30 missingness indicators, −0.0883 with median imputation, −0.0384 with multiple imputation (m = 5) and −0.0197 with single chained-equation imputation; XGBoost was selected in 96 of 100 cells across the four strategies.

Scope: one cohort from one institution, synthetic missingness mechanisms with fixed functional forms, one dominant model family. MNAR-Y at OR 4 is an upper stress level, not a model of a clinical process. The strategy comparisons use one rate (q = 0.30) and are unadjusted.

## Repository layout

```
code/                         analysis code, exactly as run
  main_study/                 main grid and attribution analyses, one folder per phase
    PHASE_4_RQ1_RQ2/          rate sweep and MNAR-Y strength sweep (selection, AUROC, displacement, regret)
    PHASE_5_RQ3/              frozen source-to-target mechanism shift at q = 0.30 (and MCAR source at 0.10, 0.50)
    PHASE_6_RQ4/              mask-only and count-only baselines, test-time mask permutation, indicator ablation
    PHASE_6_SECTION_B_REPEATS/ 30 permutation draws per fold
    PHASE_8_MNAR_X/           value-dependent missingness (sensitivity)
    PHASE_8_CALIBRATION/      calibration slope and intercept (sensitivity)
    PHASE_8_S1_SENSITIVITY/   outer split without patient grouping (sensitivity)
    PHASE_STATS_LAYER/        fold-level tests, bootstrap intervals, Benjamini–Hochberg adjustment
    CASE_LEVEL_BOOTSTRAP_CI/  pooled case-level AUROC/AP intervals
    Figures/                  single-panel figures and the three two-panel paper figures
  run_r1_missingness_handling.py   median, median + indicators, chained-equation imputation
  run_r1_multiple_imputation.py    multiple imputation (--m 5) and the single-draw arm (--m 1)
  run_r1_repeated_outer_cv.py      five split and mask seeds (25 folds) for the main grid and q = 0.30 shifts
  run_r1_permutation_rate_shift.py permutation at OR 1.5/2/4 (q = 0.30) and OR 2/4 (q = 0.40); 3 x 3 shift at q = 0.10, 0.30, 0.50
  analyse_r1_*.py                  read-only analyses that print every reported number
  test_r1_*.py                     known-answer tests
docs/README.md                which script and output file supports each table and figure of the paper
feature_manifest.csv          role, timing, native missingness and model use of every column
requirements.txt              package versions recorded in the run logs
```

Each phase folder under `code/main_study/` carries its own copy of the shared modules (`missingness_generator_v2.py`, `selection_v2.py`, `metrics_v2.py`, `jcsse_audit_runner_tqdm_hardened.py`, `run_phase4_rq1_rq2.py`). Each copy is the version imported when that phase produced its results. `selection_v2.py` loads `jcsse_audit_runner_tqdm_hardened.py` from its own folder; only its model definitions, preprocessing pipeline (`split_columns_A`, `build_preprocessor`) and calibrated prediction are used.

## Data access

The analytic dataset (`full_analytic_dataset_mortality_all_admissions.csv`, 14,081 rows, 43 columns) is derived from MIMIC-IV v3.1 and is not included. MIMIC-IV is a credentialed-access database under the PhysioNet Data Use Agreement, and no admission-level data, predictions or masks are redistributed here.

1. Register at https://physionet.org, complete the required CITI training and request access to MIMIC-IV (https://physionet.org/content/mimiciv/).
2. Build the analytic file with the columns listed in `feature_manifest.csv`: `subject_id` (grouping key), `hadm_id`, `label_mortality` (1 = in-hospital death), 10 demographic, administrative and temporal columns, and 30 laboratory columns `lab_*` from the first 24 hours of the admission (the adopted landmark of the study).

Drivers find the file through `--data-path`, or at `Dataset/full_analytic_dataset_mortality_all_admissions.csv` relative to the parent of the driver's folder: `code/main_study/Dataset/` for the phase drivers and `Dataset/` at the repository root for the drivers in `code/`.

## Reproducing the results

Python 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`). The recorded runs used NumPy 2.2.5, pandas 3.0.2, scikit-learn 1.8.0 and XGBoost 3.2.0. pandas 3 reads text columns with dtype `str`; the code treats `str` columns as categorical and stops with an error if the seven categorical columns are not detected, so that every model receives all 39 inputs.

Every driver accepts `--smoke-test` (a short end-to-end run that writes to separate smoke folders) and resumes from its checkpoints when restarted. Run times below are the wall times recorded in the run logs on one desktop machine.

```bash
# 1. main grid (about 1.7 h); later phases read its results
cd code/main_study/PHASE_4_RQ1_RQ2 && python run_phase4_rq1_rq2.py --data-path /path/to/csv

# 2. after step 1, in any order
cd code/main_study/PHASE_5_RQ3            && python run_phase5_rq3.py            --data-path /path/to/csv
cd code/main_study/PHASE_6_RQ4            && python run_phase6_rq4.py            --data-path /path/to/csv
cd code/main_study/PHASE_8_CALIBRATION    && python run_phase8_calibration.py    --data-path /path/to/csv
cd code/main_study/PHASE_8_MNAR_X         && python run_phase8_mnar_x.py         --data-path /path/to/csv
cd code/main_study/PHASE_8_S1_SENSITIVITY && python run_phase8_s1_sensitivity.py --data-path /path/to/csv

# 3. after PHASE_6_RQ4
cd code/main_study/PHASE_6_SECTION_B_REPEATS && python run_phase6_section_b_repeats.py --data-path /path/to/csv

# 4. statistics, case-level intervals, figures (minutes)
cd code/main_study/PHASE_STATS_LAYER && python phase_stats_layer.py --data-root .. \
    --perm-repeats-csv ../PHASE_6_SECTION_B_REPEATS/results/phase6_section_b_repeats/phase8_section_b_repeats_full.csv \
    --out-prefix phase_stats_layer_v2
cd code/main_study/CASE_LEVEL_BOOTSTRAP_CI && python case_level_bootstrap_ci.py --data-root ..
cd code/main_study/Figures && python make_figures.py

# 5. from code/: handling comparison, multiple imputation, repeated CV, permutation and rate shift
python run_r1_missingness_handling.py                # about 2.5 h
python run_r1_multiple_imputation.py --m 5           # about 10 h
python run_r1_multiple_imputation.py --m 1           # about 2 h
python run_r1_repeated_outer_cv.py                   # about 8 h
python run_r1_permutation_rate_shift.py --part perm  # about 35 min
python run_r1_permutation_rate_shift.py --part rate  # about 1 h

# 6. from code/: analyses (read-only; each stops if a reproduction check fails)
python analyse_r1_missingness_handling.py   > ../Results/R1_missingness_handling/r1_handling_analysis.txt
python analyse_r1_repeated_outer_cv.py      > ../Results/R1_repeated_outer_cv/r1_repeated_cv_analysis.txt
python analyse_r1_permutation_rate_shift.py > ../Results/R1_permutation_rate_shift/r1_perm_rate_analysis.txt
cd main_study/Figures && python make_manuscript_figures.py
```

The phase drivers write to `results/` inside their own folder; the drivers in `code/` write to `Results/` at the repository root. Both are ignored by git.

Built-in reproduction checks: the median arm of the handling comparison reproduces the main grid (125 family-fold AUROCs and 25 selections), repeat 0 of the repeated cross-validation uses the main seeds and reproduces the main grid, and the permutation and rate-shift driver reproduces the OR 1.5 fits, the 600 OR 2 and OR 4 permutation draws and the 75 phase-5 shift cells; the analysis scripts assert these before printing results.

Tests: `pytest test_r1_*.py` in `code/` (fast known-answer tests), and `pytest` in a phase folder for the generator, selection and metric modules; some of the phase tests are integration tests that need the analytic file or the phase's saved results and take several minutes.

## License

MIT, see `LICENSE`. The license covers the code and documentation in this repository only, not MIMIC-IV.
