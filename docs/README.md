# Where each result of the paper comes from

Paths are relative to `code/`. `main_study/<PHASE>/results/` and `../Results/` are created by the drivers; they are not part of the repository (see the data-access section of the top-level README).

| Paper item | Driver | Output read for the reported values |
|---|---|---|
| Cohort and feature counts (Section II-A) | `main_study/PHASE_4_RQ1_RQ2/run_phase4_rq1_rq2.py` | `run.log` and the analytic file; per-column roles in `../feature_manifest.csv` |
| Realised injected rates (Section II-C) | same | `results/phase4/phase4_manipulation_check.csv` |
| Fig. 1, Section III-A (AUROC against rate, displacement, regret) | same | `results/phase4/phase4_fold_summary.csv`, `phase4_full_table.csv` |
| Section III-B, Table I rows at q = 0.30 (strength sweep) | same, plus `main_study/PHASE_STATS_LAYER/phase_stats_layer.py` | `phase4_fold_summary.csv`; `phase_stats_layer_v2_results.csv` |
| Table I, permutation column and q = 0.40 rows | `run_r1_permutation_rate_shift.py --part perm`; `main_study/PHASE_6_SECTION_B_REPEATS/run_phase6_section_b_repeats.py` | `../Results/R1_permutation_rate_shift/r1_perm_rate_analysis.txt` §2; `phase8_section_b_repeats_full.csv` |
| Table I, pooled case-level AUROC | `main_study/CASE_LEVEL_BOOTSTRAP_CI/case_level_bootstrap_ci.py`; `analyse_r1_permutation_rate_shift.py` (q = 0.40, OR 4) | `case_level_bootstrap_ci.csv`; `r1_perm_rate_analysis.txt` §2 |
| Fig. 2(a), Section III-C (shift matrix at q = 0.30) | `main_study/PHASE_5_RQ3/run_phase5_rq3.py` | `results/phase5/phase5_shift_summary.csv`, `phase5_full_table.csv` |
| Fig. 2(b), Section III-C (shift over q = 0.10, 0.30, 0.50; slopes) | `run_r1_permutation_rate_shift.py --part rate` | `../Results/R1_permutation_rate_shift/rate_shift_eval.csv`; `r1_perm_rate_analysis.txt` §3 |
| Fig. 3(a), Section III-D (mask-only and count-only baselines) | `main_study/PHASE_6_RQ4/run_phase6_rq4.py` | `results/phase6/phase6_baselines.csv` |
| Fig. 3(b), Section III-D (permutation over 150 draws) | `main_study/PHASE_6_SECTION_B_REPEATS/run_phase6_section_b_repeats.py` | `phase8_section_b_repeats_full.csv`; `phase_stats_layer_v2_results.csv` (`RQ4.*`) |
| Section III-D (indicator ablation with intervals) | `main_study/PHASE_6_RQ4/run_phase6_rq4.py` | `phase6_indicator_with.csv`, `phase4_full_table.csv`; `r1_perm_rate_analysis.txt` §4 |
| Table II, Section III-E (handling strategies) | `run_r1_missingness_handling.py`, `run_r1_multiple_imputation.py --m 5` and `--m 1` | `../Results/R1_missingness_handling/r1_handling_analysis.txt` §2–§7 |
| Sections III-A to III-C, Table III (25-fold contrasts) | `run_r1_repeated_outer_cv.py` | `../Results/R1_repeated_outer_cv/r1_repeated_cv_analysis.txt` |
| Table III (S1 split, AP criterion, MNAR-X, calibration) | `main_study/PHASE_8_S1_SENSITIVITY/`, `PHASE_4_RQ1_RQ2/` (inner AP), `PHASE_8_MNAR_X/`, `PHASE_8_CALIBRATION/` | each folder's `results/`; `phase_stats_layer_v2_results.csv` |
| Fold-level tests, Benjamini–Hochberg over 24 tests | `main_study/PHASE_STATS_LAYER/phase_stats_layer.py` | `phase_stats_layer_v2_results.csv` |
| Figs. 1–3 as printed | `main_study/Figures/make_manuscript_figures.py` | `main_study/Figures/manuscript/*.pdf`, `_manuscript_numbers.json` |

Seeds: outer split 2026, masks 20260917 (repeats r = 0–4 use 2026 + r and 20260917 + r), permutation draws `(20260917, 555, fold, round(100 q), round(10 OR))` plus the draw index, fold-level bootstrap 20260919, case-level bootstrap 20260920, imputations 2026–2030.
