# RQ4 Results — Shortcut Attribution / Permutation Ablations

Real experiment, executed with `code/PHASE_6_RQ4/run_phase6_rq4.py` (`smoke_test=False`, `n_outer=5`, wall time ~1h06m, zero warnings/errors), plus a 30-repeats-per-cell validation run of Section B (`code/PHASE_6_SECTION_B_REPEATS/run_phase6_section_b_repeats.py`).

## 1. Independent correctness verification

- **Structural completeness**: baseline table has exactly 75 rows (15 conditions × 5 folds); permutation table has 20 rows (4 (rate,OR) cells × 5 folds); the two indicator-ablation tables each have 30 rows (6 (mechanism,rate) cells × 5 folds).
- **Full raw-checkpoint reconstruction**: all 125 checkpoint files were reloaded and every numeric column in all four output tables was recomputed from them directly — 0 mismatches, max absolute difference 0.0.
- **Cross-verified against the RQ1/RQ2 run**: the without-indicator side of the ablation (rate=0, reused rows) matches the RQ1/RQ2 run's own results to better than `1e-9` in every row.
- **Permutation-seed collision check**: the two fold-size collision pairs (folds 1&2, folds 4&5) show genuinely different post-permutation AUROC values within every (rate, OR) group — 0 duplicate values in all four groups.
- **Model-selection integrity**: Section B selected `xgb` in 20/20 cells; Section C selected `xgb` in 19/30, `extratrees` in 7/30, `rf` in 4/30 — all from the inner-CV-only selection path, structurally incapable of seeing outer-test data.

## 2. Section B — test-only permutation ablation: the flagship RQ4 finding

Single-draw results (5 folds/cell):

| Rate (q) | OR | Mean AUROC before | Mean AUROC after | Mean Δ | Range of Δ | Folds negative |
|---|---:|---:|---:|---:|---|---|
| 0.30 | 2.0 (main) | 0.9336 | 0.9234 | −0.0102 | −0.0158 to −0.0036 | 5/5 |
| 0.40 | 2.0 (main) | 0.9235 | 0.9111 | −0.0124 | −0.0289 to +0.0007 | 4/5 |
| 0.30 | 4.0 (sensitivity) | 0.9653 | 0.8763 | −0.0890 | −0.1102 to −0.0778 | 5/5 |
| 0.40 | 4.0 (sensitivity) | 0.9576 | 0.8492 | −0.1085 | −0.1415 to −0.0920 | 5/5 |

Shuffling the synthetic mask across admissions at test time — same fold, same values, same model, only the row-to-row pattern of *which* labs are missing is destroyed — costs the model about 1 point of AUROC at the main strength (OR=2) but 9–11 points at the sensitivity strength (OR=4), roughly an 8–9× larger collapse. At OR=4, a substantial share of the model's AUROC advantage is not about the lab values at all — permuting the mask pushes its score below OR=2's own level (see §5), pointing to the missingness pattern itself as a major driver. At the frozen main OR=2 setting, the model is not meaningfully shortcut-reliant.

## 3. Section A — mask-only / count-only / synth-only baselines, and the eligibility-confound floor

| Mechanism | q=0.10 | q=0.20 | q=0.30 | q=0.40 |
|---|---:|---:|---:|---:|
| MCAR (synth-only mask AUROC) | 0.673 | 0.741 | 0.789 | 0.819 |
| MAR (synth-only mask AUROC) | 0.676 | 0.750 | 0.794 | 0.828 |
| MNAR-Y (synth-only mask AUROC) | 0.873 | 0.914 | 0.937 | 0.940 |

MCAR and MAR — whose injection *probability* has zero built-in outcome-dependence — climb from ~0.67 to ~0.82-0.83 AUROC purely from a mechanical eligibility-confound floor: injection can only land on natively-observed cells, so more injected cells means a stronger correlation between synthetic-count and native-missingness-count, which is itself informative about the outcome. MNAR-Y sits above this floor at every rate (0.87–0.94, reaching 0.982 at OR=4), though the gap narrows as rate rises (~0.20 at q=0.10 to ~0.12 at q=0.40 against the MCAR/MAR average).

MNAR-Y OR sweep at q=0.30:

| MNAR-Y strength | synth-only mask AUROC |
|---|---:|
| OR=1.5 | 0.883 |
| OR=2.0 (main) | 0.937 |
| OR=4.0 (sensitivity) | 0.982 |

## 4. Section C — explicit-indicator-concat ablation

| Mechanism | Rate | With-indicator AUROC | Without-indicator AUROC | Δ |
|---|---:|---:|---:|---:|
| MCAR | 0.00 | 0.9555 | 0.9555 | −0.00005 |
| MCAR | 0.30 | 0.9342 | 0.9316 | +0.0026 |
| MCAR | 0.40 | 0.9267 | 0.9300 | −0.0033 |
| MNAR-Y | 0.00 | 0.9555 | 0.9555 | −0.00005 |
| MNAR-Y | 0.30 | 0.9441 | 0.9336 | +0.0105 |
| MNAR-Y | 0.40 | 0.9387 | 0.9235 | +0.0151 |

Adding explicit "is this lab missing" indicators gives no consistent benefit under MCAR but a small, consistently positive, rate-growing benefit under MNAR-Y.

## 5. Synthesis

All three sections corroborate: at the frozen main setting (OR=2), the model's apparent AUROC gain under MNAR-Y is mostly genuine — the mask-pattern shortcut contributes on the order of 1 AUROC point when tested by permutation. At the OR=4 sensitivity condition, the shortcut is the dominant contributor — permuting OR=4's mask drives its post-permutation AUROC (0.876 at q=0.30, 0.849 at q=0.40) *below* OR=2's own post-permutation AUROC (0.923, 0.911), even though the raw OR2→OR4 AUROC gap is only ~3.2–3.4 points. This is consistent with the OR=4 model having partially substituted genuine value-based signal for the easier-to-fit missingness-pattern shortcut during training, rather than learning genuine signal plus an add-on.

## 6. Repeated-permutation-draws validation of Section B (30 draws/cell, 150 observations/cell)

A companion driver repeats Section B's reshuffle-and-rescore step 30× per cell using the same fitted model (no retraining), giving 150 observations per (rate, OR) cell instead of 5. The built-in correctness gate (draw 0 must reproduce the single-draw `auroc_after` for that cell) passed with 0.000000 drift in all 20/20 cells.

| Rate | OR | Single-draw mean (n=5) | Repeated-draw mean (n=150) | Repeated-draw sd | Repeated-draw 95% CI | Folds negative (of 5) |
|---|---:|---:|---:|---:|---|---|
| 0.30 | 2.0 | −0.0102 | −0.0080 | 0.0128 | [−0.0101, −0.0060] | 3/5 |
| 0.30 | 4.0 | −0.0890 | −0.0960 | 0.0150 | [−0.0984, −0.0936] | 5/5 |
| 0.40 | 2.0 | −0.0124 | −0.0072 | 0.0106 | [−0.0089, −0.0056] | 4/5 |
| 0.40 | 4.0 | −0.1085 | −0.1083 | 0.0173 | [−0.1111, −0.1055] | 5/5 |

**OR=4.0 is the most robustly validated number in the project**: every one of the 5 folds is negative at both rates, each with a tight within-fold sd relative to the between-fold spread — the effect is a stable property of each fold's fitted model, not an artifact of a single reshuffle. The repeated-draw pooled mean is, if anything, slightly more negative than the original single-draw estimate.

**OR=2.0's weak effect is genuinely fold-heterogeneous, not draw noise**: 2 of 5 folds at q=0.30 (1 of 5 at q=0.40) show a real, repeatable small *increase* in AUROC under permutation, confirmed by a tight within-fold sd around a positive mean across all 30 of that fold's reshuffles; the remaining folds show a real, repeatable small decrease. The pooled 95% CI still excludes zero at both rates, but this is an aggregate effect borrowing strength across heterogeneous folds, not a uniformly-directional one.

**Terminology note**: the repeated-draw numbers are a Monte Carlo characterization of the ablation effect's own sampling distribution under reshuffling — not a null-hypothesis significance test.
