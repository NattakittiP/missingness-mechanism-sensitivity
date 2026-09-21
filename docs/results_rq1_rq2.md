# RQ1 + RQ2 Results — Rate Sweep and MNAR-Y Strength Sweep

Real experiment, executed on the full dataset (`code/PHASE_4_RQ1_RQ2/run_phase4_rq1_rq2.py`). Every number below was independently recomputed from the driver's own output tables, not copied from a run log.

## 1. Run-integrity checks (before trusting any scientific number)

- **Total wall time**: 1h 30m.
- **Zero warnings or errors** anywhere in the run log across all 14 conditions × 5 folds × 5 model families.
- **Row counts match exactly**: 450 rows in the full results table (18 condition-entries × 5 folds × 5 families); 90 rows in the fold-summary table (18 × 5). No missing or duplicated fold results.
- **q=0 reuse verified bit-identical** across the four mechanisms' shared q=0 baseline, and matched against the independently-validated Phase 3 checkpoint.
- **No checkpoint-skip messages** — a genuine fresh, complete run, not a partial resume.

## 2. Manipulation-check table — generator calibration verified at full scale

Averaged realized rate vs. target rate (train partition), across all 5 folds:

| mechanism | target q | realized r_inject (train) | realized r_inject (test) |
|---|---:|---:|---:|
| MCAR | 0.10 | 0.10033 | 0.09994 |
| MCAR | 0.20 | 0.20047 | 0.20080 |
| MCAR | 0.30 | 0.30018 | 0.30022 |
| MCAR | 0.40 | 0.40066 | 0.40063 |
| MAR | 0.10 | 0.10046 | 0.10014 |
| MAR | 0.20 | 0.20040 | 0.20090 |
| MAR | 0.30 | 0.30050 | 0.30079 |
| MAR | 0.40 | 0.40069 | 0.40080 |
| MNAR-Y | 0.10 | 0.10046 | 0.10012 |
| MNAR-Y | 0.20 | 0.20065 | 0.20114 |
| MNAR-Y | 0.30 | 0.30007 | 0.29995 |
| MNAR-Y | 0.40 | 0.40053 | 0.40046 |

Calibration is tight everywhere — realized rate is within ±0.001 of target, train and test alike, for every mechanism and rate.

**MNAR-Y outcome-dependence, checked against the pre-registered theoretical table** (q=0.30):

| OR | theoretical P(S=1\|Y=0) | realized P(S=1\|Y=0) | theoretical P(S=1\|Y=1) | realized P(S=1\|Y=1) |
|---:|---:|---:|---:|---:|
| 1.5 | 0.298 | 0.297 | 0.389 | 0.383–0.405 |
| 2 (main) | 0.296 | 0.294–0.295 | 0.457 | 0.439–0.468 |
| 4 | 0.292 | 0.289 | 0.622 | 0.604–0.636 |

The real, full-scale run reproduces the pre-registered theoretical table closely at every OR.

**MAR X-side dependence**: mask-count-per-patient ↔ admission-context-driver correlation is positive and grows with rate (0.42–0.45 at q=0.1 → 0.60–0.62 at q=0.4), never touching the outcome by construction.

## 3. RQ1 — Rate sweep (mechanism strength fixed): AUROC declines everywhere, but selection stability diverges sharply by mechanism

Selected-family outer-test AUROC, mean ± SD across the 5 outer folds:

| mechanism | q=0.0 | q=0.1 | q=0.2 | q=0.3 | q=0.4 |
|---|---:|---:|---:|---:|---:|
| MCAR | 0.9555 ± 0.0091 | 0.9459 ± 0.0114 | 0.9396 ± 0.0103 | 0.9316 ± 0.0067 | 0.9300 ± 0.0059 |
| MAR | 0.9555 ± 0.0091 | 0.9482 ± 0.0109 | 0.9345 ± 0.0069 | 0.9346 ± 0.0069 | 0.9240 ± 0.0165 |
| MNAR-Y | 0.9555 ± 0.0091 | 0.9426 ± 0.0116 | 0.9335 ± 0.0119 | 0.9336 ± 0.0120 | 0.9235 ± 0.0125 |

AUROC declines with rate under all three mechanisms, at a broadly comparable pace (a ~2.5–3 point drop from q=0 to q=0.4).

**The interesting result is in `baseline_selection_displacement_rate`** — how often the selected model family changes relative to the q=0 baseline, same fold:

| mechanism | q=0.1 | q=0.2 | q=0.3 | q=0.4 |
|---|---:|---:|---:|---:|
| MCAR | 0/5 (0%) | 0/5 (0%) | 2/5 (40%) | 4/5 (80%) |
| MAR | 0/5 (0%) | 1/5 (20%) | 2/5 (40%) | 2/5 (40%) |
| MNAR-Y | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |

**Outcome-independent missingness (MCAR/MAR) destabilizes model-family selection as it increases; outcome-dependent missingness (MNAR-Y) does not.** By q=0.4, MCAR displaces the baseline selection in 4 of 5 folds (mostly a shift toward `extratrees`); MNAR-Y's selection (`xgb`, always) never changes across the entire rate sweep.

## 4. RQ2 — MNAR-Y strength sweep (rate fixed at q=0.30): AUROC rises with OR — read as a shortcut signature, not "better prediction"

| OR | mean AUROC (5 folds) | selection: xgb chosen | selection entropy | eval margin (top1−top2) |
|---:|---:|---:|---:|---:|
| 1.5 | 0.9258 ± 0.0130 | 5/5 | 0.0 | 0.0114 |
| 2.0 (main) | 0.9336 ± 0.0120 | 5/5 | 0.0 | 0.0266 |
| 4.0 | 0.9653 ± 0.0055 | 5/5 | 0.0 | 0.0424 |

AUROC rises monotonically with OR, and the evaluation margin roughly quadruples from OR=1.5 to OR=4. As the outcome-driven missingness mechanism strengthens, the pattern of missingness itself becomes an increasingly strong proxy for the label (a synthetic-mask-only baseline already reaches AUROC≈0.97 at OR=4 — see `results_rq4.md` §3). The rising AUROC is consistent with the model exploiting that shortcut more effectively as it strengthens, not with the task becoming genuinely easier — see the RQ3 deployment-shift results and the RQ4 permutation ablation, which test this directly.

## 5. What's actually happening to selection under MCAR at high rates

Per-fold selected family counts (5 folds each):

| condition | selected-family distribution |
|---|---|
| mcar q10 / q20 | xgb: 5/5 |
| mcar q30 | xgb: 3, extratrees: 1, rf: 1 |
| mcar q40 | extratrees: 4, xgb: 1 |
| mar q10 | xgb: 5/5 |
| mar q20 | xgb: 4, extratrees: 1 |
| mar q30 / q40 | xgb: 3, extratrees: 1, rf: 1 |
| mnar_y (all rates, all ORs) | xgb: 5/5, always |

At q=0.4, MCAR's modal selected family flips from `xgb` to `extratrees` — consistent with the sharp jump in displacement rate at that condition.

Figures for this section (AUROC vs. rate, selection displacement vs. rate, and the RQ2 strength-sweep summary) can be regenerated from the tables above with `code/FIGURES/make_figures.py` — see the top-level `README.md`.

## 6. Caveats

1. Selection-stability metrics (entropy, Kendall's τ_b) use the 5 outer folds as the "repeat" dimension (one mask-seed per condition) — they quantify across-fold stability entangled with mask-realization variance, not isolated mask-seed noise.
2. RQ2's OR=4 condition (and to a lesser extent OR=2) is a shortcut-learning phenomenon (see §4) — not an unqualified performance improvement.

## 7. Case-level bootstrap CIs on headline AUROC/AP

Every AUROC/AP point estimate above has a companion 95% case-level bootstrap CI, computed from raw per-case predictions (pooled across all 5 folds, N=14,081, every patient counted once):

| Condition | AUROC (95% CI) | AP (95% CI) |
|---|---|---|
| q=0 shared baseline | 0.9521 (0.9412–0.9618) | 0.5452 (0.4941–0.5948) |
| mcar_q10_main | 0.9439 (0.9326–0.9540) | 0.5047 (0.4574–0.5557) |
| mcar_q20_main | 0.9392 (0.9269–0.9501) | 0.4725 (0.4223–0.5236) |
| mcar_q30_main | 0.9258 (0.9117–0.9389) | 0.4456 (0.3972–0.4970) |
| mcar_q40_main | 0.9212 (0.9067–0.9344) | 0.3919 (0.3413–0.4463) |
| mar_q10_main | 0.9449 (0.9328–0.9560) | 0.4968 (0.4472–0.5474) |
| mar_q20_main | 0.9308 (0.9165–0.9439) | 0.4480 (0.3988–0.5017) |
| mar_q30_main | 0.9309 (0.9191–0.9422) | 0.4232 (0.3751–0.4782) |
| mar_q40_main | 0.9157 (0.9001–0.9314) | 0.4042 (0.3580–0.4588) |
| mnar_y_q10_main | 0.9400 (0.9270–0.9512) | 0.4794 (0.4315–0.5342) |
| mnar_y_q20_main | 0.9325 (0.9188–0.9449) | 0.4628 (0.4158–0.5146) |
| mnar_y_q30_main (OR=2.0) | 0.9333 (0.9209–0.9449) | 0.4487 (0.3993–0.4997) |
| mnar_y_q30_OR1.5 | 0.9251 (0.9131–0.9374) | 0.4167 (0.3658–0.4685) |
| mnar_y_q30_OR4.0 | 0.9646 (0.9563–0.9723) | 0.5692 (0.5205–0.6208) |
| mnar_y_q40_main | 0.9230 (0.9099–0.9348) | 0.3848 (0.3394–0.4337) |

**Scope note**: this is a finite-test-set sampling CI on each point estimate — a different, non-substituting source of uncertainty from the fold-level paired significance tests in the internal statistical layer (which, with only n=5 paired folds, has a floor of p=0.0625 for a two-sided sign test no matter how consistent the effect is).
