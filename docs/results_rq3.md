# RQ3 Results — Source→Target Deployment (Mechanism-Shift) Study

Real experiment, executed with `code/PHASE_5_RQ3/run_phase5_rq3.py`: train under one missingness mechanism, evaluate under another, holding the model/hyperparameters/calibration frozen. Every number below was independently recomputed from the driver's own output tables.

## 1. Run-integrity checks

- **Wall time**: 32m 47s.
- **Zero warnings or errors** across all 25 fold-fits × up to 3 targets × 5 families.
- **Row counts match exactly**: 375 rows in the full table (75 (rate,fold,source,target) cells × 5 families); 75 rows in the shift-summary table; 45 rows in the manipulation-check table. Source/target counts match the frozen design exactly: r=0.10 and r=0.50 have MCAR as the only source (15 rows each); r=0.30 has all 3 sources (45 rows).
- **Structural guarantee verified on real data**: the selected family is identical across all targets within every (rate, fold, source) cell (0/25 violations) — direct empirical confirmation that target-mechanism swapping never influences which family was selected.

## 2. Independent cross-check against the RQ1/RQ2 run — bit-for-bit match

Every diagonal cell (source == target) at r=0.10 and r=0.30 reproduces the RQ1/RQ2 run's real results exactly: all 100 overlapping (condition, fold, family) cells match to better than `1e-9`, including `xgb`. This is strong, direct evidence that the fold-safe masking and the frozen-selection design function correctly at full scale on the real 14,081-row dataset.

## 3. Manipulation-check table — calibration verified at full scale, across all 3 rates

| mechanism | target q | realized r_inject (train) | realized r_inject (test) |
|---|---:|---:|---:|
| MCAR | 0.10 | 0.10033 | 0.09994 |
| MCAR | 0.30 | 0.30018 | 0.30022 |
| MCAR | 0.50 | 0.50066 | 0.50173 |
| MAR | 0.10 | 0.10046 | 0.10014 |
| MAR | 0.30 | 0.30050 | 0.30079 |
| MAR | 0.50 | 0.50001 | 0.50019 |
| MNAR-Y | 0.10 | 0.10046 | 0.10012 |
| MNAR-Y | 0.30 | 0.30007 | 0.29995 |
| MNAR-Y | 0.50 | 0.50050 | 0.50165 |

Max deviation from target anywhere: 0.0021 (test-side, r=0.50).

## 4. Selected-family stability — reproduces the RQ1/RQ2 pattern

Diagonal-cell selected family per fold:

| rate, source | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---|---|---|---|---|
| q10, MCAR | xgb | xgb | xgb | xgb | xgb |
| q30, MCAR | xgb | xgb | extratrees | rf | xgb |
| q30, MAR | xgb | xgb | extratrees | rf | xgb |
| q30, MNAR-Y | xgb | xgb | xgb | xgb | xgb |
| q50, MCAR | extratrees | extratrees | extratrees | extratrees | xgb |

MNAR-Y stays perfectly stable (5/5 xgb) at q=0.30. At q=0.50, MCAR's modal family has flipped further toward `extratrees` (4/5) — a continuation of the flip-toward-extratrees trend first seen at q=0.40.

## 5. Primary result (r=0.30): mechanism-shift penalty is asymmetric

Full 3×3 matrix, mean over 5 folds:

| source → target | selected AUROC (mean) | Δ AUROC vs. diagonal | selection regret (mean) | displacement rate |
|---|---:|---:|---:|---:|
| MCAR → MCAR (diagonal) | 0.9316 | — | 0.0044 | 40% |
| MCAR → MAR | 0.9298 | −0.0018 | 0.0040 | 80% |
| MCAR → MNAR-Y | 0.9003 | −0.0313 | 0.0083 | 60% |
| MAR → MAR (diagonal) | 0.9346 | — | 0.0028 | 40% |
| MAR → MCAR | 0.9349 | +0.0002 | 0.0040 | 40% |
| MAR → MNAR-Y | 0.9048 | −0.0298 | 0.0105 | 40% |
| MNAR-Y → MNAR-Y (diagonal) | 0.9336 | — | 0.0000 | 0% |
| MNAR-Y → MCAR | 0.9163 | −0.0173 | 0.0134 | 100% |
| MNAR-Y → MAR | 0.9175 | −0.0161 | 0.0120 | 100% |

The direction is consistent across all folds tested for each pair: `MCAR→MNAR-Y` negative in 5/5 folds; `MAR→MNAR-Y` negative in 5/5; `MNAR-Y→MCAR` and `MNAR-Y→MAR` each negative in 5/5.

**Two distinct findings, in different directions:**

1. **Raw performance drops roughly twice as hard when an outcome-independent-trained model meets an outcome-dependent deployment environment than the reverse.** A model trained under MCAR/MAR loses ~0.030 AUROC when deployed where missingness is actually MNAR-Y; a model trained under MNAR-Y loses only ~0.016–0.017 AUROC deployed under MCAR/MAR. Read with the RQ2 shortcut-learning finding, this shows the shortcut, once learned, degrades gracefully rather than collapsing when its statistical basis disappears at test time.
2. **Model-family selection integrity is far more fragile under this shift than raw AUROC is.** `MNAR-Y → {MCAR, MAR}` shows 100% displacement — the family selected under MNAR-Y (always `xgb`) is never the outer-test oracle once deployed under MCAR/MAR; the true oracle is `extratrees` in 8 of these 10 cells, `rf` in the remaining 2 — and the largest selection-regret values in the whole table.

## 6. Rate-sensitivity result (MCAR source, all 3 rates): the MNAR-Y-target shift penalty grows sharply with rate

| rate | diagonal AUROC (MCAR→MCAR) | shifted AUROC (MCAR→MNAR-Y) | Δ AUROC | mean regret (shifted) |
|---:|---:|---:|---:|---:|
| 0.10 | 0.9459 | 0.9379 | −0.0080 | 0.0001 |
| 0.30 | 0.9316 | 0.9003 | −0.0313 | 0.0083 |
| 0.50 | 0.9186 | 0.8445 | −0.0741 | 0.0209 |

Monotonic and accelerating — negative in all 15/15 folds tested across the three rates.

## 7. Caveats

1. Only MCAR was run as a source at r=0.10/r=0.50 (a deliberate compute-bounding design choice) — the rate-trend result in §6 is available only for the MCAR→MNAR-Y direction.
2. Selection regret is defined relative to the source's own trained candidate set, not a hypothetically-retrained-on-target oracle — RQ3 answers "how much does freezing cost you," not "what's the best possible score achievable on the target."
3. As with RQ1/RQ2, selection-stability figures entangle patient-split variance with missingness-realization variance across the 5 outer folds.
