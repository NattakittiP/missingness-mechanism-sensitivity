# Missingness-Mechanism Sensitivity in Mortality Prediction

Code and results for a study of how the mechanism of missing data, not just its rate, affects model-family selection stability, deployment robustness, and shortcut learning in an in-hospital mortality prediction task on MIMIC-IV.

## Headline findings

- **RQ1 (mechanism vs. selection stability)**: at a matched injected missingness rate, outcome-independent mechanisms (MCAR, MAR) progressively destabilize which model family gets selected (MCAR reaches 80% selection displacement by q=0.4), while outcome-dependent missingness (MNAR-Y) never displaces the selected family, at any rate tested.
- **RQ2 (MNAR-Y strength)**: AUROC rises monotonically with MNAR-Y strength, but this is largely a shortcut-learning signature, not a genuine improvement in predictive power — confirmed directly by RQ4.
- **RQ3 (deployment shift)**: a model trained under MCAR/MAR loses about twice as much raw AUROC when deployed where missingness is actually MNAR-Y (~−0.030) as the reverse (~−0.016 to −0.017) — but the *selection* decision breaks the other way: a model selected under MNAR-Y is never the correct choice once deployed under MCAR/MAR (100% displacement).
- **RQ4 (shortcut attribution)**: destroying the missingness pattern at test time (permuting which labs are "missing," holding values and model fixed) costs the model ~1 AUROC point at the frozen main MNAR-Y strength, but 9–11 AUROC points at a stronger sensitivity setting — direct, quantitative evidence that the model is partly reading the missingness pattern itself as a proxy for the outcome, and that this effect is highly strength-dependent.

Full results, and caveats: see `docs/`.

## Repository structure

```
code/           Every driver and shared module actually run to produce the results,
                organized by phase (see "Reproducing the pipeline" below).
docs/           Methodology and results write-ups (start at docs/README.md).
feature_manifest.csv Per-feature table: type, native missingness %, masking eligibility.
```

See "Data access" below for how to obtain the MIMIC-IV-derived dataset (not included in this repository).

### `code/` layout

Each `PHASE_*` subfolder is self-contained and corresponds to one stage of the pipeline, matching the order the experiments were actually run in:

| Folder | What it does |
|---|---|
| `PHASE_2_GENERATOR_V2/` | Missingness generator (MCAR/MAR/MNAR-Y injection with bisection-calibrated rates) + unit tests + the manipulation-check pilot. |
| `PHASE_3_SELECTION_FIX/` | The inner-CV-only model-family selection procedure (`selection_v2.py`) + unit tests. |
| `PHASE_4_RQ1_RQ2/` | RQ1 (rate sweep) and RQ2 (MNAR-Y strength sweep) driver. |
| `PHASE_5_RQ3/` | RQ3 (source→target deployment-shift) driver. |
| `PHASE_6_RQ4/` | RQ4 (shortcut-attribution / permutation-ablation) driver. |
| `PHASE_6_SECTION_B_REPEATS/` | 30-repeats-per-cell validation of RQ4 Section B's permutation ablation. |
| `PHASE_7_METRIC_LAYER/` | The stability/attribution metric functions (`metrics_v2.py`: displacement rate, selection entropy, Kendall's τ, selection regret) + unit tests. |
| `PHASE_8_CALIBRATION/` | Calibration-slope/intercept backfill across the full experimental grid. Also carries the canonical, most-complete copy of every shared module and its full test suite. |
| `PHASE_8_MNAR_X/` | MNAR-X (value-dependent missingness) sensitivity experiment. |
| `PHASE_8_S1_SENSITIVITY/` | Legacy-split (no patient separation) sensitivity check. |
| `PHASE_STATS_LAYER/` | Driver-independent statistical-significance layer (paired fold-level tests, bootstrap CIs) computed directly from the result tables above. |
| `CASE_LEVEL_BOOTSTRAP_CI/` | Case-level (patient-level) bootstrap confidence intervals on headline AUROC/AP, computed from saved per-case predictions. |

**Why each phase folder carries its own copy of the shared modules**: `missingness_generator_v2.py`, `selection_v2.py`, and `metrics_v2.py` evolved slightly across the project (bug fixes, refactors, added functionality). Each folder's copy is exactly the version that was actually imported when that phase's results were produced. This is deliberate, not duplication left in by accident — it guarantees that running any driver in this repository reproduces the results reported in `docs/`, without depending on a module version from a different phase of the project. `PHASE_4_RQ1_RQ2/` and `PHASE_8_CALIBRATION/` additionally vendor `jcsse_audit_runner_tqdm_hardened.py`, a large model-training/preprocessing utility module inherited from an earlier, unrelated project; only a handful of its functions (model definitions, the preprocessing pipeline, calibrated-probability prediction) are actually used by `selection_v2.py`, but it is included byte-for-byte unmodified — the same file that was actually imported at run time — rather than manually extracted, to eliminate any risk of a reproduction discrepancy.

## Reproducing the pipeline

1. Install dependencies: `pip install -r requirements.txt`
2. Obtain MIMIC-IV v3.1 access and build the analytic dataset — see "Data access" below.
3. Run each phase's driver from inside its own folder, in order, pointing `--data-path` at your local analytic CSV:

```bash
cd code/PHASE_4_RQ1_RQ2 && python run_phase4_rq1_rq2.py --data-path /path/to/full_analytic_dataset_mortality_all_admissions.csv
cd code/PHASE_5_RQ3     && python run_phase5_rq3.py     --data-path /path/to/...csv
cd code/PHASE_6_RQ4     && python run_phase6_rq4.py     --data-path /path/to/...csv
cd code/PHASE_6_SECTION_B_REPEATS && python run_phase6_section_b_repeats.py --data-path /path/to/...csv
cd code/PHASE_8_CALIBRATION  && python run_phase8_calibration.py  --data-path /path/to/...csv
cd code/PHASE_8_MNAR_X       && python run_phase8_mnar_x.py       --data-path /path/to/...csv
cd code/PHASE_8_S1_SENSITIVITY && python run_phase8_s1_sensitivity.py --data-path /path/to/...csv
```

Every driver also accepts `--smoke-test` for a fast, reduced-scale run (useful for verifying your environment before committing to a full run, which can take 30 minutes to well over an hour depending on the phase — see the wall-time figures reported in `docs/results_rq1_rq2.md`, `docs/results_rq3.md`, and `docs/results_rq4.md`).

Unit tests (per phase, using `pytest`):

```bash
cd code/PHASE_2_GENERATOR_V2 && pytest test_missingness_generator_v2.py
cd code/PHASE_3_SELECTION_FIX && pytest test_selection_v2.py
cd code/PHASE_7_METRIC_LAYER && pytest test_metrics_v2.py
```

(`PHASE_8_CALIBRATION/` carries the full, latest test suite for all three modules together.)

Statistical layer, run after the drivers above have produced their result tables:

```bash
cd code/PHASE_STATS_LAYER && python phase_stats_layer.py --data-root /path/to/project/root
```

## Data access

The analytic dataset used in this study (`full_analytic_dataset_mortality_all_admissions.csv`, N=14,081 admissions) is **not included** in this repository. MIMIC-IV is a restricted dataset governed by the PhysioNet Data Use Agreement, and per that agreement no derived patient-level data is redistributed here.

**To obtain access:**

1. Register at [https://physionet.org](https://physionet.org)
2. Complete the required CITI "Data or Specimens Only Research" training
3. Apply for access to MIMIC-IV at [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
4. Once approved, download MIMIC-IV v3.1 and build the analytic dataset with the schema below

**Expected file format:**

- Shape: (14081, 43)
- Label column: `label_mortality` (binary; 1 = in-hospital death)
- Group column: `subject_id` (used for the `StratifiedGroupKFold` split)
- Features: `age`, `gender`, `race`, `marital_status`, `admission_type`, `anchor_year_group`, `anchor_age`, `anchor_year`, `admission_location`, `insurance`, plus 30 laboratory/vital-sign columns (`lab_*`)
- Prediction landmark and lab-observation window: first 24 hours of admission

See `feature_manifest.csv` for the full per-feature table (type, native missingness %, model-input status, masking eligibility), and `docs/methodology.md` for how features are used. Every driver under `code/` accepts a `--data-path` argument pointing at your local copy of the analytic CSV once you have built it — see "Reproducing the pipeline" above.


## License

MIT — see `LICENSE`. This license covers the code and documentation in this repository only; it does not extend to the MIMIC-IV dataset itself (see "Data access" above).
