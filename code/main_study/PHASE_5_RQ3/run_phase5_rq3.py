"""
Phase 5 driver -- RQ3 (source -> target missingness-mechanism deployment shift).
Protocol v2.1 Phase 5 ("Source -> Target Deployment Shift"):

    train on source, tune on source, select family on source, fit
    preprocessing on source, fit calibration on source if used, freeze
    everything, apply the target mechanism only at outer-test evaluation.
    Do not retrain, retune, or recalibrate after seeing the target
    environment.

RQ3 (methodology-redesign-spec.md Sec.0): "Does a model SELECTED under
mechanism A retain performance when the deployment-time missingness
mechanism shifts to B?" -- train_mechanism vs test_mechanism, at fixed rate.

WHAT THIS RUNS

  Mechanism grid: {mcar, mar, mnar_y} x {mcar, mar, mnar_y} (source x target).
  MNAR-X is explicitly OUT OF SCOPE here -- protocol_v2_1_frozen.md's Final
  Implementation Order places MNAR-X at item 11 (Phase 6), strictly AFTER
  RQ3 (item 9). So this is a 3x3 grid, not the 4x4 mentioned as an
  originally-considered design in methodology-redesign-spec.md Sec.7.

  Strength: theta = ln(2), OR = 2 for MAR and MNAR-Y -- the SAME "main"
  strength used throughout RQ1 (protocol_v2.yaml's frozen main setting).
  RQ3 is a mechanism-shift study, not a strength-sweep study (RQ2 already
  owns strength); shifting BOTH rate/strength AND mechanism at once would
  confound what is being attributed to the mechanism shift itself.

  Rate design -- RESOLVED AMBIGUITY, read this before changing it:
  methodology-redesign-spec.md Sec.7 (verbatim): "4x4 grid ... primary at
  r=30%, sensitivity at r=10% and r=50% (source-baseline -> all 4 targets
  only, not full 4x4, to keep compute bounded)." The primary source protocol
  document (Protocol_v2_1_Experimental_Design_Decisions.md, Phase 5 section)
  does not itself specify a rate grid at all -- this reduction is therefore
  this project's own resolution of an underspecified design note, not a
  literal frozen instruction, and is recorded here for the paper trail:
    - r=0.30 (primary, matches RQ1/RQ2's main rate): FULL 3x3 source x
      target matrix -- all 3 mechanisms trained as source, each evaluated
      against all 3 targets (9 source-target pairs x 5 folds = 45 result
      cells, from 3 x 5 = 15 fit_and_select_all_families calls).
    - r=0.10 and r=0.50 (rate sensitivity): reduced grid, ONE source only,
      evaluated against all 3 targets, per the spec's own "not full
      NxN, to keep compute bounded" instruction. "source-baseline" is read
      here as MCAR: in the standard missing-data taxonomy (Rubin), MCAR is
      the textbook baseline/null mechanism against which MAR and MNAR are
      defined as deviations -- so MCAR is the literal "baseline" mechanism
      among the three candidates, not a scientifically-motivated pick tied
      to any one result. (3 target pairs x 5 folds x 2 rates = 30 more
      result cells, from 1 x 5 x 2 = 10 more fit_and_select_all_families
      calls.) This keeps total fresh model-family-tuning compute to 25
      fold-fits, comparable to or less than Phase 4's 70 fold-fits (14
      conditions x 5 folds), despite covering a 2-dimensional grid.
    - rate=0.0 (Natural) is excluded from RQ3 entirely: with no injected
      missingness there is no mechanism to shift between, so a "source vs
      target mechanism" comparison is not meaningful at that rate.

  IMPORTANT STRUCTURAL FACT (a direct consequence of the protocol's own
  "freeze everything, do not retrain after seeing the target" rule):
  `selected_family` is a property of (rate, source, fold) ONLY -- it is
  decided via `fit_and_select_all_families()`'s inner-CV step BEFORE any
  target mechanism is ever touched, so it is IDENTICAL across all 3 targets
  for a fixed (rate, source, fold). Only the outer-test evaluation (AUROC/
  AP/Brier, oracle family, regret) varies by target. This is enforced
  structurally, not just by convention: `evaluate_frozen_model()` never
  fits anything (see selection_v2.py), so there is no code path through
  which a target could influence the selected family.

FOLD-SAFETY: identical convention to Phase 4 -- for each (rate, fold), every
one of the 3 mechanisms' alpha_j is calibrated on THAT fold's own outer-train
partition only (via missingness_generator_v2's fit_mcar/fit_mar_context/
fit_mnar_y, reused verbatim), then applied to both outer-train and
outer-test of that SAME fold. Outer-test rows never influence any alpha_j,
for any mechanism, source or target.

SEED REUSE -- A BUILT-IN CROSS-CHECK: this driver reuses Phase 4's own
OUTER_SEED (2026) and MASK_SEED (20260917) verbatim (imported directly from
run_phase4_rq1_rq2, not redefined), and the SAME per-(mechanism,rate,fold)
"fresh rng, train draw then test draw" masking convention. Consequence: at
r=0.10 and r=0.30 (both of which Phase 4 also ran, at OR=2 main strength),
this driver's DIAGONAL cells (source==target) reproduce Phase 4's own
mcar_q10_main / mar_q10_main / mnar_y_q10_main / *_q30_main results
BIT-IDENTICALLY. This was exploited as an independent correctness check
during the smoke test (see phase5-driver-summary.md) rather than trusted by
construction alone.

CHECKPOINTING: one checkpoint file per (rate, fold) -- results/phase5/
r<RR>_fold<F>.pkl -- containing that cell's full-table rows, manipulation-
check diagnostics, and shift-summary rows for every source trained in that
(rate, fold) cell. Masks for all 3 mechanisms are computed once per (rate,
fold) and reused across every source trained at that (rate, fold), since
masking (bisection alpha calibration) is cheap relative to GridSearchCV
family tuning -- the actual compute bottleneck. Re-running this script
skips any (rate, fold) cell whose checkpoint already exists.

Run with:   python3 run_phase5_rq3.py
Smoke-test: python3 run_phase5_rq3.py --smoke-test
            (primary rate only, r=0.30, 2 outer folds instead of 5 -- still
            exercises the FULL 3x3 source x target matrix, all 3
            fit_and_select_all_families calls per fold, every evaluate_
            frozen_model() call, and the complete aggregation/output code
            path. Writes to results/phase5_smoke/, never touches
            results/phase5/.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as genmod
import selection_v2 as selv2
import metrics_v2 as met

# Reuses Phase 4's driver for: _prepare_X_y_groups, _fit_generator_and_drivers,
# _resolve_data_path, make_logger, OUTER_SEED, MASK_SEED -- see module
# docstring's "SEED REUSE" note for why reusing (not re-deriving) the seeds
# specifically matters here. Importing this module also re-applies the
# warnings-visibility fix (warnings.resetwarnings() + simplefilter("once"))
# that undoes the base runner's global warning suppression -- see
# run_phase4_rq1_rq2.py's own docstring for the full explanation. Re-applied
# explicitly below too, defensively, so this driver's warning behavior does
# not silently depend on import order or on run_phase4_rq1_rq2.py's internals.
import run_phase4_rq1_rq2 as p4

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

OUTER_SEED = p4.OUTER_SEED       # 2026 -- SAME as Phase 4, required for the diagonal cross-check (see module docstring)
MASK_SEED = p4.MASK_SEED         # 20260917 -- SAME as Phase 4, ditto
N_OUTER = 5

MECHANISMS = ["mcar", "mar", "mnar_y"]     # MNAR-X deferred to Phase 6 (protocol_v2_1_frozen.md item 11)
THETA_MAIN = 0.693147                       # ln(2), OR=2 -- frozen main strength, shared with RQ1
RATES = [0.10, 0.30, 0.50]
PRIMARY_RATE = 0.30
SENSITIVITY_SOURCE = "mcar"                 # "source-baseline" per methodology-redesign-spec.md Sec.7 -- see module docstring

RESULTS_DIR = Path("results/phase5")


def sources_for_rate(rate: float) -> List[str]:
    if rate == PRIMARY_RATE:
        return list(MECHANISMS)
    return [SENSITIVITY_SOURCE]


# ---------------------------------------------------------------------------
# Fold-safe masking: all 3 mechanisms, one outer fold, one rate.
# ---------------------------------------------------------------------------

def _mask_all_mechanisms_for_fold(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray,
    df_tr: pd.DataFrame, df_te: pd.DataFrame, rate: float, mask_seed: int,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], Dict[str, Dict[str, Any]]]:
    """For ONE outer fold and ONE rate, fold-safe-calibrate and apply all 3
    mechanisms' generators (MCAR; MAR-context and MNAR-Y at theta=THETA_MAIN)
    to that SAME fold's outer-train/outer-test partitions. Every mechanism's
    alpha_j is calibrated on this fold's own outer-train only -- identical
    fold-safety discipline to run_phase4_rq1_rq2.run_condition_with_injection,
    generalized here to compute all 3 mechanisms' masked copies of the SAME
    fold at once (RQ3 needs a target mechanism's test mask available
    regardless of which mechanism(s) are trained as sources at this
    (rate, fold) cell -- see sources_for_rate()).

    Returns (masked, diagnostics):
      masked[mechanism] = (X_tr_masked, X_te_masked)
      diagnostics[mechanism] = manipulation-check dict (r_inject/r_total,
        plus the MNAR-Y Y-split / MAR H_i-correlation diagnostics Phase 4
        already reports, computed identically here).
    """
    nat_mask_tr = genmod.compute_natural_mask(df_tr)
    nat_mask_te = genmod.compute_natural_mask(df_te)

    masked: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for mech in MECHANISMS:
        theta = 0.0 if mech == "mcar" else THETA_MAIN
        gen, driver_tr, driver_te = p4._fit_generator_and_drivers(
            mech, rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
        )
        # Fresh rng per (mechanism, rate, fold) cell -- same CRN convention as
        # Phase 4 (module docstring there): train draw first, then test draw,
        # one continuing stream.
        rng = np.random.default_rng(mask_seed)
        syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
        syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
        final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
        final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

        X_tr_masked = X_tr.copy()
        X_te_masked = X_te.copy()
        for col in genmod.ELIGIBLE_LAB_COLS:
            X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
            X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan
        masked[mech] = (X_tr_masked, X_te_masked)

        rates_tr = genmod.compute_rates(nat_mask_tr, syn_mask_tr)
        rates_te = genmod.compute_rates(nat_mask_te, syn_mask_te)
        diag: Dict[str, Any] = {
            "mechanism": mech, "target_rate": rate, "theta": theta,
            "r_inject_train": rates_tr["r_inject"], "r_total_train": rates_tr["r_total"],
            "r_inject_test": rates_te["r_inject"], "r_total_test": rates_te["r_total"],
        }
        if mech == "mnar_y":
            syn_te_np = syn_mask_te.to_numpy()
            y1 = y_te == 1
            y0 = y_te == 0
            elig_te = genmod.compute_eligible_mask(nat_mask_te).to_numpy()
            diag["r_inject_among_y1"] = float(syn_te_np[y1].sum() / max(elig_te[y1].sum(), 1))
            diag["r_inject_among_y0"] = float(syn_te_np[y0].sum() / max(elig_te[y0].sum(), 1))
        if mech == "mar":
            mask_count_per_patient = syn_mask_te.to_numpy().sum(axis=1)
            if np.std(driver_te) > 0 and np.std(mask_count_per_patient) > 0:
                diag["corr_mask_count_vs_H_i"] = float(np.corrcoef(mask_count_per_patient, driver_te)[0, 1])
        diagnostics[mech] = diag

    return masked, diagnostics


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_logger(log_path: Path):
    return p4.make_logger(log_path)


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None):
    results_dir = Path("results/phase5_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    # Smoke test uses the primary rate only -- it alone already exercises the
    # full 3x3 source x target matrix (all 3 sources, all 3 targets), which is
    # the part of this driver with the most new code relative to Phase 4;
    # the sensitivity rates reuse that exact same per-source/per-target loop
    # with a shorter `sources` list, so they add no new code path to validate.
    rates = [PRIMARY_RATE] if smoke_test else RATES
    log(f"=== Phase 5 driver start (smoke_test={smoke_test}, n_outer={n_outer}, rates={rates}) ===")

    data_path = p4._resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    all_full_rows: List[Dict[str, Any]] = []
    all_diag_rows: List[Dict[str, Any]] = []
    all_shift_rows: List[Dict[str, Any]] = []

    for rate in rates:
        rate_tag = f"r{int(round(rate * 100)):02d}"
        sources = sources_for_rate(rate)
        log(f"--- rate={rate} ({rate_tag}): sources={sources}, targets={MECHANISMS} ---")

        X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
        outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
        fold_splits = list(enumerate(outer_splitter.split(X, y, groups), start=1))

        for fold_id, (tr_idx, te_idx) in fold_splits:
            ckpt = results_dir / f"{rate_tag}_fold{fold_id}.pkl"
            if ckpt.exists():
                log(f"  [{rate_tag} fold {fold_id}] checkpoint already exists, skipping")
                with open(ckpt, "rb") as f:
                    fold_full_rows, fold_diag_rows, fold_shift_rows = pickle.load(f)
                all_full_rows.extend(fold_full_rows)
                all_diag_rows.extend(fold_diag_rows)
                all_shift_rows.extend(fold_shift_rows)
                continue

            t_fold0 = time.time()
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y[tr_idx]
            g_tr = groups[tr_idx]
            X_te = X.iloc[te_idx].reset_index(drop=True)
            y_te = y[te_idx]
            assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"
            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_te = df.iloc[te_idx].reset_index(drop=True)

            masked, diagnostics = _mask_all_mechanisms_for_fold(
                X_tr, y_tr, X_te, y_te, df_tr, df_te, rate, MASK_SEED
            )
            fold_diag_rows = [{"rate": rate, "fold_id": fold_id, **d} for d in diagnostics.values()]

            fold_full_rows: List[Dict[str, Any]] = []
            fold_shift_rows: List[Dict[str, Any]] = []
            for source in sources:
                X_tr_masked, _ = masked[source]
                t0 = time.time()
                fit_result = selv2.fit_and_select_all_families(
                    X_tr_masked, y_tr, g_tr, num_cols, cat_cols, seed=OUTER_SEED
                )
                fit_elapsed = time.time() - t0
                selected_family = fit_result.selected_family
                log(f"  [{rate_tag} fold {fold_id}] source={source}: fit+select done in "
                    f"{fit_elapsed:.1f}s, selected={selected_family}")

                per_target: Dict[str, Dict[str, Any]] = {}
                for target in MECHANISMS:
                    _, X_te_masked_t = masked[target]
                    family_scores: Dict[str, Tuple[float, float, float]] = {}
                    for model_key in selv2.MODELS:
                        auroc, ap, brier = selv2.evaluate_frozen_model(
                            fit_result.final_models[model_key], X_te_masked_t, y_te
                        )
                        family_scores[model_key] = (auroc, ap, brier)
                        fold_full_rows.append(dict(
                            rate=rate, fold_id=fold_id, source_mechanism=source, target_mechanism=target,
                            model_key=model_key,
                            inner_auroc_mean=fit_result.inner_metrics[model_key][0],
                            inner_ap_mean=fit_result.inner_metrics[model_key][1],
                            outer_test_auroc=auroc, outer_test_ap=ap, outer_test_brier=brier,
                            selected_family=selected_family,
                        ))
                    # Diagnostic oracle for THIS target: best of the 5 models
                    # that were ALL trained on `source`'s data, evaluated on
                    # `target`. Same candidate-set convention selection_v2.py
                    # uses for the single-condition (A==B) case, generalized:
                    # oracle and selected are always compared over the SAME
                    # frozen-model set, which is what keeps regret >= 0 by
                    # construction (metrics_v2.selection_regret's docstring).
                    oracle_family = selv2.select_family(family_scores)
                    sel_auroc, sel_ap, sel_brier = family_scores[selected_family]
                    ora_auroc, ora_ap, ora_brier = family_scores[oracle_family]
                    regret = met.selection_regret(ora_auroc, sel_auroc)
                    per_target[target] = dict(
                        selected_auroc=sel_auroc, selected_ap=sel_ap, selected_brier=sel_brier,
                        oracle_family=oracle_family, oracle_auroc=ora_auroc, oracle_ap=ora_ap, oracle_brier=ora_brier,
                        regret=regret, displaced=(selected_family != oracle_family),
                    )

                # source is always itself a target (MECHANISMS is the target
                # list for every source), so per_target[source] is always
                # defined -- this is the "no shift" diagonal reference point.
                diagonal_auroc = per_target[source]["selected_auroc"]
                for target in MECHANISMS:
                    pt = per_target[target]
                    fold_shift_rows.append(dict(
                        rate=rate, fold_id=fold_id, source_mechanism=source, target_mechanism=target,
                        is_diagonal=(source == target), selected_family=selected_family,
                        selected_outer_test_auroc=pt["selected_auroc"], selected_outer_test_ap=pt["selected_ap"],
                        selected_outer_test_brier=pt["selected_brier"],
                        oracle_family=pt["oracle_family"], oracle_outer_test_auroc=pt["oracle_auroc"],
                        selection_regret=pt["regret"], displaced=pt["displaced"],
                        shift_delta_auroc=pt["selected_auroc"] - diagonal_auroc,
                        fit_seconds=fit_elapsed,
                    ))
                    log(f"    -> target={target}: selected_auroc={pt['selected_auroc']:.4f} "
                        f"(delta_vs_diagonal={pt['selected_auroc'] - diagonal_auroc:+.4f}) "
                        f"oracle={pt['oracle_family']} regret={pt['regret']:.4f} displaced={pt['displaced']}")

            with open(ckpt, "wb") as f:
                pickle.dump((fold_full_rows, fold_diag_rows, fold_shift_rows), f)
            all_full_rows.extend(fold_full_rows)
            all_diag_rows.extend(fold_diag_rows)
            all_shift_rows.extend(fold_shift_rows)
            log(f"  [{rate_tag} fold {fold_id}] done in {time.time() - t_fold0:.1f}s, checkpointed to {ckpt}")

    log("All (rate, fold) cells complete. Aggregating...")
    aggregate_and_save(all_full_rows, all_diag_rows, all_shift_rows, results_dir, log)
    log("=== Phase 5 driver finished ===")
    log.close()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_and_save(all_full_rows, all_diag_rows, all_shift_rows, results_dir: Path, log):
    full_table = pd.DataFrame(all_full_rows)
    full_table.to_csv(results_dir / "phase5_full_table.csv", index=False)
    log(f"Wrote {results_dir / 'phase5_full_table.csv'} ({len(full_table)} rows)")

    diag_df = pd.DataFrame(all_diag_rows)
    diag_df.to_csv(results_dir / "phase5_manipulation_check.csv", index=False)
    log(f"Wrote {results_dir / 'phase5_manipulation_check.csv'} ({len(diag_df)} rows)")

    shift_df = pd.DataFrame(all_shift_rows)
    shift_df.to_csv(results_dir / "phase5_shift_summary.csv", index=False)
    log(f"Wrote {results_dir / 'phase5_shift_summary.csv'} ({len(shift_df)} rows) -- this is the primary RQ3 result table")

    # Per (rate, source, target): mean/SD selected AUROC, mean regret, mean
    # shift_delta_auroc (vs that source's own diagonal, same fold), fraction
    # of folds displaced (selected != oracle for that target).
    metrics_summary: Dict[str, Any] = {}
    for (rate, source, target), g in shift_df.groupby(["rate", "source_mechanism", "target_mechanism"]):
        key = f"r{rate}__{source}__to__{target}"
        metrics_summary[key] = dict(
            rate=rate, source=source, target=target, is_diagonal=bool(g["is_diagonal"].iloc[0]),
            n_folds=int(len(g)),
            selected_auroc_mean=float(g["selected_outer_test_auroc"].mean()),
            selected_auroc_sd=float(g["selected_outer_test_auroc"].std(ddof=0)),
            selection_regret_mean=float(g["selection_regret"].mean()),
            shift_delta_auroc_mean=float(g["shift_delta_auroc"].mean()),
            displacement_rate=float(g["displaced"].mean()),
            modal_selected_family=g["selected_family"].mode().iat[0],
        )
    with open(results_dir / "phase5_metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    log(f"Wrote {results_dir / 'phase5_metrics_summary.json'}")

    with open(results_dir / "phase5_all_results.pkl", "wb") as f:
        pickle.dump({"full_table": full_table, "diag_df": diag_df, "shift_df": shift_df}, f)
    log(f"Wrote {results_dir / 'phase5_all_results.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run the primary rate (r=0.30) only, with 2 outer folds instead of 5, "
                              "to validate the full 3x3 source-target pipeline mechanically before "
                              "committing to the full run. Writes to results/phase5_smoke/.")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv. "
                              "If omitted, tries ../Dataset/<that file> relative to this script first, "
                              "then the original cloud-sandbox path.")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path)
