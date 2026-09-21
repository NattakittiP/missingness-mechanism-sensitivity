"""
Phase 4 driver — RQ1 (rate sweep) + RQ2 (mechanism-strength sweep).
Protocol v2.1 Phase 4: "After selection integrity is correct, run the first
full experiment ... Do not begin with the full source-target matrix."

WHAT THIS RUNS (14 fresh conditions; full derivation in phase4-driver-summary.md):

  RQ1 (rate sweep, strength held at the frozen main setting theta=ln(2), OR=2):
    mechanism in {mcar, mar, mnar_y} x rate in {0.10, 0.20, 0.30, 0.40}      -> 12 conditions
    (rate=0.0 is mechanism-independent by construction -- fit_alphas() gives
     alpha=-inf for target_rate=0.0 regardless of mechanism/driver -- so it is
     REUSED from the already-completed Phase 3 Natural/q=0 validation run
     instead of recomputed 3 more times.)

  RQ2 (strength sweep, mnar_y only per protocol_v2.yaml's frozen
       mnar_y.sensitivity_or list; rate held at the frozen main setting q=0.30):
    OR in {1.5, 4.0}                                                         -> 2 conditions
    (OR=2.0 @ q=0.30 is already produced by RQ1's mnar_y rate sweep above --
     reused, not rerun.)

  Total fresh compute: 14 conditions x 5 StratifiedGroupKFold(subject_id)
  outer folds x 5 model families (with full GridSearchCV inner tuning) each.
  Measured cost of one such condition on this codebase (Phase 3's Natural/q=0
  validation, same fold/family/tuning cost): 493.4s. Budget accordingly
  (~1.9-2h at that per-condition rate; will vary with the machine this runs on).

METHODOLOGICAL DECISIONS MADE HERE THAT WERE NOT FULLY PINNED DOWN ELSEWHERE
(both confirmed with the user before this script was written -- see
phase4-driver-summary.md for the full paper trail):

  1. Mask-seed repeats per (mechanism, rate) condition = 1 (not the
     provisionally-proposed, never-frozen "20 seeds" from
     methodology-redesign-spec.md's pre-Phase-2 open-items list). The 5 outer
     StratifiedGroupKFold folds serve as the repeated realizations used for
     entropy / Kendall's tau_b stability summaries -- same convention already
     used and reported in Phase 3 / Phase 7. These metrics therefore quantify
     ACROSS-FOLD stability (patient-split variation entangled with
     missingness-realization variation), not isolated mask-seed variance.
     Neither RQ1 nor RQ2 as frozen in the source protocol makes a claim that
     requires separating those two noise sources, so this is not a
     limitation relative to what RQ1/RQ2 actually ask.
  2. Mask RNG convention: for every (mechanism, rate, fold) cell, a FRESH
     `np.random.default_rng(MASK_SEED)` is instantiated (same MASK_SEED used
     throughout, matching phase2_3_manipulation_pilot.py's documented
     common-random-numbers convention), then used for the fold's train draw
     first and its test draw second (one continuing stream, not two
     independently-seeded ones) -- identical pattern to the pilot, now applied
     per-fold instead of the pilot's single fold.

FOLD-SAFETY (Protocol v2.1 Phase 2.2 Test 6 / methodology-redesign-spec.md
Section 9): for every (mechanism, rate, fold) cell, alpha_j is calibrated
(fit_mcar / fit_mar_context / fit_mnar_y) on THAT FOLD'S OWN outer-train
partition only, then the SAME fitted generator (same alpha_j, same theta) is
applied via apply_synthetic_mask() separately to that fold's outer-train and
outer-test rows. Outer-test rows never influence alpha_j. This is why
selection_v2.run_condition() (which expects an already-masked, single global
df) is NOT used directly here -- it does its own internal splitting and has
no per-fold masking hook. This driver re-implements the same X/y/groups setup
and outer-split logic as run_condition(), then inserts the per-fold masking
step before calling selection_v2.run_outer_fold() (Phase 3's fold-safe
selection routine, reused verbatim and unmodified).

CHECKPOINTING: each condition's result is pickled to results/phase4/<id>.pkl
immediately after that condition finishes. Re-running this script skips any
condition whose checkpoint file already exists, so an interrupted ~2h run can
simply be restarted from where it left off, not from scratch.

Run with:   python3 run_phase4_rq1_rq2.py
Smoke-test: python3 run_phase4_rq1_rq2.py --smoke-test
            (1 condition, 2 outer folds instead of 5, exercises the exact
            same code path -- masking, selection_v2, GridSearchCV, metrics_v2
            aggregation -- at a scale that finishes in a couple of minutes,
            to validate the pipeline mechanically before committing to the
            full run. Writes to results/phase4_smoke/, never touches
            results/phase4/, so it cannot corrupt or count towards the real run.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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

# ---------------------------------------------------------------------------
# Importing selection_v2 dynamically execs the base runner module, which
# calls warnings.filterwarnings("ignore", category=UserWarning/FutureWarning)
# GLOBALLY for the rest of this process (confirmed by the Phase 4 pre-flight
# independent code review). Left alone, a multi-hour unattended run would get
# zero visibility into sklearn ConvergenceWarning (a UserWarning subclass) or
# anything else, across its entire duration. Restore default visibility here,
# right after the import that causes it.
# ---------------------------------------------------------------------------
warnings.resetwarnings()
# "once": print each distinct (message, category) exactly once for the whole
# process, not once per call site. Verified during the smoke test that plain
# "default" reprints sklearn's FutureWarning dozens of times per fold (once
# per internal call site inside GridSearchCV's repeated fits) -- across the
# full ~14-condition x 5-fold x 5-family run that would bury a genuinely new
# warning (e.g. ConvergenceWarning) under thousands of already-known,
# already-benign FutureWarning lines. "once" still guarantees a first-time
# ConvergenceWarning (or anything else) is shown exactly once, which is the
# actual goal -- visibility, not silence.
warnings.simplefilter("once")
# Verified via the smoke test that plain "once" does NOT fully collapse these
# two specific sklearn FutureWarnings (they still reprint ~20x across a
# handful of GridSearchCV fits -- sklearn's internal warnings.warn() calls
# apparently don't share one __warningregistry__ across repeated fits here).
# Both are benign, sklearn-version-only API-deprecation notices about lr_l2's
# hyperparameters (nothing about data, convergence, or result validity), so
# they are explicitly silenced by message text -- everything else (including
# a first-time ConvergenceWarning, which WOULD matter) still surfaces via the
# "once" filter above.
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

# Path resolution (same portability fix applied to selection_v2.py's base-runner
# path -- the original hardcoded path is specific to the cloud sandbox this
# driver was developed in and does not exist once this script ships to run
# elsewhere, e.g. the user's own machine). Resolved at runtime in run_all()
# via _resolve_data_path(), in order: (1) an explicit --data-path CLI arg,
# (2) "../Dataset/full_analytic_dataset_mortality_all_admissions.csv" relative
# to this script -- matches the delivery layout where this driver ships into
# a sibling folder of the existing Dataset/ folder, (3) the original
# cloud-sandbox absolute path, for continuity there without needing a
# relative layout.
_CONTAINER_DATA_PATH = "/mnt/user-data/uploads/DASA2026/Dataset/full_analytic_dataset_mortality_all_admissions.csv"
_RELATIVE_DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "full_analytic_dataset_mortality_all_admissions.csv"


def _resolve_data_path(cli_arg: Optional[str]) -> str:
    if cli_arg:
        if not Path(cli_arg).exists():
            raise FileNotFoundError(f"--data-path was given as {cli_arg!r} but that file does not exist")
        return cli_arg
    if _RELATIVE_DATA_PATH.exists():
        return str(_RELATIVE_DATA_PATH)
    if Path(_CONTAINER_DATA_PATH).exists():
        return _CONTAINER_DATA_PATH
    raise FileNotFoundError(
        "Could not find full_analytic_dataset_mortality_all_admissions.csv. Tried "
        f"{_RELATIVE_DATA_PATH} (relative to this script) and {_CONTAINER_DATA_PATH!r} "
        "(cloud-sandbox path). Pass --data-path /full/path/to/the/csv explicitly."
    )
OUTER_SEED = 2026            # SAME seed as Phase 3's Natural/q=0 validation -- required for q=0 reuse + identical fold partitions across all conditions
MASK_SEED = 20260917         # SAME mask seed as phase2_3_manipulation_pilot.py (documented CRN convention)
N_OUTER = 5

THETA_MAIN = 0.693147        # ln(2), OR=2 -- frozen MAR + MNAR-Y main strength (protocol_v2.yaml)
RATES_MAIN = [0.10, 0.20, 0.30, 0.40]
RQ2_RATE = 0.30
RQ2_EXTRA_ORS = {1.5: float(np.log(1.5)), 4.0: float(np.log(4.0))}  # OR=2.0 @ q=0.30 reused from RQ1

RESULTS_DIR = Path("results/phase4")
NATURAL_PICKLE_CANDIDATES = [
    Path("phase3_validation_natural_results.pkl"),
    Path(__file__).resolve().parent / "phase3_validation_natural_results.pkl",
]


# ---------------------------------------------------------------------------
# Condition grid
# ---------------------------------------------------------------------------

def build_condition_grid() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for mech in ["mcar", "mar", "mnar_y"]:
        theta = 0.0 if mech == "mcar" else THETA_MAIN
        or_value = None if mech == "mcar" else 2.0
        for r in RATES_MAIN:
            conditions.append(dict(
                rq="RQ1", mechanism=mech, rate=r, theta=theta, or_value=or_value,
                condition_id=f"{mech}_q{int(round(r * 100)):02d}_main",
            ))
    for or_val, theta in RQ2_EXTRA_ORS.items():
        conditions.append(dict(
            rq="RQ2", mechanism="mnar_y", rate=RQ2_RATE, theta=theta, or_value=or_val,
            condition_id=f"mnar_y_q{int(round(RQ2_RATE * 100)):02d}_OR{or_val}",
        ))
    return conditions


# ---------------------------------------------------------------------------
# Fold-safe masking + one condition's full 5-fold nested-CV run
# ---------------------------------------------------------------------------

def _prepare_X_y_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    y = df[genmod.LABEL_COL].astype(int).to_numpy()
    groups = df[genmod.GROUP_COL].to_numpy()
    X = df.drop(columns=[genmod.LABEL_COL], errors="ignore").copy()
    for c in genmod.EXCLUDED_FROM_MODEL_ENTIRELY:
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")
    num_cols, cat_cols = selv2.base_runner.split_columns_A(df)
    present = set(X.columns)
    num_cols = [c for c in num_cols if c in present]
    cat_cols = [c for c in cat_cols if c in present]
    return X, y, groups, num_cols, cat_cols


def _fit_generator_and_drivers(mechanism: str, rate: float, theta: float,
                                df_tr: pd.DataFrame, y_tr: np.ndarray,
                                nat_mask_tr: pd.DataFrame, df_te: pd.DataFrame,
                                y_te: np.ndarray):
    if rate == 0.0:
        gen = genmod.fit_natural_baseline()
        return gen, None, None
    if mechanism == "mcar":
        gen = genmod.fit_mcar(nat_mask_tr, target_rate=rate)
        return gen, None, None
    if mechanism == "mar":
        gen = genmod.fit_mar_context(nat_mask_tr, df_tr["admission_type"], theta=theta, target_rate=rate)
        driver_tr = genmod.compute_H_i(df_tr["admission_type"]).to_numpy().astype(float)
        driver_te = genmod.compute_H_i(df_te["admission_type"]).to_numpy().astype(float)
        return gen, driver_tr, driver_te
    if mechanism == "mnar_y":
        gen = genmod.fit_mnar_y(nat_mask_tr, df_tr[genmod.LABEL_COL], theta=theta, target_rate=rate)
        return gen, y_tr.astype(float), y_te.astype(float)
    raise ValueError(f"unknown mechanism: {mechanism}")


def run_condition_with_injection(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    theta: float,
    seed: int = OUTER_SEED,
    mask_seed: int = MASK_SEED,
    n_outer: int = N_OUTER,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Fold-safe: for each outer fold, calibrate the generator on THAT fold's
    own outer-train partition only, apply the same fitted generator to both
    outer-train and outer-test of that fold, then run selection_v2's
    (inner-CV-only) family selection + outer-test evaluation."""
    X, y, groups, num_cols, cat_cols = _prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    results = []
    diagnostics = []
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splitter.split(X, y, groups), start=1):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        g_tr = groups[tr_idx]
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]
        assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        nat_mask_tr = genmod.compute_natural_mask(df_tr)
        nat_mask_te = genmod.compute_natural_mask(df_te)

        gen, driver_tr, driver_te = _fit_generator_and_drivers(
            mechanism, rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
        )

        rng = np.random.default_rng(mask_seed)  # fresh per (mechanism, rate, fold) cell -- see module docstring
        syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
        syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)

        final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
        final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

        X_tr_masked = X_tr.copy()
        X_te_masked = X_te.copy()
        for col in genmod.ELIGIBLE_LAB_COLS:
            X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
            X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan

        rates_tr = genmod.compute_rates(nat_mask_tr, syn_mask_tr)
        rates_te = genmod.compute_rates(nat_mask_te, syn_mask_te)

        diag = {
            "fold_id": fold_id, "mechanism": mechanism, "target_rate": rate, "theta": theta,
            "r_inject_train": rates_tr["r_inject"], "r_total_train": rates_tr["r_total"],
            "r_inject_test": rates_te["r_inject"], "r_total_test": rates_te["r_total"],
        }
        if mechanism == "mnar_y" and rate > 0.0:
            syn_te_np = syn_mask_te.to_numpy()
            y1 = y_te == 1
            y0 = y_te == 0
            elig_te = genmod.compute_eligible_mask(nat_mask_te).to_numpy()
            diag["r_inject_among_y1"] = float(syn_te_np[y1].sum() / max(elig_te[y1].sum(), 1))
            diag["r_inject_among_y0"] = float(syn_te_np[y0].sum() / max(elig_te[y0].sum(), 1))
        if mechanism == "mar" and rate > 0.0:
            mask_count_per_patient = syn_mask_te.to_numpy().sum(axis=1)
            if np.std(driver_te) > 0 and np.std(mask_count_per_patient) > 0:
                diag["corr_mask_count_vs_H_i"] = float(np.corrcoef(mask_count_per_patient, driver_te)[0, 1])
        diagnostics.append(diag)

        result = selv2.run_outer_fold(
            X_tr_masked, y_tr, g_tr, X_te_masked, y_te, num_cols, cat_cols, seed=seed, fold_id=fold_id
        )
        results.append(result)

    return results, diagnostics


# ---------------------------------------------------------------------------
# q=0 reuse from Phase 3
# ---------------------------------------------------------------------------

def load_or_compute_natural_q0(df: pd.DataFrame, log, n_outer: int = N_OUTER) -> Tuple[List[Any], List[Dict[str, Any]]]:
    for p in NATURAL_PICKLE_CANDIDATES:
        if p.exists():
            log(f"q=0 (Natural): reusing existing Phase 3 validation results from {p}")
            with open(p, "rb") as f:
                results = pickle.load(f)
            diagnostics = [
                {"fold_id": r.fold_id, "mechanism": "natural", "target_rate": 0.0, "theta": 0.0,
                 "r_inject_train": 0.0, "r_total_train": None, "r_inject_test": 0.0, "r_total_test": None}
                for r in results
            ]
            return results, diagnostics
    log("q=0 (Natural): no existing pickle found, computing fresh via run_condition_with_injection(rate=0.0)")
    return run_condition_with_injection(df, mechanism="mcar", rate=0.0, theta=0.0, n_outer=n_outer)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    log.close = f.close  # exposed so run_all() can close the handle explicitly when done
    return log


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None):
    results_dir = Path("results/phase4_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    log(f"=== Phase 4 driver start (smoke_test={smoke_test}, n_outer={n_outer}) ===")

    data_path = _resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    conditions = build_condition_grid()
    if smoke_test:
        # Use the mnar_y condition specifically (not mcar) -- it exercises the
        # most code: outcome-driven alpha fitting, the Y-split diagnostics
        # branch, and is the mechanism with the highest label-leakage risk if
        # fold-safety were ever broken, so it's the most informative single
        # condition to smoke-test.
        conditions = [c for c in conditions if c["condition_id"] == "mnar_y_q10_main"]
    log(f"Condition grid: {len(conditions)} fresh conditions")
    for c in conditions:
        log(f"  {c['condition_id']}: rq={c['rq']} mechanism={c['mechanism']} rate={c['rate']} theta={c['theta']:.6f}")

    all_results: Dict[str, List[Any]] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    condition_meta: Dict[str, Dict[str, Any]] = {}

    # --- q=0 (Natural), shared across all 3 mechanisms, computed/reused once ---
    q0_ckpt = results_dir / "natural_q0.pkl"
    if q0_ckpt.exists():
        log("q=0 (Natural): checkpoint already exists, loading")
        with open(q0_ckpt, "rb") as f:
            q0_results, q0_diag = pickle.load(f)
    else:
        t0 = time.time()
        q0_results, q0_diag = load_or_compute_natural_q0(df, log, n_outer=n_outer)
        with open(q0_ckpt, "wb") as f:
            pickle.dump((q0_results, q0_diag), f)
        log(f"q=0 (Natural): done in {time.time() - t0:.1f}s, checkpointed to {q0_ckpt}")
    all_results["natural_q00"] = q0_results
    all_diagnostics.extend(q0_diag)
    condition_meta["natural_q00"] = dict(rq="RQ1", mechanism="natural", rate=0.0, theta=0.0, or_value=None)
    # q=0 is reused verbatim as the rate=0 point for all 3 mechanisms in RQ1 plots/tables.
    for mech in ["mcar", "mar", "mnar_y"]:
        all_results[f"{mech}_q00"] = q0_results
        condition_meta[f"{mech}_q00"] = dict(rq="RQ1", mechanism=mech, rate=0.0, theta=0.0, or_value=(None if mech == "mcar" else 2.0))

    # --- fresh conditions ---
    for i, c in enumerate(conditions, start=1):
        cid = c["condition_id"]
        ckpt = results_dir / f"{cid}.pkl"
        condition_meta[cid] = c
        if ckpt.exists():
            log(f"[{i}/{len(conditions)}] {cid}: checkpoint already exists, skipping")
            with open(ckpt, "rb") as f:
                r, d = pickle.load(f)
            all_results[cid] = r
            all_diagnostics.extend(d)
            continue

        log(f"[{i}/{len(conditions)}] {cid}: starting (rq={c['rq']} mechanism={c['mechanism']} "
            f"rate={c['rate']} theta={c['theta']:.6f} OR={c['or_value']})")
        t0 = time.time()
        r, d = run_condition_with_injection(
            df, mechanism=c["mechanism"], rate=c["rate"], theta=c["theta"], n_outer=n_outer,
        )
        elapsed = time.time() - t0
        with open(ckpt, "wb") as f:
            pickle.dump((r, d), f)
        all_results[cid] = r
        all_diagnostics.extend(d)
        for fold_result in r:
            log(f"    fold {fold_result.fold_id}: selected={fold_result.selected_family} "
                f"outer_test_auroc={fold_result.selected_outer_test_auroc:.4f} "
                f"oracle={fold_result.oracle_family} regret={fold_result.selection_regret:.4f} "
                f"displaced={fold_result.displaced}")
        log(f"[{i}/{len(conditions)}] {cid}: done in {elapsed:.1f}s, checkpointed to {ckpt}")

    log("All conditions complete. Aggregating...")
    aggregate_and_save(all_results, all_diagnostics, condition_meta, results_dir, log)
    log("=== Phase 4 driver finished ===")
    log.close()


# ---------------------------------------------------------------------------
# Aggregation: per-fold-per-family table, metrics_v2 layer, manipulation table
# ---------------------------------------------------------------------------

def aggregate_and_save(all_results, all_diagnostics, condition_meta, results_dir: Path, log):
    # 1) Full per-fold-per-family table
    rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            for fam, fres in fr.families.items():
                rows.append({
                    "condition_id": cid, "rq": meta["rq"], "mechanism": meta["mechanism"],
                    "rate": meta["rate"], "theta": meta["theta"], "or_value": meta["or_value"],
                    "fold_id": fr.fold_id, "model_key": fam,
                    "inner_auroc_mean": fres.inner_auroc_mean, "inner_ap_mean": fres.inner_ap_mean,
                    "outer_test_auroc": fres.outer_test_auroc, "outer_test_ap": fres.outer_test_ap,
                    "outer_test_brier": fres.outer_test_brier,
                    "selected_family": fr.selected_family, "oracle_family": fr.oracle_family,
                    "selection_regret": fr.selection_regret, "displaced": fr.displaced,
                })
    full_table = pd.DataFrame(rows)
    full_table.to_csv(results_dir / "phase4_full_table.csv", index=False)
    log(f"Wrote {results_dir / 'phase4_full_table.csv'} ({len(full_table)} rows)")

    # 2) Fold-summary (one row per condition x fold, selected family only)
    summary_rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            summary_rows.append({
                "condition_id": cid, "rq": meta["rq"], "mechanism": meta["mechanism"],
                "rate": meta["rate"], "or_value": meta["or_value"], "fold_id": fr.fold_id,
                "selected_family": fr.selected_family, "selected_outer_test_auroc": fr.selected_outer_test_auroc,
                "oracle_family": fr.oracle_family, "oracle_outer_test_auroc": fr.oracle_outer_test_auroc,
                "selection_regret": fr.selection_regret, "displaced": fr.displaced,
            })
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_csv(results_dir / "phase4_fold_summary.csv", index=False)
    log(f"Wrote {results_dir / 'phase4_fold_summary.csv'}")

    # 3) Manipulation-check / rate-diagnostics table
    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(results_dir / "phase4_manipulation_check.csv", index=False)
    log(f"Wrote {results_dir / 'phase4_manipulation_check.csv'}")

    # 4) metrics_v2 layer: entropy + Kendall tau_b per condition; margins per fold;
    #    displacement rate per mechanism across the rate sweep (incl. RQ2's mnar_y OR sweep separately)
    metrics_summary: Dict[str, Any] = {}
    for cid, fold_results in all_results.items():
        selected = met.selected_family_by_fold(fold_results)
        entropy = met.within_condition_selection_entropy(list(selected.values()), selv2.MODELS)
        scores_by_repeat = [met.outer_test_auroc_by_family(r) for r in fold_results]
        kendall = met.kendall_tau_b_rank_stability(scores_by_repeat) if len(scores_by_repeat) >= 2 else None
        sel_margins = [met.top1_minus_top2_margin(met.inner_auroc_by_family(r)) for r in fold_results]
        eval_margins = [met.top1_minus_top2_margin(met.outer_test_auroc_by_family(r)) for r in fold_results]
        metrics_summary[cid] = dict(
            entropy_normalized=entropy.normalized_entropy, modal_family=entropy.modal_family,
            modal_frequency=entropy.modal_frequency,
            kendall_tau_b_mean=(kendall.mean_tau_b if kendall else None),
            kendall_tau_b_min=(kendall.min_tau_b if kendall else None),
            kendall_tau_b_max=(kendall.max_tau_b if kendall else None),
            selection_margin_mean=float(np.mean(sel_margins)),
            evaluation_margin_mean=float(np.mean(eval_margins)),
        )

    # Baseline-selection displacement rate D_{m,r}: per mechanism, selected family
    # at rate r vs that SAME fold's selection at r=0, across the RQ1 rate sweep.
    displacement_by_mechanism: Dict[str, Any] = {}
    for mech in ["mcar", "mar", "mnar_y"]:
        by_fold_and_rate: Dict[int, Dict[float, str]] = {}
        for r in [0.0] + RATES_MAIN:
            cid = f"{mech}_q00" if r == 0.0 else f"{mech}_q{int(round(r * 100)):02d}_main"
            if cid not in all_results:
                continue
            for fr in all_results[cid]:
                by_fold_and_rate.setdefault(fr.fold_id, {})[r] = fr.selected_family
        disp = met.baseline_selection_displacement_rate(by_fold_and_rate, baseline_rate=0.0)
        displacement_by_mechanism[mech] = {
            r: dict(displacement_rate=v.displacement_rate, n_folds=v.n_folds, n_displaced=v.n_displaced,
                     displaced_fold_ids=v.displaced_fold_ids)
            for r, v in disp.items()
        }

    with open(results_dir / "phase4_metrics_summary.json", "w") as f:
        json.dump({"per_condition": metrics_summary, "displacement_by_mechanism": displacement_by_mechanism}, f, indent=2, default=str)
    log(f"Wrote {results_dir / 'phase4_metrics_summary.json'}")

    # 5) Raw results pickle (everything, for later reanalysis)
    with open(results_dir / "phase4_all_results.pkl", "wb") as f:
        pickle.dump({"all_results": all_results, "condition_meta": condition_meta, "all_diagnostics": all_diagnostics}, f)
    log(f"Wrote {results_dir / 'phase4_all_results.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run 1 condition x 2 outer folds only, to validate the pipeline mechanically "
                              "before committing to the full ~2h run. Writes to results/phase4_smoke/.")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv. "
                              "If omitted, tries ../Dataset/<that file> relative to this script first, "
                              "then the original cloud-sandbox path.")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path)
