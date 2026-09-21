"""
Phase 8, Part 2 of 3 — Calibration slope/intercept backfill for MCAR/MAR/
MNAR-Y (implementation-order item 11).

WHY THIS EXISTS: Phase 4/5/6's saved result pickles were found, during Phase
8 planning, to contain only aggregate AUROC/AP/Brier per (condition, fold,
family) -- never the raw per-case predicted probabilities calibration
slope/intercept actually needs. `protocol_v2.yaml` already freezes
`calibration_primary: [slope, intercept]` (from Phase 7's metric layer) but
it has never actually been computed on real data, because the raw
predictions to compute it from were never persisted anywhere.

WHAT THIS DOES, AND WHY IT IS CHEAP (NOT a re-run of Phase 4): Phase 4 (and,
transitively, Phase 5's diagonal cells and Phase 6's without-indicator
cells -- both independently verified bit-for-bit identical to Phase 4 in the
Phase 6.5 audit) already recorded, for every (condition, fold), exactly
WHICH family was selected and exactly WHAT hyperparameters GridSearchCV
picked for it (`phase4_all_results.pkl`'s `FamilyResult.best_params`). This
driver does not repeat the tuning search across 5 families -- it directly
refits ONLY the already-known selected family, with its already-known best
hyperparameters, on the exact same fold-safe masked data Phase 4 used (same
OUTER_SEED/MASK_SEED, same StratifiedGroupKFold(subject_id) outer split,
same GroupShuffleSplit(test_size=0.25) train_sub/cal_sub calibration split).
That means: 1 model fit + 1 calibration fit per (condition, fold), versus
Phase 4's original 5 families x (3-inner-fold x up-to-9-hyperparam-combo)
GridSearchCV per (condition, fold) -- dramatically cheaper, while being a
BUILT-IN correctness check at the same time: the refit outer-test AUROC is
compared against Phase 4's own recorded value for that exact (condition,
fold, family) and logged if it drifts beyond floating-point noise (it
should not, since every input -- data, split, mask, hyperparameters -- is
byte-identical to what produced Phase 4's original number).

KNOWN, EMPIRICALLY-DIAGNOSED DRIFT CAVEAT (found during this driver's own
smoke-testing, cross-environment): re-running this exact refit logic in an
environment other than the one that produced `phase4_all_results.pkl` can
show a small (~1e-4 to ~2e-3) `auroc_drift` isolated ONLY to folds where
`selected_family == "xgb"`. This was root-caused, not assumed: for a
cross-environment reproduction test on real data (`mar_q20_main` fold 1),
all FOUR non-xgb families (`lr_l2`, `svm_linear_cal`, `rf`, `extratrees`)
reproduced Phase 4's saved outer-test AUROC bit-for-bit (drift exactly
0.0), while `xgb` alone drifted by ~5.7e-4 -- proving the fold-safe
masking/split/data pipeline is reproduced exactly and isolating the
discrepancy to XGBoost's own histogram-tree-construction internals, which
are not guaranteed bit-reproducible across xgboost library versions/builds
(`n_jobs=1` is already set, ruling out within-process thread-order
nondeterminism; two repeated fresh reruns in the SAME environment were
bit-identical to each other, ruling out true run-to-run randomness -- the
difference is specifically environment-to-environment). Because of this,
`auroc_drift` above 1e-6 is logged as INFO (not treated as a bug) up to a
much larger practical threshold; only drift beyond that threshold -- which
would need to appear on non-xgb families too, or be far larger than any
plausible xgboost-version artifact, to indicate a real masking/pipeline
bug rather than this benign library-version effect -- is escalated to an
actionable WARNING. See `DRIFT_INFO_THRESHOLD` / `DRIFT_WARN_THRESHOLD`
below and the Part-2 driver summary for the full diagnosis.

SCOPE: every condition in Phase 4's real completed grid (18 conditions --
natural_q00 + the 3-mechanism q00 reuse + 12 RQ1 rate-sweep conditions + 2
RQ2 OR-sweep conditions -- exactly what `phase4_all_results.pkl` contains).
This single backfill run therefore ALSO covers calibration for every Phase 5
diagonal cell and Phase 6 without-indicator cell that reuses one of these
exact (condition, fold) combinations -- no separate backfill is needed for
those, since they are already proven to be the identical fitted computation
(see phase8-driver-summary.md and the Phase 6.5 audit for the bit-for-bit
verification this relies on).

CALIBRATION METRIC: calibration slope/intercept via the standard logistic
recalibration definition -- fit `y ~ b0 + b1 * logit(clip(p, eps, 1-eps))`
by unregularized logistic regression on the outer-test predictions; b1 is
the calibration slope (1.0 = perfectly calibrated dispersion), b0 is the
calibration intercept (0.0 = no systematic over/under-prediction). Reported
per (condition, fold) and aggregated (mean/SD) per condition.

CHECKPOINTING: one checkpoint per condition, results/phase8_calibration/<id>.pkl.

Run with:   python3 run_phase8_calibration.py
Smoke-test: python3 run_phase8_calibration.py --smoke-test
            (1 condition, 2 outer folds instead of 5.)
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

import missingness_generator_v2 as genmod
import selection_v2 as selv2

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

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
        f"{_RELATIVE_DATA_PATH} (relative to this script) and {_CONTAINER_DATA_PATH!r}. "
        "Pass --data-path explicitly."
    )


def _phase4_results_pkl_candidates() -> List[Path]:
    """Mirrors run_phase6_rq4.py's multi-candidate Phase-4-artifact search
    (that driver's own adversarial-review fix), applied to the results
    pickle instead of the CSV -- this driver needs FamilyResult.best_params,
    which only the pickle carries (never written to any CSV)."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [script_dir.parent, script_dir, cwd.parent, cwd]
    suffixes = [Path("PHASE_4_RQ1_RQ2") / "results" / "phase4" / "phase4_all_results.pkl",
                Path("results") / "phase4" / "phase4_all_results.pkl"]
    seen: List[Path] = []
    for root in roots:
        for suf in suffixes:
            p = (root / suf).resolve()
            if p not in seen:
                seen.append(p)
    return seen


def _resolve_phase4_results_pkl(cli_arg: Optional[str], log) -> Path:
    if cli_arg:
        p = Path(cli_arg)
        if not p.exists():
            raise FileNotFoundError(f"--phase4-results-pkl was given as {cli_arg!r} but that file does not exist")
        return p
    for p in _phase4_results_pkl_candidates():
        if p.exists():
            log(f"Resolved Phase 4 results pickle via candidate search: {p}")
            return p
    raise FileNotFoundError(
        "Could not find phase4_all_results.pkl. Tried:\n" +
        "\n".join(f"  {p}" for p in _phase4_results_pkl_candidates()) +
        "\nPass --phase4-results-pkl /full/path/to/phase4_all_results.pkl explicitly. "
        "This file is required (not just the CSVs) because it carries FamilyResult.best_params, "
        "which this driver needs to refit the exact already-selected model without re-running "
        "GridSearchCV from scratch."
    )


OUTER_SEED = 2026
MASK_SEED = 20260917
N_OUTER = 5
THETA_MAIN = 0.693147
RESULTS_DIR = Path("results/phase8_calibration")

# auroc_drift is ALWAYS recorded in the output CSV regardless of these
# thresholds; these only control log-line severity (INFO vs WARNING).
#
# Family-aware, per an independent adversarial review's own empirical
# re-derivation (not just this driver's own initial 2-cell smoke test): the
# reviewer refit ALL 5 families (not just the selected one) against Phase
# 4's recorded best_params for 5 separate (condition, fold) cells, and then
# refit xgb alone across the FULL 90-cell grid. Result: non-xgb families
# (lr_l2, svm_linear_cal, rf, extratrees) were bit-for-bit identical to
# Phase 4's saved values in EVERY case (drift exactly 0.0, never nonzero even
# once) -- proving the masking/split/hyperparameter pipeline is reproduced
# exactly. xgb alone drifted, up to a max of 0.009146 across all 90 cells,
# symmetric around zero (33 positive / 37 negative / 20 exactly-zero
# drifts, no directional bias) -- the signature of xgboost's own
# cross-environment floating-point noise in histogram-based tree
# construction (n_jobs=1 is already set, and two same-environment reruns
# were bit-identical, ruling out true run-to-run randomness as the cause),
# not a pipeline bug. Because non-xgb families have NEVER been observed to
# drift at all, a single non-xgb drift above DRIFT_INFO_THRESHOLD already
# deserves attention and is escalated immediately; xgb gets a wider band
# (DRIFT_WARN_THRESHOLD_XGB, > 3x the empirically observed ceiling) since
# noise up to ~0.009 is expected there.
DRIFT_INFO_THRESHOLD = 1e-6
DRIFT_WARN_THRESHOLD_XGB = 0.03
DRIFT_WARN_THRESHOLD_OTHER = 1e-6  # any non-xgb drift above INFO is already a WARNING

# Every condition Phase 4 actually produced (mirrors run_phase4_rq1_rq2.py's
# build_condition_grid() + its q00 reuse-for-all-3-mechanisms convention).
CONDITIONS_TO_BACKFILL = (
    ["natural_q00", "mcar_q00", "mar_q00", "mnar_y_q00"] +
    [f"{m}_q{r:02d}_main" for m in ["mcar", "mar", "mnar_y"] for r in [10, 20, 30, 40]] +
    ["mnar_y_q30_OR1.5", "mnar_y_q30_OR4.0"]
)


# ---------------------------------------------------------------------------
# Fold-safe masking, identical logic to run_phase4_rq1_rq2.py's
# _fit_generator_and_drivers/run_condition_with_injection (duplicated here
# rather than imported, so this backfill driver has no import-time
# dependency on that script -- it only needs the already-completed Phase 4
# OUTPUT, never Phase 4's own driver code, keeping this a standalone,
# minimal-dependency script per this project's "ships to a sibling folder"
# delivery convention).
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


def _parse_condition(cid: str) -> Tuple[str, float, float, Optional[float]]:
    """condition_id -> (mechanism, rate, theta, or_value). Mirrors the exact
    literals run_phase4_rq1_rq2.py's build_condition_grid()/q00-reuse produced.

    NOTE: `natural_q00` reports mechanism="natural" (matching
    phase4_all_results.pkl's own condition_meta["natural_q00"]["mechanism"]
    exactly -- verified against all 18 conditions' condition_meta entries
    during adversarial review), NOT "mcar". This only affects the
    `mechanism` label written to the output CSV -- `_fit_generator_and_drivers`
    always short-circuits to `fit_natural_baseline()` at rate==0.0 before
    ever branching on mechanism, so the masking itself is unaffected either
    way. Getting the label right matters because downstream analysis may
    group/filter the CSV by `mechanism` rather than `condition_id`, and a
    mislabeled natural baseline would silently be miscounted as MCAR.
    """
    if cid == "natural_q00":
        return "natural", 0.0, 0.0, None
    if cid in ("mcar_q00", "mar_q00", "mnar_y_q00"):
        mech = cid.split("_q00")[0]
        return mech, 0.0, 0.0, (None if mech == "mcar" else 2.0)
    if cid.startswith("mnar_y_q30_OR"):
        or_val = float(cid.split("_OR")[1])
        return "mnar_y", 0.30, float(np.log(or_val)), or_val
    mech, rest = cid.rsplit("_q", 1)
    rate = int(rest.split("_")[0]) / 100.0
    theta = 0.0 if mech == "mcar" else THETA_MAIN
    or_val = None if mech == "mcar" else 2.0
    return mech, rate, theta, or_val


def _fit_generator_and_drivers(mechanism: str, rate: float, theta: float,
                                df_tr: pd.DataFrame, y_tr: np.ndarray,
                                nat_mask_tr: pd.DataFrame, df_te: pd.DataFrame, y_te: np.ndarray):
    if rate == 0.0:
        return genmod.fit_natural_baseline(), None, None
    if mechanism == "mcar":
        return genmod.fit_mcar(nat_mask_tr, target_rate=rate), None, None
    if mechanism == "mar":
        gen = genmod.fit_mar_context(nat_mask_tr, df_tr["admission_type"], theta=theta, target_rate=rate)
        driver_tr = genmod.compute_H_i(df_tr["admission_type"]).to_numpy().astype(float)
        driver_te = genmod.compute_H_i(df_te["admission_type"]).to_numpy().astype(float)
        return gen, driver_tr, driver_te
    if mechanism == "mnar_y":
        gen = genmod.fit_mnar_y(nat_mask_tr, df_tr[genmod.LABEL_COL], theta=theta, target_rate=rate)
        return gen, y_tr.astype(float), y_te.astype(float)
    raise ValueError(f"unknown mechanism: {mechanism}")


def _mask_fold(df: pd.DataFrame, mechanism: str, rate: float, theta: float,
                tr_idx: np.ndarray, te_idx: np.ndarray):
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    nat_mask_tr = genmod.compute_natural_mask(df_tr)
    nat_mask_te = genmod.compute_natural_mask(df_te)
    y_tr_full = df[genmod.LABEL_COL].astype(int).to_numpy()[tr_idx]
    y_te_full = df[genmod.LABEL_COL].astype(int).to_numpy()[te_idx]

    gen, driver_tr, driver_te = _fit_generator_and_drivers(
        mechanism, rate, theta, df_tr, y_tr_full, nat_mask_tr, df_te, y_te_full
    )
    rng = np.random.default_rng(MASK_SEED)
    syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
    syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
    final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
    final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)
    return final_mask_tr, final_mask_te


# ---------------------------------------------------------------------------
# Direct refit of the known-selected family with its known-best hyperparams
# (no GridSearchCV) -- the cheap step this whole driver exists to do.
# ---------------------------------------------------------------------------

def _refit_selected_family_and_predict(
    X_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    num_cols: List[str], cat_cols: List[str],
    model_key: str, best_params: Dict[str, Any], seed: int,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 13)
    tr_sub, cal_sub = next(gss.split(np.zeros_like(y_train), y_train, groups_train))
    X_tune, y_tune = X_train.iloc[tr_sub], y_train[tr_sub]
    X_cal, y_cal = X_train.iloc[cal_sub], y_train[cal_sub]

    pre = selv2.base_runner.build_preprocessor(num_cols, cat_cols, include_imputer=True, include_scaler=True)
    base_model, _grid, do_cal = selv2.base_runner.make_model_and_grid(model_key, seed)
    pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])
    pipe.set_params(**best_params)
    pipe.fit(X_tune, y_tune)

    if do_cal:
        calibrator = selv2.base_runner.PrefitCalibrator(pipe, method="sigmoid")
        calibrator.fit(X_cal, y_cal)
        final_model = calibrator
    else:
        final_model = pipe

    return selv2.evaluate_frozen_model_with_predictions(final_model, X_test, y_test)


# ---------------------------------------------------------------------------
# Calibration slope/intercept
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """Standard logistic recalibration: fit y ~ b0 + b1*logit(clip(p)) by
    unregularized logistic regression. b1 = slope (1.0 = ideal dispersion),
    b0 = intercept (0.0 = no systematic over/under-prediction)."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


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

    log.close = f.close
    return log


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None, phase4_pkl_arg: Optional[str] = None):
    results_dir = Path("results/phase8_calibration_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")
    log(f"=== Phase 8 (calibration backfill) driver start (smoke_test={smoke_test}) ===")

    data_path = _resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"

    phase4_pkl = _resolve_phase4_results_pkl(phase4_pkl_arg, log)
    with open(phase4_pkl, "rb") as f:
        phase4_data = pickle.load(f)
    phase4_results = phase4_data["all_results"]  # cid -> List[OuterFoldResult]

    conditions = CONDITIONS_TO_BACKFILL[:2] if smoke_test else CONDITIONS_TO_BACKFILL
    log(f"Backfilling calibration for {len(conditions)} conditions")

    X, y, groups, num_cols, cat_cols = _prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=OUTER_SEED)
    fold_splits = list(outer_splitter.split(X, y, groups))
    n_outer = 2 if smoke_test else N_OUTER

    all_rows: List[Dict[str, Any]] = []
    all_predictions: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    skipped_conditions: List[str] = []

    for i, cid in enumerate(conditions, start=1):
        ckpt = results_dir / f"{cid}.pkl"
        if ckpt.exists():
            log(f"[{i}/{len(conditions)}] {cid}: checkpoint already exists, skipping")
            with open(ckpt, "rb") as f:
                rows, preds = pickle.load(f)
            all_rows.extend(rows)
            all_predictions[cid] = preds
            continue

        if cid not in phase4_results:
            log(f"[{i}/{len(conditions)}] {cid}: WARNING -- NOT FOUND in {phase4_pkl}, skipping "
                f"(results incomplete? this condition will be MISSING from the final table)")
            skipped_conditions.append(cid)
            continue

        mechanism, rate, theta, or_value = _parse_condition(cid)
        log(f"[{i}/{len(conditions)}] {cid}: starting (mechanism={mechanism} rate={rate} theta={theta:.6f})")
        t0 = time.time()
        rows: List[Dict[str, Any]] = []
        preds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        fold_results = phase4_results[cid]
        for fr in fold_results[:n_outer]:
            fold_id = fr.fold_id
            tr_idx, te_idx = fold_splits[fold_id - 1]
            selected_family = fr.selected_family
            best_params = fr.families[selected_family].best_params
            original_auroc = fr.families[selected_family].outer_test_auroc

            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y[tr_idx]
            g_tr = groups[tr_idx]
            X_te = X.iloc[te_idx].reset_index(drop=True)
            y_te = y[te_idx]

            final_mask_tr, final_mask_te = _mask_fold(df, mechanism, rate, theta, tr_idx, te_idx)
            X_tr_masked, X_te_masked = X_tr.copy(), X_te.copy()
            for col in genmod.ELIGIBLE_LAB_COLS:
                X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
                X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan

            refit_auroc, refit_ap, refit_brier, y_sel, p_sel = _refit_selected_family_and_predict(
                X_tr_masked, y_tr, g_tr, X_te_masked, y_te, num_cols, cat_cols,
                selected_family, best_params, seed=OUTER_SEED,
            )
            drift = abs(refit_auroc - original_auroc)
            warn_threshold = DRIFT_WARN_THRESHOLD_XGB if selected_family == "xgb" else DRIFT_WARN_THRESHOLD_OTHER
            if drift > warn_threshold:
                extra = ("" if selected_family == "xgb" else
                         " -- non-xgb families have NEVER been observed to drift at all in independent "
                         "review; this is unexpected and should be investigated first")
                log(f"    WARNING fold {fold_id} ({selected_family}): refit AUROC {refit_auroc:.6f} vs Phase 4's "
                    f"original {original_auroc:.6f} (drift={drift:.6f}) -- larger than the known noise band"
                    f"{extra}")
            elif drift > DRIFT_INFO_THRESHOLD:
                log(f"    INFO fold {fold_id} ({selected_family}): refit AUROC {refit_auroc:.6f} vs Phase 4's "
                    f"original {original_auroc:.6f} (drift={drift:.6f}) -- within the expected xgboost "
                    f"cross-environment noise band (see module docstring); not a concern on its own")

            slope, intercept = calibration_slope_intercept(y_sel, p_sel)
            rows.append(dict(
                condition_id=cid, mechanism=mechanism, rate=rate, or_value=or_value, fold_id=fold_id,
                selected_family=selected_family, refit_outer_test_auroc=refit_auroc,
                phase4_outer_test_auroc=original_auroc, auroc_drift=drift,
                calibration_slope=slope, calibration_intercept=intercept, n_test=len(y_sel),
            ))
            preds[fold_id] = (y_sel, p_sel)
            log(f"    fold {fold_id}: family={selected_family} refit_auroc={refit_auroc:.4f} "
                f"(phase4={original_auroc:.4f}) slope={slope:.3f} intercept={intercept:.3f}")

        elapsed = time.time() - t0
        with open(ckpt, "wb") as f:
            pickle.dump((rows, preds), f)
        all_rows.extend(rows)
        all_predictions[cid] = preds
        log(f"[{i}/{len(conditions)}] {cid}: done in {elapsed:.1f}s, checkpointed to {ckpt}")

    expected_rows = len(conditions) * n_outer
    table = pd.DataFrame(all_rows)
    table.to_csv(results_dir / "phase8_calibration_table.csv", index=False)
    if len(table) != expected_rows:
        skip_note = skipped_conditions if skipped_conditions else (
            "(none -- check per-fold logs above for a different cause, e.g. a condition with fewer "
            "than n_outer saved folds)"
        )
        log(f"WARNING: expected {expected_rows} rows ({len(conditions)} conditions x {n_outer} folds) "
            f"but wrote {len(table)}. Skipped conditions: {skip_note}")
    else:
        log(f"Wrote {results_dir / 'phase8_calibration_table.csv'} ({len(table)} rows, "
            f"matches expected {expected_rows} = {len(conditions)} conditions x {n_outer} folds)")

    if len(table):
        summary = table.groupby("condition_id").agg(
            mean_slope=("calibration_slope", "mean"), sd_slope=("calibration_slope", "std"),
            mean_intercept=("calibration_intercept", "mean"), sd_intercept=("calibration_intercept", "std"),
            max_auroc_drift=("auroc_drift", "max"),
        ).round(4)
        summary.to_json(results_dir / "phase8_calibration_summary.json", orient="index", indent=2)
        log(f"Wrote {results_dir / 'phase8_calibration_summary.json'}")

    with open(results_dir / "phase8_calibration_predictions.pkl", "wb") as f:
        pickle.dump(all_predictions, f)
    log(f"Wrote {results_dir / 'phase8_calibration_predictions.pkl'}")
    log("=== Phase 8 (calibration backfill) driver finished ===")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--phase4-results-pkl", default=None,
                         help="Explicit path to phase4_all_results.pkl. If omitted, searched automatically "
                              "(same sibling-folder convention as run_phase6_rq4.py's Phase-4-CSV resolution).")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path, phase4_pkl_arg=args.phase4_results_pkl)
