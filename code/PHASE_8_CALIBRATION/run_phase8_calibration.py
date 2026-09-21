"""Phase 8 driver, part 2 of 3: backfills calibration slope/intercept for MCAR/MAR/MNAR-Y by refitting only the already-selected family."""

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
    """Searches multiple candidate paths for the Phase 4 results pickle, mirroring run_phase6_rq4.py's own search."""
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

DRIFT_INFO_THRESHOLD = 1e-6
DRIFT_WARN_THRESHOLD_XGB = 0.03
DRIFT_WARN_THRESHOLD_OTHER = 1e-6

CONDITIONS_TO_BACKFILL = (
    ["natural_q00", "mcar_q00", "mar_q00", "mnar_y_q00"] +
    [f"{m}_q{r:02d}_main" for m in ["mcar", "mar", "mnar_y"] for r in [10, 20, 30, 40]] +
    ["mnar_y_q30_OR1.5", "mnar_y_q30_OR4.0"]
)



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
    """Parses a condition_id into (mechanism, rate, theta, or_value), matching Phase 4's condition-grid literals."""
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



def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """Standard logistic recalibration: fits slope and intercept of the outcome against the predicted logit."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept



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
    phase4_results = phase4_data["all_results"]

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
