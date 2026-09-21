"""Fixes the outer-test selection-leakage bug: model-family selection now uses inner-CV scores only, per Protocol v2.1 Phase 3."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

import missingness_generator_v2 as genmod


_BASE_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_model_runner.py")

if not os.path.exists(_BASE_RUNNER_PATH):
    raise FileNotFoundError(
        "selection_v2.py cannot find base_model_runner.py. Place a copy of the "
        f"base runner script next to selection_v2.py (expected at {_BASE_RUNNER_PATH!r}) to fix this."
    )

_spec = importlib.util.spec_from_file_location("base_model_runner", _BASE_RUNNER_PATH)
base_runner = importlib.util.module_from_spec(_spec)
sys.modules["base_model_runner"] = base_runner
_spec.loader.exec_module(base_runner)

MODELS: List[str] = base_runner.MODELS
INNER_FOLDS: int = base_runner.INNER_FOLDS


@dataclasses.dataclass
class FamilyResult:
    model_key: str
    best_params: Dict[str, Any]
    inner_auroc_mean: float
    inner_ap_mean: float
    inner_brier_mean: float
    outer_test_auroc: float
    outer_test_ap: float
    outer_test_brier: float


@dataclasses.dataclass
class OuterFoldResult:
    fold_id: int
    families: Dict[str, FamilyResult]
    selected_family: str
    oracle_family: str
    selected_outer_test_auroc: float
    oracle_outer_test_auroc: float
    selection_regret: float
    displaced: bool


@dataclasses.dataclass
class FitAndSelectResult:
    """Everything produced by tuning and selecting on outer-train only, before outer-test data is touched."""
    selected_family: str
    final_models: Dict[str, Any]
    best_params: Dict[str, Dict[str, Any]]
    inner_scores: Dict[str, Tuple[float, float, float]]
    inner_metrics: Dict[str, Tuple[float, float, float]]


def _rank_key(auroc: float, ap: float, brier: float) -> Tuple[float, float, float]:
    return base_runner.rank_key(auroc, ap, brier)



def _fit_and_score_family_inner_cv(
    model_key: str,
    pre,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    groups_tune: np.ndarray,
    seed: int,
) -> Tuple[Pipeline, Dict[str, Any], float, float, float, bool]:
    """Tunes hyperparameters for one model family via inner CV and returns the refit pipeline plus its inner-CV scores."""
    from sklearn.model_selection import GridSearchCV

    base_model, grid, do_cal = base_runner.make_model_and_grid(model_key, seed)
    base_pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])

    inner_splitter = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 7)
    cv_iter = list(inner_splitter.split(X_tune, y_tune, groups_tune))

    scoring = {
        "auroc": "roc_auc",
        "ap": "average_precision",
    }
    gs = GridSearchCV(
        estimator=base_pipe,
        param_grid=grid,
        scoring=scoring,
        cv=cv_iter,
        refit="auroc",
        n_jobs=1,
    )
    gs.fit(X_tune, y_tune)

    best_idx = gs.best_index_
    inner_auroc_mean = float(gs.cv_results_["mean_test_auroc"][best_idx])
    inner_ap_mean = float(gs.cv_results_["mean_test_ap"][best_idx])
    inner_brier_mean = float("nan")

    return gs.best_estimator_, gs.best_params_, inner_auroc_mean, inner_ap_mean, inner_brier_mean, do_cal


def select_family(inner_scores: Dict[str, Tuple[float, float, float]]) -> str:
    """Selects the winning family from inner-CV scores only; no outer-test data is ever visible to this function."""
    best_key = None
    best_rank = None
    for model_key, (auroc, ap, brier) in inner_scores.items():
        rk = _rank_key(auroc, ap, brier)
        if best_rank is None or rk > best_rank:
            best_rank = rk
            best_key = model_key
    return best_key



def fit_and_select_all_families(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    calibration_test_size: float = 0.25,
) -> FitAndSelectResult:
    gss = GroupShuffleSplit(n_splits=1, test_size=calibration_test_size, random_state=seed + 13)
    tr_sub, cal_sub = next(gss.split(np.zeros_like(y_train), y_train, groups_train))

    X_tune = X_train.iloc[tr_sub]
    y_tune = y_train[tr_sub]
    g_tune = groups_train[tr_sub]
    X_cal = X_train.iloc[cal_sub]
    y_cal = y_train[cal_sub]

    pre = base_runner.build_preprocessor(num_cols, cat_cols, include_imputer=True, include_scaler=True)

    inner_scores: Dict[str, Tuple[float, float, float]] = {}
    inner_metrics: Dict[str, Tuple[float, float, float]] = {}
    best_params_by_family: Dict[str, Dict[str, Any]] = {}
    final_models: Dict[str, Any] = {}

    for model_key in MODELS:
        best_pipe, best_params, inner_auroc, inner_ap, inner_brier, do_cal = _fit_and_score_family_inner_cv(
            model_key, pre, X_tune, y_tune, g_tune, seed
        )
        inner_scores[model_key] = (inner_auroc, inner_ap, 0.0)
        inner_metrics[model_key] = (inner_auroc, inner_ap, inner_brier)
        best_params_by_family[model_key] = best_params

        if do_cal:
            calibrator = base_runner.PrefitCalibrator(best_pipe, method="sigmoid")
            calibrator.fit(X_cal, y_cal)
            final_model = calibrator
        else:
            final_model = best_pipe
        final_models[model_key] = final_model

    selected_family = select_family(inner_scores)

    return FitAndSelectResult(
        selected_family=selected_family,
        final_models=final_models,
        best_params=best_params_by_family,
        inner_scores=inner_scores,
        inner_metrics=inner_metrics,
    )


def evaluate_frozen_model(final_model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[float, float, float]:
    """Scores an already-fitted model on held-out data; no fitting happens here."""
    p_te = base_runner.predict_proba_safe(final_model, X_test)
    auroc = float(roc_auc_score(y_test, p_te))
    ap = float(average_precision_score(y_test, p_te))
    brier = float(brier_score_loss(y_test, np.clip(p_te, 0, 1)))
    return auroc, ap, brier



def run_outer_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    fold_id: int,
    calibration_test_size: float = 0.25,
) -> OuterFoldResult:
    fit_result = fit_and_select_all_families(
        X_train, y_train, groups_train, num_cols, cat_cols, seed, calibration_test_size
    )

    families: Dict[str, FamilyResult] = {}
    for model_key in MODELS:
        outer_auroc, outer_ap, outer_brier = evaluate_frozen_model(fit_result.final_models[model_key], X_test, y_test)
        inner_auroc, inner_ap, inner_brier = fit_result.inner_metrics[model_key]

        families[model_key] = FamilyResult(
            model_key=model_key,
            best_params=fit_result.best_params[model_key],
            inner_auroc_mean=inner_auroc,
            inner_ap_mean=inner_ap,
            inner_brier_mean=inner_brier,
            outer_test_auroc=outer_auroc,
            outer_test_ap=outer_ap,
            outer_test_brier=outer_brier,
        )

    selected_family = fit_result.selected_family

    oracle_scores = {k: (v.outer_test_auroc, v.outer_test_ap, v.outer_test_brier) for k, v in families.items()}
    oracle_family = select_family(oracle_scores)

    selected_auroc = families[selected_family].outer_test_auroc
    oracle_auroc = families[oracle_family].outer_test_auroc
    regret = oracle_auroc - selected_auroc

    return OuterFoldResult(
        fold_id=fold_id,
        families=families,
        selected_family=selected_family,
        oracle_family=oracle_family,
        selected_outer_test_auroc=selected_auroc,
        oracle_outer_test_auroc=oracle_auroc,
        selection_regret=regret,
        displaced=(selected_family != oracle_family),
    )



def run_condition(
    df: pd.DataFrame,
    seed: int = 2026,
    n_outer: int = 5,
) -> List[OuterFoldResult]:
    y = df[genmod.LABEL_COL].astype(int).to_numpy()
    groups = df[genmod.GROUP_COL].to_numpy()

    X = df.drop(columns=[genmod.LABEL_COL], errors="ignore").copy()
    for c in genmod.EXCLUDED_FROM_MODEL_ENTIRELY:
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")

    num_cols, cat_cols = base_runner.split_columns_A(df)
    present = set(X.columns)
    num_cols = [c for c in num_cols if c in present]
    cat_cols = [c for c in cat_cols if c in present]

    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    results: List[OuterFoldResult] = []
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splitter.split(X, y, groups), start=1):
        X_tr, y_tr, g_tr = X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], groups[tr_idx]
        X_te, y_te = X.iloc[te_idx].reset_index(drop=True), y[te_idx]
        assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"

        result = run_outer_fold(X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=seed, fold_id=fold_id)
        results.append(result)

    return results
