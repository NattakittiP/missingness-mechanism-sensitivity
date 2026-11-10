"""
Phase 3 — Fix Winner-Selection Integrity (Protocol v2.1).

THE BUG (confirmed by direct code reading of
`PHASE 4/jcsse_audit_runner_tqdm_hardened.py`):

    run_config() computes, for every (model family, outer fold), the AUROC
    on the OUTER-TEST partition of that fold (`p_te` is predicted on `X_te`,
    the held-out outer-test rows). summarize_configs() averages this outer-test
    AUROC across outer folds per model family. compute_winners() then picks,
    for each experimental condition, the model family with the single best
    MEAN OUTER-TEST AUROC as the "winner".

    This means the outer-test partition is used FIVE TIMES (once per model
    family) and the best of the five is reported -- the outer test answers
    "which model should be selected?", which Protocol v2.1 Phase 3 explicitly
    forbids: "The outer test can answer: how well did the already-selected
    model perform? It must never answer: which model should be selected?"

    Within-family hyperparameter tuning (fit_best_model_nested / GridSearchCV)
    WAS already nested correctly -- that part is not the bug. The bug is
    specifically the ACROSS-FAMILY selection step.

THE FIX implemented here, exactly per Protocol v2.1 Phase 3's mandated structure:

    outer train
      -> inner grouped CV (StratifiedGroupKFold on subject_id, within outer-train)
      -> tune each of the 5 model families (GridSearchCV over each family's grid)
      -> the SAME inner-CV run also yields each family's inner-CV score
         (multi-metric: AUROC primary, AP secondary, Brier tertiary -- same
         rank_key tie-break convention as the original code)
      -> select the family with the single best inner-CV score  <-- SELECTION HAPPENS HERE, OUTER-TEST NEVER TOUCHED
      -> refit the selected family (best hyperparams) on train_sub of outer-train
      -> calibrate (if applicable) on a held-out calibration split of outer-train
      -> outer test ONCE, for the already-selected family only

For diagnostics (selection regret / displacement / oracle), the 4
non-selected families are ALSO refit and evaluated on outer-test -- but
this is a *post-hoc, read-only* computation that happens strictly AFTER
`select_family()` has already committed to a family using inner-CV scores
alone. `select_family()`'s function signature takes no outer-test data at
all, so this is enforced structurally (see test_selection_v2.py Test A).

This module reuses the exact model definitions, hyperparameter grids,
preprocessing pipeline, and calibration wrapper from the existing base
runner (imported dynamically below, since its path contains a space) to
guarantee the refactor changes ONLY the selection logic and nothing else
about how each model family is built or evaluated.

One deliberate, documented departure from the base runner: BOTH the outer
and inner splits here use StratifiedGroupKFold(subject_id) rather than the
base runner's "S2" GroupKFold (which doesn't guarantee stratification by
label under this dataset's 2.54% prevalence). This is the split upgrade
already frozen in protocol_v2.yaml (`split.outer: StratifiedGroupKFold`)
and methodology-redesign-spec.md, not an accidental behavior change bundled
into the selection fix.
"""

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

# ---------------------------------------------------------------------------
# Dynamically import the existing base runner (path contains a space, so a
# normal `import` statement can't reach it). This reuses MODELS,
# make_model_and_grid, build_preprocessor, PrefitCalibrator, predict_proba_safe,
# rank_key, split_columns_A verbatim -- Phase 3 changes ONLY the cross-family
# selection logic, nothing about how individual models are built.
#
# Path resolution (fixed during the Phase 4 pre-flight review): the original
# hardcoded absolute Linux container path only exists inside the cloud
# sandbox this code was developed in. Once this module ships to a different
# machine (e.g. the user's own Windows box, to run the Phase 4 driver there),
# that path never exists and the import fails immediately. Resolve in order:
#   1. A vendored copy sitting next to this file (jcsse_audit_runner_tqdm_hardened.py
#      in the same directory) -- this is what gets shipped alongside selection_v2.py.
#   2. The original absolute container path, for continuity inside this
#      cloud sandbox without needing a vendored copy there too.
# Raises a clear, actionable error if neither exists, rather than the opaque
# AttributeError/TypeError spec_from_file_location would otherwise raise.
# ---------------------------------------------------------------------------

_VENDORED_BASE_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jcsse_audit_runner_tqdm_hardened.py")
_CONTAINER_BASE_RUNNER_PATH = "/mnt/user-data/uploads/DASA2026/PHASE 4/jcsse_audit_runner_tqdm_hardened.py"

if os.path.exists(_VENDORED_BASE_RUNNER_PATH):
    _BASE_RUNNER_PATH = _VENDORED_BASE_RUNNER_PATH
elif os.path.exists(_CONTAINER_BASE_RUNNER_PATH):
    _BASE_RUNNER_PATH = _CONTAINER_BASE_RUNNER_PATH
else:
    raise FileNotFoundError(
        "selection_v2.py cannot find jcsse_audit_runner_tqdm_hardened.py. "
        f"Looked for a vendored copy at {_VENDORED_BASE_RUNNER_PATH!r} (same "
        f"directory as selection_v2.py) and the original cloud-sandbox path "
        f"at {_CONTAINER_BASE_RUNNER_PATH!r}. Place a copy of the base runner "
        "script next to selection_v2.py to fix this."
    )

_spec = importlib.util.spec_from_file_location("jcsse_base_runner", _BASE_RUNNER_PATH)
base_runner = importlib.util.module_from_spec(_spec)
sys.modules["jcsse_base_runner"] = base_runner
_spec.loader.exec_module(base_runner)

MODELS: List[str] = base_runner.MODELS  # ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]
INNER_FOLDS: int = base_runner.INNER_FOLDS  # 3

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

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
    selected_family: str          # chosen using INNER-CV scores only
    oracle_family: str            # diagnostic only: argmax of OUTER-TEST scores
    selected_outer_test_auroc: float
    oracle_outer_test_auroc: float
    selection_regret: float       # oracle_outer_test_auroc - selected_outer_test_auroc, >= 0
    displaced: bool               # selected_family != oracle_family


@dataclasses.dataclass
class FitAndSelectResult:
    """Everything produced by tuning+selecting on outer-train ONLY, before any
    outer-test data is touched. Added for Phase 5 (RQ3 source->target shift),
    which needs the actual frozen fitted model object for the selected family
    (run_outer_fold below discards it after computing metrics) so it can be
    evaluated against several different outer-test copies without retraining.
    Factored out of run_outer_fold rather than duplicated -- run_outer_fold
    now calls this function internally, so there is exactly one implementation
    of "tune all families on outer-train, select from inner-CV only" and
    Phase 3/4's already-validated behavior is preserved by construction (see
    test_selection_v2.py, all of which still exercises this same code path)."""
    selected_family: str
    final_models: Dict[str, Any]                       # family -> frozen, predict-ready model/calibrator
    best_params: Dict[str, Dict[str, Any]]
    inner_scores: Dict[str, Tuple[float, float, float]]  # family -> (auroc, ap, 0.0), what select_family() saw
    inner_metrics: Dict[str, Tuple[float, float, float]]  # family -> (inner_auroc_mean, inner_ap_mean, inner_brier_mean)


def _rank_key(auroc: float, ap: float, brier: float) -> Tuple[float, float, float]:
    # identical convention to base_runner.rank_key: AUROC desc, AP desc, Brier asc
    return base_runner.rank_key(auroc, ap, brier)


# ---------------------------------------------------------------------------
# Core: tune + inner-CV-score ONE model family on outer-train ONLY
# ---------------------------------------------------------------------------

def _fit_and_score_family_inner_cv(
    model_key: str,
    pre,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    groups_tune: np.ndarray,
    seed: int,
) -> Tuple[Pipeline, Dict[str, Any], float, float, float, bool]:
    """Tune hyperparameters for `model_key` via inner StratifiedGroupKFold on
    (X_tune, y_tune, groups_tune) -- a subset of outer-train only -- and
    return the refit-on-train_sub best pipeline PLUS its inner-CV scores
    (auroc/ap), which are what cross-family selection reads.

    Scoring is restricted to AUROC (primary) + average precision (secondary),
    matching protocol_v2.yaml's frozen selection criterion
    (`primary_selection_criterion: AUROC`, `secondary: [average_precision]`).
    Brier/ECE are deliberately NOT part of the inner-CV selection score: at
    this point `svm_linear_cal` (LinearSVC) has no `predict_proba` yet (only
    `decision_function` -- it is only calibrated into a probabilistic model
    AFTER a family is selected, on the held-out calibration split), so a
    pre-calibration Brier score isn't a well-posed comparison across families
    in the first place. Brier/ECE ARE still computed and reported at the
    outer-test stage below, where every family (including svm_linear_cal) has
    gone through its PrefitCalibrator and produces genuine probabilities.

    No outer-test data is passed into or visible from this function.
    """
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
    inner_brier_mean = float("nan")  # not computed pre-calibration; see docstring

    return gs.best_estimator_, gs.best_params_, inner_auroc_mean, inner_ap_mean, inner_brier_mean, do_cal


def select_family(inner_scores: Dict[str, Tuple[float, float, float]]) -> str:
    """THE SELECTION STEP. Takes ONLY inner-CV scores (family -> (auroc, ap, brier)),
    never outer-test data -- there is no parameter through which outer-test
    performance could even be passed in. Returns the selected model_key.

    Structural enforcement of Protocol v2.1 Phase 3's central requirement:
    "outer test selects family? Never."
    """
    best_key = None
    best_rank = None
    for model_key, (auroc, ap, brier) in inner_scores.items():
        rk = _rank_key(auroc, ap, brier)
        if best_rank is None or rk > best_rank:
            best_rank = rk
            best_key = model_key
    return best_key


# ---------------------------------------------------------------------------
# Tune + select (inner-CV only) all 5 families on outer-train. Outer-test is
# never referenced anywhere in this function -- structurally identical
# guarantee as select_family() itself, just at the "fit everything" level.
# ---------------------------------------------------------------------------

def fit_and_select_all_families(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    calibration_test_size: float = 0.25,
) -> FitAndSelectResult:
    # 1) Carve a calibration split out of outer-train ONLY (identical
    #    mechanics to the base runner's calibration_split_indices for split_key
    #    "S2": GroupShuffleSplit by subject_id). train_sub is what every family
    #    is tuned AND finally fit on; cal_sub is reserved for Platt calibration.
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
        # Selection tuple uses AUROC (primary) + AP (secondary) only, per
        # protocol_v2.yaml; the third slot is a constant so _rank_key's
        # (-brier) tie-break term never differentiates families here (Brier
        # is not well-posed pre-calibration -- see _fit_and_score_family_inner_cv).
        inner_scores[model_key] = (inner_auroc, inner_ap, 0.0)
        inner_metrics[model_key] = (inner_auroc, inner_ap, inner_brier)
        best_params_by_family[model_key] = best_params

        # Refit best hyperparams on train_sub (already done by GridSearchCV's
        # refit=True internally via best_estimator_, which is already fit on
        # X_tune/y_tune -- no further fitting needed here).
        if do_cal:
            calibrator = base_runner.PrefitCalibrator(best_pipe, method="sigmoid")
            calibrator.fit(X_cal, y_cal)
            final_model = calibrator
        else:
            final_model = best_pipe
        final_models[model_key] = final_model

    # --- SELECTION: inner-CV scores only ---
    selected_family = select_family(inner_scores)

    return FitAndSelectResult(
        selected_family=selected_family,
        final_models=final_models,
        best_params=best_params_by_family,
        inner_scores=inner_scores,
        inner_metrics=inner_metrics,
    )


def evaluate_frozen_model_with_predictions(
    final_model: Any, X_test: pd.DataFrame, y_test: np.ndarray
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Identical scoring to evaluate_frozen_model, but ALSO returns the raw
    (y_test, p_test) pair -- added for Phase 8's calibration-sensitivity
    analysis, which needs actual predicted probabilities per case (calibration
    slope/intercept cannot be computed from aggregate AUROC/AP/Brier alone).
    Purely additive: evaluate_frozen_model itself is untouched, so every
    already-validated call site (Phase 5) is unaffected by this addition."""
    p_te = base_runner.predict_proba_safe(final_model, X_test)
    auroc = float(roc_auc_score(y_test, p_te))
    ap = float(average_precision_score(y_test, p_te))
    brier = float(brier_score_loss(y_test, np.clip(p_te, 0, 1)))
    return auroc, ap, brier, np.asarray(y_test), np.asarray(p_te)


def evaluate_frozen_model(final_model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[float, float, float]:
    """Score an already-fitted model on X_test/y_test. No fitting happens
    here -- this is the "freeze everything, apply target mechanism only at
    outer-test evaluation" step Phase 5 (RQ3) needs (Protocol v2.1 Phase 5:
    "Do not retrain, retune, or recalibrate after seeing the target
    environment."). Identical scoring logic to run_outer_fold's inline
    version, factored out so Phase 5 can call it once per target mechanism
    against the SAME frozen model without duplicating the metric formulas."""
    p_te = base_runner.predict_proba_safe(final_model, X_test)
    auroc = float(roc_auc_score(y_test, p_te))
    ap = float(average_precision_score(y_test, p_te))
    brier = float(brier_score_loss(y_test, np.clip(p_te, 0, 1)))
    return auroc, ap, brier


# ---------------------------------------------------------------------------
# One full outer fold: tune+select (inner-CV only) all 5 families, THEN
# evaluate all 5 on outer-test (diagnostics), reporting the selected family's
# outer-test score as the condition's result.
# ---------------------------------------------------------------------------

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
        # Outer-test evaluation -- computed for EVERY family here so the
        # diagnostics (oracle/regret/displacement) below can be reported, but
        # NOTE: fit_and_select_all_families() above already committed to
        # `selected_family` using inner_scores alone, before any of this ran.
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

    # --- Diagnostics only (never fed back into selection) ---
    oracle_scores = {k: (v.outer_test_auroc, v.outer_test_ap, v.outer_test_brier) for k, v in families.items()}
    oracle_family = select_family(oracle_scores)  # same rank_key logic, applied to outer-test scores -- diagnostic use only

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


def run_outer_fold_with_predictions(
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
) -> Tuple[OuterFoldResult, np.ndarray, np.ndarray]:
    """Identical to run_outer_fold in every respect (same fit_and_select_all_
    families() call, same per-family outer-test scoring for all 5 families,
    same selection/oracle/regret/displacement computation) -- added for Phase
    8's calibration-sensitivity analysis, which additionally needs the raw
    (y_test, p_test) pair for the SELECTED family specifically (calibration
    slope/intercept requires actual predicted probabilities per case, not
    just the aggregate AUROC/AP/Brier that OuterFoldResult already carries).

    This is a deliberate near-duplicate of run_outer_fold rather than a
    parameter added to it: run_outer_fold is exercised by the full existing
    Phase 3/4/5/6 test suite and result set, and this project's standing
    practice (see selection_v2.py's own Phase 5 additions) is to add new
    functions alongside proven ones rather than modify a validated function's
    signature/behavior, so every prior phase's results remain reproducible
    from code that is provably unchanged. The only difference from
    run_outer_fold's body: the selected family's evaluation call goes through
    evaluate_frozen_model_with_predictions instead of evaluate_frozen_model,
    to additionally capture (y_test, p_test) for that one family. All other
    families still use evaluate_frozen_model (their predictions are never
    needed for calibration, which is defined on the DEPLOYED/selected model
    only), so this costs zero extra fits or scoring calls beyond run_outer_fold's."""
    fit_result = fit_and_select_all_families(
        X_train, y_train, groups_train, num_cols, cat_cols, seed, calibration_test_size
    )

    families: Dict[str, FamilyResult] = {}
    selected_family = fit_result.selected_family
    y_test_selected: Optional[np.ndarray] = None
    p_test_selected: Optional[np.ndarray] = None

    for model_key in MODELS:
        if model_key == selected_family:
            outer_auroc, outer_ap, outer_brier, y_sel, p_sel = evaluate_frozen_model_with_predictions(
                fit_result.final_models[model_key], X_test, y_test
            )
            y_test_selected, p_test_selected = y_sel, p_sel
        else:
            outer_auroc, outer_ap, outer_brier = evaluate_frozen_model(
                fit_result.final_models[model_key], X_test, y_test
            )
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

    oracle_scores = {k: (v.outer_test_auroc, v.outer_test_ap, v.outer_test_brier) for k, v in families.items()}
    oracle_family = select_family(oracle_scores)

    selected_auroc = families[selected_family].outer_test_auroc
    oracle_auroc = families[oracle_family].outer_test_auroc
    regret = oracle_auroc - selected_auroc

    result = OuterFoldResult(
        fold_id=fold_id,
        families=families,
        selected_family=selected_family,
        oracle_family=oracle_family,
        selected_outer_test_auroc=selected_auroc,
        oracle_outer_test_auroc=oracle_auroc,
        selection_regret=regret,
        displaced=(selected_family != oracle_family),
    )
    assert y_test_selected is not None and p_test_selected is not None  # selected_family is always in MODELS
    return result, y_test_selected, p_test_selected


# ---------------------------------------------------------------------------
# Full nested-CV run across all 5 StratifiedGroupKFold(subject_id) outer folds
# for ONE experimental condition (one missingness mechanism/rate/strength).
# ---------------------------------------------------------------------------

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
