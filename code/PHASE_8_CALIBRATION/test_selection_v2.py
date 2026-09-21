"""Unit tests for selection_v2.py's inner-CV-only family selection (Protocol v2.1 Phase 3)."""

import inspect

import numpy as np
import pandas as pd
import pytest

import selection_v2 as sel



def test_a_select_family_signature_has_no_outer_test_parameter():
    sig = inspect.signature(sel.select_family)
    param_names = set(sig.parameters.keys())
    forbidden = {"x_test", "y_test", "outer_test", "test_scores", "test_auroc"}
    assert param_names.isdisjoint(forbidden), (
        f"select_family must not accept any outer-test-labeled parameter, got {param_names}"
    )
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert len(sig.parameters) == 1


def test_a2_select_family_is_a_pure_function_of_its_argument():
    """Checks select_family is a pure function of its inner-CV-score argument."""
    scores = {"lr_l2": (0.70, 0.30, 0.10), "rf": (0.75, 0.35, 0.09), "xgb": (0.72, 0.32, 0.11)}
    r1 = sel.select_family(scores)
    r2 = sel.select_family(dict(scores))
    assert r1 == r2 == "rf"


def test_a3_select_family_matches_rank_key_tie_break_convention():
    scores = {
        "a": (0.80, 0.20, 0.05),
        "b": (0.80, 0.25, 0.05),
        "c": (0.80, 0.25, 0.03),
    }
    assert sel.select_family(scores) == "c"



@pytest.fixture(scope="module")
def small_df():
    """Small synthetic stand-in for Dataset A's shape, used to keep this test fast."""
    rng = np.random.default_rng(0)
    n_subjects = 400
    rows_per_subject = 2
    n = n_subjects * rows_per_subject
    subject_id = np.repeat(np.arange(n_subjects), rows_per_subject)
    lab_cols = {f"lab_{i}": rng.normal(size=n) for i in range(6)}
    admission_type = rng.choice(["EW EMER.", "URGENT"], size=n)
    y = (rng.random(n) < 0.15).astype(int)
    df = pd.DataFrame({
        "subject_id": subject_id,
        "hadm_id": np.arange(n),
        "label_mortality": y,
        "admission_type": admission_type,
        "gender": rng.choice(["M", "F"], size=n),
        **lab_cols,
    })
    return df


def test_b_selected_family_is_invariant_to_outer_test_perturbation(small_df):
    """Checks the selected family is unaffected by any change to the outer-test partition."""
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(1)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te_a, y_te_a = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    y_te_b = 1 - y_te_a
    X_te_b = X_te_a.copy()
    for c in num_cols:
        X_te_b[c] = rng.permutation(X_te_b[c].to_numpy())

    result_a = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te_a, y_te_a, num_cols, cat_cols, seed=42, fold_id=1)
    result_b = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te_b, y_te_b, num_cols, cat_cols, seed=42, fold_id=1)

    assert result_a.selected_family == result_b.selected_family, (
        "selected_family changed when ONLY the outer-test partition changed -- "
        "this means outer-test data is leaking into selection, reintroducing the bug"
    )
    for model_key in sel.MODELS:
        fam_a, fam_b = result_a.families[model_key], result_b.families[model_key]
        assert fam_a.inner_auroc_mean == fam_b.inner_auroc_mean
        assert fam_a.inner_ap_mean == fam_b.inner_ap_mean
        assert fam_a.best_params == fam_b.best_params



def test_c_regret_is_nonnegative_and_displacement_flag_is_consistent(small_df):
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(2)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    result = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=7, fold_id=1)

    assert result.selection_regret >= -1e-9
    assert result.displaced == (result.selected_family != result.oracle_family)
    if not result.displaced:
        assert abs(result.selection_regret) < 1e-9
    assert set(result.families.keys()) == set(sel.MODELS)


def test_c2_run_condition_produces_one_result_per_outer_fold(small_df):
    results = sel.run_condition(small_df, seed=11, n_outer=3)
    assert len(results) == 3
    fold_ids = [r.fold_id for r in results]
    assert fold_ids == [1, 2, 3]
    for r in results:
        assert r.selected_family in sel.MODELS
        assert r.oracle_family in sel.MODELS



def test_d_fit_and_select_plus_evaluate_matches_run_outer_fold_exactly(small_df):
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(3)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    seed = 99
    direct = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=seed, fold_id=1)

    fit_result = sel.fit_and_select_all_families(X_tr, y_tr, g_tr, num_cols, cat_cols, seed=seed)
    assert fit_result.selected_family == direct.selected_family
    for model_key in sel.MODELS:
        auroc, ap, brier = sel.evaluate_frozen_model(fit_result.final_models[model_key], X_te, y_te)
        assert auroc == pytest.approx(direct.families[model_key].outer_test_auroc, abs=1e-12)
        assert ap == pytest.approx(direct.families[model_key].outer_test_ap, abs=1e-12)
        assert brier == pytest.approx(direct.families[model_key].outer_test_brier, abs=1e-12)
        assert fit_result.inner_metrics[model_key][0] == pytest.approx(direct.families[model_key].inner_auroc_mean, abs=1e-12)
        assert fit_result.best_params[model_key] == direct.families[model_key].best_params


def test_d2_fit_and_select_all_families_signature_has_no_outer_test_parameter():
    """Checks fit_and_select_all_families has no parameter through which outer-test data could enter."""
    sig = inspect.signature(sel.fit_and_select_all_families)
    param_names = set(sig.parameters.keys())
    forbidden = {"x_test", "y_test", "outer_test", "test_scores", "test_auroc"}
    assert param_names.isdisjoint(forbidden)



def test_e_golden_file_regression_on_fixed_seed(small_df):
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(3)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    result = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=99, fold_id=1)

    assert result.selected_family == "extratrees"

    expected = {
        "lr_l2": dict(inner_auroc=0.49937576160407665, inner_ap=0.18988240234738593, outer_auroc=0.4330380406461699, outer_ap=0.14253499870026154, outer_brier=0.1358758302746312),
        "svm_linear_cal": dict(inner_auroc=0.49985939008371716, inner_ap=0.18978044596509183, outer_auroc=0.43251693590411666, outer_ap=0.14238020908030347, outer_brier=0.13573145030803005),
        "rf": dict(inner_auroc=0.5327661855702733, inner_ap=0.19226287423362395, outer_auroc=0.5225377800937988, outer_ap=0.16189466105821118, outer_brier=0.14319258101851853),
        "xgb": dict(inner_auroc=0.5124605548434062, inner_ap=0.2105350617714045, outer_auroc=0.556539864512767, outer_ap=0.23073004475207082, outer_brier=0.13593104481697083),
        "extratrees": dict(inner_auroc=0.5469840082925826, inner_ap=0.19039065868662508, outer_auroc=0.5626628452318916, outer_ap=0.20178563560291882, outer_brier=0.14058046296296295),
    }
    for model_key, exp in expected.items():
        fam = result.families[model_key]
        assert fam.inner_auroc_mean == pytest.approx(exp["inner_auroc"], abs=1e-12)
        assert fam.inner_ap_mean == pytest.approx(exp["inner_ap"], abs=1e-12)
        assert fam.outer_test_auroc == pytest.approx(exp["outer_auroc"], abs=1e-12)
        assert fam.outer_test_ap == pytest.approx(exp["outer_ap"], abs=1e-12)
        assert fam.outer_test_brier == pytest.approx(exp["outer_brier"], abs=1e-12)



def test_f_run_outer_fold_with_predictions_matches_run_outer_fold_exactly(small_df):
    """Checks run_outer_fold_with_predictions is a pure superset of run_outer_fold's behavior."""
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(3)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    seed = 99
    direct = sel.run_outer_fold(X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=seed, fold_id=1)
    with_pred, y_captured, p_captured = sel.run_outer_fold_with_predictions(
        X_tr, y_tr, g_tr, X_te, y_te, num_cols, cat_cols, seed=seed, fold_id=1
    )

    assert with_pred.selected_family == direct.selected_family
    for model_key in sel.MODELS:
        a, b = with_pred.families[model_key], direct.families[model_key]
        assert a.outer_test_auroc == pytest.approx(b.outer_test_auroc, abs=1e-12)
        assert a.outer_test_ap == pytest.approx(b.outer_test_ap, abs=1e-12)
        assert a.outer_test_brier == pytest.approx(b.outer_test_brier, abs=1e-12)
        assert a.inner_auroc_mean == pytest.approx(b.inner_auroc_mean, abs=1e-12)
    assert with_pred.oracle_family == direct.oracle_family
    assert with_pred.selection_regret == pytest.approx(direct.selection_regret, abs=1e-12)

    from sklearn.metrics import roc_auc_score, average_precision_score
    assert len(y_captured) == len(y_te) == len(p_captured)
    assert set(np.unique(y_captured)) <= {0, 1}
    recomputed_auroc = roc_auc_score(y_captured, p_captured)
    recomputed_ap = average_precision_score(y_captured, p_captured)
    assert recomputed_auroc == pytest.approx(with_pred.families[with_pred.selected_family].outer_test_auroc, abs=1e-9)
    assert recomputed_ap == pytest.approx(with_pred.families[with_pred.selected_family].outer_test_ap, abs=1e-9)


def test_f2_evaluate_frozen_model_with_predictions_matches_evaluate_frozen_model(small_df):
    """Checks the scalar score triple from the _with_predictions variant is bit-identical to the original."""
    y = small_df["label_mortality"].to_numpy()
    groups = small_df["subject_id"].to_numpy()
    X = small_df.drop(columns=["label_mortality", "hadm_id"])
    num_cols = [c for c in X.columns if c.startswith("lab_")]
    cat_cols = ["admission_type", "gender"]

    rng = np.random.default_rng(3)
    idx = np.arange(len(small_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]
    X_tr, y_tr, g_tr = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_te, y_te = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    fit_result = sel.fit_and_select_all_families(X_tr, y_tr, g_tr, num_cols, cat_cols, seed=99)
    model_key = fit_result.selected_family
    final_model = fit_result.final_models[model_key]

    auroc1, ap1, brier1 = sel.evaluate_frozen_model(final_model, X_te, y_te)
    auroc2, ap2, brier2, y2, p2 = sel.evaluate_frozen_model_with_predictions(final_model, X_te, y_te)

    assert auroc1 == pytest.approx(auroc2, abs=1e-15)
    assert ap1 == pytest.approx(ap2, abs=1e-15)
    assert brier1 == pytest.approx(brier2, abs=1e-15)
    assert len(y2) == len(y_te) == len(p2)
