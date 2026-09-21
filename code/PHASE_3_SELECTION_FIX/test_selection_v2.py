"""
Unit tests for selection_v2.py — Protocol v2.1 Phase 3
("outer test selects family? Never.").

These are regression guards against the EXACT bug confirmed in
`jcsse_audit_runner_tqdm_hardened.py`'s `compute_winners()`, which picked
the model family with the best MEAN OUTER-TEST AUROC.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import selection_v2 as sel


# ---------------------------------------------------------------------------
# Test A — select_family() cannot structurally see outer-test data
# ---------------------------------------------------------------------------

def test_a_select_family_signature_has_no_outer_test_parameter():
    sig = inspect.signature(sel.select_family)
    param_names = set(sig.parameters.keys())
    forbidden = {"x_test", "y_test", "outer_test", "test_scores", "test_auroc"}
    assert param_names.isdisjoint(forbidden), (
        f"select_family must not accept any outer-test-labeled parameter, got {param_names}"
    )
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    # Only a single positional parameter is accepted, and by construction (see
    # module docstring) it is populated from inner-CV scores only.
    assert len(sig.parameters) == 1


def test_a2_select_family_is_a_pure_function_of_its_argument():
    """Calling select_family with identical inner-CV scores must always
    return the identical family, regardless of any global/outer-test state --
    i.e. it cannot be reading anything from outside its argument."""
    scores = {"lr_l2": (0.70, 0.30, 0.10), "rf": (0.75, 0.35, 0.09), "xgb": (0.72, 0.32, 0.11)}
    r1 = sel.select_family(scores)
    r2 = sel.select_family(dict(scores))  # fresh dict object, same content
    assert r1 == r2 == "rf"  # highest AUROC wins


def test_a3_select_family_matches_rank_key_tie_break_convention():
    # AUROC ties -> AP breaks the tie -> Brier breaks remaining ties (lower is better)
    scores = {
        "a": (0.80, 0.20, 0.05),
        "b": (0.80, 0.25, 0.05),  # same AUROC as a, higher AP -> wins over a
        "c": (0.80, 0.25, 0.03),  # same AUROC and AP as b, lower Brier -> wins overall
    }
    assert sel.select_family(scores) == "c"


# ---------------------------------------------------------------------------
# Test B — behavioral: perturbing outer-test data must not change the
# selected family (only the diagnostic oracle/regret numbers may change)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """A small, fast, synthetic-but-realistic stand-in for Dataset A's shape,
    used only to keep this behavioral test fast; the real end-to-end
    validation against the actual Dataset A (all 5 outer folds, all 5 real
    model families) lives in run_phase3_selection_validation.py, not here."""
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
    """The core regression guard: shuffle/replace outer-TEST labels and
    features entirely -- since selection only reads inner-CV scores computed
    from outer-TRAIN, the selected family for a given outer-train partition
    must be identical no matter what the outer-test partition looks like."""
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

    # An adversarially different "test partition B": same rows, but labels
    # flipped and lab values scrambled -- a stand-in for "outer-test could be
    # anything" since selection must never depend on it.
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
    # Sanity: the inner-CV scores themselves (computed from outer-train only)
    # must also be bit-identical between the two runs, since outer-train was
    # untouched. Checks BOTH components select_family() actually reads
    # (auroc AND ap) -- an earlier version of this test only checked auroc,
    # which would have missed a hypothetical bug that leaked outer-test
    # influence specifically into the AP computation.
    for model_key in sel.MODELS:
        fam_a, fam_b = result_a.families[model_key], result_b.families[model_key]
        assert fam_a.inner_auroc_mean == fam_b.inner_auroc_mean
        assert fam_a.inner_ap_mean == fam_b.inner_ap_mean
        assert fam_a.best_params == fam_b.best_params


# ---------------------------------------------------------------------------
# Test C — diagnostics (oracle / regret / displacement) are well-formed
# ---------------------------------------------------------------------------

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

    # Oracle is BY DEFINITION the best outer-test performer, so regret >= 0.
    assert result.selection_regret >= -1e-9
    # displaced flag must agree with selected != oracle
    assert result.displaced == (result.selected_family != result.oracle_family)
    if not result.displaced:
        assert abs(result.selection_regret) < 1e-9
    # every one of the 5 families must have been evaluated (diagnostics need all 5)
    assert set(result.families.keys()) == set(sel.MODELS)


def test_c2_run_condition_produces_one_result_per_outer_fold(small_df):
    results = sel.run_condition(small_df, seed=11, n_outer=3)
    assert len(results) == 3
    fold_ids = [r.fold_id for r in results]
    assert fold_ids == [1, 2, 3]
    for r in results:
        assert r.selected_family in sel.MODELS
        assert r.oracle_family in sel.MODELS
