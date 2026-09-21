"""Unit tests for metrics_v2.py, the Phase 7 metric layer, including an integration check against real Phase 3 results."""

import pickle

import numpy as np
import pytest
from scipy.stats import kendalltau

import metrics_v2 as met



def test_selection_regret_hand_example():
    assert met.selection_regret(oracle_score_on_target=0.95, selected_score_on_target=0.93) == pytest.approx(0.02)


def test_selection_regret_zero_when_selected_equals_oracle():
    assert met.selection_regret(0.90, 0.90) == 0.0



def test_displacement_rate_hand_example():
    data = {
        1: {0.0: "xgb", 0.3: "rf"},
        2: {0.0: "xgb", 0.3: "xgb"},
        3: {0.0: "rf",  0.3: "extratrees"},
        4: {0.0: "xgb", 0.3: "xgb"},
    }
    out = met.baseline_selection_displacement_rate(data, baseline_rate=0.0)
    assert 0.3 in out
    r = out[0.3]
    assert r.n_folds == 4
    assert r.n_displaced == 2
    assert r.displacement_rate == pytest.approx(0.5)
    assert set(r.displaced_fold_ids) == {1, 3}


def test_displacement_rate_zero_when_all_folds_stable():
    data = {i: {0.0: "xgb", 0.1: "xgb", 0.2: "xgb"} for i in range(5)}
    out = met.baseline_selection_displacement_rate(data, baseline_rate=0.0)
    assert out[0.1].displacement_rate == 0.0
    assert out[0.2].displacement_rate == 0.0


def test_displacement_rate_skips_folds_missing_a_rate():
    data = {
        1: {0.0: "xgb", 0.3: "rf"},
        2: {0.0: "xgb"},
    }
    out = met.baseline_selection_displacement_rate(data, baseline_rate=0.0)
    assert out[0.3].n_folds == 1
    assert out[0.3].n_displaced == 1
    assert out[0.3].displacement_rate == 1.0


def test_displacement_rate_raises_on_empty_input():
    with pytest.raises(ValueError):
        met.baseline_selection_displacement_rate({})



def test_entropy_zero_when_always_same_family():
    result = met.within_condition_selection_entropy(
        selected_families=["xgb"] * 10,
        all_families=["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"],
    )
    assert result.entropy_nats == pytest.approx(0.0)
    assert result.normalized_entropy == pytest.approx(0.0)
    assert result.modal_family == "xgb"
    assert result.modal_frequency == pytest.approx(1.0)


def test_entropy_one_when_uniform_across_all_five():
    families = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]
    selected = families * 2
    result = met.within_condition_selection_entropy(selected, families)
    assert result.normalized_entropy == pytest.approx(1.0, abs=1e-9)


def test_entropy_intermediate_case_hand_computed():
    families = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]
    selected = ["xgb", "xgb", "rf"]
    result = met.within_condition_selection_entropy(selected, families)
    p1, p2 = 2 / 3, 1 / 3
    expected_H = -(p1 * np.log(p1) + p2 * np.log(p2))
    expected_H_norm = expected_H / np.log(5)
    assert result.entropy_nats == pytest.approx(expected_H)
    assert result.normalized_entropy == pytest.approx(expected_H_norm)
    assert result.modal_family == "xgb"
    assert result.modal_frequency == pytest.approx(2 / 3)


def test_entropy_raises_on_family_not_in_roster():
    with pytest.raises(ValueError):
        met.within_condition_selection_entropy(["not_a_real_model"], ["lr_l2", "rf"])


def test_entropy_raises_on_empty_repeats():
    with pytest.raises(ValueError):
        met.within_condition_selection_entropy([], ["lr_l2", "rf"])



def test_kendall_tau_b_perfect_agreement_across_repeats():
    scores = [
        {"a": 0.9, "b": 0.8, "c": 0.7},
        {"a": 0.95, "b": 0.85, "c": 0.75},
    ]
    result = met.kendall_tau_b_rank_stability(scores)
    assert result.mean_tau_b == pytest.approx(1.0)
    assert result.n_pairs == 1


def test_kendall_tau_b_perfect_disagreement():
    scores = [
        {"a": 0.9, "b": 0.8, "c": 0.7},
        {"a": 0.7, "b": 0.8, "c": 0.9},
    ]
    result = met.kendall_tau_b_rank_stability(scores)
    assert result.mean_tau_b == pytest.approx(-1.0)


def test_kendall_tau_b_matches_scipy_directly_on_three_repeats():
    scores = [
        {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6},
        {"a": 0.6, "b": 0.9, "c": 0.7, "d": 0.8},
        {"a": 0.7, "b": 0.6, "c": 0.9, "d": 0.8},
    ]
    result = met.kendall_tau_b_rank_stability(scores)
    families = sorted(scores[0].keys())
    expected_pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            xi = [scores[i][f] for f in families]
            xj = [scores[j][f] for f in families]
            tau, _ = kendalltau(xi, xj, variant="b")
            expected_pairs.append(tau)
    assert result.mean_tau_b == pytest.approx(np.mean(expected_pairs))
    assert result.n_pairs == 3


def test_kendall_tau_b_raises_on_single_repeat():
    with pytest.raises(ValueError):
        met.kendall_tau_b_rank_stability([{"a": 0.9, "b": 0.8}])


def test_kendall_tau_b_raises_on_mismatched_family_sets():
    with pytest.raises(ValueError):
        met.kendall_tau_b_rank_stability([{"a": 0.9, "b": 0.8}, {"a": 0.9, "c": 0.8}])



def test_margin_hand_example():
    assert met.top1_minus_top2_margin({"xgb": 0.95, "rf": 0.93, "lr_l2": 0.85}) == pytest.approx(0.02)


def test_margin_nan_with_fewer_than_two_families():
    assert np.isnan(met.top1_minus_top2_margin({"xgb": 0.95}))



@pytest.fixture(scope="module")
def real_phase3_results():
    with open("phase3_validation_natural_results.pkl", "rb") as f:
        return pickle.load(f)


def test_integration_entropy_on_real_natural_condition_results(real_phase3_results):
    """Checks entropy is exactly 0 for the real Natural condition, where the same family was selected in all 5 folds."""
    selected = met.selected_family_by_fold(real_phase3_results)
    families = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]
    result = met.within_condition_selection_entropy(list(selected.values()), families)
    assert result.modal_family == "xgb"
    assert result.modal_frequency == pytest.approx(1.0)
    assert result.normalized_entropy == pytest.approx(0.0)


def test_integration_kendall_tau_b_on_real_natural_condition_results(real_phase3_results):
    """Computes rank-stability on real Phase 3 outer-test AUROC rankings, not synthetic data."""
    scores_by_repeat = [met.outer_test_auroc_by_family(r) for r in real_phase3_results]
    result = met.kendall_tau_b_rank_stability(scores_by_repeat)
    assert result.n_repeats == 5
    assert result.n_pairs == 10
    assert -1.0 <= result.mean_tau_b <= 1.0
    for (_, _, tau, _) in result.pairwise_tau_b:
        assert -1.0 <= tau <= 1.0


def test_integration_regret_matches_selection_v2_per_fold(real_phase3_results):
    """Checks selection_regret() reproduces exactly what selection_v2.py already computed per fold."""
    for r in real_phase3_results:
        recomputed = met.selection_regret(r.oracle_outer_test_auroc, r.selected_outer_test_auroc)
        assert recomputed == pytest.approx(r.selection_regret, abs=1e-12)


def test_integration_margin_on_real_fold_1_outer_test_scores(real_phase3_results):
    """Checks the evaluation margin on fold 1, the one real displaced fold, is small and positive."""
    fold1 = [r for r in real_phase3_results if r.fold_id == 1][0]
    scores = met.outer_test_auroc_by_family(fold1)
    margin = met.top1_minus_top2_margin(scores)
    assert margin > 0
    assert margin < 0.02
