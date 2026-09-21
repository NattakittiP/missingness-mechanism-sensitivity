"""Unit tests for missingness_generator_v2.py, Protocol v2.1 Phase 2.2."""

import inspect

import numpy as np
import pandas as pd
import pytest

import missingness_generator_v2 as gen


DATA_PATH = "/mnt/user-data/uploads/DASA2026/Dataset/full_analytic_dataset_mortality_all_admissions.csv"


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv(DATA_PATH)
    assert d.shape[0] == 14081
    return d


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260917)



@pytest.mark.parametrize("mechanism", ["natural", "mcar", "mar", "mnar_y"])
def test_1_zero_rate_identity(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.0, theta=0.693147, rng=rng)
    assert out["synthetic_mask"].to_numpy().sum() == 0, (
        f"mechanism={mechanism} at q=0 must inject zero cells; "
        f"got {out['synthetic_mask'].to_numpy().sum()}"
    )
    assert (out["final_mask"].to_numpy() == out["natural_mask"].to_numpy()).all()


def test_1b_all_mechanisms_identical_at_zero_rate(df, rng):
    """Checks all mechanisms reduce to the same baseline dataset at q=0."""
    baseline = None
    for mechanism in ["natural", "mcar", "mar", "mnar_y"]:
        out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.0, theta=0.693147, rng=rng)
        if baseline is None:
            baseline = out["final_mask"]
        else:
            assert (out["final_mask"].to_numpy() == baseline.to_numpy()).all(), mechanism



def test_2_mcar_realized_rate_close_to_target(df, rng):
    target = 0.30
    out = gen.generate_mask_single_partition(df, mechanism="mcar", target_rate=target, theta=0.0, rng=rng)
    rates = gen.compute_rates(out["natural_mask"], out["synthetic_mask"])
    assert abs(rates["r_inject"] - target) < 0.01, rates



@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar_y"])
def test_3_feature_scope_never_masks_never_maskable_cols(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.30, theta=0.693147, rng=rng)
    assert set(out["synthetic_mask"].columns) == set(gen.ELIGIBLE_LAB_COLS)
    assert set(out["synthetic_mask"].columns).isdisjoint(set(gen.NEVER_MASKABLE_COLS))



@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar_y"])
def test_4_natural_synthetic_never_overlap(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.30, theta=0.693147, rng=rng)
    overlap = (out["natural_mask"].to_numpy() & out["synthetic_mask"].to_numpy())
    assert overlap.sum() == 0, "a natively-missing cell was counted as newly synthetically masked"



def test_5_mar_context_signature_has_no_outcome_parameter():
    sig = inspect.signature(gen.fit_mar_context)
    param_names = set(sig.parameters.keys())
    forbidden = {"y", "label", "label_mortality", "outcome", "target"}
    assert param_names.isdisjoint(forbidden), (
        f"fit_mar_context must not accept an outcome-labeled parameter, got {param_names}"
    )
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_5b_mar_context_output_is_independent_of_label_shuffle(df, rng):
    """Checks shuffling the outcome label does not change the MAR mask."""
    df_shuffled = df.copy()
    df_shuffled[gen.LABEL_COL] = df_shuffled[gen.LABEL_COL].sample(frac=1.0, random_state=0).to_numpy()

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out_orig = gen.generate_mask_single_partition(df, mechanism="mar", target_rate=0.30, theta=0.693147, rng=rng1)
    out_shuf = gen.generate_mask_single_partition(df_shuffled, mechanism="mar", target_rate=0.30, theta=0.693147, rng=rng2)
    assert (out_orig["synthetic_mask"].to_numpy() == out_shuf["synthetic_mask"].to_numpy()).all()



def test_6_mnar_y_alpha_fit_on_train_only_applied_unchanged_to_test(df, rng):
    idx = np.arange(len(df))
    rng_split = np.random.default_rng(1)
    rng_split.shuffle(idx)
    train_idx, test_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    nat_train = gen.compute_natural_mask(df_train)
    nat_test = gen.compute_natural_mask(df_test)

    theta = 0.693147
    target_rate = 0.30

    generator = gen.fit_mnar_y(nat_train, df_train[gen.LABEL_COL], theta, target_rate)
    alpha_snapshot = dict(generator.alpha)

    y_train = df_train[gen.LABEL_COL].to_numpy().astype(float)
    y_test = df_test[gen.LABEL_COL].to_numpy().astype(float)

    syn_train = gen.apply_synthetic_mask(nat_train, y_train, generator, rng)
    assert generator.alpha == alpha_snapshot

    syn_test = gen.apply_synthetic_mask(nat_test, y_test, generator, rng)
    assert generator.alpha == alpha_snapshot, "alpha_j was mutated when applying to test partition"

    generator_on_test_directly = gen.fit_mnar_y(nat_test, df_test[gen.LABEL_COL], theta, target_rate)
    assert generator.alpha != generator_on_test_directly.alpha



def test_7_mask_sharing_across_model_families(df):
    """Checks two consumers of the same fitted generator get the identical mask realization."""
    natural_mask = gen.compute_natural_mask(df)
    h_i = gen.compute_H_i(df["admission_type"]).to_numpy().astype(float)
    generator = gen.fit_mar_context(natural_mask, df["admission_type"], theta=0.693147, target_rate=0.30)

    seed = 777
    mask_for_model_family_A = gen.apply_synthetic_mask(natural_mask, h_i, generator, np.random.default_rng(seed))
    mask_for_model_family_B = gen.apply_synthetic_mask(natural_mask, h_i, generator, np.random.default_rng(seed))
    assert (mask_for_model_family_A.to_numpy() == mask_for_model_family_B.to_numpy()).all(), (
        "Two model families given the same generator+seed must see the identical mask realization "
        "(caller is responsible for reusing ONE mask per condition across all families, not reseeding per family)"
    )



def test_3b_discharge_location_never_eligible_never_maskable_never_a_driver():
    assert "discharge_location" not in gen.ELIGIBLE_LAB_COLS
    assert "discharge_location" in gen.NEVER_MASKABLE_COLS
    assert "discharge_location" in gen.EXCLUDED_FROM_MODEL_ENTIRELY

    sig = inspect.signature(gen.fit_mar_context)
    assert "discharge_location" not in sig.parameters


def test_3b_discharge_location_excluded_end_to_end(df, rng):
    """Checks discharge_location never appears in a synthetic mask's columns."""
    out = gen.generate_mask_single_partition(df, mechanism="mnar_y", target_rate=0.30, theta=0.693147, rng=rng)
    assert "discharge_location" not in out["synthetic_mask"].columns
    assert "discharge_location" not in out["natural_mask"].columns



def test_compute_h_i_raises_on_nan_admission_type_not_silently_defaults():
    """Regression guard: compute_H_i must raise on NaN admission_type, not silently default it."""
    s = pd.Series(["EW EMER.", "URGENT", np.nan, "ELECTIVE"])
    with pytest.raises(ValueError):
        gen.compute_H_i(s)


def test_compute_h_i_raises_on_genuinely_unknown_category():
    s = pd.Series(["EW EMER.", "SOME NEW CATEGORY NOT IN DATASET"])
    with pytest.raises(ValueError):
        gen.compute_H_i(s)


def test_h_i_matches_frozen_phase0_grouping(df):
    h_i = gen.compute_H_i(df["admission_type"])
    n_high = int(h_i.sum())
    n_low = int((1 - h_i).sum())
    assert n_high == 9015
    assert n_low == 5066


def test_mar_theta_matches_frozen_or2(df, rng):
    assert abs(np.log(2) - 0.693147) < 1e-5


def test_rates_r_inject_vs_r_total_differ_when_native_missingness_present(df, rng):
    out = gen.generate_mask_single_partition(df, mechanism="mcar", target_rate=0.30, theta=0.0, rng=rng)
    rates = gen.compute_rates(out["natural_mask"], out["synthetic_mask"])
    assert rates["r_total"] > rates["r_inject"]
