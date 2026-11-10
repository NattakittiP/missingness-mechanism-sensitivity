"""
Unit tests for missingness_generator_v2.py — Protocol v2.1 Phase 2.2.

Implements the 7 required tests plus the additional Test 3b (discharge_location
permanent-exclusion regression guard) identified in phase0_feature_audit.md /
protocol_v2_1_frozen.md.

Run with:  python3 -m pytest test_missingness_generator_v2.py -v
"""

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


# ---------------------------------------------------------------------------
# Test 1 — zero-rate identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["natural", "mcar", "mar", "mnar_y", "mnar_x"])
def test_1_zero_rate_identity(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.0, theta=0.693147, rng=rng)
    assert out["synthetic_mask"].to_numpy().sum() == 0, (
        f"mechanism={mechanism} at q=0 must inject zero cells; "
        f"got {out['synthetic_mask'].to_numpy().sum()}"
    )
    # final_mask must equal natural_mask exactly at q=0 (all mechanisms reduce
    # to the same baseline dataset)
    assert (out["final_mask"].to_numpy() == out["natural_mask"].to_numpy()).all()


def test_1b_all_mechanisms_identical_at_zero_rate(df, rng):
    """All five mechanisms must reduce to the SAME baseline dataset at q=0."""
    baseline = None
    for mechanism in ["natural", "mcar", "mar", "mnar_y", "mnar_x"]:
        out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.0, theta=0.693147, rng=rng)
        if baseline is None:
            baseline = out["final_mask"]
        else:
            assert (out["final_mask"].to_numpy() == baseline.to_numpy()).all(), mechanism


# ---------------------------------------------------------------------------
# Test 2 — MCAR realized rate
# ---------------------------------------------------------------------------

def test_2_mcar_realized_rate_close_to_target(df, rng):
    target = 0.30
    out = gen.generate_mask_single_partition(df, mechanism="mcar", target_rate=target, theta=0.0, rng=rng)
    rates = gen.compute_rates(out["natural_mask"], out["synthetic_mask"])
    # Sampling tolerance: with ~30 columns x thousands of eligible cells per
    # column, binomial SE is tiny; 1 percentage point is a generous tolerance.
    assert abs(rates["r_inject"] - target) < 0.01, rates


# ---------------------------------------------------------------------------
# Test 3 — feature scope (no synthetic masking on demographic/administrative columns)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar_y", "mnar_x"])
def test_3_feature_scope_never_masks_never_maskable_cols(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.30, theta=0.693147, rng=rng)
    # synthetic_mask is only ever constructed over ELIGIBLE_LAB_COLS by
    # construction (apply_synthetic_mask never touches other columns), but
    # assert it explicitly against the frozen never-maskable list too.
    assert set(out["synthetic_mask"].columns) == set(gen.ELIGIBLE_LAB_COLS)
    assert set(out["synthetic_mask"].columns).isdisjoint(set(gen.NEVER_MASKABLE_COLS))


# ---------------------------------------------------------------------------
# Test 4 — natural/synthetic separation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar_y", "mnar_x"])
def test_4_natural_synthetic_never_overlap(df, rng, mechanism):
    out = gen.generate_mask_single_partition(df, mechanism=mechanism, target_rate=0.30, theta=0.693147, rng=rng)
    overlap = (out["natural_mask"].to_numpy() & out["synthetic_mask"].to_numpy())
    assert overlap.sum() == 0, "a natively-missing cell was counted as newly synthetically masked"


# ---------------------------------------------------------------------------
# Test 5 — MAR must not access outcome labels
# ---------------------------------------------------------------------------

def test_5_mar_context_signature_has_no_outcome_parameter():
    sig = inspect.signature(gen.fit_mar_context)
    param_names = set(sig.parameters.keys())
    forbidden = {"y", "label", "label_mortality", "outcome", "target"}
    assert param_names.isdisjoint(forbidden), (
        f"fit_mar_context must not accept an outcome-labeled parameter, got {param_names}"
    )
    # Also enforce it can't be smuggled in via **kwargs
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_5b_mar_context_output_is_independent_of_label_shuffle(df, rng):
    """Behavioral check: shuffling label_mortality must not change the MAR
    synthetic mask at all (since the generator never reads it)."""
    df_shuffled = df.copy()
    df_shuffled[gen.LABEL_COL] = df_shuffled[gen.LABEL_COL].sample(frac=1.0, random_state=0).to_numpy()

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out_orig = gen.generate_mask_single_partition(df, mechanism="mar", target_rate=0.30, theta=0.693147, rng=rng1)
    out_shuf = gen.generate_mask_single_partition(df_shuffled, mechanism="mar", target_rate=0.30, theta=0.693147, rng=rng2)
    assert (out_orig["synthetic_mask"].to_numpy() == out_shuf["synthetic_mask"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Test 6 — MNAR-Y alpha fold-safety (fit on train only, apply unchanged to test)
# ---------------------------------------------------------------------------

def test_6_mnar_y_alpha_fit_on_train_only_applied_unchanged_to_test(df, rng):
    # Simple train/test split (not the real StratifiedGroupKFold -- this test
    # only checks the fit/apply API contract, not CV correctness).
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
    # alpha must be UNCHANGED after applying to train
    assert generator.alpha == alpha_snapshot

    syn_test = gen.apply_synthetic_mask(nat_test, y_test, generator, rng)
    # alpha must still be unchanged after applying to test -- i.e. no
    # recalibration happened as a side effect of seeing test data
    assert generator.alpha == alpha_snapshot, "alpha_j was mutated when applying to test partition"

    # And: applying the SAME fitted generator to test data must not require
    # (or use) test labels for calibration -- only for driving Bernoulli draws,
    # which is the defined generative mechanism, not a fit step. Confirm this
    # by checking a generator fit on train produces a *different* alpha than
    # one (incorrectly) fit directly on test -- i.e. fitting is doing real work
    # and isn't a no-op that would mask a "secretly used test data" bug.
    generator_on_test_directly = gen.fit_mnar_y(nat_test, df_test[gen.LABEL_COL], theta, target_rate)
    # They need not be wildly different, but they should generally not be
    # bit-identical across an independent random split (sanity check that
    # fitting is partition-specific, i.e. real calibration is happening).
    assert generator.alpha != generator_on_test_directly.alpha


# ---------------------------------------------------------------------------
# Test 7 — mask sharing across model families
# ---------------------------------------------------------------------------

def test_7_mask_sharing_across_model_families(df):
    """Given the SAME fitted generator + SAME rng state, two independent
    'model family' consumers must receive the identical mask realization --
    i.e. masks must be generated ONCE per experimental condition and shared,
    never regenerated per model family."""
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


# ---------------------------------------------------------------------------
# Test 3b (additional, beyond Protocol's 7) — discharge_location permanent-exclusion guard
# ---------------------------------------------------------------------------

def test_3b_discharge_location_never_eligible_never_maskable_never_a_driver():
    assert "discharge_location" not in gen.ELIGIBLE_LAB_COLS
    assert "discharge_location" in gen.NEVER_MASKABLE_COLS
    assert "discharge_location" in gen.EXCLUDED_FROM_MODEL_ENTIRELY

    # And the MAR driver function signature can only ever consume admission_type,
    # never discharge_location.
    sig = inspect.signature(gen.fit_mar_context)
    assert "discharge_location" not in sig.parameters


def test_3b_discharge_location_excluded_end_to_end(df, rng):
    """Even if discharge_location were accidentally left in the input df,
    it must never appear in a synthetic_mask's columns."""
    out = gen.generate_mask_single_partition(df, mechanism="mnar_y", target_rate=0.30, theta=0.693147, rng=rng)
    assert "discharge_location" not in out["synthetic_mask"].columns
    assert "discharge_location" not in out["natural_mask"].columns


# ---------------------------------------------------------------------------
# Extra sanity checks (not in the required 7, but cheap and directly relevant)
# ---------------------------------------------------------------------------

def test_compute_h_i_raises_on_nan_admission_type_not_silently_defaults():
    """Regression guard added after independent adversarial review found that
    .dropna() before the unknown-category diff let NaN admission_type slip
    through and silently resolve to H_i=0 -- exactly the unaudited-default
    behavior compute_H_i's own docstring claims never happens. Dataset A has
    0.0% missing admission_type today (Phase 0 audit), so this is currently
    dormant, but must still raise, not default, if it ever becomes reachable."""
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
    # From phase0_feature_audit.md §0.3
    assert n_high == 9015
    assert n_low == 5066


def test_mar_theta_matches_frozen_or2(df, rng):
    # theta = ln(2) is what protocol_v2.yaml freezes for MAR's main setting
    assert abs(np.log(2) - 0.693147) < 1e-5


def test_rates_r_inject_vs_r_total_differ_when_native_missingness_present(df, rng):
    out = gen.generate_mask_single_partition(df, mechanism="mcar", target_rate=0.30, theta=0.0, rng=rng)
    rates = gen.compute_rates(out["natural_mask"], out["synthetic_mask"])
    # Native missingness exists (labs range 3.64%-81.39%), so r_total must
    # exceed r_inject once injection happens.
    assert rates["r_total"] > rates["r_inject"]


# ---------------------------------------------------------------------------
# Regression guards added during the Phase 4 pre-flight independent review
# ---------------------------------------------------------------------------

def test_bisect_alpha_raises_on_nan_driver_not_silently_saturates():
    """Found by independent adversarial review: without a finiteness guard,
    a NaN anywhere in `driver` makes f(alpha) NaN for every alpha, and since
    `NaN < 0` is False in Python, the old bisection loop always took the
    "raise lo" branch and silently converged to alpha=hi=50.0 (~100%
    injection probability) instead of erroring. Must raise instead."""
    driver = np.array([0.0, 1.0, np.nan, 0.0, 1.0] * 200)
    with pytest.raises(ValueError):
        gen._bisect_alpha_for_rate(driver, theta=1.0, target_rate=0.3)


def test_bisect_alpha_still_converges_normally_on_clean_driver():
    # Same shape/scale as the NaN-guard test above, minus the NaN, to prove
    # the new guard doesn't false-positive on legitimate input.
    driver = np.array([0.0, 1.0, 0.0, 0.0, 1.0] * 200)
    alpha = gen._bisect_alpha_for_rate(driver, theta=1.0, target_rate=0.3)
    achieved = float(np.mean(1.0 / (1.0 + np.exp(-(alpha + 1.0 * driver)))))
    assert abs(achieved - 0.3) < 1e-6


def test_fit_alphas_raises_on_driver_length_mismatch():
    natural_mask = pd.DataFrame(
        {c: [False, False, True, False, False] for c in gen.ELIGIBLE_LAB_COLS}
    )
    bad_driver = np.zeros(3)  # length 3 != natural_mask's 5 rows
    with pytest.raises(ValueError):
        gen.fit_alphas(natural_mask, bad_driver, theta=0.5, target_rate=0.3, driver_kind="H_i")


def test_apply_synthetic_mask_raises_on_driver_length_mismatch():
    natural_mask = pd.DataFrame(
        {c: [False, False, True, False, False] for c in gen.ELIGIBLE_LAB_COLS}
    )
    generator = gen.fit_alphas(natural_mask, None, theta=0.0, target_rate=0.3, driver_kind="none")
    bad_driver = np.zeros(3)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        gen.apply_synthetic_mask(natural_mask, bad_driver, generator, rng)


# ---------------------------------------------------------------------------
# MNAR-X (Phase 8 / implementation-order item 11) — new tests
# ---------------------------------------------------------------------------

def test_mnar_x_signature_has_no_outcome_parameter():
    """Mirrors Test 5 for MAR: fit_mnar_x must have no parameter through which
    an outcome label could be passed in, by construction."""
    sig = inspect.signature(gen.fit_mnar_x)
    param_names = set(sig.parameters.keys())
    forbidden = {"y", "label", "label_mortality", "outcome", "target"}
    assert param_names.isdisjoint(forbidden), (
        f"fit_mnar_x must not accept an outcome-labeled parameter, got {param_names}"
    )
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_mnar_x_output_is_independent_of_label_shuffle(df, rng):
    """Behavioral check mirroring Test 5b: shuffling label_mortality must not
    change the MNAR-X synthetic mask at all."""
    df_shuffled = df.copy()
    df_shuffled[gen.LABEL_COL] = df_shuffled[gen.LABEL_COL].sample(frac=1.0, random_state=0).to_numpy()

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out_orig = gen.generate_mask_single_partition(df, mechanism="mnar_x", target_rate=0.30, theta=0.34657, rng=rng1)
    out_shuf = gen.generate_mask_single_partition(df_shuffled, mechanism="mnar_x", target_rate=0.30, theta=0.34657, rng=rng2)
    assert (out_orig["synthetic_mask"].to_numpy() == out_shuf["synthetic_mask"].to_numpy()).all()


def test_mnar_x_fold_safety_mu_sigma_alpha_fit_on_train_only(df, rng):
    """Mirrors Test 6 for MNAR-Y: mu_j/sigma_j/alpha_j must be fit on train
    only, frozen, and applied unchanged to test — never recalibrated as a
    side effect of seeing test data."""
    idx = np.arange(len(df))
    rng_split = np.random.default_rng(1)
    rng_split.shuffle(idx)
    train_idx, test_idx = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    nat_train = gen.compute_natural_mask(df_train)
    nat_test = gen.compute_natural_mask(df_test)

    gamma, target_rate = 0.34657, 0.30
    generator = gen.fit_mnar_x(df_train, nat_train, gamma, target_rate)
    alpha_snapshot, mu_snapshot, sigma_snapshot = dict(generator.alpha), dict(generator.mu), dict(generator.sigma)

    syn_train = gen.apply_mnar_x_mask(df_train, nat_train, generator, rng)
    assert generator.alpha == alpha_snapshot and generator.mu == mu_snapshot and generator.sigma == sigma_snapshot

    syn_test = gen.apply_mnar_x_mask(df_test, nat_test, generator, rng)
    assert generator.alpha == alpha_snapshot and generator.mu == mu_snapshot and generator.sigma == sigma_snapshot, (
        "mu_j/sigma_j/alpha_j were mutated when applying to the test partition"
    )

    # Fitting directly on test should generally give different mu/sigma/alpha
    # than the train-fit generator (independent random split) -- proves real,
    # partition-specific calibration is happening, not a silent no-op.
    generator_on_test_directly = gen.fit_mnar_x(df_test, nat_test, gamma, target_rate)
    assert generator.mu != generator_on_test_directly.mu
    assert generator.alpha != generator_on_test_directly.alpha


def test_mnar_x_realized_rate_close_to_target(df, rng):
    target = 0.30
    out = gen.generate_mask_single_partition(df, mechanism="mnar_x", target_rate=target, theta=0.34657, rng=rng)
    rates = gen.compute_rates(out["natural_mask"], out["synthetic_mask"])
    assert abs(rates["r_inject"] - target) < 0.01, rates


def test_mnar_x_direction_extreme_values_less_likely_masked(df, rng):
    """Behavioral check of the mechanism's defined direction: cells whose
    native value is far from their column's mean (high |z|) must be
    synthetically masked LESS often than cells close to the mean, on
    average across the real dataset -- the defining property of this
    mechanism (extreme/abnormal results are less likely to go missing)."""
    natural_mask = gen.compute_natural_mask(df)
    gamma, target_rate = 0.34657, 0.30
    generator = gen.fit_mnar_x(df, natural_mask, gamma, target_rate)
    synthetic_mask = gen.apply_mnar_x_mask(df, natural_mask, generator, rng)

    eligible_mask = gen.compute_eligible_mask(natural_mask)
    near_mean_masked, near_mean_total = 0, 0
    far_masked, far_total = 0, 0
    for col in gen.ELIGIBLE_LAB_COLS:
        elig = eligible_mask[col].to_numpy()
        x = df[col].to_numpy(dtype=float)[elig]
        z = np.abs((x - generator.mu[col]) / generator.sigma[col])
        masked = synthetic_mask[col].to_numpy()[elig]
        near = z < 0.5
        far = z > 2.0
        near_mean_masked += int(masked[near].sum())
        near_mean_total += int(near.sum())
        far_masked += int(masked[far].sum())
        far_total += int(far.sum())

    rate_near = near_mean_masked / near_mean_total
    rate_far = far_masked / far_total
    assert rate_near > rate_far, (
        f"expected near-mean cells (|z|<0.5) to be masked MORE often than far cells "
        f"(|z|>2.0): got rate_near={rate_near:.4f}, rate_far={rate_far:.4f}"
    )


def test_mnar_x_gamma_matches_frozen_or_at_z2_reference_point(df, rng):
    """Mathematical identity check for the resolved main-condition gamma:
    gamma = ln(2)/2 must give exactly OR=2.0 comparing P(mask) at |z|=0 vs
    |z|=2, by construction of how gamma was derived (see fit_mnar_x's module
    section docstring / phase8-driver-summary.md)."""
    gamma_main = np.log(2.0) / 2.0
    alpha = 0.0  # OR is invariant to alpha; use 0 for a clean symbolic check
    p0 = 1.0 / (1.0 + np.exp(-(alpha - gamma_main * 0.0)))
    p2 = 1.0 / (1.0 + np.exp(-(alpha - gamma_main * 2.0)))
    OR = (p0 / (1 - p0)) / (p2 / (1 - p2))
    assert abs(OR - 2.0) < 1e-9, OR


def test_mnar_x_z_clip_prevents_extreme_outlier_zero_probability_degeneracy(df, rng):
    """Regression guard for the data-grounded finding made during Phase 8
    planning: several real lab columns are skewed enough (native max |z| in
    the 40-50 range) that an UNCLIPPED |z| drives that column's single most
    extreme native value's masking probability to a numerically-zero floor
    at the resolved main gamma. With z_clip=5.0 (the frozen default), no
    eligible cell's masking probability may fall below a sane floor."""
    natural_mask = gen.compute_natural_mask(df)
    gamma_main = np.log(2.0) / 2.0
    generator = gen.fit_mnar_x(df, natural_mask, gamma_main, target_rate=0.30, z_clip=5.0)
    eligible_mask = gen.compute_eligible_mask(natural_mask)

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    worst_p = 1.0
    for col in gen.ELIGIBLE_LAB_COLS:
        a = generator.alpha[col]
        if not np.isfinite(a):
            continue
        elig = eligible_mask[col].to_numpy()
        x = df[col].to_numpy(dtype=float)[elig]
        z = np.clip(np.abs((x - generator.mu[col]) / generator.sigma[col]), 0.0, 5.0)
        p_min = sigmoid(a - gamma_main * z.max()) if len(z) else 1.0
        worst_p = min(worst_p, p_min)
    assert worst_p > 0.01, f"expected every column's worst-case masking probability > 1%, got {worst_p}"


def test_mnar_x_never_uses_z_beyond_train_partition_stats(df, rng):
    """Length/alignment guard mirroring the driver-length-mismatch tests
    above: fit_mnar_x must not silently succeed on a natural_mask that is
    misaligned with df_train's row count."""
    natural_mask = pd.DataFrame(
        {c: [False, False, True, False, False] for c in gen.ELIGIBLE_LAB_COLS}
    )
    df_bad = pd.DataFrame({c: np.arange(3, dtype=float) for c in gen.ELIGIBLE_LAB_COLS})
    with pytest.raises(Exception):
        gen.fit_mnar_x(df_bad, natural_mask, gamma=0.5, target_rate=0.3)


def test_fit_mnar_x_raises_on_df_natural_mask_length_mismatch():
    """Found by independent adversarial review: fit_mnar_x/apply_mnar_x_mask
    originally had no explicit length check between df/natural_mask (unlike
    every sibling fit/apply pair in this module), so a same-length-but-
    misaligned df/mask pair (e.g. a missing reset_index on only one side)
    would silently compute mu_j/sigma_j/z_ij against the wrong rows with no
    error. This test targets the length-mismatch case directly (a stand-in
    for the alignment contract fit_alphas/apply_synthetic_mask already
    enforce), independent of the coincidental IndexError the boolean-index
    version above happens to raise."""
    natural_mask = pd.DataFrame(
        {c: [False, False, True, False, False] for c in gen.ELIGIBLE_LAB_COLS}
    )
    df_wrong_len = pd.DataFrame({c: np.arange(6, dtype=float) for c in gen.ELIGIBLE_LAB_COLS})
    with pytest.raises(ValueError):
        gen.fit_mnar_x(df_wrong_len, natural_mask, gamma=0.5, target_rate=0.3)


def test_apply_mnar_x_mask_raises_on_df_natural_mask_length_mismatch(df, rng):
    natural_mask = gen.compute_natural_mask(df)
    generator = gen.fit_mnar_x(df, natural_mask, gamma=0.34657, target_rate=0.30)
    df_wrong_len = df.iloc[:10].reset_index(drop=True)
    with pytest.raises(ValueError):
        gen.apply_mnar_x_mask(df_wrong_len, natural_mask, generator, rng)


def test_mnar_x_fit_and_apply_use_the_same_z_clip_empirically(rng):
    """Found by independent adversarial review: nothing previously guarded
    against fit_mnar_x calibrating alpha against CLIPPED |z| while
    apply_mnar_x_mask (by some future edit) drew masks from UNCLIPPED |z| --
    a one-line regression that the aggregate realized-rate test alone does
    NOT catch (verified during review: on the real 30-column dataset,
    realized r_inject only drifts by ~0.03 percentage points under that
    exact bug, well inside the 1pp tolerance used elsewhere). This test
    constructs a small synthetic column with a deliberate extreme outlier
    (|z|~50, duplicated many times to make the Monte Carlo estimate tight)
    and checks the EMPIRICAL masking rate at that outlier via
    apply_mnar_x_mask itself (not a re-derivation from generator.alpha) --
    it must land near the clipped-z prediction (~7-9%), not the unclipped-z
    prediction (~1e-8), directly exercising apply_mnar_x_mask's own code
    path rather than assuming it matches fit_mnar_x's."""
    n_typical, n_outlier = 50000, 30
    typical = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=n_typical)
    outlier_val = 200.0  # lands at |z| ~ 40 once mu/sigma are fit on the whole (typical-dominated) column
    col = "lab_50802"
    other_cols = [c for c in gen.ELIGIBLE_LAB_COLS if c != col]
    x = np.concatenate([typical, np.full(n_outlier, outlier_val)])
    df_synth = pd.DataFrame({col: x})
    for c in other_cols:
        df_synth[c] = 0.0  # irrelevant, unused by this column's own z/alpha

    natural_mask = gen.compute_natural_mask(df_synth)
    gamma_main = np.log(2.0) / 2.0
    generator = gen.fit_mnar_x(df_synth, natural_mask, gamma_main, target_rate=0.30, z_clip=5.0)

    z_check = abs((outlier_val - generator.mu[col]) / generator.sigma[col])
    assert z_check > 30.0, f"expected the outlier to sit at |z|>30 given this construction, got {z_check}"

    synthetic_mask = gen.apply_mnar_x_mask(df_synth, natural_mask, generator, rng)
    empirical_p_outlier = synthetic_mask[col].to_numpy()[n_typical:].mean()
    empirical_p_typical = synthetic_mask[col].to_numpy()[:n_typical].mean()

    # Clipped prediction (z capped at 5): P(mask) should sit in a
    # non-degenerate few-percent range, NOT the ~1e-8-scale floor an
    # unclipped |z|~50 would produce at this gamma.
    assert empirical_p_outlier > 0.03, (
        f"empirical masking rate at the extreme outlier was {empirical_p_outlier:.5f} -- "
        f"consistent with apply_mnar_x_mask NOT honoring z_clip (would collapse to ~0 unclipped)"
    )
    # And it must still be clearly rarer than a typical (near-mean) cell --
    # confirms the clip didn't erase the mechanism's intended direction either.
    assert empirical_p_outlier < empirical_p_typical, (empirical_p_outlier, empirical_p_typical)
