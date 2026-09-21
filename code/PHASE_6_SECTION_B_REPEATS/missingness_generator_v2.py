"""
Missingness Generator v2 — Protocol v2.1 (frozen), Phase 2 + Phase 8 (MNAR-X).

Implements all five mechanisms now that MNAR-X (deferred at Phase 2 per
Protocol v2.1: "Do not implement MNAR-X first") is in scope (item 11 of the
frozen implementation order):
    1. Natural baseline   (no synthetic injection)
    2. MCAR                (theta = 0, driver-free)
    3. MAR-context          (driver = H_i, from admission_type; NEVER sees y)
    4. MNAR-Y               (driver = Y_i, the outcome)
    5. MNAR-X               (driver = |z_ij|, the cell's OWN value's distance
                              from its column's train-fold mean, in SD units;
                              never sees y -- see the MNAR-X section below)

Design invariants enforced throughout (Protocol v2.1 Phase 2.1):
    synthetic_mask & natural_mask == False   (never "re-mask" an already-missing cell)
    synthetic_mask outside lab columns == 0  (only the 30 eligible lab_* columns
                                               may ever be synthetically masked)

Rate semantics (frozen in protocol_v2.yaml):
    r_inject : newly-masked cells / originally-observed eligible cells (target, calibrated)
    r_total  : all missing cells (native + injected) / all eligible cells (realized)

Calibration: for each eligible lab column j independently, alpha_j is found by
bisection so that
    mean_i [ sigmoid(alpha_j + theta * driver_i) ]  ==  q     (over eligible rows i for column j)
Then each eligible cell (i, j) is synthetically masked with probability
    P(S_ij = 1) = sigmoid(alpha_j + theta * driver_i)
via an independent Bernoulli draw. This gives realized r_inject ~= q by
construction (Test 1, Test 2), while allowing the mechanism to be
outcome/driver-dependent within that fixed marginal budget.

Fold safety (Protocol v2.1 Phase 2.2, Test 6): alpha_j (and, for consistency,
the MAR driver mapping) must be fit on the outer-TRAIN partition only, then
applied unchanged (same alpha_j, same theta) to generate masks on outer-TEST
rows. This module never recalibrates on test data — callers achieve this by
calling `fit_alphas(...)` on the train subset and passing the fitted alphas
into `apply_synthetic_mask(...)` for both train and test subsets.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen feature roles (Protocol v2.1 §1.10, phase0_feature_audit.md, protocol_v2.yaml)
# ---------------------------------------------------------------------------

ELIGIBLE_LAB_COLS = [
    "lab_50802", "lab_50820", "lab_50821", "lab_50861", "lab_50863",
    "lab_50868", "lab_50878", "lab_50882", "lab_50885", "lab_50893",
    "lab_50902", "lab_50912", "lab_50931", "lab_50960", "lab_50970",
    "lab_50971", "lab_50983", "lab_51006", "lab_51221", "lab_51222",
    "lab_51248", "lab_51249", "lab_51250", "lab_51265", "lab_51274",
    "lab_51275", "lab_51277", "lab_51279", "lab_51301", "lab_52172",
]
assert len(ELIGIBLE_LAB_COLS) == 30

NEVER_MASKABLE_COLS = [
    "gender", "anchor_age", "race", "marital_status", "insurance",
    "admission_type", "admission_location", "discharge_location",
    "anchor_year", "anchor_year_group",
]

EXCLUDED_FROM_MODEL_ENTIRELY = ["hadm_id", "subject_id", "discharge_location"]

LABEL_COL = "label_mortality"
GROUP_COL = "subject_id"

# ---------------------------------------------------------------------------
# Frozen MAR driver H_i (Protocol v2.1 Phase 0 audit, X-side only, y never used)
# ---------------------------------------------------------------------------

H_I_HIGH_INTENSITY_ADMISSION_TYPES = frozenset({
    "OBSERVATION ADMIT", "ELECTIVE", "EW EMER.", "DIRECT EMER.",
})
H_I_LOW_INTENSITY_ADMISSION_TYPES = frozenset({
    "SURGICAL SAME DAY ADMISSION", "EU OBSERVATION", "URGENT",
    "DIRECT OBSERVATION", "AMBULATORY OBSERVATION",
})


def compute_H_i(admission_type: pd.Series) -> pd.Series:
    """Binary admission-context driver, X-side only. Never touches the outcome.

    Raises if an admission_type value is seen that isn't in either predeclared
    group, rather than silently defaulting it — Protocol v2.1 §1.7 requires the
    mapping be based on inspected categories, not assumed ones. This includes
    NaN/missing admission_type: Phase 0 confirmed admission_type is 0.0%
    natively missing in Dataset A today, but a silent NaN->H_i=0 default would
    be exactly the kind of unaudited assumption §1.7 forbids if that ever
    changes (e.g. a future data pull, or MNAR-X reusing this driver) -- so a
    NaN is treated as an unrecognized category and raises, not silently mapped.
    """
    observed = admission_type.unique()
    unknown = set(observed) - H_I_HIGH_INTENSITY_ADMISSION_TYPES - H_I_LOW_INTENSITY_ADMISSION_TYPES
    unknown = {u for u in unknown if not (isinstance(u, float) and np.isnan(u))} | (
        {"<NaN>"} if admission_type.isna().any() else set()
    )
    if unknown:
        raise ValueError(
            f"compute_H_i: admission_type contains categories outside the frozen "
            f"H_i mapping (including NaN if listed): {sorted(unknown, key=str)}. "
            f"The mapping must be re-audited before use -- do not silently default."
        )
    return admission_type.isin(H_I_HIGH_INTENSITY_ADMISSION_TYPES).astype(int)


# ---------------------------------------------------------------------------
# Mask-state primitives (Protocol v2.1 Phase 2.1)
# ---------------------------------------------------------------------------

def compute_natural_mask(df: pd.DataFrame, lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS) -> pd.DataFrame:
    """natural_mask[i, j] = True iff cell (i, j) was NOT observed in the raw data."""
    lab_cols = list(lab_cols)
    return df[lab_cols].isna()


def compute_eligible_mask(natural_mask: pd.DataFrame) -> pd.DataFrame:
    """eligible_mask[i, j] = True iff cell (i, j) was natively observed
    (only originally-observed cells are eligible for synthetic masking)."""
    return ~natural_mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Alpha calibration (per-feature bisection)
# ---------------------------------------------------------------------------

def _bisect_alpha_for_rate(
    driver: np.ndarray,
    theta: float,
    target_rate: float,
    lo: float = -50.0,
    hi: float = 50.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Find alpha such that mean(sigmoid(alpha + theta*driver)) == target_rate.

    mean(sigmoid(...)) is strictly increasing in alpha (sigmoid is monotonic),
    so a scalar bisection is well-posed and always converges for target_rate
    in (0, 1). Callers handle target_rate == 0 as a special case upstream
    (alpha := -inf, i.e. no injection) rather than calling this function.
    """
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"_bisect_alpha_for_rate requires target_rate in (0, 1), got {target_rate}")
    if not np.all(np.isfinite(driver)):
        # Found during the independent Phase 4 pre-flight review: without this
        # guard, a NaN anywhere in `driver` makes f(alpha) NaN for every alpha,
        # and since `NaN < 0` is False in Python, the bisection loop below
        # silently always takes the "raise lo" branch and converges to
        # alpha=hi=50.0 (~100% injection probability) instead of erroring --
        # exactly the kind of silent-default-on-bad-input this module's own
        # compute_H_i explicitly refuses to do elsewhere. Fail loud instead.
        raise ValueError(
            "_bisect_alpha_for_rate: driver contains non-finite values (NaN/inf); "
            "refusing to silently calibrate against them. Audit the driver source "
            "(H_i / Y / native-missingness pattern) before proceeding."
        )

    def f(alpha: float) -> float:
        return float(np.mean(_sigmoid(alpha + theta * driver))) - target_rate

    flo, fhi = f(lo), f(hi)
    # Widen the bracket if the target rate is extreme enough that +-50 logits
    # (already ~1e-22 / 1-1e-22 in probability) don't bracket it -- should not
    # happen for any q in [0, 1) but guarded defensively.
    while flo > 0 and lo > -1e6:
        lo *= 2
        flo = f(lo)
    while fhi < 0 and hi < 1e6:
        hi *= 2
        fhi = f(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


@dataclasses.dataclass
class CalibratedGenerator:
    """Frozen, fold-safe generator state: one alpha_j per eligible column,
    fit on a single (outer-train) partition, plus the shared theta.

    `driver_kind` records what was used to fit alpha_j, purely for
    bookkeeping / assertions (e.g. Test 5 below checks this is never "y"
    for a MAR-context generator instantiated the correct way).
    """
    theta: float
    target_rate: float
    driver_kind: str  # "none" (MCAR), "H_i" (MAR-context), "Y" (MNAR-Y)
    alpha: Dict[str, float]


def fit_alphas(
    natural_mask: pd.DataFrame,
    driver: Optional[np.ndarray],
    theta: float,
    target_rate: float,
    driver_kind: str,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """Calibrate one alpha_j per lab column on the given (train) partition only.

    driver: length-N array aligned to natural_mask's rows, or None for MCAR
            (theta is then ignored / forced to 0 for MCAR by convention of the
            caller passing theta=0.0, driver=zeros).
    target_rate == 0.0 is handled directly: alpha_j := -inf (no cell is ever
    injected), satisfying Test 1 (zero-rate identity) exactly, not just
    approximately.
    """
    lab_cols = list(lab_cols)
    eligible_mask = compute_eligible_mask(natural_mask)

    if driver is None:
        driver = np.zeros(len(natural_mask))
    elif len(driver) != len(natural_mask):
        raise ValueError(
            f"fit_alphas: driver length ({len(driver)}) != natural_mask row count "
            f"({len(natural_mask)}) -- driver and natural_mask must be positionally "
            f"aligned (same partition, same row order, e.g. both reset_index(drop=True))."
        )

    alpha: Dict[str, float] = {}
    for col in lab_cols:
        elig = eligible_mask[col].to_numpy()
        if target_rate == 0.0:
            alpha[col] = -np.inf
            continue
        col_driver = driver[elig]
        if len(col_driver) == 0:
            # No eligible (natively-observed) cells at all for this column in
            # this partition -- nothing to calibrate; alpha is irrelevant
            # since there are no cells to inject into.
            alpha[col] = -np.inf
            continue
        alpha[col] = _bisect_alpha_for_rate(col_driver, theta, target_rate)

    return CalibratedGenerator(theta=theta, target_rate=target_rate, driver_kind=driver_kind, alpha=alpha)


def apply_synthetic_mask(
    natural_mask: pd.DataFrame,
    driver: Optional[np.ndarray],
    generator: CalibratedGenerator,
    rng: np.random.Generator,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> pd.DataFrame:
    """Draw the synthetic mask for one partition (train OR test) using an
    ALREADY-FITTED generator (see fit_alphas). No calibration happens here --
    this is the fold-safe application step (Test 6).

    Returns a boolean DataFrame, same shape/index as natural_mask[lab_cols],
    True where a cell is newly synthetically masked. By construction this is
    always a subset of eligible_mask (synthetic_mask & natural_mask == False
    holds automatically, since sampling only occurs over eligible cells).
    """
    lab_cols = list(lab_cols)
    eligible_mask = compute_eligible_mask(natural_mask)

    if driver is None:
        driver = np.zeros(len(natural_mask))
    elif len(driver) != len(natural_mask):
        raise ValueError(
            f"apply_synthetic_mask: driver length ({len(driver)}) != natural_mask row "
            f"count ({len(natural_mask)}) -- driver and natural_mask must be "
            f"positionally aligned (same partition, same row order)."
        )

    out = pd.DataFrame(False, index=natural_mask.index, columns=lab_cols)
    for col in lab_cols:
        a = generator.alpha[col]
        elig = eligible_mask[col].to_numpy()
        n_elig = int(elig.sum())
        if n_elig == 0 or not np.isfinite(a):
            continue  # alpha=-inf (q=0 or no eligible cells) -> no injection
        p = _sigmoid(a + generator.theta * driver[elig])
        draws = rng.random(n_elig) < p
        col_out = np.zeros(len(natural_mask), dtype=bool)
        col_out[np.where(elig)[0]] = draws
        out[col] = col_out
    return out


def compute_final_mask(natural_mask: pd.DataFrame, synthetic_mask: pd.DataFrame) -> pd.DataFrame:
    return natural_mask | synthetic_mask


# ---------------------------------------------------------------------------
# High-level per-mechanism entry points
# ---------------------------------------------------------------------------
# Every mechanism below returns a CalibratedGenerator (fit on the partition
# passed in) plus a function-application step identical to apply_synthetic_mask.
# Splitting fit/apply like this is what makes fold-safety (Test 6) possible:
# callers fit on outer-train, then re-use the SAME CalibratedGenerator object to
# apply() on outer-test.

def fit_natural_baseline() -> CalibratedGenerator:
    """No synthetic injection at all. q is fixed at 0 regardless of caller input."""
    return CalibratedGenerator(theta=0.0, target_rate=0.0, driver_kind="none", alpha={c: -np.inf for c in ELIGIBLE_LAB_COLS})


def fit_mcar(natural_mask: pd.DataFrame, target_rate: float, lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS) -> CalibratedGenerator:
    """MCAR: theta = 0 (no driver dependence at all) -> P(S=1) = sigmoid(alpha_j) = q for every eligible cell."""
    return fit_alphas(natural_mask, driver=None, theta=0.0, target_rate=target_rate, driver_kind="none", lab_cols=lab_cols)


def fit_mar_context(
    natural_mask: pd.DataFrame,
    admission_type: pd.Series,
    theta: float,
    target_rate: float,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """MAR-context: driver = H_i computed from admission_type only.

    Signature deliberately takes `admission_type`, NOT `y` / the outcome --
    this is Test 5 (MAR must not access outcome labels), enforced at the
    function-signature level: there is no parameter through which a label
    could even be passed in.
    """
    h_i = compute_H_i(admission_type).to_numpy().astype(float)
    return fit_alphas(natural_mask, driver=h_i, theta=theta, target_rate=target_rate, driver_kind="H_i", lab_cols=lab_cols)


def fit_mnar_y(
    natural_mask: pd.DataFrame,
    y: pd.Series,
    theta: float,
    target_rate: float,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """MNAR-Y: driver = Y_i, the outcome itself. Must be fit on outer-TRAIN
    only and then applied unchanged to outer-TEST (Test 6) -- this function
    does the fitting half; apply_synthetic_mask does the (fold-safe) other half."""
    y_arr = y.to_numpy().astype(float)
    return fit_alphas(natural_mask, driver=y_arr, theta=theta, target_rate=target_rate, driver_kind="Y", lab_cols=lab_cols)


# ---------------------------------------------------------------------------
# MNAR-X (Phase 8 / implementation-order item 11) -- value-dependent mechanism
# ---------------------------------------------------------------------------
# Unlike MCAR/MAR/MNAR-Y, MNAR-X's driver is NOT a single length-N vector
# shared across all 30 lab columns -- it is column j's OWN cell value,
# expressed as a distance (in SD units) from that column's own train-fold
# mean:  z_ij = (x_ij - mu_j^train) / sigma_j^train,  |z_ij| is the driver.
# mu_j/sigma_j are therefore per-column, fold-safe statistics that must be
# estimated on the train partition and frozen before ever touching test rows
# (identical fold-safety contract to alpha_j elsewhere in this module -- see
# the module docstring's "Fold safety" paragraph). Because the driver differs
# by column, MNAR-X cannot reuse fit_alphas/apply_synthetic_mask's shared-
# driver-array signature; it gets its own fit/apply pair below, structurally
# unable to accept a `y` parameter (mirrors fit_mar_context's Test-5-style
# signature guarantee: there is no argument through which a label could leak
# in), and its own dataclass (MnarXGenerator) so it can never be passed by
# mistake into the shared-driver apply_synthetic_mask path, which would
# silently misinterpret its per-column mu/sigma state.
#
# Generator form (methodology-redesign-spec.md Sec.4.3, gamma resolved by
# data-driven analysis during Phase 8 planning -- see phase8-driver-summary.md
# Sec.1 for the full derivation):
#     P(S_ij = 1) = sigmoid(alpha_j - gamma * clip(|z_ij|, 0, z_clip))
# "Values far from a column's own typical range are LESS likely to be
# synthetically masked" (extreme/abnormal results are clinically re-checked
# and tracked more; the direction matches the spec's own framing). gamma is
# frozen per-condition at gamma = ln(OR) / 2, so that OR is exactly the
# multiplicative odds of masking at |z|=0 (a perfectly typical value) versus
# |z|=2 (a "moderately extreme", 2-SD-from-mean value) -- this makes MNAR-X's
# OR directly comparable, at that fixed 2-SD reference point, to MAR's and
# MNAR-Y's own main-condition OR=2 binary-driver effect sizes. z_clip=5.0 is
# a data-grounded safeguard (not in the original spec draft): 4 of the 30 real
# lab columns are heavily right-skewed enough (native max |z| in the 40-50
# range) that an UNCLIPPED |z| would drive that column's single most extreme
# native value's masking probability to a numerically-indistinguishable-from-
# zero floor (verified: <1e-6 for lab_50861/50878/50893/51301 at the resolved
# main gamma) -- i.e. that one real cell could structurally never be injected
# as synthetically missing in any repeat. Clipping the *driver* fed to the
# sigmoid (never the underlying data) at |z|<=5 keeps every column's worst-
# case masking probability at a non-degenerate 7-13% floor instead, while
# leaving every column's typical/moderate range (|z| up to its own p99,
# always well under 5 for every one of the 30 columns) completely untouched.


@dataclasses.dataclass
class MnarXGenerator:
    """Frozen, fold-safe MNAR-X generator state: one alpha_j, mu_j, sigma_j per
    eligible column, all fit on a single (outer-train) partition."""
    gamma: float
    target_rate: float
    z_clip: Optional[float]
    alpha: Dict[str, float]
    mu: Dict[str, float]
    sigma: Dict[str, float]


def _col_abs_zscore(x: np.ndarray, mu: float, sigma: float, z_clip: Optional[float]) -> np.ndarray:
    z = np.abs((x - mu) / sigma)
    if z_clip is not None:
        z = np.clip(z, 0.0, z_clip)
    return z


def fit_mnar_x(
    df_train: pd.DataFrame,
    natural_mask_train: pd.DataFrame,
    gamma: float,
    target_rate: float,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
    z_clip: Optional[float] = 5.0,
) -> MnarXGenerator:
    """Fit mu_j/sigma_j (on this partition's natively-observed values only,
    pre-masking, per methodology-redesign-spec.md Sec.4.3) and alpha_j (via the
    same bisection primitive used for MAR/MNAR-Y, with driver=|z| and
    theta=-gamma) for every eligible column. Must be called on outer-TRAIN
    only; apply the returned generator to BOTH train and test via
    apply_mnar_x_mask (fold-safe, Test-6-equivalent for MNAR-X).

    Signature takes df_train (feature values) and natural_mask_train, never a
    `y`/label argument -- there is no parameter through which an outcome
    label could leak into this driver, by construction.
    """
    if len(df_train) != len(natural_mask_train):
        raise ValueError(
            f"fit_mnar_x: df_train length ({len(df_train)}) != natural_mask_train row "
            f"count ({len(natural_mask_train)}) -- df_train and natural_mask_train must be "
            f"positionally aligned (same partition, same row order, e.g. both "
            f"reset_index(drop=True)). A same-length-but-misaligned df/mask pair (e.g. one "
            f"side missing a reset_index) would NOT be caught by this length check alone -- "
            f"callers remain responsible for alignment, exactly as fit_alphas/apply_synthetic_mask "
            f"already require elsewhere in this module."
        )
    lab_cols = list(lab_cols)
    eligible_mask = compute_eligible_mask(natural_mask_train)
    alpha: Dict[str, float] = {}
    mu: Dict[str, float] = {}
    sigma: Dict[str, float] = {}

    for col in lab_cols:
        elig = eligible_mask[col].to_numpy()
        x_obs = df_train[col].to_numpy(dtype=float)[elig]
        if len(x_obs) < 2:
            # Too few (or zero) natively-observed values in this column, in
            # this partition, to estimate a variance -- no injection rather
            # than a divide-by-zero/undefined z-score.
            mu[col], sigma[col], alpha[col] = float("nan"), float("nan"), -np.inf
            continue
        m = float(np.mean(x_obs))
        s = float(np.std(x_obs, ddof=1))
        mu[col], sigma[col] = m, s
        if target_rate == 0.0 or not np.isfinite(s) or s <= 0.0:
            alpha[col] = -np.inf
            continue
        z = _col_abs_zscore(x_obs, m, s, z_clip)
        alpha[col] = _bisect_alpha_for_rate(z, -gamma, target_rate)

    return MnarXGenerator(gamma=gamma, target_rate=target_rate, z_clip=z_clip, alpha=alpha, mu=mu, sigma=sigma)


def apply_mnar_x_mask(
    df: pd.DataFrame,
    natural_mask: pd.DataFrame,
    generator: MnarXGenerator,
    rng: np.random.Generator,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> pd.DataFrame:
    """Draw the synthetic mask for one partition (train OR test) using an
    ALREADY-FITTED MnarXGenerator (mu_j/sigma_j/alpha_j all frozen from
    outer-train). No re-estimation happens here -- this is the fold-safe
    application step, mirroring apply_synthetic_mask's contract exactly."""
    if len(df) != len(natural_mask):
        raise ValueError(
            f"apply_mnar_x_mask: df length ({len(df)}) != natural_mask row count "
            f"({len(natural_mask)}) -- df and natural_mask must be positionally aligned "
            f"(same partition, same row order)."
        )
    lab_cols = list(lab_cols)
    eligible_mask = compute_eligible_mask(natural_mask)
    out = pd.DataFrame(False, index=natural_mask.index, columns=lab_cols)

    for col in lab_cols:
        a = generator.alpha[col]
        elig = eligible_mask[col].to_numpy()
        n_elig = int(elig.sum())
        if n_elig == 0 or not np.isfinite(a):
            continue
        mu_j, sigma_j = generator.mu[col], generator.sigma[col]
        x = df[col].to_numpy(dtype=float)[elig]
        z = _col_abs_zscore(x, mu_j, sigma_j, generator.z_clip)
        p = _sigmoid(a - generator.gamma * z)
        draws = rng.random(n_elig) < p
        col_out = np.zeros(len(natural_mask), dtype=bool)
        col_out[np.where(elig)[0]] = draws
        out[col] = col_out
    return out


# ---------------------------------------------------------------------------
# Convenience: single-partition, single-call generation (used by simple tests
# and by the manipulation pilot in Phase 2.3; full nested-CV usage fits on
# train and applies separately to train/test, see module docstring).
# ---------------------------------------------------------------------------

def generate_mask_single_partition(
    df: pd.DataFrame,
    mechanism: str,
    target_rate: float,
    theta: float,
    rng: np.random.Generator,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
    z_clip: float = 5.0,
) -> Dict[str, pd.DataFrame]:
    """Fit-and-apply on the SAME partition. Convenient for single-fold pilots
    and unit tests; NOT fold-safe for a real nested-CV MNAR-Y/MNAR-X run (use
    fit_*() on train + apply_*() on train and test separately for that).

    mechanism in {"natural", "mcar", "mar", "mnar_y", "mnar_x"}. For
    mechanism="mnar_x", `theta` is interpreted as gamma (kept as one shared
    parameter name across mechanisms for this convenience wrapper only --
    fit_mnar_x itself always names it gamma).
    """
    natural_mask = compute_natural_mask(df, lab_cols)

    if mechanism == "natural":
        gen = fit_natural_baseline()
        driver = None
    elif mechanism == "mcar":
        gen = fit_mcar(natural_mask, target_rate, lab_cols)
        driver = None
    elif mechanism == "mar":
        gen = fit_mar_context(natural_mask, df["admission_type"], theta, target_rate, lab_cols)
        driver = compute_H_i(df["admission_type"]).to_numpy().astype(float)
    elif mechanism == "mnar_y":
        gen = fit_mnar_y(natural_mask, df[LABEL_COL], theta, target_rate, lab_cols)
        driver = df[LABEL_COL].to_numpy().astype(float)
    elif mechanism == "mnar_x":
        gen = fit_mnar_x(df, natural_mask, gamma=theta, target_rate=target_rate, lab_cols=lab_cols, z_clip=z_clip)
        synthetic_mask = apply_mnar_x_mask(df, natural_mask, gen, rng, lab_cols)
        final_mask = compute_final_mask(natural_mask, synthetic_mask)
        return {
            "natural_mask": natural_mask,
            "synthetic_mask": synthetic_mask,
            "final_mask": final_mask,
            "generator": gen,
        }
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")

    synthetic_mask = apply_synthetic_mask(natural_mask, driver, gen, rng, lab_cols)
    final_mask = compute_final_mask(natural_mask, synthetic_mask)

    return {
        "natural_mask": natural_mask,
        "synthetic_mask": synthetic_mask,
        "final_mask": final_mask,
        "generator": gen,
    }


# ---------------------------------------------------------------------------
# Rate bookkeeping (protocol_v2.yaml masking.rate_definitions)
# ---------------------------------------------------------------------------

def compute_rates(natural_mask: pd.DataFrame, synthetic_mask: pd.DataFrame) -> Dict[str, float]:
    """r_inject: newly-masked / originally-observed eligible cells.
    r_total:  all missing (native+injected) / all eligible cells."""
    eligible_mask = compute_eligible_mask(natural_mask)
    n_eligible = int(eligible_mask.to_numpy().sum())
    n_injected = int(synthetic_mask.to_numpy().sum())
    n_total_missing = int(natural_mask.to_numpy().sum()) + n_injected
    n_all_cells = natural_mask.size
    return {
        "r_inject": (n_injected / n_eligible) if n_eligible > 0 else float("nan"),
        "r_total": (n_total_missing / n_all_cells) if n_all_cells > 0 else float("nan"),
        "n_eligible_cells": n_eligible,
        "n_injected_cells": n_injected,
        "n_native_missing_cells": int(natural_mask.to_numpy().sum()),
    }
