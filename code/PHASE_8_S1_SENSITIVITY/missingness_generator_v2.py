"""Missingness generator implementing all five mechanisms (adds MNAR-X) per Protocol v2.1 Phase 2 + Phase 8."""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


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


H_I_HIGH_INTENSITY_ADMISSION_TYPES = frozenset({
    "OBSERVATION ADMIT", "ELECTIVE", "EW EMER.", "DIRECT EMER.",
})
H_I_LOW_INTENSITY_ADMISSION_TYPES = frozenset({
    "SURGICAL SAME DAY ADMISSION", "EU OBSERVATION", "URGENT",
    "DIRECT OBSERVATION", "AMBULATORY OBSERVATION",
})


def compute_H_i(admission_type: pd.Series) -> pd.Series:
    """Computes the binary admission-context driver from admission_type, raising on any unrecognized category rather than defaulting."""
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



def compute_natural_mask(df: pd.DataFrame, lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS) -> pd.DataFrame:
    """Returns the boolean mask of natively-unobserved cells."""
    lab_cols = list(lab_cols)
    return df[lab_cols].isna()


def compute_eligible_mask(natural_mask: pd.DataFrame) -> pd.DataFrame:
    """Returns the boolean mask of cells eligible for synthetic masking (natively observed cells only)."""
    return ~natural_mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))



def _bisect_alpha_for_rate(
    driver: np.ndarray,
    theta: float,
    target_rate: float,
    lo: float = -50.0,
    hi: float = 50.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Finds alpha by bisection such that the mean injection probability equals the target rate."""
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"_bisect_alpha_for_rate requires target_rate in (0, 1), got {target_rate}")
    if not np.all(np.isfinite(driver)):
        raise ValueError(
            "_bisect_alpha_for_rate: driver contains non-finite values (NaN/inf); "
            "refusing to silently calibrate against them. Audit the driver source "
            "(H_i / Y / native-missingness pattern) before proceeding."
        )

    def f(alpha: float) -> float:
        return float(np.mean(_sigmoid(alpha + theta * driver))) - target_rate

    flo, fhi = f(lo), f(hi)
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
    """Frozen, fold-safe generator state: one alpha per eligible column, fit on a single partition."""
    theta: float
    target_rate: float
    driver_kind: str
    alpha: Dict[str, float]


def fit_alphas(
    natural_mask: pd.DataFrame,
    driver: Optional[np.ndarray],
    theta: float,
    target_rate: float,
    driver_kind: str,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """Calibrates one alpha per lab column on the given (train) partition only."""
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
    """Draws the synthetic mask for one partition using an already-fitted generator; the fold-safe application step."""
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
            continue
        p = _sigmoid(a + generator.theta * driver[elig])
        draws = rng.random(n_elig) < p
        col_out = np.zeros(len(natural_mask), dtype=bool)
        col_out[np.where(elig)[0]] = draws
        out[col] = col_out
    return out


def compute_final_mask(natural_mask: pd.DataFrame, synthetic_mask: pd.DataFrame) -> pd.DataFrame:
    return natural_mask | synthetic_mask



def fit_natural_baseline() -> CalibratedGenerator:
    """No synthetic injection; q is fixed at 0."""
    return CalibratedGenerator(theta=0.0, target_rate=0.0, driver_kind="none", alpha={c: -np.inf for c in ELIGIBLE_LAB_COLS})


def fit_mcar(natural_mask: pd.DataFrame, target_rate: float, lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS) -> CalibratedGenerator:
    """MCAR: constant injection probability with no driver dependence."""
    return fit_alphas(natural_mask, driver=None, theta=0.0, target_rate=target_rate, driver_kind="none", lab_cols=lab_cols)


def fit_mar_context(
    natural_mask: pd.DataFrame,
    admission_type: pd.Series,
    theta: float,
    target_rate: float,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """MAR-context: driver is admission_type-derived only, never the outcome."""
    h_i = compute_H_i(admission_type).to_numpy().astype(float)
    return fit_alphas(natural_mask, driver=h_i, theta=theta, target_rate=target_rate, driver_kind="H_i", lab_cols=lab_cols)


def fit_mnar_y(
    natural_mask: pd.DataFrame,
    y: pd.Series,
    theta: float,
    target_rate: float,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
) -> CalibratedGenerator:
    """MNAR-Y: driver is the outcome itself; must be fit on outer-train only."""
    y_arr = y.to_numpy().astype(float)
    return fit_alphas(natural_mask, driver=y_arr, theta=theta, target_rate=target_rate, driver_kind="Y", lab_cols=lab_cols)




@dataclasses.dataclass
class MnarXGenerator:
    """Frozen, fold-safe MNAR-X generator state: alpha, mu, and sigma per eligible column, fit on one partition."""
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
    """Fits mu/sigma/alpha for the MNAR-X driver on outer-train only; never sees the outcome label."""
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
    """Draws the synthetic mask for one partition using an already-fitted MNAR-X generator."""
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



def generate_mask_single_partition(
    df: pd.DataFrame,
    mechanism: str,
    target_rate: float,
    theta: float,
    rng: np.random.Generator,
    lab_cols: Iterable[str] = ELIGIBLE_LAB_COLS,
    z_clip: float = 5.0,
) -> Dict[str, pd.DataFrame]:
    """Fit-and-apply on the same partition, across all five mechanisms; convenient for pilots/unit tests, not fold-safe for CV."""
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



def compute_rates(natural_mask: pd.DataFrame, synthetic_mask: pd.DataFrame) -> Dict[str, float]:
    """Computes the realized injection rate and total missingness rate."""
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
