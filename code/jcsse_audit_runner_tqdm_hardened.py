# jcsse_audit_runner.py (REV2) — Locked to JCSSE-Idea + fixes requested (100%)
#
# Fixes included:
# 1) Compute winner flip % (RQ3) and export winner_flip_summary.csv + per-seed winners
# 2) P2 leaks ONLY scaling (no implicit global imputation). Uses nanmean/nanstd; keeps NaN; fold-imputer handles NaN.
# 3) P3 fixed feature mapping (no index mismatch): P3 is explicitly "global transform + selection"
#    - Fit GLOBAL preprocessor (impute+scale+onehot) on full data
#    - Compute MI on global transformed space, select top-k indices
#    - In CV, ALWAYS transform using the SAME global preprocessor, then select indices
#    => Consistent feature space, no fold encoder mismatch. This does leak global transform; logged & saved.
# 4) LeakageArtifacts saved to JSON per config (results/leakage_artifacts/*.json)
# 5) Synthetic control runs BOTH Synthetic-clean and Synthetic-MISS (MCAR 15%) for P0 vs P1 (as in Idea)
#
# Datasets:
#   A: full_analytic_dataset_mortality_all_admissions.csv (label_mortality, subject_id)
#   B: Synthetic_Dataset_1500_Patients_precise.csv (TG4h top quartile label; optional id col)
#
# Run:
#   python jcsse_audit_runner.py

import os
import json
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------
# Progress bars (tqdm)
# -----------------------
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    # Minimal fallback so the script still runs even if tqdm is not installed.
    def tqdm(iterable=None, total=None, desc=None, leave=True, position=0, dynamic_ncols=True, **kwargs):
        if iterable is None:
            return range(total or 0)
        return iterable

    tqdm.write = print  # type: ignore

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GridSearchCV,
    StratifiedShuffleSplit, GroupShuffleSplit
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.isotonic import IsotonicRegression

class PrefitCalibrator:
    """Calibration wrapper for an already-fitted estimator (works on sklearn>=1.4 where cv='prefit' is removed).

    Uses either a sigmoid (Platt scaling via logistic regression) or isotonic regression on a 1D score.
    The score is taken from decision_function if available, otherwise from predict_proba[:,1].
    """
    def __init__(self, base_estimator, method: str = "sigmoid", eps: float = 1e-7):
        self.base_estimator = base_estimator
        self.method = method
        self.eps = eps
        self._lr = None
        self._iso = None

    def _score(self, X):
        if hasattr(self.base_estimator, "decision_function"):
            s = self.base_estimator.decision_function(X)
            return np.asarray(s).ravel()
        if hasattr(self.base_estimator, "predict_proba"):
            p = self.base_estimator.predict_proba(X)[:, 1]
            # Use logit(p) as a more stable score for sigmoid calibration
            p = np.clip(np.asarray(p).ravel(), self.eps, 1.0 - self.eps)
            return np.log(p / (1.0 - p))
        # Fallback: raw predictions (not ideal, but keeps pipeline from crashing)
        return np.asarray(self.base_estimator.predict(X)).ravel()

    def fit(self, X, y):
        y = np.asarray(y).astype(int).ravel()
        s = self._score(X).reshape(-1, 1)
        if self.method == "sigmoid":
            self._lr = LogisticRegression(solver="lbfgs", max_iter=2000)
            self._lr.fit(s, y)
        elif self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(s.ravel(), y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def predict_proba(self, X):
        s = self._score(X).reshape(-1, 1)
        if self.method == "sigmoid":
            p = self._lr.predict_proba(s)[:, 1]
        else:
            p = self._iso.predict(s.ravel())
        p = np.clip(np.asarray(p).ravel(), 0.0, 1.0)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# -----------------------
# Config locked to IDEA
# -----------------------
DATASET_A_PATH = "full_analytic_dataset_mortality_all_admissions.csv"
DATASET_B_PATH = "Synthetic_Dataset_1500_Patients_precise.csv"

RESULTS_DIR = "results"
LEAK_DIR = os.path.join(RESULTS_DIR, "leakage_artifacts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LEAK_DIR, exist_ok=True)

LABEL_A = "label_mortality"
GROUP_A = "subject_id"
DROP_A = {"hadm_id", "discharge_location", LABEL_A, GROUP_A}

B_TG4H_COL = "TG4h"
B_ID_COL_CANDIDATES = ["ID", "Id", "id", "patient_id", "PatientID", "subject_id"]
B_MISS_FEATURES = ["Age", "BMI", "TG0h", "HDL", "LDL", "Hematocrit", "TotalProtein", "WBV"]
B_MISS_RATE = 0.15
B_MISS_SEED = 777  # fixed for reproducibility

# Missingness mode for synthetic controls (Phase 3)
# - 'targeted': only B_MISS_FEATURES
# - 'global_numeric': all numeric columns
MISSING_MODE = "targeted"  # "targeted" | "global_numeric"

PROTOCOLS = ["P0", "P1", "P2", "P3"]
SPLITS = ["S1", "S2"]

# 5 models locked (as before)
MODELS = ["lr_l2", "svm_linear_cal", "rf", "xgb", "extratrees"]

K_A = 25
K_B = 6

OUTER_FOLDS = 5
INNER_FOLDS = 3

# Repro seeds locked to 20
SEEDS_20 = list(range(1001, 1021))  # 20 seeds

# Phase seeds locked
SEED_PHASE1 = 2026
SEED_PHASE3 = 2040

# ranking tie-break (locked)
# rank by AUROC desc, then AP desc, then Brier asc
def rank_key(mean_auc: float, mean_ap: float, mean_brier: float) -> Tuple[float, float, float]:
    return (mean_auc, mean_ap, -mean_brier)

# -----------------------
# Utilities
# -----------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def safe_json_dump(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in B_ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names defensively (prevents KeyError due to hidden spaces/BOM).
    - Strips leading/trailing whitespace from column names.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

# -----------------------
# XGBoost (locked model)
# -----------------------
def build_xgb_model(random_state: int):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError(
            "xgboost is required for the 'xgb' model per JCSSE-Idea. "
            "Install: pip install xgboost"
        ) from e
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=random_state,
    )

# -----------------------
# Preprocessing builders
# -----------------------

# -----------------------
# Safe Imputers (no-NaN guarantee)
# -----------------------
class SafeMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer that never outputs NaN and never errors on pandas StringDtype.

    Notes:
    - Works on any array-like / DataFrame input.
    - Coerces each column to numeric (errors='coerce') so that accidental string columns will become NaN,
      then fills them with the fallback value instead of crashing.
    """
    def __init__(self, fill_value: float = 0.0):
        self.fill_value = float(fill_value)
        self.statistics_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        stats = []
        for j in range(X_df.shape[1]):
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            if np.any(~np.isnan(col)):
                med = float(np.nanmedian(col))
            else:
                med = float(self.fill_value)
            if not np.isfinite(med):
                med = float(self.fill_value)
            stats.append(med)
        self.statistics_ = np.asarray(stats, dtype=float)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SafeMedianImputer must be fitted before calling transform().")
        X_df = pd.DataFrame(X)
        out = np.empty((X_df.shape[0], X_df.shape[1]), dtype=float)
        for j in range(X_df.shape[1]):
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            fill = float(self.statistics_[j]) if j < len(self.statistics_) else float(self.fill_value)
            if not np.isfinite(fill):
                fill = float(self.fill_value)
            col = np.where(np.isnan(col), fill, col)
            col = np.where(np.isfinite(col), col, float(self.fill_value))
            out[:, j] = col
        return out

class SafeMostFrequentImputer(BaseEstimator, TransformerMixin):
    """Most-frequent imputer for categorical that never outputs NaN.
    If a column has all missing values in the fit data, it falls back to fill_value (default 'MISSING').
    """
    def __init__(self, fill_value: str = "MISSING"):
        self.fill_value = fill_value
        self.fill_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        fills = []
        for j in range(X_df.shape[1]):
            col = X_df.iloc[:, j]
            mode = col.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) > 0 else self.fill_value
            fills.append(fill)
        self.fill_ = fills
        return self

    def transform(self, X):
        if self.fill_ is None:
            raise RuntimeError("SafeMostFrequentImputer is not fitted.")
        X_df = pd.DataFrame(X).copy()
        for j in range(X_df.shape[1]):
            fill = self.fill_[j]
            X_df.iloc[:, j] = X_df.iloc[:, j].fillna(fill)
        return X_df.values

def _final_sanitize_numeric(X):
    """Last-resort guard: replace NaN/inf with 0.0."""
    X = np.asarray(X)
    X = np.where(np.isfinite(X), X, 0.0)
    X = np.where(np.isnan(X), 0.0, X)
    return X

def split_columns_A(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[LABEL_A], errors="ignore")
    for c in list(DROP_A):
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")
    cat_cols = [c for c in X.columns if ( (lambda _dt: (_dt in ["object","category","bool","string"] or _dt.startswith("string")))(str(X[c].dtype)) )]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str], *,
                       include_imputer: bool, include_scaler: bool) -> ColumnTransformer:
    num_steps = []
    cat_steps = []
    if include_imputer:
        num_steps.append(("imputer", SafeMedianImputer(fill_value=0.0)))
        cat_steps.append(("imputer", SafeMostFrequentImputer(fill_value="MISSING")))
    if include_scaler:
        num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        num_steps.append(("sanitize", FunctionTransformer(_final_sanitize_numeric, feature_names_out="one-to-one")))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

    num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"
    cat_pipe = Pipeline(steps=cat_steps) if cat_steps else "passthrough"

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )

def build_global_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Global preprocessor used ONLY for P3 (explicitly global transform leakage) to fix feature space.
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SafeMedianImputer(fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("sanitize", FunctionTransformer(_final_sanitize_numeric, feature_names_out="one-to-one")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SafeMostFrequentImputer(fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )

# -----------------------
# Leakage artifacts (saved)
# -----------------------
@dataclass
class LeakageArtifacts:
    dataset: str
    split: str
    protocol: str
    model: str
    seed: int
    phase: str
    notes: str

    # P1
    global_num_imputer_stats: Optional[Dict[str, float]] = None
    global_cat_imputer_fill: Optional[Dict[str, Any]] = None

    # P2
    global_scaler_mean: Optional[Dict[str, float]] = None
    global_scaler_scale: Optional[Dict[str, float]] = None

    # P3
    p3_k: Optional[int] = None
    p3_selected_idx: Optional[List[int]] = None
    p3_feature_names: Optional[List[str]] = None  # safe, interpretable mapping
    p3_declared_leakage: Optional[str] = None     # explicit: "global transform + selection"

def save_leakage_artifacts(art: LeakageArtifacts) -> None:
    fname = f"{art.phase}_{art.dataset}_{art.split}_{art.protocol}_{art.model}_seed{art.seed}.json"
    path = os.path.join(LEAK_DIR, fname)
    safe_json_dump(path, art.__dict__)

# -----------------------
# Leakage transforms (global) for P1/P2
# -----------------------
def apply_global_imputation(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """P1 global imputation (explicit leakage), hardened to never leave NaN/inf behind.
    - Numeric: median; if all-missing -> 0.0
    - Categorical: mode; if all-missing -> 'MISSING'
    - Drop columns that are all-missing globally (no information content)
    """
    X2 = X.copy()
    num_stats: Dict[str, float] = {}
    cat_fill: Dict[str, Any] = {}

    # Drop globally all-NaN numeric columns first (prevents nanmedian issues)
    drop_num = []
    for c in num_cols:
        if c not in X2.columns:
            continue
        col = pd.to_numeric(X2[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col.isna().all():
            drop_num.append(c)
    if drop_num:
        X2 = X2.drop(columns=drop_num, errors="ignore")
        num_cols = [c for c in num_cols if c not in drop_num]

    for c in tqdm(num_cols, desc="P1 global impute (num)", leave=False):
        col = pd.to_numeric(X2[c], errors="coerce").replace([np.inf, -np.inf], np.nan).values.astype(float)
        med = float(np.nanmedian(col)) if np.any(~np.isnan(col)) else 0.0
        if not np.isfinite(med):
            med = 0.0
        num_stats[c] = med
        X2[c] = pd.to_numeric(X2[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float).fillna(med)
        # final guard
        X2[c] = X2[c].replace([np.inf, -np.inf], med).fillna(med)

    for c in tqdm(cat_cols, desc="P1 global impute (cat)", leave=False):
        if c not in X2.columns:
            continue
        mode = X2[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) > 0 else "MISSING"
        cat_fill[c] = fill
        X2[c] = X2[c].fillna(fill).astype(object)

    # Drop any remaining all-NaN columns (defensive)
    all_nan_cols = [c for c in X2.columns if X2[c].isna().all()]
    if all_nan_cols:
        X2 = X2.drop(columns=all_nan_cols, errors="ignore")

    return X2, num_stats, cat_fill


def apply_global_scaling_only(X: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    P2 FIX: leak ONLY scaling (no implicit global imputation).
    - Compute nanmean/nanstd on full data per column.
    - Apply scaling only where values are non-NaN; keep NaN untouched.
    - Fold-level imputer (median) handles NaNs later.
    """
    X2 = X.copy()
    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for c in tqdm(num_cols, desc="P2 global scale-only (num)", leave=False):
        col = pd.to_numeric(X2[c], errors="coerce").values.astype(float)
        mu = float(np.nanmean(col))
        sd = float(np.nanstd(col))
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sd)) or sd == 0.0:
            sd = 1.0
        means[c] = mu
        scales[c] = sd
        # scale only observed values; keep NaN
        out = col.copy()
        mask = ~np.isnan(out)
        out[mask] = (out[mask] - mu) / sd
        X2[c] = out
    return X2, means, scales

# -----------------------
# P3: global transform + selection (fixed feature space)
# -----------------------
def get_feature_names_from_global_preprocessor(global_pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """
    Build feature names in the same order as global_pre.transform output.
    """
    # numeric names
    names = []
    names.extend([f"num__{c}" for c in num_cols])

    # categorical onehot names
    if len(cat_cols) > 0:
        ohe: OneHotEncoder = global_pre.named_transformers_["cat"].named_steps["onehot"]
        # sklearn provides get_feature_names_out
        cat_names = ohe.get_feature_names_out(cat_cols)
        names.extend([f"cat__{n}" for n in cat_names.tolist()])
    return names

def p3_fit_global_transform_and_select(
    X: pd.DataFrame, y: np.ndarray, num_cols: List[str], cat_cols: List[str], k: int, seed: int
) -> Tuple[ColumnTransformer, np.ndarray, List[str]]:
    """
    P3 FIX: explicitly global transform + selection.
    - Fit global preprocessor (impute+scale+onehot) on FULL dataset
    - Transform full dataset
    - MI on that fixed global space -> top-k indices
    - Return (global_preprocessor, idx, feature_names_for_mapping)
    """
    global_pre = build_global_preprocessor(num_cols, cat_cols)
    X_full = global_pre.fit_transform(X)

    mi = mutual_info_classif(X_full, y, discrete_features=False, random_state=seed)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(mi)[::-1][:k].astype(int)

    feat_names = get_feature_names_from_global_preprocessor(global_pre, num_cols, cat_cols)
    # safety: length match
    if len(feat_names) != X_full.shape[1]:
        # fallback: no names (but should not happen)
        feat_names = [f"f{i}" for i in range(X_full.shape[1])]

    return global_pre, idx, feat_names

# -----------------------
# Model factory + grids (nested tuning)
# -----------------------
def make_model_and_grid(model_key: str, seed: int):
    if model_key == "lr_l2":
        model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, n_jobs=1, random_state=seed)
        grid = {"clf__C": [0.1, 1.0, 10.0]}
        return model, grid, True
    if model_key == "svm_linear_cal":
        model = LinearSVC(C=1.0, random_state=seed)
        grid = {"clf__C": [0.1, 1.0, 10.0]}
        return model, grid, True
    if model_key == "rf":
        model = RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=1)
        grid = {"clf__max_depth": [None, 6, 12]}
        return model, grid, False
    if model_key == "xgb":
        model = build_xgb_model(seed)
        grid = {"clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.03, 0.05]}
        return model, grid, False
    if model_key == "extratrees":
        model = ExtraTreesClassifier(n_estimators=600, random_state=seed, n_jobs=1)
        grid = {"clf__max_depth": [None, 6, 12]}
        return model, grid, False
    raise ValueError(f"Unknown model key: {model_key}")

# -----------------------
# CV / calibration splitters
# -----------------------
def get_outer(split_key: str, seed: int):
    if split_key == "S1":
        return StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    if split_key == "S2":
        return GroupKFold(n_splits=OUTER_FOLDS)
    raise ValueError(split_key)

def get_inner(split_key: str, seed: int):
    if split_key == "S1":
        return StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 7)
    if split_key == "S2":
        return GroupKFold(n_splits=INNER_FOLDS)
    raise ValueError(split_key)

def calibration_split_indices(split_key: str, y_train: np.ndarray, groups_train: Optional[np.ndarray], seed: int):
    """
    Split outer-train into train_sub / cal_sub (inside outer-train only).
    """
    if split_key == "S1":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 13)
        tr_sub, cal_sub = next(sss.split(np.zeros_like(y_train), y_train))
        return tr_sub, cal_sub
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 13)
    tr_sub, cal_sub = next(gss.split(np.zeros_like(y_train), y_train, groups_train))
    return tr_sub, cal_sub

def predict_proba_safe(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return clf.predict(X).astype(float)

def fit_best_model_nested(base_pipe: Pipeline, grid: Dict[str, List[Any]],
                          split_key: str, X_train, y_train, groups_train, seed: int):
    inner = get_inner(split_key, seed)
    cv_iter = inner.split(X_train, y_train, groups_train) if split_key == "S2" else inner.split(X_train, y_train)
    gs = GridSearchCV(
        estimator=base_pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv_iter,
        refit=True,
        n_jobs=1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

# -----------------------
# Dataset loaders
# -----------------------
def load_dataset_A():
    df = normalize_columns(pd.read_csv(DATASET_A_PATH))
    if LABEL_A not in df.columns:
        raise ValueError(f"Dataset A missing label column: {LABEL_A}")
    if GROUP_A not in df.columns:
        raise ValueError(f"Dataset A missing group column: {GROUP_A}")

    y = df[LABEL_A].astype(int).values
    groups = df[GROUP_A].values

    X = df.drop(columns=[LABEL_A], errors="ignore").copy()
    if "hadm_id" in X.columns:
        X = X.drop(columns=["hadm_id"], errors="ignore")
    if GROUP_A in X.columns:
        X = X.drop(columns=[GROUP_A], errors="ignore")

    num_cols, cat_cols = split_columns_A(df)
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]
    # Drop globally all-NaN columns (prevents fold/global imputer corner cases)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols, errors="ignore")
        num_cols = [c for c in num_cols if c not in all_nan_cols]
        cat_cols = [c for c in cat_cols if c not in all_nan_cols]

    return X, y, groups, num_cols, cat_cols

def load_dataset_B_make_label():
    df = normalize_columns(pd.read_csv(DATASET_B_PATH))
    if B_TG4H_COL not in df.columns:
        raise ValueError(f"Dataset B missing required column: {B_TG4H_COL}")

    tg4h = pd.to_numeric(df[B_TG4H_COL], errors="coerce").values.astype(float)
    thr = float(np.nanpercentile(tg4h, 75))
    y = (tg4h >= thr).astype(int)

    id_col = detect_id_col(df)
    drop_cols = [B_TG4H_COL]
    if id_col is not None:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # แปลงเฉพาะคอลัมน์ที่ไม่ใช่ object/category/bool/string
    for c in X.columns:
        dt = str(X[c].dtype)
        if (dt in ["object", "category", "bool", "string"] or dt.startswith("string")):
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

# กันพังคอลัมน์ที่ NaN ทั้งแท่ง
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    return X, y, thr

def make_synthetic_miss(X: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(B_MISS_SEED)
    X2 = X.copy()

    if MISSING_MODE == "targeted":
        cols = B_MISS_FEATURES
    elif MISSING_MODE == "global_numeric":
        cols = X2.select_dtypes(include=[np.number]).columns.tolist()
    else:
        raise ValueError("Unknown MISSING_MODE")

    for c in tqdm(cols, desc=f"Synthetic MCAR {int(B_MISS_RATE*100)}%", leave=False):
        if c not in X2.columns:
            continue
        m = rng.random(len(X2)) < B_MISS_RATE
        X2.loc[m, c] = np.nan

    return X2

# -----------------------
# Core run: one config
# -----------------------
def run_config(
    phase: str,
    dataset_tag: str,
    X: pd.DataFrame,
    y: np.ndarray,
    split_key: str,
    protocol: str,
    model_key: str,
    seed: int,
    groups: Optional[np.ndarray],
    num_cols: List[str],
    cat_cols: List[str],
    config_tag: str,
    store_oof: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:

    outer = get_outer(split_key, seed)

    # protocol views / knobs
    X_view = X.copy()
    include_imputer = True
    include_scaler = True

    # P3 objects (global)
    p3_global_pre = None
    p3_idx = None
    p3_feat_names = None

    # leakage artifact record (always saved for P1/P2/P3; optional for P0)
    notes = ""
    leak = LeakageArtifacts(
        dataset=dataset_tag, split=split_key, protocol=protocol, model=model_key,
        seed=seed, phase=phase, notes=""
    )

    if protocol == "P0":
        leak.notes = "Proper: all preprocessing within folds; nested tuning; calibration split within outer-train."
    elif protocol == "P1":
        X_view, num_stats, cat_fill = apply_global_imputation(X_view, num_cols, cat_cols)
        include_imputer = False  # already imputed globally
        leak.global_num_imputer_stats = num_stats
        leak.global_cat_imputer_fill = cat_fill
        leak.notes = "P1: Global imputation leakage only (median/mode) applied BEFORE split; scaling remains fold-safe."
    elif protocol == "P2":
        X_view, means, scales = apply_global_scaling_only(X_view, num_cols)
        include_scaler = False  # already scaled globally (NaNs preserved)
        leak.global_scaler_mean = means
        leak.global_scaler_scale = scales
        leak.notes = "P2: Global scaling leakage ONLY using nanmean/nanstd; NaNs preserved; imputation remains fold-safe."
    elif protocol == "P3":
        k = K_A if dataset_tag.startswith("A") else K_B
        p3_global_pre, p3_idx, p3_feat_names = p3_fit_global_transform_and_select(
            X_view, y, num_cols, cat_cols, k=k, seed=seed
        )
        leak.p3_k = int(k)
        leak.p3_selected_idx = [int(i) for i in p3_idx.tolist()]
        # map idx -> feature names
        leak.p3_feature_names = [p3_feat_names[i] for i in p3_idx.tolist()]
        leak.p3_declared_leakage = "P3 = GLOBAL transform (impute+scale+onehot fit on full data) + pre-split selection on that global feature space."
        leak.notes = "P3: Explicit global transform + selection to fix onehot feature space; avoids index mismatch; logged as broader leakage than selection-only."
    else:
        raise ValueError(protocol)

    # Ensure column lists match X_view (prevents KeyError if any column was dropped/renamed)
    present_cols = set(X_view.columns)
    num_cols = [c for c in num_cols if c in present_cols]
    cat_cols = [c for c in cat_cols if c in present_cols]

    # Build fold-safe preprocessor (used for P0/P1/P2 only)
    pre = build_preprocessor(num_cols, cat_cols, include_imputer=include_imputer, include_scaler=include_scaler)

    oof_store = None
    if store_oof:
        oof_store = {
            "y_true": y.astype(int),
            "y_prob": np.full(len(y), np.nan, dtype=float),
            "fold_id": np.full(len(y), -1, dtype=int),
        }

    rows: List[Dict[str, Any]] = []

    # outer CV
    split_iter = outer.split(X_view, y, groups) if split_key == "S2" else outer.split(X_view, y)
    split_list = list(split_iter)
    for fold_id, (tr_idx, te_idx) in enumerate(
        tqdm(split_list, desc=f"OuterCV {dataset_tag}-{split_key}-{protocol}-{model_key}-seed{seed}", leave=False),
        start=1
    ):
        X_tr = X_view.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_te = X_view.iloc[te_idx].copy()
        y_te = y[te_idx]
        g_tr = groups[tr_idx] if groups is not None else None

        base_model, grid, do_cal = make_model_and_grid(model_key, seed)

        if protocol != "P3":
            # fold-safe pipeline (or modified knobs for P1/P2)
            base_pipe = Pipeline(steps=[("pre", pre), ("clf", base_model)])

            # 1) split BEFORE tuning (cal_sub must NOT be seen in hyperparameter search)
            tr_sub, cal_sub = calibration_split_indices(split_key, y_tr, g_tr, seed)

            X_tune = X_tr.iloc[tr_sub]
            y_tune = y_tr[tr_sub]
            g_tune = g_tr[tr_sub] if g_tr is not None else None

            # 2) nested tuning on train_sub ONLY
            best_pipe, best_params = fit_best_model_nested(
                base_pipe,
                grid,
                split_key,
                X_tune,
                y_tune,
                g_tune,
                seed
            )

            # 3) fit best model on train_sub
            best_pipe.fit(X_tune, y_tune)

            # 4) calibration on cal_sub
            if do_cal:
                calibrator = PrefitCalibrator(best_pipe, method="sigmoid")
                calibrator.fit(X_tr.iloc[cal_sub], y_tr[cal_sub])
                final_model = calibrator
            else:
                final_model = best_pipe

            p_te = predict_proba_safe(final_model, X_te)

        else:
            # P3: global transform + selection used for BOTH tuning and final fits (consistent feature space)
            # build matrices
            Xtr_full = p3_global_pre.transform(X_tr)
            Xte_full = p3_global_pre.transform(X_te)
            Xtr_sel = Xtr_full[:, p3_idx]
            Xte_sel = Xte_full[:, p3_idx]

            # inner CV for tuning (on selected features only)
            inner = get_inner(split_key, seed)
            cv_iter = inner.split(Xtr_sel, y_tr, g_tr) if split_key == "S2" else inner.split(Xtr_sel, y_tr)

            # estimator without pipeline
            # we'll emulate GridSearchCV by fitting clones via setting params
            # easiest: wrap in a trivial Pipeline with passthrough to use existing grid keys
            from sklearn.base import clone
            # Make a "fake" pipeline to keep grid structure stable
            fake = Pipeline(steps=[("clf", base_model)])
            gs = GridSearchCV(
                estimator=fake,
                param_grid={"clf__" + k.split("clf__", 1)[1]: v for k, v in grid.items()},
                scoring="roc_auc",
                cv=cv_iter,
                refit=True,
                n_jobs=1,
            )
            gs.fit(Xtr_sel, y_tr)
            best_fake = gs.best_estimator_
            best_params = gs.best_params_

            tr_sub, cal_sub = calibration_split_indices(split_key, y_tr, g_tr, seed)

            # fit best model on train_sub
            best_clf = clone(best_fake.named_steps["clf"])
            # set best params
            best_clf.set_params(**{k.replace("clf__", ""): v for k, v in best_params.items()})
            best_clf.fit(Xtr_sel[tr_sub], y_tr[tr_sub])

            if do_cal:
                cal = PrefitCalibrator(best_clf, method="sigmoid")
                cal.fit(Xtr_sel[cal_sub], y_tr[cal_sub])
                p_te = cal.predict_proba(Xte_sel)[:, 1]
            else:
                if hasattr(best_clf, "predict_proba"):
                    p_te = best_clf.predict_proba(Xte_sel)[:, 1]
                else:
                    s = best_clf.decision_function(Xte_sel)
                    p_te = 1.0 / (1.0 + np.exp(-s))

        auc = roc_auc_score(y_te, p_te)
        ap = average_precision_score(y_te, p_te)
        brier = brier_score_loss(y_te, np.clip(p_te, 0, 1))
        ece = compute_ece(y_te, p_te, n_bins=10)

        if oof_store is not None:
            oof_store["y_prob"][te_idx] = p_te
            oof_store["fold_id"][te_idx] = fold_id

        rows.append({
            "timestamp": now_ts(),
            "phase": phase,
            "dataset": dataset_tag,
            "split": split_key,
            "protocol": protocol,
            "model": model_key,
            "seed": seed,
            "config_tag": config_tag,
            "fold": fold_id,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "prevalence_test": float(np.mean(y_te)),
            "auroc": float(auc),
            "ap": float(ap),
            "brier": float(brier),
            "ece": float(ece),
            "best_params": json.dumps(best_params),
        })

    # Save leakage artifacts for P1/P2/P3 (and also P0 if you want audit trail)
    if protocol in ["P1", "P2", "P3"]:
        save_leakage_artifacts(leak)

    # sanity: if store_oof, ensure no NaN
    if oof_store is not None and np.any(np.isnan(oof_store["y_prob"])):
        raise RuntimeError(f"OOF contains NaN for {dataset_tag}-{split_key}-{protocol}-{model_key}-seed{seed}")

    return rows, oof_store

# -----------------------
# Aggregation + winners + flips
# -----------------------
def summarize_configs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["phase", "dataset", "split", "protocol", "model", "seed", "config_tag"]
    agg = metrics_df.groupby(gcols).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
    ).reset_index()
    for c in ["auroc_std", "ap_std", "brier_std", "ece_std"]:
        agg[c] = agg[c].fillna(0.0)
    return agg

def compute_winners(summary_df: pd.DataFrame) -> pd.DataFrame:
    winners = []
    group_cols = ["phase", "dataset", "split", "protocol", "seed", "config_tag"]
    n_groups = summary_df.groupby(group_cols).ngroups
    for keys, sub in tqdm(summary_df.groupby(group_cols), total=n_groups, desc="Compute winners", leave=False):
        best_row = None
        best_k = None
        for _, r in sub.iterrows():
            k = rank_key(r["auroc_mean"], r["ap_mean"], r["brier_mean"])
            if best_k is None or k > best_k:
                best_k = k
                best_row = r
        winners.append({
            "phase": keys[0],
            "dataset": keys[1],
            "split": keys[2],
            "protocol": keys[3],
            "seed": int(keys[4]),
            "config_tag": keys[5],
            "winner_model": best_row["model"],
            "winner_auroc": float(best_row["auroc_mean"]),
            "winner_ap": float(best_row["ap_mean"]),
            "winner_brier": float(best_row["brier_mean"]),
            "winner_ece": float(best_row["ece_mean"]),
        })
    return pd.DataFrame(winners)

def compute_winner_flip(summary_df: pd.DataFrame, baseline_winners: pd.DataFrame) -> pd.DataFrame:
    """
    RQ3: winner flip % across seeds for P0 only (Phase2_REPRO), relative to a baseline winner.
    Baseline: Phase1_MAIN, dataset A, protocol P0, seed SEED_PHASE1, per split.
    """
    # build baseline dict per split
    base = {}
    for split_key in ["S1", "S2"]:
        row = baseline_winners[
            (baseline_winners["phase"] == "PHASE1_MAIN") &
            (baseline_winners["dataset"] == "A") &
            (baseline_winners["split"] == split_key) &
            (baseline_winners["protocol"] == "P0") &
            (baseline_winners["seed"] == SEED_PHASE1)
        ]
        if len(row) == 0:
            continue
        base[split_key] = row.iloc[0]["winner_model"]

    # winners per seed in Phase2 for P0
    w2 = compute_winners(summary_df)
    w2 = w2[
        (w2["phase"] == "PHASE2_REPRO") &
        (w2["dataset"] == "A") &
        (w2["protocol"] == "P0")
    ].copy()

    # per-split flip
    out = []
    for split_key, sub in w2.groupby("split"):
        if split_key not in base:
            continue
        baseline = base[split_key]
        flips = (sub["winner_model"] != baseline).astype(int).values
        flip_pct = 100.0 * float(np.mean(flips)) if len(flips) else 0.0
        out.append({
            "dataset": "A",
            "protocol": "P0",
            "split": split_key,
            "baseline_seed": SEED_PHASE1,
            "baseline_winner": baseline,
            "n_seeds": int(len(flips)),
            "winner_flip_pct": float(flip_pct),
        })

    # also export per-seed listing for audit
    per_seed = w2[["split", "seed", "winner_model", "winner_auroc", "winner_ap", "winner_brier", "winner_ece"]].copy()
    per_seed = per_seed.sort_values(["split", "seed"]).reset_index(drop=True)
    per_seed.to_csv(os.path.join(RESULTS_DIR, "winner_by_seed_phase2.csv"), index=False)

    return pd.DataFrame(out)

# -----------------------
# Main runner: Phase 1–3 locked to Idea
# -----------------------
def main():
    config_log_path = os.path.join(RESULTS_DIR, "config_log.jsonl")
    metrics_path = os.path.join(RESULTS_DIR, "metrics_all.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary_by_config.csv")
    winners_path = os.path.join(RESULTS_DIR, "winner_by_config.csv")
    flip_path = os.path.join(RESULTS_DIR, "winner_flip_summary.csv")

    # reset logs
    if os.path.exists(config_log_path):
        os.remove(config_log_path)

    all_rows: List[Dict[str, Any]] = []

    # ---------- Load datasets ----------
    tqdm.write("[" + now_ts() + "] Loading Dataset A...")
    X_A, y_A, g_A, num_A, cat_A = load_dataset_A()
    tqdm.write("[" + now_ts() + "] Loading Dataset B + building label...")
    X_B, y_B, thr_B = load_dataset_B_make_label()
    tqdm.write("[" + now_ts() + "] Generating Synthetic MISS (MCAR)...")
    X_B_miss = make_synthetic_miss(X_B)

    # Dataset B: detect categorical vs numeric columns (so Sex stays categorical and is one-hot encoded)
    cat_B = [c for c in X_B.columns if ( (lambda _dt: (_dt in ["object","category","bool","string"] or _dt.startswith("string")))(str(X_B[c].dtype)) )]
    num_B = [c for c in X_B.columns if c not in cat_B]

    # ---------- Phase 1: Main Matrix (Dataset A) ----------
    # 40 configs: A × (S1,S2) × (P0–P3) × (5 models)
    phase = "PHASE1_MAIN"
    total_runs = 2 * 4 * len(MODELS)  # (S1,S2) x (P0-P3) x 5 models
    pbar = tqdm(total=total_runs, desc="PHASE1_MAIN (A matrix)", dynamic_ncols=True)
    for split_key in ["S1", "S2"]:
        for protocol in ["P0", "P1", "P2", "P3"]:
            for model_key in MODELS:
                cfg = {
                    "timestamp": now_ts(),
                    "phase": phase,
                    "dataset": "A",
                    "split": split_key,
                    "protocol": protocol,
                    "model": model_key,
                    "seed": SEED_PHASE1,
                }
                jsonl_append(config_log_path, cfg)

                store_oof = (protocol == "P0")  # store OOF for all P0 models (audit-friendly)
                rows, oof = run_config(
                    phase=phase,
                    dataset_tag="A",
                    X=X_A,
                    y=y_A,
                    split_key=split_key,
                    protocol=protocol,
                    model_key=model_key,
                    seed=SEED_PHASE1,
                    groups=g_A if split_key == "S2" else None,
                    num_cols=num_A,
                    cat_cols=cat_A,
                    config_tag=phase,
                    store_oof=store_oof,
                )
                all_rows.extend(rows)
                pbar.update(1)

                if store_oof and oof is not None:
                    npz_path = os.path.join(RESULTS_DIR, f"oof_P0_{split_key}_{model_key}.npz")
                    np.savez_compressed(npz_path, **oof)

    pbar.close()

    # ---------- Phase 3: Synthetic control (Dataset B) ----------
    # MUST include Synthetic-clean AND Synthetic-MISS, S1 only; Protocol P0 vs P1; Models 5
    phase = "PHASE3_SYN_CONTROL"

    # 3a) Synthetic clean
    total_runs = 2 * len(MODELS)
    pbar = tqdm(total=total_runs, desc="PHASE3_SYN_CONTROL (B_CLEAN)", dynamic_ncols=True)
    for protocol in ["P0", "P1"]:
        for model_key in MODELS:
            cfg = {
                "timestamp": now_ts(),
                "phase": phase,
                "dataset": "B_CLEAN",
                "split": "S1",
                "protocol": protocol,
                "model": model_key,
                "seed": SEED_PHASE3,
                "y_syn_def": f"TG4h>=global_p75 (thr={thr_B:.6f})",
                "missingness": "none",
            }
            jsonl_append(config_log_path, cfg)

            rows, _ = run_config(
                phase=phase,
                dataset_tag="B_CLEAN",
                X=X_B,
                y=y_B,
                split_key="S1",
                protocol=protocol,
                model_key=model_key,
                seed=SEED_PHASE3,
                groups=None,
                num_cols=num_B,
                cat_cols=cat_B,
                config_tag=phase,
                store_oof=False,
            )
            all_rows.extend(rows)
            pbar.update(1)

    pbar.close()

    # 3b) Synthetic MISS
    total_runs = 2 * len(MODELS)
    pbar = tqdm(total=total_runs, desc="PHASE3_SYN_CONTROL (B_SYN_MISS)", dynamic_ncols=True)
    for protocol in ["P0", "P1"]:
        for model_key in MODELS:
            cfg = {
                "timestamp": now_ts(),
                "phase": phase,
                "dataset": "B_SYN_MISS",
                "split": "S1",
                "protocol": protocol,
                "model": model_key,
                "seed": SEED_PHASE3,
                "y_syn_def": f"TG4h>=global_p75 (thr={thr_B:.6f})",
                "missingness": {"type": "MCAR", "rate": B_MISS_RATE, "seed": B_MISS_SEED, "features": B_MISS_FEATURES},
            }
            jsonl_append(config_log_path, cfg)

            rows, _ = run_config(
                phase=phase,
                dataset_tag="B_SYN_MISS",
                X=X_B_miss,
                y=y_B,
                split_key="S1",
                protocol=protocol,
                model_key=model_key,
                seed=SEED_PHASE3,
                groups=None,
                num_cols=num_B,
                cat_cols=cat_B,
                config_tag=phase,
                store_oof=False,
            )
            all_rows.extend(rows)
            pbar.update(1)

    pbar.close()

    # ---------- Phase 2: Reproducibility (Dataset A, P0 only, 20 seeds) ----------
    # A × (S1,S2) × P0 × (20 seeds) × (5 models) = 200 runs
    phase = "PHASE2_REPRO"
    total_runs = 2 * len(SEEDS_20) * len(MODELS)
    pbar = tqdm(total=total_runs, desc="PHASE2_REPRO (A, P0, 20 seeds)", dynamic_ncols=True)
    for split_key in ["S1", "S2"]:
        for seed in SEEDS_20:
            for model_key in MODELS:
                cfg = {
                    "timestamp": now_ts(),
                    "phase": phase,
                    "dataset": "A",
                    "split": split_key,
                    "protocol": "P0",
                    "model": model_key,
                    "seed": seed,
                }
                jsonl_append(config_log_path, cfg)

                rows, _ = run_config(
                    phase=phase,
                    dataset_tag="A",
                    X=X_A,
                    y=y_A,
                    split_key=split_key,
                    protocol="P0",
                    model_key=model_key,
                    seed=seed,
                    groups=g_A if split_key == "S2" else None,
                    num_cols=num_A,
                    cat_cols=cat_A,
                    config_tag=phase,
                    store_oof=False,
                )
                all_rows.extend(rows)
                pbar.update(1)

    pbar.close()

    # ---------- Save outputs ----------
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(metrics_path, index=False)

    summary_df = summarize_configs(metrics_df)
    summary_df.to_csv(summary_path, index=False)

    winners_df = compute_winners(summary_df)
    winners_df.to_csv(winners_path, index=False)

    # ---------- Canonical OOF files for P0 S1/S2 (as per Idea) ----------
    # Copy the Phase1 winner model OOF into canonical oof_P0_S1.npz / oof_P0_S2.npz
    for split_key in ["S1", "S2"]:
        row = winners_df[
            (winners_df["phase"] == "PHASE1_MAIN") &
            (winners_df["dataset"] == "A") &
            (winners_df["split"] == split_key) &
            (winners_df["protocol"] == "P0") &
            (winners_df["seed"] == SEED_PHASE1)
        ]
        if len(row) == 0:
            continue
        winner_model = row.iloc[0]["winner_model"]
        src = os.path.join(RESULTS_DIR, f"oof_P0_{split_key}_{winner_model}.npz")
        dst = os.path.join(RESULTS_DIR, f"oof_P0_{split_key}.npz")
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

    # ---------- RQ3 output: winner flip % ----------
    flip_df = compute_winner_flip(summary_df, winners_df)
    flip_df.to_csv(flip_path, index=False)

    print("DONE.")
    print(f"- metrics:     {metrics_path}")
    print(f"- summary:     {summary_path}")
    print(f"- winners:     {winners_path}")
    print(f"- flip:        {flip_path}")
    print(f"- oof canon:   {os.path.join(RESULTS_DIR, 'oof_P0_S1.npz')} and {os.path.join(RESULTS_DIR, 'oof_P0_S2.npz')}")
    print(f"- leakage dir: {LEAK_DIR}")
    print(f"- config log:  {config_log_path}")

if __name__ == "__main__":
    main()