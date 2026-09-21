"""
Phase 8 driver, part 1 of 3 — MNAR-X full experiment (RQ1-equivalent rate
sweep + RQ2-equivalent strength sweep), the primary piece of implementation-
order item 11 ("MNAR-X + calibration + remaining sensitivities").

Mirrors run_phase4_rq1_rq2.py's structure and conventions exactly (same
OUTER_SEED/MASK_SEED, same StratifiedGroupKFold(subject_id) outer split, same
checkpoint-per-condition design, same aggregation/output shape), applied to
the one mechanism MCAR/MAR/MNAR-Y's original Phase 4 run deliberately left
out (Protocol v2.1 Phase 2: "Do not implement MNAR-X first").

WHAT THIS RUNS (6 fresh conditions; full derivation of gamma/z_clip in
phase8-driver-summary.md):

  RQ1-equivalent (rate sweep, strength held at the resolved main gamma):
    mnar_x x rate in {0.10, 0.20, 0.30, 0.40}                -> 4 conditions
    (rate=0.0 is mechanism-independent by construction -- fit_mnar_x gives
     alpha_j=-inf for target_rate=0.0 regardless of gamma -- so it is REUSED
     from the same natural_q00 checkpoint Phase 4 already produced, exactly
     like Phase 4 reused Phase 3's Natural/q=0 run.)

  RQ2-equivalent (strength sweep, rate held at the frozen main rate q=0.30):
    gamma in {ln(1.5)/2, ln(4)/2}   (OR=1.5, OR=4.0 at the |z|=2 reference
    point -- same OR grid MNAR-Y's own RQ2 sweep uses)              -> 2 conditions
    (gamma=ln(2)/2 (OR=2.0) @ q=0.30 is already produced by the RQ1-
     equivalent sweep above -- reused, not rerun.)

  Total fresh compute: 6 conditions x 5 StratifiedGroupKFold(subject_id)
  outer folds x 5 model families (full GridSearchCV inner tuning) each -- 30
  fold-fits, versus Phase 4's 70 (14 conditions x 5 folds), since this covers
  one mechanism instead of three. Expect roughly 30/70 x Phase 4's measured
  wall time (~1.9-2h) -> **very roughly 45-55 minutes**, will vary with the
  machine this runs on (measure the first condition and extrapolate).

WHY GAMMA/Z_CLIP ARE WHAT THEY ARE (full derivation, verified against the
real 30-column dataset, in phase8-driver-summary.md Sec.1 -- summarized here
for anyone reading only this file):
  - gamma_main = ln(2)/2 ~ 0.34657, chosen so that P(mask) at a perfectly
    typical value (|z|=0) versus a value 2 SD from its column's own mean
    (|z|=2) has OR=2.0 exactly -- the same reference-point OR the MAR and
    MNAR-Y main conditions use for their own binary drivers, so all three
    mechanisms' "main" strength is comparable at that fixed reference point.
  - RQ2-equivalent sensitivity gammas follow the identical gamma=ln(OR)/2
    rule at OR in {1.5, 4.0} -- exactly mirroring MNAR-Y's own OR grid.
  - Z_CLIP=5.0: 4 of the 30 real lab columns are skewed enough (native max
    |z| in the 40-50 range) that an UNCLIPPED |z| would drive that column's
    single most extreme native value's masking probability to a numerically-
    zero floor at gamma_main -- i.e. that cell could never be selected for
    synthetic masking in ANY repeat, a subtle, undocumented bias in the
    eligible pool. Clipping the driver (never the underlying data) at
    |z|<=5 keeps every column's worst-case masking probability at a
    non-degenerate 7-13% floor instead. See missingness_generator_v2.py's
    MNAR-X section docstring for the full numeric verification.

FOLD-SAFETY: identical contract to every other mechanism in this project --
for every (rate, fold) cell, mu_j/sigma_j/alpha_j are all calibrated
(fit_mnar_x) on THAT FOLD'S OWN outer-train partition only, then the SAME
fitted generator is applied (apply_mnar_x_mask) separately to that fold's
outer-train and outer-test rows. Outer-test rows never influence mu_j,
sigma_j, or alpha_j.

CALIBRATION DATA CAPTURE (feeds Phase 8 part 2, calibration-sensitivity):
unlike Phase 4/5/6, this driver captures the SELECTED family's raw
(y_test, p_test) pair for every (condition, fold) cell, via
selection_v2.run_outer_fold_with_predictions (additive, zero extra model
fits versus run_outer_fold -- see selection_v2.py's docstring on that
function). Saved to results/phase8_mnar_x/phase8_mnar_x_predictions.pkl.
This was found to be necessary during Phase 8 planning: Phase 4/5/6's saved
result pickles only ever contained aggregate AUROC/AP/Brier, never raw
per-case predicted probabilities, so calibration slope/intercept could not
actually be computed "for free" from them as originally assumed -- see
phase8-driver-summary.md Sec.2 for the full correction.

CHECKPOINTING: each condition's result is pickled to
results/phase8_mnar_x/<id>.pkl immediately after that condition finishes.
Re-running this script skips any condition whose checkpoint already exists.

Run with:   python3 run_phase8_mnar_x.py
Smoke-test: python3 run_phase8_mnar_x.py --smoke-test
            (1 condition, 2 outer folds instead of 5, exercises the exact
            same code path. Writes to results/phase8_mnar_x_smoke/, never
            touches results/phase8_mnar_x/.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as genmod
import selection_v2 as selv2
import metrics_v2 as met

# Same warnings-visibility fix as run_phase4_rq1_rq2.py (see that file's
# comment for the full rationale) -- importing selection_v2 silences
# UserWarning/FutureWarning globally as a side effect of the vendored base
# runner it execs; restore visibility for this multi-condition unattended run.
warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

_CONTAINER_DATA_PATH = "/mnt/user-data/uploads/DASA2026/Dataset/full_analytic_dataset_mortality_all_admissions.csv"
_RELATIVE_DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "full_analytic_dataset_mortality_all_admissions.csv"


def _resolve_data_path(cli_arg: Optional[str]) -> str:
    if cli_arg:
        if not Path(cli_arg).exists():
            raise FileNotFoundError(f"--data-path was given as {cli_arg!r} but that file does not exist")
        return cli_arg
    if _RELATIVE_DATA_PATH.exists():
        return str(_RELATIVE_DATA_PATH)
    if Path(_CONTAINER_DATA_PATH).exists():
        return _CONTAINER_DATA_PATH
    raise FileNotFoundError(
        "Could not find full_analytic_dataset_mortality_all_admissions.csv. Tried "
        f"{_RELATIVE_DATA_PATH} (relative to this script) and {_CONTAINER_DATA_PATH!r} "
        "(cloud-sandbox path). Pass --data-path /full/path/to/the/csv explicitly."
    )


# ---------------------------------------------------------------------------
# Phase-4-checkpoint candidate search (for q=0 reuse), mirroring run_phase6_
# rq4.py's multi-candidate Phase-4-CSV resolution pattern (the fix from that
# driver's own adversarial review) rather than a single hardcoded guess.
# ---------------------------------------------------------------------------

def _natural_q0_candidates() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [script_dir.parent, script_dir, cwd.parent, cwd]
    suffixes = [
        Path("PHASE_4_RQ1_RQ2") / "results" / "phase4" / "natural_q0.pkl",
        Path("results") / "phase4" / "natural_q0.pkl",
        Path("phase3_validation_natural_results.pkl"),
    ]
    seen: List[Path] = []
    for root in roots:
        for suf in suffixes:
            p = (root / suf).resolve()
            if p not in seen:
                seen.append(p)
    return seen


OUTER_SEED = 2026            # SAME seed as Phase 3/4/5/6 -- required for q=0 reuse + identical fold partitions
MASK_SEED = 20260917         # SAME mask seed used throughout this project (documented CRN convention)
N_OUTER = 5

Z_CLIP = 5.0
GAMMA_MAIN = float(np.log(2.0) / 2.0)          # OR=2.0 @ |z|=2 -- main strength, comparable to MAR/MNAR-Y's OR=2
GAMMA_SENS = {1.5: float(np.log(1.5) / 2.0), 4.0: float(np.log(4.0) / 2.0)}  # RQ2-equivalent, same OR grid as MNAR-Y
RATES_MAIN = [0.10, 0.20, 0.30, 0.40]
RQ2_RATE = 0.30

RESULTS_DIR = Path("results/phase8_mnar_x")


# ---------------------------------------------------------------------------
# Condition grid
# ---------------------------------------------------------------------------

def build_condition_grid() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for r in RATES_MAIN:
        conditions.append(dict(
            rq="RQ1", mechanism="mnar_x", rate=r, gamma=GAMMA_MAIN, or_value=2.0,
            condition_id=f"mnar_x_q{int(round(r * 100)):02d}_main",
        ))
    for or_val, gamma in GAMMA_SENS.items():
        conditions.append(dict(
            rq="RQ2", mechanism="mnar_x", rate=RQ2_RATE, gamma=gamma, or_value=or_val,
            condition_id=f"mnar_x_q{int(round(RQ2_RATE * 100)):02d}_OR{or_val}",
        ))
    return conditions


# ---------------------------------------------------------------------------
# Fold-safe masking + one condition's full 5-fold nested-CV run
# ---------------------------------------------------------------------------

def _prepare_X_y_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    y = df[genmod.LABEL_COL].astype(int).to_numpy()
    groups = df[genmod.GROUP_COL].to_numpy()
    X = df.drop(columns=[genmod.LABEL_COL], errors="ignore").copy()
    for c in genmod.EXCLUDED_FROM_MODEL_ENTIRELY:
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")
    num_cols, cat_cols = selv2.base_runner.split_columns_A(df)
    present = set(X.columns)
    num_cols = [c for c in num_cols if c in present]
    cat_cols = [c for c in cat_cols if c in present]
    return X, y, groups, num_cols, cat_cols


def run_condition_with_injection(
    df: pd.DataFrame,
    rate: float,
    gamma: float,
    seed: int = OUTER_SEED,
    mask_seed: int = MASK_SEED,
    n_outer: int = N_OUTER,
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Fold-safe: for each outer fold, calibrate mu_j/sigma_j/alpha_j on THAT
    fold's own outer-train partition only (fit_mnar_x), apply the same frozen
    generator to both outer-train and outer-test of that fold (apply_mnar_x_
    mask), then run selection_v2's fold-safe family selection + evaluation,
    capturing the selected family's raw predictions for calibration."""
    X, y, groups, num_cols, cat_cols = _prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    results = []
    diagnostics = []
    predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for fold_id, (tr_idx, te_idx) in enumerate(outer_splitter.split(X, y, groups), start=1):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        g_tr = groups[tr_idx]
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]
        assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        nat_mask_tr = genmod.compute_natural_mask(df_tr)
        nat_mask_te = genmod.compute_natural_mask(df_te)

        if rate == 0.0:
            gen = genmod.MnarXGenerator(
                gamma=gamma, target_rate=0.0, z_clip=Z_CLIP,
                alpha={c: -np.inf for c in genmod.ELIGIBLE_LAB_COLS},
                mu={c: float("nan") for c in genmod.ELIGIBLE_LAB_COLS},
                sigma={c: float("nan") for c in genmod.ELIGIBLE_LAB_COLS},
            )
        else:
            gen = genmod.fit_mnar_x(df_tr, nat_mask_tr, gamma=gamma, target_rate=rate, z_clip=Z_CLIP)

        rng = np.random.default_rng(mask_seed)  # fresh per (rate, gamma, fold) cell -- matches this project's CRN convention
        syn_mask_tr = genmod.apply_mnar_x_mask(df_tr, nat_mask_tr, gen, rng)
        syn_mask_te = genmod.apply_mnar_x_mask(df_te, nat_mask_te, gen, rng)

        final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
        final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

        X_tr_masked = X_tr.copy()
        X_te_masked = X_te.copy()
        for col in genmod.ELIGIBLE_LAB_COLS:
            X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
            X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan

        rates_tr = genmod.compute_rates(nat_mask_tr, syn_mask_tr)
        rates_te = genmod.compute_rates(nat_mask_te, syn_mask_te)

        diag = {
            "fold_id": fold_id, "mechanism": "mnar_x", "target_rate": rate, "gamma": gamma,
            "r_inject_train": rates_tr["r_inject"], "r_total_train": rates_tr["r_total"],
            "r_inject_test": rates_te["r_inject"], "r_total_test": rates_te["r_total"],
        }
        if rate > 0.0:
            # Manipulation-check diagnostic specific to MNAR-X (protocol_v2.yaml/
            # methodology-redesign-spec.md Sec.8's "corr(mask, |z|) -- MNAR-X only"
            # row): pooled correlation between a cell's own |z| and whether it was
            # synthetically masked, across all eligible cells on the test partition.
            # Expected sign: NEGATIVE (higher |z| -> less likely masked, by design).
            elig_te = genmod.compute_eligible_mask(nat_mask_te)
            z_list, m_list = [], []
            for col in genmod.ELIGIBLE_LAB_COLS:
                e = elig_te[col].to_numpy()
                if e.sum() == 0 or not np.isfinite(gen.sigma[col]) or gen.sigma[col] <= 0:
                    continue
                x = df_te[col].to_numpy(dtype=float)[e]
                z = np.clip(np.abs((x - gen.mu[col]) / gen.sigma[col]), 0.0, Z_CLIP)
                m = syn_mask_te[col].to_numpy()[e].astype(float)
                z_list.append(z)
                m_list.append(m)
            if z_list:
                z_all = np.concatenate(z_list)
                m_all = np.concatenate(m_list)
                if np.std(z_all) > 0 and np.std(m_all) > 0:
                    diag["corr_mask_vs_abs_z"] = float(np.corrcoef(z_all, m_all)[0, 1])
        diagnostics.append(diag)

        result, y_sel, p_sel = selv2.run_outer_fold_with_predictions(
            X_tr_masked, y_tr, g_tr, X_te_masked, y_te, num_cols, cat_cols, seed=seed, fold_id=fold_id
        )
        results.append(result)
        predictions[fold_id] = (y_sel, p_sel)

    return results, diagnostics, predictions


# ---------------------------------------------------------------------------
# q=0 reuse from Phase 4's natural_q0.pkl (mechanism-independent at rate=0)
# ---------------------------------------------------------------------------

def load_or_compute_natural_q0(df: pd.DataFrame, log, n_outer: int = N_OUTER) -> Tuple[List[Any], List[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    for p in _natural_q0_candidates():
        if p.exists():
            log(f"q=0: reusing existing Phase 3/4 natural_q0 checkpoint from {p}")
            with open(p, "rb") as f:
                loaded = pickle.load(f)
            # Phase 4's natural_q0.pkl is (results, diagnostics) -- no predictions
            # captured there (predates this driver's calibration-capture addition).
            # Phase 3's raw pickle is a bare list of OuterFoldResult.
            if isinstance(loaded, tuple) and len(loaded) == 2:
                results, diag = loaded
            else:
                results = loaded
                diag = [{"fold_id": r.fold_id, "mechanism": "mnar_x", "target_rate": 0.0, "gamma": 0.0,
                          "r_inject_train": 0.0, "r_total_train": None, "r_inject_test": 0.0, "r_total_test": None}
                         for r in results]
            log(f"q=0: no calibration predictions available for the reused q=0 point "
                f"(natural_q0 predates prediction-capture) -- calibration analysis for "
                f"mnar_x will start from q>0 conditions only, consistent with q=0 carrying "
                f"no injected missingness to calibrate against in the first place.")
            return results, diag, {}
    log("q=0: no existing natural_q0 checkpoint found anywhere searched, computing fresh")
    return run_condition_with_injection(df, rate=0.0, gamma=GAMMA_MAIN, n_outer=n_outer)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    log.close = f.close
    return log


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None):
    results_dir = Path("results/phase8_mnar_x_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    log(f"=== Phase 8 (MNAR-X) driver start (smoke_test={smoke_test}, n_outer={n_outer}) ===")
    log(f"GAMMA_MAIN={GAMMA_MAIN:.6f} (OR=2.0 @ |z|=2), GAMMA_SENS={GAMMA_SENS}, Z_CLIP={Z_CLIP}")

    data_path = _resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    conditions = build_condition_grid()
    if smoke_test:
        conditions = [c for c in conditions if c["condition_id"] == "mnar_x_q10_main"]
    log(f"Condition grid: {len(conditions)} fresh conditions")
    for c in conditions:
        log(f"  {c['condition_id']}: rq={c['rq']} rate={c['rate']} gamma={c['gamma']:.6f} OR={c['or_value']}")

    all_results: Dict[str, List[Any]] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    all_predictions: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    condition_meta: Dict[str, Dict[str, Any]] = {}

    # --- q=0, reused from Phase 4's natural_q0.pkl ---
    q0_ckpt = results_dir / "natural_q0.pkl"
    if q0_ckpt.exists():
        log("q=0: checkpoint already exists, loading")
        with open(q0_ckpt, "rb") as f:
            q0_results, q0_diag, q0_pred = pickle.load(f)
    else:
        t0 = time.time()
        q0_results, q0_diag, q0_pred = load_or_compute_natural_q0(df, log, n_outer=n_outer)
        with open(q0_ckpt, "wb") as f:
            pickle.dump((q0_results, q0_diag, q0_pred), f)
        log(f"q=0: done in {time.time() - t0:.1f}s, checkpointed to {q0_ckpt}")
    all_results["mnar_x_q00"] = q0_results
    all_diagnostics.extend(q0_diag)
    all_predictions["mnar_x_q00"] = q0_pred
    condition_meta["mnar_x_q00"] = dict(rq="RQ1", mechanism="mnar_x", rate=0.0, gamma=0.0, or_value=None)

    # --- fresh conditions ---
    for i, c in enumerate(conditions, start=1):
        cid = c["condition_id"]
        ckpt = results_dir / f"{cid}.pkl"
        condition_meta[cid] = c
        if ckpt.exists():
            log(f"[{i}/{len(conditions)}] {cid}: checkpoint already exists, skipping")
            with open(ckpt, "rb") as f:
                r, d, p = pickle.load(f)
            all_results[cid] = r
            all_diagnostics.extend(d)
            all_predictions[cid] = p
            continue

        log(f"[{i}/{len(conditions)}] {cid}: starting (rq={c['rq']} rate={c['rate']} "
            f"gamma={c['gamma']:.6f} OR={c['or_value']})")
        t0 = time.time()
        r, d, p = run_condition_with_injection(df, rate=c["rate"], gamma=c["gamma"], n_outer=n_outer)
        elapsed = time.time() - t0
        with open(ckpt, "wb") as f:
            pickle.dump((r, d, p), f)
        all_results[cid] = r
        all_diagnostics.extend(d)
        all_predictions[cid] = p
        for fold_result in r:
            log(f"    fold {fold_result.fold_id}: selected={fold_result.selected_family} "
                f"outer_test_auroc={fold_result.selected_outer_test_auroc:.4f} "
                f"oracle={fold_result.oracle_family} regret={fold_result.selection_regret:.4f} "
                f"displaced={fold_result.displaced}")
        log(f"[{i}/{len(conditions)}] {cid}: done in {elapsed:.1f}s, checkpointed to {ckpt}")

    log("All conditions complete. Aggregating...")
    aggregate_and_save(all_results, all_diagnostics, all_predictions, condition_meta, results_dir, log)
    log("=== Phase 8 (MNAR-X) driver finished ===")
    log.close()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_and_save(all_results, all_diagnostics, all_predictions, condition_meta, results_dir: Path, log):
    rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            for fam, fres in fr.families.items():
                rows.append({
                    "condition_id": cid, "rq": meta["rq"], "mechanism": "mnar_x",
                    "rate": meta["rate"], "gamma": meta["gamma"], "or_value": meta["or_value"],
                    "fold_id": fr.fold_id, "model_key": fam,
                    "inner_auroc_mean": fres.inner_auroc_mean, "inner_ap_mean": fres.inner_ap_mean,
                    "outer_test_auroc": fres.outer_test_auroc, "outer_test_ap": fres.outer_test_ap,
                    "outer_test_brier": fres.outer_test_brier,
                    "selected_family": fr.selected_family, "oracle_family": fr.oracle_family,
                    "selection_regret": fr.selection_regret, "displaced": fr.displaced,
                })
    full_table = pd.DataFrame(rows)
    full_table.to_csv(results_dir / "phase8_mnar_x_full_table.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_full_table.csv'} ({len(full_table)} rows)")

    summary_rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            summary_rows.append({
                "condition_id": cid, "rq": meta["rq"], "rate": meta["rate"], "or_value": meta["or_value"],
                "fold_id": fr.fold_id, "selected_family": fr.selected_family,
                "selected_outer_test_auroc": fr.selected_outer_test_auroc,
                "oracle_family": fr.oracle_family, "oracle_outer_test_auroc": fr.oracle_outer_test_auroc,
                "selection_regret": fr.selection_regret, "displaced": fr.displaced,
            })
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_csv(results_dir / "phase8_mnar_x_fold_summary.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_fold_summary.csv'}")

    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(results_dir / "phase8_mnar_x_manipulation_check.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_manipulation_check.csv'}")

    metrics_summary: Dict[str, Any] = {}
    for cid, fold_results in all_results.items():
        selected = met.selected_family_by_fold(fold_results)
        entropy = met.within_condition_selection_entropy(list(selected.values()), selv2.MODELS)
        scores_by_repeat = [met.outer_test_auroc_by_family(r) for r in fold_results]
        kendall = met.kendall_tau_b_rank_stability(scores_by_repeat) if len(scores_by_repeat) >= 2 else None
        sel_margins = [met.top1_minus_top2_margin(met.inner_auroc_by_family(r)) for r in fold_results]
        eval_margins = [met.top1_minus_top2_margin(met.outer_test_auroc_by_family(r)) for r in fold_results]
        metrics_summary[cid] = dict(
            entropy_normalized=entropy.normalized_entropy, modal_family=entropy.modal_family,
            modal_frequency=entropy.modal_frequency,
            kendall_tau_b_mean=(kendall.mean_tau_b if kendall else None),
            kendall_tau_b_min=(kendall.min_tau_b if kendall else None),
            kendall_tau_b_max=(kendall.max_tau_b if kendall else None),
            selection_margin_mean=float(np.mean(sel_margins)),
            evaluation_margin_mean=float(np.mean(eval_margins)),
        )

    # Baseline-selection displacement rate for mnar_x across its own rate sweep
    by_fold_and_rate: Dict[int, Dict[float, str]] = {}
    for r in [0.0] + RATES_MAIN:
        cid = "mnar_x_q00" if r == 0.0 else f"mnar_x_q{int(round(r * 100)):02d}_main"
        if cid not in all_results:
            continue
        for fr in all_results[cid]:
            by_fold_and_rate.setdefault(fr.fold_id, {})[r] = fr.selected_family
    disp = met.baseline_selection_displacement_rate(by_fold_and_rate, baseline_rate=0.0)
    displacement = {
        r: dict(displacement_rate=v.displacement_rate, n_folds=v.n_folds, n_displaced=v.n_displaced,
                 displaced_fold_ids=v.displaced_fold_ids)
        for r, v in disp.items()
    }

    with open(results_dir / "phase8_mnar_x_metrics_summary.json", "w") as f:
        json.dump({"per_condition": metrics_summary, "displacement_mnar_x": displacement}, f, indent=2, default=str)
    log(f"Wrote {results_dir / 'phase8_mnar_x_metrics_summary.json'}")

    with open(results_dir / "phase8_mnar_x_all_results.pkl", "wb") as f:
        pickle.dump({"all_results": all_results, "condition_meta": condition_meta, "all_diagnostics": all_diagnostics}, f)
    log(f"Wrote {results_dir / 'phase8_mnar_x_all_results.pkl'}")

    # Predictions, saved separately (kept out of the main results pickle since
    # it holds raw per-case arrays, not just aggregate metrics) -- feeds Phase
    # 8 part 2 (calibration slope/intercept).
    with open(results_dir / "phase8_mnar_x_predictions.pkl", "wb") as f:
        pickle.dump({"predictions": all_predictions, "condition_meta": condition_meta}, f)
    log(f"Wrote {results_dir / 'phase8_mnar_x_predictions.pkl'} "
        f"({sum(len(v) for v in all_predictions.values())} (condition, fold) prediction arrays)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run 1 condition x 2 outer folds only. Writes to results/phase8_mnar_x_smoke/.")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv.")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path)
