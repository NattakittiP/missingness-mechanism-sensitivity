"""
Phase 8, Part 4 of 4 — Alternative-split (S1) sensitivity (implementation-order
item 11, final sub-part).

WHAT THIS IS: `protocol_v2.yaml`'s `split:` block freezes the headline split as
`StratifiedGroupKFold(subject_id)` (patients never split across outer-train/
outer-test) and separately records a `legacy_sensitivity_split: StratifiedKFold`
with the note `"no patient separation -- do not use as headline result"`. Every
result in Phases 3-8 Parts 1-3 uses the group-safe split. This driver runs the
SAME RQ1 rate-sweep grid Phase 4 ran, but under plain `StratifiedKFold` (no
`subject_id` grouping) instead -- the sensitivity question is: does RQ1's
headline finding (baseline_selection_displacement_rate rises sharply with rate
for MCAR/MAR but stays at 0% for MNAR-Y) survive when patient-level separation
between outer-train and outer-test is removed?

WHY THIS MATTERS METHODOLOGICALLY: Dataset A is one row per admission, and a
single patient (subject_id) can have multiple admissions. Under
StratifiedGroupKFold, a patient's admissions are guaranteed to land entirely
in outer-train or entirely in outer-test. Under plain StratifiedKFold, the
same patient's admissions can be split across both -- if a patient's
admissions are similar to each other (same baseline labs, same eventual
outcome), the model could partially "recognize" a test patient from having
seen a different admission of theirs in training, inflating outer-test
performance relative to genuine generalization to NEW patients. This is
exactly the leakage `StratifiedGroupKFold` was adopted specifically to
prevent (Phase 0 audit). S1 quantifies how much this would have mattered if
that precaution had not been taken.

SCOPE (a deliberate reduction from Phase 4's full 14-condition grid, to keep
this the last and most expensive of the 4 item-11 sub-parts affordable):
RQ1's rate sweep ONLY -- 3 mechanisms (mcar, mar, mnar_y) x 4 rates
(0.10, 0.20, 0.30, 0.40) = 12 conditions, PLUS one fresh q=0 baseline
(mechanism-independent by construction, reused across all 3 mechanism labels,
exactly Phase 4's own q00-reuse convention) = 13 conditions total x 5 outer
folds x 5 model families with full GridSearchCV tuning each -- the same
per-condition cost as Phase 4, so expect similar wall time (Phase 4 measured
~493s/condition on the user's machine, i.e. budget ~1.8h for 13 conditions).
RQ2's OR-sweep is explicitly OUT OF SCOPE here: RQ2 asks a strength question
(how does MNAR-Y's effect size change AUROC/displacement), which is
orthogonal to what S1 is testing (does the OUTER SPLIT mechanism change the
answer) -- re-deriving RQ2 under S1 as well would double this driver's cost
for a question this project has not asked. If the user wants it later, this
script's condition-building logic (`build_condition_grid`) can be extended
the same way `run_phase4_rq1_rq2.py`'s was, following the same pattern.

CRITICAL DIFFERENCE FROM PHASE 4's DRIVER (besides the splitter itself): Phase
4's `run_condition_with_injection` asserts
`set(groups[tr_idx]).isdisjoint(set(groups[te_idx]))` -- a group-safety
invariant that is BY DESIGN violated under S1 (that violation is the entire
point of this sensitivity check), so that assertion is replaced here with a
per-fold DIAGNOSTIC that measures and logs exactly how much subject_id
overlap plain StratifiedKFold introduces (fraction of outer-test admissions
whose subject_id also appears in that fold's outer-train), written to
`phase8_s1_overlap_diagnostics.csv`. Everything else -- fold-safe masking
(alpha_j fit on outer-train only, same generator applied unchanged to
outer-train/outer-test), selection_v2's inner-CV-only family selection
(`run_outer_fold`, reused verbatim, unmodified), the metrics_v2 layer,
checkpointing -- is identical in mechanics to Phase 4's driver; only the
outer splitter class and its `.split(...)` call signature differ.

Run with:   python3 run_phase8_s1_sensitivity.py
Smoke-test: python3 run_phase8_s1_sensitivity.py --smoke-test
            (1 condition, 2 outer folds instead of 5.)
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
from sklearn.model_selection import StratifiedKFold

import missingness_generator_v2 as genmod
import selection_v2 as selv2
import metrics_v2 as met

# Same warning-visibility restoration as run_phase4_rq1_rq2.py/run_phase6_rq4.py
# -- importing selection_v2 execs base_runner, which globally silences
# UserWarning/FutureWarning for the rest of the process; restore default
# visibility (minus two known-benign, already-audited sklearn deprecation
# notices) so a multi-hour run doesn't hide a genuine ConvergenceWarning.
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
        f"{_RELATIVE_DATA_PATH} (relative to this script) and {_CONTAINER_DATA_PATH!r}. "
        "Pass --data-path explicitly."
    )


OUTER_SEED = 2026             # same seed as every other Phase 3-8 driver, for consistency (NOT for fold-identity --
                               # StratifiedKFold and StratifiedGroupKFold produce different partitions even at the same seed)
MASK_SEED = 20260917          # same CRN convention as every other driver
N_OUTER = 5
THETA_MAIN = 0.693147         # ln(2), OR=2 -- frozen MAR + MNAR-Y main strength
RATES_MAIN = [0.10, 0.20, 0.30, 0.40]

RESULTS_DIR = Path("results/phase8_s1_sensitivity")


# ---------------------------------------------------------------------------
# Condition grid (RQ1 rate sweep only -- see module docstring for scope note)
# ---------------------------------------------------------------------------

def build_condition_grid() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for mech in ["mcar", "mar", "mnar_y"]:
        theta = 0.0 if mech == "mcar" else THETA_MAIN
        or_value = None if mech == "mcar" else 2.0
        for r in RATES_MAIN:
            conditions.append(dict(
                rq="RQ1_S1", mechanism=mech, rate=r, theta=theta, or_value=or_value,
                condition_id=f"{mech}_q{int(round(r * 100)):02d}_main",
            ))
    return conditions


# ---------------------------------------------------------------------------
# Fold-safe masking -- IDENTICAL logic to run_phase4_rq1_rq2.py's
# _prepare_X_y_groups / _fit_generator_and_drivers (copied verbatim; only the
# outer splitter differs in this driver -- see run_condition_with_injection).
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


def _fit_generator_and_drivers(mechanism: str, rate: float, theta: float,
                                df_tr: pd.DataFrame, y_tr: np.ndarray,
                                nat_mask_tr: pd.DataFrame, df_te: pd.DataFrame,
                                y_te: np.ndarray):
    if rate == 0.0:
        gen = genmod.fit_natural_baseline()
        return gen, None, None
    if mechanism == "mcar":
        gen = genmod.fit_mcar(nat_mask_tr, target_rate=rate)
        return gen, None, None
    if mechanism == "mar":
        gen = genmod.fit_mar_context(nat_mask_tr, df_tr["admission_type"], theta=theta, target_rate=rate)
        driver_tr = genmod.compute_H_i(df_tr["admission_type"]).to_numpy().astype(float)
        driver_te = genmod.compute_H_i(df_te["admission_type"]).to_numpy().astype(float)
        return gen, driver_tr, driver_te
    if mechanism == "mnar_y":
        gen = genmod.fit_mnar_y(nat_mask_tr, df_tr[genmod.LABEL_COL], theta=theta, target_rate=rate)
        return gen, y_tr.astype(float), y_te.astype(float)
    raise ValueError(f"unknown mechanism: {mechanism}")


def run_condition_with_injection(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    theta: float,
    seed: int = OUTER_SEED,
    mask_seed: int = MASK_SEED,
    n_outer: int = N_OUTER,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Same fold-safe masking + inner-CV-only selection as
    run_phase4_rq1_rq2.py's run_condition_with_injection, but split with
    plain StratifiedKFold(y) instead of StratifiedGroupKFold(y, subject_id)
    -- S1, the legacy/sensitivity split per protocol_v2.yaml. subject_id is
    still computed (needed for the overlap diagnostic and passed through to
    selection_v2.run_outer_fold's internal calibration carve-out, which
    remains group-aware -- S1's definition concerns the OUTER split only, per
    protocol_v2.yaml's `legacy_sensitivity_split` entry)."""
    X, y, groups, num_cols, cat_cols = _prepare_X_y_groups(df)
    outer_splitter = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    results = []
    diagnostics = []
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splitter.split(X, y), start=1):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        g_tr = groups[tr_idx]
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]

        # S1's defining property: subject_id is explicitly NOT used for the
        # split, so overlap is expected -- measure and log it rather than
        # asserting disjointness (that assertion, present in Phase 4's
        # driver, would fail here by construction).
        train_subjects = set(groups[tr_idx])
        test_subjects = set(groups[te_idx])
        overlap_subjects = train_subjects & test_subjects
        overlap_admissions_frac = float(np.isin(groups[te_idx], list(overlap_subjects)).mean()) if overlap_subjects else 0.0

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        nat_mask_tr = genmod.compute_natural_mask(df_tr)
        nat_mask_te = genmod.compute_natural_mask(df_te)

        gen, driver_tr, driver_te = _fit_generator_and_drivers(
            mechanism, rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
        )

        rng = np.random.default_rng(mask_seed)  # fresh per (mechanism, rate, fold) cell -- same CRN convention
        syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
        syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)

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
            "fold_id": fold_id, "mechanism": mechanism, "target_rate": rate, "theta": theta,
            "r_inject_train": rates_tr["r_inject"], "r_total_train": rates_tr["r_total"],
            "r_inject_test": rates_te["r_inject"], "r_total_test": rates_te["r_total"],
            "n_train_subjects": len(train_subjects), "n_test_subjects": len(test_subjects),
            "n_overlap_subjects": len(overlap_subjects),
            "overlap_test_admissions_frac": overlap_admissions_frac,
        }
        if mechanism == "mnar_y" and rate > 0.0:
            syn_te_np = syn_mask_te.to_numpy()
            y1 = y_te == 1
            y0 = y_te == 0
            elig_te = genmod.compute_eligible_mask(nat_mask_te).to_numpy()
            diag["r_inject_among_y1"] = float(syn_te_np[y1].sum() / max(elig_te[y1].sum(), 1))
            diag["r_inject_among_y0"] = float(syn_te_np[y0].sum() / max(elig_te[y0].sum(), 1))
        if mechanism == "mar" and rate > 0.0:
            mask_count_per_patient = syn_mask_te.to_numpy().sum(axis=1)
            if np.std(driver_te) > 0 and np.std(mask_count_per_patient) > 0:
                diag["corr_mask_count_vs_H_i"] = float(np.corrcoef(mask_count_per_patient, driver_te)[0, 1])
        diagnostics.append(diag)

        result = selv2.run_outer_fold(
            X_tr_masked, y_tr, g_tr, X_te_masked, y_te, num_cols, cat_cols, seed=seed, fold_id=fold_id
        )
        results.append(result)

    return results, diagnostics


def load_or_compute_natural_q0(df: pd.DataFrame, n_outer: int = N_OUTER) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Unlike Phase 4's driver, this is NEVER reused from an earlier phase's
    pickle -- Phase 3's Natural/q=0 validation used StratifiedGroupKFold, a
    DIFFERENT split from S1's plain StratifiedKFold, so the fold partitions
    (and therefore the results) are not interchangeable. Always computed
    fresh here, once, then reused across the 3 mechanism labels exactly like
    Phase 4's own q00-reuse convention."""
    return run_condition_with_injection(df, mechanism="mcar", rate=0.0, theta=0.0, n_outer=n_outer)


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
    results_dir = Path("results/phase8_s1_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    log(f"=== Phase 8 (S1 alternative-split sensitivity) driver start (smoke_test={smoke_test}, n_outer={n_outer}) ===")

    data_path = _resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    conditions = build_condition_grid()
    if smoke_test:
        conditions = [c for c in conditions if c["condition_id"] == "mnar_y_q10_main"]
    log(f"Condition grid: {len(conditions)} fresh conditions (RQ1 rate sweep only -- see module docstring for scope)")
    for c in conditions:
        log(f"  {c['condition_id']}: mechanism={c['mechanism']} rate={c['rate']} theta={c['theta']:.6f}")

    all_results: Dict[str, List[Any]] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    condition_meta: Dict[str, Dict[str, Any]] = {}

    # --- q=0 (Natural under S1), shared across all 3 mechanisms ---
    q0_ckpt = results_dir / "natural_q0.pkl"
    if q0_ckpt.exists():
        log("q=0 (Natural, S1): checkpoint already exists, loading")
        with open(q0_ckpt, "rb") as f:
            q0_results, q0_diag = pickle.load(f)
    else:
        t0 = time.time()
        q0_results, q0_diag = load_or_compute_natural_q0(df, n_outer=n_outer)
        with open(q0_ckpt, "wb") as f:
            pickle.dump((q0_results, q0_diag), f)
        log(f"q=0 (Natural, S1): done in {time.time() - t0:.1f}s, checkpointed to {q0_ckpt}")
    all_results["natural_q00"] = q0_results
    all_diagnostics.extend(q0_diag)
    condition_meta["natural_q00"] = dict(rq="RQ1_S1", mechanism="natural", rate=0.0, theta=0.0, or_value=None)
    for mech in ["mcar", "mar", "mnar_y"]:
        all_results[f"{mech}_q00"] = q0_results
        condition_meta[f"{mech}_q00"] = dict(rq="RQ1_S1", mechanism=mech, rate=0.0, theta=0.0,
                                               or_value=(None if mech == "mcar" else 2.0))

    # --- fresh conditions ---
    for i, c in enumerate(conditions, start=1):
        cid = c["condition_id"]
        ckpt = results_dir / f"{cid}.pkl"
        condition_meta[cid] = c
        if ckpt.exists():
            log(f"[{i}/{len(conditions)}] {cid}: checkpoint already exists, skipping")
            with open(ckpt, "rb") as f:
                r, d = pickle.load(f)
            all_results[cid] = r
            all_diagnostics.extend(d)
            continue

        log(f"[{i}/{len(conditions)}] {cid}: starting (mechanism={c['mechanism']} "
            f"rate={c['rate']} theta={c['theta']:.6f} OR={c['or_value']})")
        t0 = time.time()
        r, d = run_condition_with_injection(
            df, mechanism=c["mechanism"], rate=c["rate"], theta=c["theta"], n_outer=n_outer,
        )
        elapsed = time.time() - t0
        with open(ckpt, "wb") as f:
            pickle.dump((r, d), f)
        all_results[cid] = r
        all_diagnostics.extend(d)
        for fold_result in r:
            log(f"    fold {fold_result.fold_id}: selected={fold_result.selected_family} "
                f"outer_test_auroc={fold_result.selected_outer_test_auroc:.4f} "
                f"oracle={fold_result.oracle_family} regret={fold_result.selection_regret:.4f} "
                f"displaced={fold_result.displaced}")
        log(f"[{i}/{len(conditions)}] {cid}: done in {elapsed:.1f}s, checkpointed to {ckpt}")

    log("All conditions complete. Aggregating...")
    aggregate_and_save(all_results, all_diagnostics, condition_meta, results_dir, log)
    log("=== Phase 8 (S1 alternative-split sensitivity) driver finished ===")
    log.close()


# ---------------------------------------------------------------------------
# Aggregation -- same shape as run_phase4_rq1_rq2.py's aggregate_and_save,
# retargeted to phase8_s1_* filenames, plus the new overlap-diagnostics CSV.
# ---------------------------------------------------------------------------

def aggregate_and_save(all_results, all_diagnostics, condition_meta, results_dir: Path, log):
    rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            for fam, fres in fr.families.items():
                rows.append({
                    "condition_id": cid, "rq": meta["rq"], "mechanism": meta["mechanism"],
                    "rate": meta["rate"], "theta": meta["theta"], "or_value": meta["or_value"],
                    "fold_id": fr.fold_id, "model_key": fam,
                    "inner_auroc_mean": fres.inner_auroc_mean, "inner_ap_mean": fres.inner_ap_mean,
                    "outer_test_auroc": fres.outer_test_auroc, "outer_test_ap": fres.outer_test_ap,
                    "outer_test_brier": fres.outer_test_brier,
                    "selected_family": fr.selected_family, "oracle_family": fr.oracle_family,
                    "selection_regret": fr.selection_regret, "displaced": fr.displaced,
                })
    full_table = pd.DataFrame(rows)
    full_table.to_csv(results_dir / "phase8_s1_full_table.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_s1_full_table.csv'} ({len(full_table)} rows)")

    summary_rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            summary_rows.append({
                "condition_id": cid, "rq": meta["rq"], "mechanism": meta["mechanism"],
                "rate": meta["rate"], "or_value": meta["or_value"], "fold_id": fr.fold_id,
                "selected_family": fr.selected_family, "selected_outer_test_auroc": fr.selected_outer_test_auroc,
                "oracle_family": fr.oracle_family, "oracle_outer_test_auroc": fr.oracle_outer_test_auroc,
                "selection_regret": fr.selection_regret, "displaced": fr.displaced,
            })
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_csv(results_dir / "phase8_s1_fold_summary.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_s1_fold_summary.csv'}")

    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(results_dir / "phase8_s1_manipulation_check.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_s1_manipulation_check.csv'}")

    # Overlap diagnostic in its own file too (the key S1-specific quantity):
    # how much patient-level leakage plain StratifiedKFold introduces, per
    # (condition, fold) -- present in every diag row via n_train_subjects/
    # n_test_subjects/n_overlap_subjects/overlap_test_admissions_frac, but
    # broken out here for direct inspection without the mechanism-specific
    # columns cluttering it.
    overlap_cols = ["fold_id", "mechanism", "target_rate", "n_train_subjects", "n_test_subjects",
                     "n_overlap_subjects", "overlap_test_admissions_frac"]
    overlap_df = diag_df[[c for c in overlap_cols if c in diag_df.columns]].drop_duplicates()
    overlap_df.to_csv(results_dir / "phase8_s1_overlap_diagnostics.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_s1_overlap_diagnostics.csv'} "
        f"(mean overlap_test_admissions_frac={overlap_df['overlap_test_admissions_frac'].mean():.4f})")

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

    displacement_by_mechanism: Dict[str, Any] = {}
    for mech in ["mcar", "mar", "mnar_y"]:
        by_fold_and_rate: Dict[int, Dict[float, str]] = {}
        for r in [0.0] + RATES_MAIN:
            cid = f"{mech}_q00" if r == 0.0 else f"{mech}_q{int(round(r * 100)):02d}_main"
            if cid not in all_results:
                continue
            for fr in all_results[cid]:
                by_fold_and_rate.setdefault(fr.fold_id, {})[r] = fr.selected_family
        disp = met.baseline_selection_displacement_rate(by_fold_and_rate, baseline_rate=0.0)
        displacement_by_mechanism[mech] = {
            r: dict(displacement_rate=v.displacement_rate, n_folds=v.n_folds, n_displaced=v.n_displaced,
                     displaced_fold_ids=v.displaced_fold_ids)
            for r, v in disp.items()
        }

    with open(results_dir / "phase8_s1_metrics_summary.json", "w") as f:
        json.dump({"per_condition": metrics_summary, "displacement_by_mechanism": displacement_by_mechanism},
                   f, indent=2, default=str)
    log(f"Wrote {results_dir / 'phase8_s1_metrics_summary.json'}")

    with open(results_dir / "phase8_s1_all_results.pkl", "wb") as f:
        pickle.dump({"all_results": all_results, "condition_meta": condition_meta, "all_diagnostics": all_diagnostics}, f)
    log(f"Wrote {results_dir / 'phase8_s1_all_results.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run 1 condition x 2 outer folds only. Writes to results/phase8_s1_smoke/.")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path)
