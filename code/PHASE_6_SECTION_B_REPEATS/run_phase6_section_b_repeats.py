"""Repeated-draw companion to Phase 6's permutation ablation, characterizing draw-to-draw variability of the RQ4 effect."""
import argparse
import hashlib
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import missingness_generator_v2 as genmod
import selection_v2 as selv2

import run_phase4_rq1_rq2 as p4

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

OUTER_SEED = p4.OUTER_SEED
MASK_SEED = p4.MASK_SEED

PERMUTATION_RATES = [0.30, 0.40]
PERMUTATION_ORS = {2.0: p4.THETA_MAIN, 4.0: float(np.log(4.0))}
PERMUTATION_SEED_OFFSET = 555

N_OUTER = 5
RESULTS_DIR = Path("results/phase6_section_b_repeats")

DRIFT_INFO_THRESHOLD = 1e-6
DRIFT_WARN_THRESHOLD_XGB = 0.03
DRIFT_WARN_THRESHOLD_OTHER = 1e-6


_PHASE6_PERMUTATION_CSV_CONTAINER = (
    "/mnt/user-data/uploads/DASA2026/PHASE_6_RQ4/results/phase6/phase6_permutation.csv"
)


def _original_permutation_csv_candidates() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [script_dir.parent, script_dir, cwd.parent, cwd]
    suffixes = [
        Path("PHASE_6_RQ4") / "results" / "phase6" / "phase6_permutation.csv",
        Path("results") / "phase6" / "phase6_permutation.csv",
    ]
    seen: List[Path] = []
    for root in roots:
        for suf in suffixes:
            p = (root / suf).resolve()
            if p not in seen:
                seen.append(p)
    return seen


def _resolve_original_permutation_csv(cli_arg: Optional[str], log) -> Optional[pd.DataFrame]:
    if cli_arg:
        p = Path(cli_arg)
        if p.exists():
            log(f"  Loaded original Section B results for cross-check from --original-permutation-csv: {p}")
            return pd.read_csv(p)
        log(f"  WARNING: --original-permutation-csv {cli_arg!r} does not exist; "
            f"draw-0 correctness cross-check will be skipped.")
        return None
    for p in _original_permutation_csv_candidates():
        if p.exists():
            log(f"  Resolved original Section B results via candidate search: {p}")
            return pd.read_csv(p)
    if Path(_PHASE6_PERMUTATION_CSV_CONTAINER).exists():
        log(f"  Resolved original Section B results via cloud-sandbox fallback path: "
            f"{_PHASE6_PERMUTATION_CSV_CONTAINER}")
        return pd.read_csv(_PHASE6_PERMUTATION_CSV_CONTAINER)
    log("  Original phase6_permutation.csv not found via candidate search; "
        "draw-0 correctness cross-check will be skipped (not fatal -- the repeats "
        "are still computed and valid, this only disables one extra safety check). "
        f"Tried: {[str(p) for p in _original_permutation_csv_candidates()]}"
        f" and {_PHASE6_PERMUTATION_CSV_CONTAINER}")
    return None


def _perm_seed_entropy(fold_id: int, rate: float, or_val: float, draw_id: int):
    """Derives the permutation seed for a given draw; draw 0 reproduces the original driver's exact seed."""
    base = (MASK_SEED, PERMUTATION_SEED_OFFSET, fold_id, int(round(rate * 100)), int(round(or_val * 10)))
    if draw_id == 0:
        return base
    return base + (draw_id,)


def run_repeats_for_cell(df: pd.DataFrame, X, y, groups, num_cols, cat_cols,
                          fold_id: int, tr_idx, te_idx, rate: float, or_val: float, theta: float,
                          n_repeats: int, results_dir: Path, log,
                          original_df: Optional[pd.DataFrame],
                          smoke_test: bool = False) -> List[Dict[str, Any]]:
    cond_id = f"mnar_y_q{int(round(rate * 100)):02d}_OR{or_val}_permtest"
    ckpt = results_dir / f"repeats_{cond_id}_fold{fold_id}.pkl"
    if ckpt.exists():
        with open(ckpt, "rb") as f:
            rows = pickle.load(f)
        if len(rows) != n_repeats:
            log(f"  [{cond_id} fold {fold_id}] WARNING: checkpoint has {len(rows)} draws but this run "
                f"requested n_repeats={n_repeats} -- returning the EXISTING {len(rows)}-draw checkpoint "
                f"UNCHANGED (not recomputed/extended). Delete {ckpt} and rerun if you want {n_repeats} "
                f"fresh draws for this cell.")
        else:
            log(f"  [{cond_id} fold {fold_id}] checkpoint already exists ({len(rows)} draws), skipping")
        return rows

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
    gen, driver_tr, driver_te = p4._fit_generator_and_drivers(
        "mnar_y", rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
    )
    rng = np.random.default_rng(MASK_SEED)
    syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
    syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
    final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
    final_mask_te_orig = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

    X_tr_masked = X_tr.copy()
    for col in genmod.ELIGIBLE_LAB_COLS:
        X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
    X_te_masked_orig = X_te.copy()
    for col in genmod.ELIGIBLE_LAB_COLS:
        X_te_masked_orig.loc[final_mask_te_orig[col].to_numpy(), col] = np.nan

    t0 = time.time()
    fit_result = selv2.fit_and_select_all_families(
        X_tr_masked, y_tr, g_tr, num_cols, cat_cols, seed=OUTER_SEED
    )
    fit_elapsed = time.time() - t0
    selected_family = fit_result.selected_family
    final_model = fit_result.final_models[selected_family]

    auroc_before, ap_before, brier_before = selv2.evaluate_frozen_model(final_model, X_te_masked_orig, y_te)

    smoke_caveat = (
        " [NOTE: under --smoke-test (n_outer=2) this compares against a DIFFERENT outer-fold "
        "partition than the original 5-fold run -- fold_id here is not the same test rows as the "
        "original's fold_id, so this is a weak/coincidental check, not the real correctness gate; "
        "run the real 5-fold split to actually exercise it]"
        if smoke_test else ""
    )

    if original_df is not None:
        orig_row = original_df[(original_df["rate"] == rate) & (original_df["or_value"] == or_val)
                                & (original_df["fold_id"] == fold_id)]
        if len(orig_row) == 1:
            orig_before = float(orig_row.iloc[0]["auroc_before"])
            drift = abs(auroc_before - orig_before)
            threshold = DRIFT_WARN_THRESHOLD_XGB if selected_family == "xgb" else DRIFT_WARN_THRESHOLD_OTHER
            if drift > threshold:
                log(f"  [{cond_id} fold {fold_id}] WARNING: refit auroc_before={auroc_before:.6f} vs "
                    f"original={orig_before:.6f} (drift={drift:.6f}) EXCEEDS the {selected_family} "
                    f"tolerance ({threshold}) -- investigate before trusting this cell's repeated draws."
                    f"{smoke_caveat}")
            elif drift > DRIFT_INFO_THRESHOLD:
                log(f"  [{cond_id} fold {fold_id}] INFO: refit auroc_before drift={drift:.6f} vs original "
                    f"(within the expected {selected_family} cross-environment noise band){smoke_caveat}")
            if orig_row.iloc[0]["selected_family"] != selected_family:
                log(f"  [{cond_id} fold {fold_id}] WARNING: refit selected_family={selected_family} "
                    f"differs from original's {orig_row.iloc[0]['selected_family']} -- this would mean "
                    f"the repeated draws below are NOT evaluating the same frozen model as the original "
                    f"Phase 6 result. Investigate before trusting this cell.")
        else:
            log(f"  [{cond_id} fold {fold_id}] NOTE: no matching row found in the original "
                f"phase6_permutation.csv for cross-check (expected exactly 1, found {len(orig_row)})")

    rows: List[Dict[str, Any]] = []
    for draw_id in range(n_repeats):
        perm_seed_entropy = _perm_seed_entropy(fold_id, rate, or_val, draw_id)
        perm_rng = np.random.default_rng(perm_seed_entropy)
        perm_idx = perm_rng.permutation(len(syn_mask_te))
        syn_mask_te_permuted = syn_mask_te.iloc[perm_idx].reset_index(drop=True)
        final_mask_te_permuted = genmod.compute_final_mask(nat_mask_te, syn_mask_te_permuted)
        X_te_masked_permuted = X_te.copy()
        for col in genmod.ELIGIBLE_LAB_COLS:
            X_te_masked_permuted.loc[final_mask_te_permuted[col].to_numpy(), col] = np.nan

        auroc_after, ap_after, brier_after = selv2.evaluate_frozen_model(final_model, X_te_masked_permuted, y_te)

        row = dict(
            condition_id=cond_id, rate=rate, or_value=or_val, fold_id=fold_id, draw_id=draw_id,
            selected_family=selected_family,
            auroc_before=auroc_before, ap_before=ap_before, brier_before=brier_before,
            auroc_after=auroc_after, ap_after=ap_after, brier_after=brier_after,
            delta_auroc=auroc_after - auroc_before,
            perm_idx_hash=hashlib.sha256(perm_idx.tobytes()).hexdigest(),
        )
        rows.append(row)

    if original_df is not None:
        orig_row = original_df[(original_df["rate"] == rate) & (original_df["or_value"] == or_val)
                                & (original_df["fold_id"] == fold_id)]
        if len(orig_row) == 1:
            orig_after = float(orig_row.iloc[0]["auroc_after"])
            draw0_after = rows[0]["auroc_after"]
            drift = abs(draw0_after - orig_after)
            threshold = DRIFT_WARN_THRESHOLD_XGB if selected_family == "xgb" else DRIFT_WARN_THRESHOLD_OTHER
            if drift > threshold:
                log(f"  [{cond_id} fold {fold_id}] WARNING: draw-0 auroc_after={draw0_after:.6f} vs "
                    f"original auroc_after={orig_after:.6f} (drift={drift:.6f}) EXCEEDS the {selected_family} "
                    f"tolerance ({threshold}) -- draw 0 should reproduce the original permutation. "
                    f"Investigate before trusting ANY of this cell's repeated draws.{smoke_caveat}")
            else:
                log(f"  [{cond_id} fold {fold_id}] draw-0 cross-check OK (drift={drift:.6f}, "
                    f"within {selected_family} tolerance {threshold}){smoke_caveat}")

    with open(ckpt, "wb") as f:
        pickle.dump(rows, f)
    deltas = np.array([r["delta_auroc"] for r in rows])
    log(f"  [{cond_id} fold {fold_id}] selected={selected_family} auroc_before={auroc_before:.4f} "
        f"n_repeats={n_repeats} mean_delta={deltas.mean():+.4f} sd_delta={deltas.std(ddof=1):.4f} "
        f"({fit_elapsed:.1f}s fit + {n_repeats} cheap rescore passes)")
    return rows


def _verify_no_duplicate_permutations(all_rows: List[Dict[str, Any]], log) -> None:
    """Confirms no two draws in a cell produced byte-identical permutation arrays."""
    df = pd.DataFrame(all_rows)
    n_dupe_groups = 0
    for (rate, orv, fid), sub in df.groupby(["rate", "or_value", "fold_id"]):
        n_dupes = sub["perm_idx_hash"].duplicated().sum()
        if n_dupes > 0:
            n_dupe_groups += 1
            log(f"  WARNING: cell (rate={rate}, OR={orv}, fold={fid}) has {n_dupes} duplicate "
                f"permutation-hash collisions among its {len(sub)} draws -- investigate seed derivation.")
    if n_dupe_groups == 0:
        log(f"  Verified: 0 duplicate permutation-hash collisions across all {len(df)} draws "
            f"({df.groupby(['rate','or_value','fold_id']).ngroups} cells).")


def run_all_repeats(n_repeats: int = 30, smoke_test: bool = False,
                     data_path_arg: Optional[str] = None,
                     original_permutation_csv_arg: Optional[str] = None):
    """Runs the repeated permutation-and-rescore passes across the rate/OR/fold grid, checkpointed per cell."""
    results_dir = Path("results/phase6_section_b_repeats_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = p4.make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    n_repeats_eff = 5 if smoke_test else n_repeats
    log(f"=== Phase 6 Section B repeated-draws driver start "
        f"(smoke_test={smoke_test}, n_outer={n_outer}, n_repeats={n_repeats_eff}) ===")

    original_df = _resolve_original_permutation_csv(original_permutation_csv_arg, log)

    data_path = p4._resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    from sklearn.model_selection import StratifiedGroupKFold
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    fold_splits = list(enumerate(outer_splitter.split(X, y, groups), start=1))

    rates = [0.30] if smoke_test else PERMUTATION_RATES
    ors = {2.0: PERMUTATION_ORS[2.0]} if smoke_test else PERMUTATION_ORS
    log(f"Cells: rates={rates}, ORs={list(ors.keys())}, {n_outer} folds each, {n_repeats_eff} repeats/cell")

    all_rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for rate in rates:
        for or_val, theta in ors.items():
            for fold_id, (tr_idx, te_idx) in fold_splits:
                rows = run_repeats_for_cell(
                    df, X, y, groups, num_cols, cat_cols, fold_id, tr_idx, te_idx,
                    rate, or_val, theta, n_repeats_eff, results_dir, log, original_df,
                    smoke_test=smoke_test,
                )
                all_rows.extend(rows)
    log(f"All cells complete in {time.time() - t_start:.1f}s")

    _verify_no_duplicate_permutations(all_rows, log)

    full_df = pd.DataFrame(all_rows)
    out_full = results_dir / "phase8_section_b_repeats_full.csv"
    full_df.to_csv(out_full, index=False)
    log(f"Wrote {out_full} ({len(full_df)} rows = "
        f"{full_df.groupby(['rate','or_value','fold_id']).ngroups} cells x {n_repeats_eff} draws)")

    summary_rows = []
    for (rate, orv, fid), sub in full_df.groupby(["rate", "or_value", "fold_id"]):
        summary_rows.append(dict(
            rate=rate, or_value=orv, fold_id=fid, selected_family=sub["selected_family"].iloc[0],
            auroc_before=sub["auroc_before"].iloc[0], n_repeats=len(sub),
            mean_delta_auroc=sub["delta_auroc"].mean(), sd_delta_auroc=sub["delta_auroc"].std(ddof=1),
            min_delta_auroc=sub["delta_auroc"].min(), max_delta_auroc=sub["delta_auroc"].max(),
        ))
    summary_df = pd.DataFrame(summary_rows)
    out_summary = results_dir / "phase8_section_b_repeats_per_fold_summary.csv"
    summary_df.to_csv(out_summary, index=False)
    log(f"Wrote {out_summary} ({len(summary_df)} rows)")

    with open(results_dir / "phase8_section_b_repeats_all.pkl", "wb") as f:
        pickle.dump({"full_df": full_df, "summary_df": summary_df}, f)
    log(f"Wrote {results_dir / 'phase8_section_b_repeats_all.pkl'}")
    log("=== Phase 6 Section B repeated-draws driver finished ===")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="2 outer folds, 1 (rate,OR) cell, 5 repeats/cell. Writes to "
                              "results/phase6_section_b_repeats_smoke/.")
    parser.add_argument("--n-repeats", type=int, default=30,
                         help="Number of independent permutation draws per (rate,OR,fold) cell "
                              "(default 30; ignored under --smoke-test, which always uses 5).")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv.")
    parser.add_argument("--original-permutation-csv", default=None,
                         help="Explicit path to the ORIGINAL run_phase6_rq4.py's phase6_permutation.csv, "
                              "used only for the draw-0 correctness cross-check. If omitted, tries "
                              "../PHASE_6_RQ4/results/phase6/phase6_permutation.csv relative to this "
                              "script. Missing this file is non-fatal (the cross-check is just skipped).")
    args = parser.parse_args()
    run_all_repeats(n_repeats=args.n_repeats, smoke_test=args.smoke_test,
                     data_path_arg=args.data_path,
                     original_permutation_csv_arg=args.original_permutation_csv)
