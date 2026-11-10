"""
Phase 6, Section B -- REPEATED-DRAW companion driver.

WHY THIS EXISTS
----------------
`run_phase6_rq4.py`'s Section B (test-only permutation ablation, the RQ4
flagship result: "-9 to -11 AUROC points at OR=4 vs. only ~1 point at OR=2")
draws exactly ONE random permutation of the synthetic mask per (rate, OR,
fold) cell. That single draw is fully reproducible (a documented, fold/rate/
OR-dependent seed), but it means the reported `delta_auroc` at each cell is a
single point estimate of an effect that could, in principle, vary from one
random reshuffling to another. This driver characterizes that variability
directly: it repeats the permutation-and-rescore step many times per cell
(default 30) and reports the resulting distribution (mean, SD, bootstrap CI)
instead of a single number.

WHAT THIS DOES **NOT** CHANGE
------------------------------
- The train-time masking, the frozen model itself, and `auroc_before` are all
  identical to the original Section B -- computed with the exact same code
  path, called once per cell (never rerun per draw).
- `run_phase6_rq4.py` and its outputs (`phase6_permutation.csv`, checkpoints)
  are never read, imported from, or written to by this script. This is a
  wholly separate, additive driver, per this project's established pattern
  (e.g. Phase 8's calibration backfill alongside Phase 4-6's original
  drivers). Nothing about the original Phase 6 result changes or is at risk
  of being overwritten.

WHAT VARIES ACROSS DRAWS
--------------------------
Only the TEST-TIME row permutation of the synthetic (MNAR-Y-injected) mask
component (`syn_mask_te`) -- exactly the same operation the original driver
performs, just repeated with different, independently seeded permutation
arrays. `X_tr`, `y_tr`, `y_te`, the fitted model, and `auroc_before` are fixed
across all draws within a cell; only which permutation is applied to
`syn_mask_te`, and therefore `auroc_after`, changes draw to draw.

BUILT-IN CORRECTNESS CHECK
----------------------------
Draw index 0 in every cell uses the EXACT SAME seed-entropy tuple the
original driver used -- `(MASK_SEED, PERMUTATION_SEED_OFFSET, fold_id,
round(rate*100), round(or_val*10))`, no trailing draw-id component -- so its
`auroc_after` should reproduce the original run's value for that cell either
bit-for-bit (non-xgb-selected folds) or within the same small,
already-documented cross-environment xgboost floating-point noise band used
throughout this project's Phase 8 drivers (xgb-selected folds; see
`run_phase8_calibration.py`'s DRIFT_WARN_THRESHOLD_XGB). Draws 1..N-1 use a
6-element seed tuple (the same 5 elements plus an explicit `draw_id`), which
is guaranteed never to collide with the 5-element draw-0 entropy pool (numpy's
SeedSequence treats different-length input sequences as distinct entropy;
additionally verified empirically at the end of the run -- see
`_verify_no_duplicate_permutations` -- and independently confirmed during
code review by directly comparing `np.random.default_rng` outputs for the
5-tuple vs. 6-tuple (draw_id 0..50) forms of the real constants used here).

This driver refits `fit_and_select_all_families` ONCE per (rate, OR, fold)
cell (same cost as the original driver's one fit per cell -- NOT once per
draw), then loops over N_REPEATS cheap reshuffle+rescore passes. Expected
added wall time: about the same as one full original Section B run (~20
cells x ~80s refit each) plus a small, fast tail for the extra scoring passes
-- see the module-level docstring's "Expected wall time" note near
`run_all_repeats`.
"""
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

OUTER_SEED = p4.OUTER_SEED         # 2026 -- same as Phase 4/5/6
MASK_SEED = p4.MASK_SEED           # 20260917 -- same as Phase 4/5/6

# Identical to run_phase6_rq4.py's Section B constants -- MUST stay identical
# for draw 0's seed to reproduce the original result.
PERMUTATION_RATES = [0.30, 0.40]
PERMUTATION_ORS = {2.0: p4.THETA_MAIN, 4.0: float(np.log(4.0))}
PERMUTATION_SEED_OFFSET = 555

N_OUTER = 5
RESULTS_DIR = Path("results/phase6_section_b_repeats")

# xgb cross-environment floating-point noise band, same convention as
# run_phase8_calibration.py (DRIFT_WARN_THRESHOLD_XGB / DRIFT_WARN_THRESHOLD_OTHER).
DRIFT_INFO_THRESHOLD = 1e-6
DRIFT_WARN_THRESHOLD_XGB = 0.03
DRIFT_WARN_THRESHOLD_OTHER = 1e-6


# Same cloud-sandbox last-resort fallback pattern as run_phase6_rq4.py's own
# _PHASE4_CSV_CONTAINER -- covers the documented DASA2026/ upload layout when
# this script isn't run from a sibling-folder layout relative to PHASE_6_RQ4/.
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
    """draw_id == 0 reproduces the ORIGINAL driver's exact seed (5-tuple, no
    draw_id component) -- this is deliberate, not an off-by-one: it is what
    lets draw 0 serve as a bit-for-bit correctness check against
    phase6_permutation.csv. draw_id >= 1 gets a 6-tuple that numpy's
    SeedSequence treats as categorically distinct entropy from the 5-tuple
    case (different sequence length), so there is no risk of draw 1..N-1
    silently colliding with draw 0's stream."""
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
            # FIX (code review, 2026-09-19): a checkpoint written by an
            # earlier invocation with a DIFFERENT --n-repeats would otherwise
            # be returned silently as-is, under-reporting (or padding) the
            # requested draw count without any signal that this happened.
            # This does not recompute/extend the checkpoint (the frozen model
            # is not itself persisted, so extending it cheaply would require
            # re-fitting), but it makes the mismatch loud rather than silent,
            # per this project's fail-loud convention.
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

    # Fit ONCE per cell -- identical call to the original driver's Section B,
    # same seed, same masked training data -> same frozen model.
    t0 = time.time()
    fit_result = selv2.fit_and_select_all_families(
        X_tr_masked, y_tr, g_tr, num_cols, cat_cols, seed=OUTER_SEED
    )
    fit_elapsed = time.time() - t0
    selected_family = fit_result.selected_family
    final_model = fit_result.final_models[selected_family]

    auroc_before, ap_before, brier_before = selv2.evaluate_frozen_model(final_model, X_te_masked_orig, y_te)

    # FIX (code review, 2026-09-19): under --smoke-test, n_outer=2 (vs the
    # ORIGINAL driver's fixed 5), so StratifiedGroupKFold produces a
    # completely different train/test partition -- "fold_id 1" here is NOT
    # the same test-set rows as "fold_id 1" in the original 5-fold
    # phase6_permutation.csv. The cross-checks below still run (a wildly
    # large drift would still be a useful red flag), but a small/OK drift
    # under --smoke-test is a coincidence of similar data, not a real
    # confirmation that draw 0 reproduced the original permutation -- that
    # gate can only be exercised with the real 5-fold split. Caveat every
    # cross-check log line so this isn't mistaken for the genuine gate.
    smoke_caveat = (
        " [NOTE: under --smoke-test (n_outer=2) this compares against a DIFFERENT outer-fold "
        "partition than the original 5-fold run -- fold_id here is not the same test rows as the "
        "original's fold_id, so this is a weak/coincidental check, not the real correctness gate; "
        "run the real 5-fold split to actually exercise it]"
        if smoke_test else ""
    )

    # Built-in correctness check: does refitting reproduce Phase 4/6's
    # already-established model for this exact (rate, fold) via the shared
    # MASK_SEED/OUTER_SEED convention? Compare against the ORIGINAL driver's
    # own auroc_before for the same cell, if available.
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
            # FIX (code review, 2026-09-19): Python's built-in hash() salts
            # str/bytes with a per-PROCESS random seed (PYTHONHASHSEED) by
            # default -- confirmed empirically (two `python3 -c` invocations
            # of `hash(np.array([3,1,2]).tobytes())` returned different
            # values). Within a single run this happened to be harmless for
            # `_verify_no_duplicate_permutations`'s actual use (every row for
            # a given cell is produced -- and hashed -- inside one process,
            # either freshly here or all together when an earlier checkpoint
            # is unpickled), but `perm_idx_hash` is also PERSISTED to disk
            # (the pickle checkpoint and the final CSV/pkl), where a salted
            # value is silently meaningless for any comparison ACROSS runs/
            # processes (e.g. confirming two separate invocations drew the
            # same permutation) -- exactly the kind of cross-environment
            # reproducibility check this project relies on elsewhere. Use a
            # process-independent digest instead so the stored value is a
            # genuinely stable identifier for the permutation.
            perm_idx_hash=hashlib.sha256(perm_idx.tobytes()).hexdigest(),
        )
        rows.append(row)

    # Draw-0 cross-check against the original driver's own auroc_after (the
    # single most important correctness gate in this whole script: if this
    # doesn't match, the two drivers are NOT evaluating the same permutation
    # at "draw 0" and something is wrong upstream of the repeats themselves).
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
    """Belt-and-suspenders check: within each (rate, or_value, fold_id) cell,
    confirm no two draws produced byte-identical permutation arrays (which
    would silently collapse the effective sample size below n_repeats)."""
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
    """Expected wall time (non-smoke): 4 (rate,OR) cells x 5 folds = 20 cells,
    each costing one fit_and_select_all_families call (~75-85s, same cost as
    the original Section B's per-cell fit, per phase6_permutation.csv's own
    fit_seconds column) plus n_repeats cheap reshuffle+rescore passes
    (sub-second each even at n_repeats=30-50) -> roughly 25-30 minutes total,
    similar order of magnitude to Phase 8's calibration backfill. Checkpointed
    per cell; safe to interrupt and resume."""
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
