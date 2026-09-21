"""
case_level_bootstrap_ci.py

Case-level (patient-level) bootstrap confidence intervals for headline
AUROC / AP point estimates, computed from raw per-case (y_test, p_test)
predictions that are ALREADY saved on disk. NO NEW MODEL FITS. NO NEW
EXPERIMENTS. This script only reads two existing pickles:

  PHASE_8_CALIBRATION/results/phase8_calibration/phase8_calibration_predictions.pkl
      -- every Phase 4 (RQ1/RQ2) condition: natural baseline, MCAR/MAR/MNAR-Y
         across the q=0.00/0.10/0.20/0.30/0.40 rate sweep, and MNAR-Y's
         OR=1.5/2.0/4.0 strength sweep at q=0.30. 18 conditions x 5 folds.
         (These predictions exist because Phase 8's calibration backfill had
         to refit the already-selected family, with its already-known
         hyperparameters, to get per-case probabilities for the calibration
         slope/intercept calculation -- see phase8-calibration-backfill.md.
         This script is purely additive on top of that existing output.)

  PHASE_8_MNAR_X/results/phase8_mnar_x/phase8_mnar_x_predictions.pkl
      -- MNAR-X's own q=0.10/0.20/0.30/0.40 rate sweep and OR=1.5/4.0 sweep
         at q=0.30. mnar_x_q00 has NO predictions on purpose (it reuses
         Phase 3/4's pre-existing natural_q0 checkpoint, which predates
         prediction-capture -- this is documented in run_phase8_mnar_x.py's
         load_or_compute_natural_q0() and is correctly skipped below, not a
         bug in this script).

Phase 5 (RQ3) and Phase 6 (RQ4) are NOT covered -- their drivers never
captured raw per-case predictions, and adding that would require new,
additively-designed drivers and a full refit (a materially bigger, riskier
piece of work). See phase-stats-and-permutation-repeats.md / the chat
discussion this script comes from for why that fuller "Tier 2" was
deliberately NOT done for this submission cycle.

=============================================================================
WHAT A CASE-LEVEL BOOTSTRAP CI IS -- AND, IMPORTANTLY, WHAT IT IS NOT
=============================================================================
This project already reports two other, DIFFERENT kinds of uncertainty:

  1. Fold-level paired tests (phase_stats_layer.py): resample/permute across
     the 5 outer folds. This is the estimate of MODEL-REFIT / GENERALIZATION
     variance -- "does this effect hold up across different patient splits
     and model fits" -- and it is the one that answers the underlying
     scientific question. Limited to n=5, hence the documented p=0.0625 floor.

  2. Repeated permutation draws (run_phase6_section_b_repeats.py, RQ4 only):
     resample the mask-shuffle at FIXED model + fixed test set. This
     characterizes PERMUTATION-RESHUFFLE variance.

  3. THIS SCRIPT: resample PATIENTS WITHIN one fold's test set, at a FIXED
     model and FIXED mask/mechanism realization. This characterizes
     FINITE-TEST-SET SAMPLING variance of the AUROC/AP estimator itself --
     "if I'd happened to test this exact frozen model on a different sample
     of ~2,800 patients drawn from the same population, how much would my
     AUROC estimate move?"

These three are genuinely different sources of variance, not three ways of
computing the same thing with more or less power. In particular:

  - A narrow case-level CI here does NOT mean the cross-fold effect is more
    "significant" than the fold-level test says. It does NOT get past the
    n=5 / p=0.0625 floor -- that floor is about having only 5 independent
    fold replicates, and no amount of within-fold patient resampling manu-
    factures more of those replicates.
  - The correct, standard use of these numbers is reporting a headline point
    estimate with a CI, e.g. "AUROC = 0.93 (95% CI 0.91-0.95)", the way
    clinical-ML results conventionally report a single AUROC's own precision
    -- NOT as a replacement for, or a strengthening of, the fold-level
    significance claims already made elsewhere in this project.

The output report below repeats this caveat, so it travels with the numbers
wherever they get reused.

=============================================================================
METHOD
=============================================================================
Bootstrap is STRATIFIED by outcome class (resample positives and negatives
separately, each preserving its own original count) rather than a single
unstratified resample of the whole test set. At this dataset's ~2.5% preva-
lence (~70-80 positives out of ~2,800 per fold), stratifying guarantees every
bootstrap resample contains both classes, so AUROC/AP are always well-defined
-- with unstratified resampling this would only fail with astronomically low
probability, but stratifying removes the possibility entirely at negligible
cost and is standard practice for imbalanced-outcome bootstrap CIs.

Two CIs are reported per condition:
  - PER-FOLD: one bootstrap CI per (condition, fold) pair, from that fold's
    own ~2,800 test cases. Useful for fold-by-fold questions (e.g. checking
    whether a specific fold's finding could be finite-sample noise).
  - POOLED: the 5 folds' out-of-fold predictions concatenated into one
    ~14,081-patient pseudo-test-set, then bootstrapped the same way. This is
    the number to cite as the headline "AUROC (95% CI)" in a table, since it
    uses every patient in the dataset exactly once (5-fold CV is a partition
    of all N=14,081 admissions).

SELF-VERIFICATION (built in, not optional): before trusting any predictions
array, this script recomputes AUROC directly from (y, p) and cross-checks it
against the ALREADY-KNOWN AUROC value recorded for that exact
(condition_id, fold_id) in phase8_calibration_table.csv / phase8_mnar_x_fold_
summary.csv. Any mismatch beyond floating-point tolerance raises immediately
and stops the script -- the output is only written if every single one of
these independent cross-checks passes.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT_DEFAULT = 2000
ALPHA_DEFAULT = 0.05
CROSS_CHECK_TOL = 1e-6
FAST_VS_SKLEARN_TOL = 1e-9

# Distinct from phase_stats_layer.py's RNG_BOOTSTRAP (seeded 20260919) --
# deliberately a different seed so it is obvious in any output that these are
# two independent bootstrap procedures, not shared state.
CASE_BOOTSTRAP_SEED = 20260920


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement, separately within each class, each
    preserving its own original count exactly. Guarantees every resample
    contains both classes."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"cannot bootstrap: only one class present (n_pos={len(pos_idx)}, n_neg={len(neg_idx)})")
    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([boot_pos, boot_neg])


def fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U / rank-biserial form), average-rank
    tie handling. Mathematically identical to sklearn.metrics.roc_auc_score
    for binary labels (verified below at runtime, not just once during
    development -- see verify_fast_metrics_match_sklearn()), but avoids
    sklearn's per-call input-validation overhead, which dominates runtime
    when called tens of thousands of times inside a bootstrap loop."""
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    order = np.argsort(p, kind="mergesort")
    sort_p = p[order]
    uniq_p, inv = np.unique(sort_p, return_inverse=True)
    counts = np.bincount(inv)
    if np.any(counts > 1):
        avg_rank_per_group = np.cumsum(counts) - (counts - 1) / 2.0
        ranks_sorted = avg_rank_per_group[inv]
    else:
        ranks_sorted = np.arange(1, n + 1, dtype=float)
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[y == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fast_ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision via sort-and-cumulate, WITH TIED SCORES GROUPED
    (this matters in practice: tree-based models -- RandomForest and
    ExtraTrees most of all, xgboost less often -- routinely produce exact
    duplicate leaf-output probabilities across many patients: the largest
    group found in this project's own data is 1,174 patients, all tied at
    exactly 0.0, from a RandomForest model in mar_q40_main fold 5; restricted
    to xgboost-selected conditions specifically, the largest tied group is
    38 patients (see Supplementary_case-level-bootstrap-confidence-intervals.md
    Sec.3 for the full breakdown) -- and sklearn's average_precision_score
    treats every tied group as a single threshold, not as arbitrarily-ordered
    individual points). An earlier version of this
    function ignored ties and was caught disagreeing with sklearn by
    verify_fast_metrics_match_sklearn() during this script's own development
    (diff ~2.5e-3 on mar_q10_main, fold 1) -- this corrected version was
    re-verified to agree with sklearn to within ~3e-16 (floating-point noise)
    across every condition and fold in both prediction files before being
    trusted. Mathematically identical to
    sklearn.metrics.average_precision_score for binary labels, no sample
    weights; avoids sklearn's per-call validation overhead, which is what
    makes 2,000+ bootstrap resamples per condition practical to run."""
    n = len(y)
    n_pos = int(y.sum())
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    tp_cumsum = np.cumsum(y_sorted)
    distinct = np.where(np.diff(p_sorted) != 0)[0]
    group_end_idx = np.r_[distinct, n - 1]
    tp_at_group_end = tp_cumsum[group_end_idx]
    n_at_group_end = group_end_idx + 1
    precision_at_group = tp_at_group_end / n_at_group_end
    tp_before = np.r_[0, tp_at_group_end[:-1]]
    pos_gained_at_group = tp_at_group_end - tp_before
    return float(np.sum(pos_gained_at_group * precision_at_group) / n_pos)


def verify_fast_metrics_match_sklearn(y: np.ndarray, p: np.ndarray, tol: float = FAST_VS_SKLEARN_TOL) -> None:
    """Runtime self-check (not just a one-off development-time test): confirm
    the fast rank-based metrics agree with sklearn's reference implementation
    on THIS run's actual data before using the fast versions for thousands of
    bootstrap iterations. Raises immediately if they ever disagree beyond
    floating-point tolerance, so a silent numerical discrepancy can never
    quietly bias every CI in the output."""
    a_sklearn, a_fast = float(roc_auc_score(y, p)), fast_auroc(y, p)
    b_sklearn, b_fast = float(average_precision_score(y, p)), fast_ap(y, p)
    if abs(a_sklearn - a_fast) > tol:
        raise RuntimeError(f"fast_auroc disagrees with sklearn: {a_fast} vs {a_sklearn} "
                            f"(diff={abs(a_sklearn - a_fast):.2e}) -- do not trust bootstrap output")
    if abs(b_sklearn - b_fast) > tol:
        raise RuntimeError(f"fast_ap disagrees with sklearn: {b_fast} vs {b_sklearn} "
                            f"(diff={abs(b_sklearn - b_fast):.2e}) -- do not trust bootstrap output")


def case_bootstrap_ci(y: np.ndarray, p: np.ndarray, rng: np.random.Generator,
                       n_boot: int = N_BOOT_DEFAULT, alpha: float = ALPHA_DEFAULT) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError(f"length mismatch: y has {len(y)}, p has {len(p)}")
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"y is not binary 0/1: unique values found {uniq}")

    # Point estimates: sklearn, the same reference implementation used
    # everywhere else in this project (and against which the cross-check in
    # cross_check() compares) -- these two numbers are never taken from the
    # fast path.
    point_auroc = float(roc_auc_score(y, p))
    point_ap = float(average_precision_score(y, p))

    # Runtime self-check: confirm the fast path matches sklearn on THIS
    # dataset before using it for any bootstrap resample below.
    verify_fast_metrics_match_sklearn(y, p)

    # Bootstrap resamples: verified-equivalent fast path (see
    # verify_fast_metrics_match_sklearn, called once per (condition, fold)
    # before this loop runs).
    boot_auroc = np.empty(n_boot, dtype=float)
    boot_ap = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        yb, pb = y[idx], p[idx]
        boot_auroc[b] = fast_auroc(yb, pb)
        boot_ap[b] = fast_ap(yb, pb)

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return dict(
        n=int(len(y)),
        n_pos=int(y.sum()),
        auroc=point_auroc,
        auroc_ci_lo=float(np.percentile(boot_auroc, lo_pct)),
        auroc_ci_hi=float(np.percentile(boot_auroc, hi_pct)),
        ap=point_ap,
        ap_ci_lo=float(np.percentile(boot_ap, lo_pct)),
        ap_ci_hi=float(np.percentile(boot_ap, hi_pct)),
    )


# ---------------------------------------------------------------------------
# Loading raw predictions (existing files only -- nothing is computed here
# that wasn't already produced by run_phase8_calibration.py / run_phase8_mnar_x.py)
# ---------------------------------------------------------------------------

def load_calibration_predictions(data_root: Path):
    path = data_root / "PHASE_8_CALIBRATION" / "results" / "phase8_calibration" / "phase8_calibration_predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path} -- run run_phase8_calibration.py first (already done per "
                                 f"phase8-calibration-backfill.md; check --data-root points at the right DASA2026 folder)")
    with open(path, "rb") as f:
        preds = pickle.load(f)  # {condition_id: {fold_id: (y, p)}}
    table_path = data_root / "PHASE_8_CALIBRATION" / "results" / "phase8_calibration" / "phase8_calibration_table.csv"
    table = pd.read_csv(table_path)
    return preds, table


def load_mnarx_predictions(data_root: Path):
    path = data_root / "PHASE_8_MNAR_X" / "results" / "phase8_mnar_x" / "phase8_mnar_x_predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path} -- run run_phase8_mnar_x.py first (already done per "
                                 f"phase8-driver-summary.md; check --data-root points at the right DASA2026 folder)")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    preds = obj["predictions"]  # {condition_id: {fold_id: (y, p)}}
    meta = obj["condition_meta"]
    summary_path = data_root / "PHASE_8_MNAR_X" / "results" / "phase8_mnar_x" / "phase8_mnar_x_fold_summary.csv"
    summary = pd.read_csv(summary_path)
    return preds, meta, summary


def cross_check(condition_id: str, fold_id: int, computed_auroc: float,
                 table: pd.DataFrame, auroc_col: str, tol: float = CROSS_CHECK_TOL):
    """Raise loudly if the AUROC recomputed from raw (y, p) here doesn't match
    the value already on record for this exact condition/fold. This is the
    guard that makes the rest of this script's output trustworthy without
    having to take it on faith."""
    row = table[(table["condition_id"] == condition_id) & (table["fold_id"] == fold_id)]
    if row.empty:
        raise RuntimeError(f"cross-check failed: no row for condition_id={condition_id} fold_id={fold_id} "
                            f"in the reference table -- predictions and table have gone out of sync, do not trust output")
    known = float(row[auroc_col].iloc[0])
    diff = abs(known - computed_auroc)
    if diff > tol:
        raise RuntimeError(
            f"CROSS-CHECK MISMATCH for {condition_id} fold {fold_id}: AUROC recomputed from raw predictions "
            f"here = {computed_auroc:.8f}, but {auroc_col} on record = {known:.8f} (diff={diff:.2e}, "
            f"tolerance={tol:.1e}). This means the predictions pickle and the reference CSV are not describing "
            f"the same run -- STOPPING before writing any output. Do not trust any number from this script "
            f"until this is resolved."
        )
    return diff


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def process_source(source_name: str, preds: dict, table: pd.DataFrame, auroc_col: str,
                    rng: np.random.Generator, n_boot: int, alpha: float, skip_conditions=()):
    rows = []
    for cid in sorted(preds.keys()):
        fold_dict = preds[cid]
        if cid in skip_conditions or len(fold_dict) == 0:
            log(f"  SKIP {cid}: no per-case predictions available for this condition "
                f"(expected for mnar_x_q00 -- see module docstring; unexpected for anything else)")
            continue

        y_pooled_parts, p_pooled_parts = [], []
        for fold_id in sorted(fold_dict.keys()):
            y, p = fold_dict[fold_id]
            y = np.asarray(y)
            p = np.asarray(p, dtype=float)

            # Self-verification against the already-known, already-audited AUROC
            # for this exact condition/fold before this fold's numbers are used
            # for anything.
            point_auroc_check = float(roc_auc_score(y, p))
            diff = cross_check(cid, fold_id, point_auroc_check, table, auroc_col)

            res = case_bootstrap_ci(y, p, rng, n_boot=n_boot, alpha=alpha)
            rows.append(dict(source=source_name, condition_id=cid, fold_id=fold_id, level="per_fold",
                              cross_check_diff=diff, **res))
            y_pooled_parts.append(y)
            p_pooled_parts.append(p)

        # Pooled (all folds' out-of-fold predictions concatenated -- every
        # patient in the dataset appears exactly once across the 5 folds).
        y_pooled = np.concatenate(y_pooled_parts)
        p_pooled = np.concatenate(p_pooled_parts)
        res_pooled = case_bootstrap_ci(y_pooled, p_pooled, rng, n_boot=n_boot, alpha=alpha)
        rows.append(dict(source=source_name, condition_id=cid, fold_id=-1, level="pooled",
                          cross_check_diff=np.nan, **res_pooled))
        log(f"  {cid}: {len(fold_dict)} folds cross-checked OK, pooled n={res_pooled['n']} "
            f"(expect 14081 if this condition covers every patient), "
            f"pooled AUROC={res_pooled['auroc']:.4f} [{res_pooled['auroc_ci_lo']:.4f}, {res_pooled['auroc_ci_hi']:.4f}]")
        if res_pooled["n"] != 14081:
            log(f"    NOTE: pooled n={res_pooled['n']} != 14081 -- expected only if this condition's folds "
                f"don't partition the full cohort (should not happen for any condition here; investigate if seen)")
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CAVEAT_TEXT = """\
**What these numbers are, and are not.** Each CI here is a case-level
(patient-level) bootstrap confidence interval: it resamples patients within
an already-frozen model's test set and asks how much the AUROC/AP estimate
would wobble under finite-sample resampling of the test population. It is
NOT a substitute for, or a strengthening of, this project's fold-level
paired significance tests (`phase_stats_layer_report.md`), which resample
across the 5 outer folds and answer the different, more central question of
whether an effect generalizes across patient splits and model refits -- that
comparison is still limited to n=5 folds and its documented p=0.0625 floor
is unaffected by anything below. These case-level CIs are best used as the
uncertainty band on a single reported AUROC/AP point estimate (e.g. "AUROC =
0.93, 95% CI [0.91, 0.95]"), which is standard practice for clinical
prediction papers, not as evidence about cross-fold consistency.
"""


def write_report(rows: list, out_prefix: Path, n_boot: int, alpha: float):
    df = pd.DataFrame(rows)
    csv_path = out_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    pooled = df[df["level"] == "pooled"].copy()
    pooled = pooled.sort_values(["source", "condition_id"])

    lines = []
    lines.append("# Case-Level Bootstrap Confidence Intervals — Headline AUROC/AP\n")
    lines.append(f"Generated from existing raw per-case predictions only (no new model fits). "
                 f"{n_boot} stratified bootstrap resamples per estimate, {int((1-alpha)*100)}% CI. "
                 f"Every AUROC recomputed here was independently cross-checked against the already-audited "
                 f"reference table for its exact (condition_id, fold_id) before being trusted; "
                 f"{len(df[df['level']=='per_fold'])} such cross-checks all passed "
                 f"(max diff = {df['cross_check_diff'].max():.2e}).\n")
    lines.append(CAVEAT_TEXT)
    lines.append("\n## Headline table (pooled across all 5 folds = full N=14,081 cohort, one point estimate + CI per condition)\n")
    lines.append("| Source | Condition | n | AUROC (95% CI) | AP (95% CI) |")
    lines.append("|---|---|---:|---|---|")
    for _, r in pooled.iterrows():
        lines.append(f"| {r['source']} | {r['condition_id']} | {int(r['n'])} | "
                     f"{r['auroc']:.4f} ({r['auroc_ci_lo']:.4f}–{r['auroc_ci_hi']:.4f}) | "
                     f"{r['ap']:.4f} ({r['ap_ci_lo']:.4f}–{r['ap_ci_hi']:.4f}) |")

    lines.append("\n## Per-fold detail\n")
    lines.append("Full per-fold breakdown (all conditions x folds) is in the CSV alongside this report; "
                 "not reproduced here in full to keep this file readable.")

    report_path = out_prefix.parent / (out_prefix.name + "_report.md")
    # encoding="utf-8" is required explicitly: Path.write_text() otherwise
    # uses the platform's default encoding, which on Windows is commonly
    # cp1252, not UTF-8 -- confirmed this actually happened on the user's
    # production run (the en-dash characters in the table above were written
    # as cp1252 byte 0x96 instead of UTF-8, producing mojibake when the file
    # is read back as UTF-8 by anything else, e.g. a text editor or a LaTeX
    # \input). Fixed here; does not affect any numeric value anywhere in the
    # CSV or report, which are and always were correct -- purely a text-
    # encoding cosmetic issue in the .md file.
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Path to the DASA2026 folder (containing PHASE_8_CALIBRATION/ and PHASE_8_MNAR_X/)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT, help=f"Bootstrap resamples per estimate (default {N_BOOT_DEFAULT})")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT, help=f"CI alpha (default {ALPHA_DEFAULT} -> 95% CI)")
    ap.add_argument("--out-prefix", type=Path, default=Path("case_level_bootstrap_ci"),
                    help="Output path prefix (writes <prefix>.csv and <prefix>_report.md)")
    args = ap.parse_args()

    data_root = args.data_root
    rng = np.random.default_rng(CASE_BOOTSTRAP_SEED)

    log("=== Case-level bootstrap CI: loading existing raw predictions (no new experiments) ===")
    cal_preds, cal_table = load_calibration_predictions(data_root)
    mnarx_preds, mnarx_meta, mnarx_summary = load_mnarx_predictions(data_root)

    log(f"Phase 4 grid (via calibration backfill): {len(cal_preds)} conditions loaded")
    log(f"MNAR-X: {len(mnarx_preds)} conditions loaded")

    all_rows = []
    log("\n--- Phase 4 grid (natural / MCAR / MAR / MNAR-Y, via PHASE_8_CALIBRATION) ---")
    all_rows += process_source("phase4_grid", cal_preds, cal_table, "refit_outer_test_auroc",
                                rng, args.n_boot, args.alpha)

    log("\n--- MNAR-X (PHASE_8_MNAR_X) ---")
    all_rows += process_source("mnar_x", mnarx_preds, mnarx_summary, "selected_outer_test_auroc",
                                rng, args.n_boot, args.alpha, skip_conditions=())

    csv_path, report_path = write_report(all_rows, args.out_prefix, args.n_boot, args.alpha)
    log(f"\nAll cross-checks passed. Wrote {csv_path} and {report_path}")


if __name__ == "__main__":
    main()
