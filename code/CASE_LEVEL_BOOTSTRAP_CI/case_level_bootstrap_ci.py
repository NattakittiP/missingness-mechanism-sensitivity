"""Computes case-level (admission-level) bootstrap confidence intervals for headline AUROC/AP from already-saved per-case predictions; runs no new experiments."""

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

CASE_BOOTSTRAP_SEED = 20260920


def log(msg: str) -> None:
    print(msg, flush=True)



def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resamples with replacement, separately within each class, preserving each class's count."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"cannot bootstrap: only one class present (n_pos={len(pos_idx)}, n_neg={len(neg_idx)})")
    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    return np.concatenate([boot_pos, boot_neg])


def fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC, mathematically identical to sklearn's but faster for repeated bootstrap calls."""
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
    """Average precision via sort-and-cumulate with tied scores grouped, matching sklearn's average_precision_score."""
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
    """Runtime self-check confirming the fast metrics agree with sklearn's reference implementation on this run's data."""
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

    point_auroc = float(roc_auc_score(y, p))
    point_ap = float(average_precision_score(y, p))

    verify_fast_metrics_match_sklearn(y, p)

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



def load_calibration_predictions(data_root: Path):
    path = data_root / "PHASE_8_CALIBRATION" / "results" / "phase8_calibration" / "phase8_calibration_predictions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path} -- run run_phase8_calibration.py first (already done per "
                                 f"phase8-calibration-backfill.md; check --data-root points at the right DASA2026 folder)")
    with open(path, "rb") as f:
        preds = pickle.load(f)
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
    preds = obj["predictions"]
    meta = obj["condition_meta"]
    summary_path = data_root / "PHASE_8_MNAR_X" / "results" / "phase8_mnar_x" / "phase8_mnar_x_fold_summary.csv"
    summary = pd.read_csv(summary_path)
    return preds, meta, summary


def cross_check(condition_id: str, fold_id: int, computed_auroc: float,
                 table: pd.DataFrame, auroc_col: str, tol: float = CROSS_CHECK_TOL):
    """Raises if the AUROC recomputed from raw (y, p) doesn't match the already-recorded value for that condition/fold."""
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

            point_auroc_check = float(roc_auc_score(y, p))
            diff = cross_check(cid, fold_id, point_auroc_check, table, auroc_col)

            res = case_bootstrap_ci(y, p, rng, n_boot=n_boot, alpha=alpha)
            rows.append(dict(source=source_name, condition_id=cid, fold_id=fold_id, level="per_fold",
                              cross_check_diff=diff, **res))
            y_pooled_parts.append(y)
            p_pooled_parts.append(p)

        y_pooled = np.concatenate(y_pooled_parts)
        p_pooled = np.concatenate(p_pooled_parts)
        res_pooled = case_bootstrap_ci(y_pooled, p_pooled, rng, n_boot=n_boot, alpha=alpha)
        rows.append(dict(source=source_name, condition_id=cid, fold_id=-1, level="pooled",
                          cross_check_diff=np.nan, **res_pooled))
        log(f"  {cid}: {len(fold_dict)} folds cross-checked OK, pooled n={res_pooled['n']} "
            f"(expect 14081 if this condition covers every admission), "
            f"pooled AUROC={res_pooled['auroc']:.4f} [{res_pooled['auroc_ci_lo']:.4f}, {res_pooled['auroc_ci_hi']:.4f}]")
        if res_pooled["n"] != 14081:
            log(f"    NOTE: pooled n={res_pooled['n']} != 14081 -- expected only if this condition's folds "
                f"don't partition the full cohort (should not happen for any condition here; investigate if seen)")
    return rows



CAVEAT_TEXT = """\
**What these numbers are, and are not.** Each CI here is a case-level
(admission-level) bootstrap confidence interval: it resamples cases within
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
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path



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
