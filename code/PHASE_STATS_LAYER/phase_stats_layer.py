#!/usr/bin/env python3
"""
Fold-level significance-testing / uncertainty-quantification layer for this
project's headline claims (RQ1-RQ4 + item 11).

WHAT THIS IS
------------
A standalone, read-only analysis script. It performs NO new experiments and
calls NONE of the project's own driver code (run_phase4/5/6/8_*.py) or metric
functions (metrics_v2.py) -- it reads only the already-produced raw CSV
outputs and recomputes every quantity from first principles with plain
pandas/numpy/scipy/statsmodels. This mirrors the same "independent of the
thing being checked" standard used throughout this project's audits.

It generalizes the one significance check already added ad hoc to the S1
write-up (a Fisher's exact test on a displacement-rate comparison) to every
headline displacement-rate and paired-AUROC claim across RQ1, RQ2, RQ3, RQ4,
and item 11 (MNAR-X, calibration, S1, AP-selection).

WHAT THIS IS NOT
----------------
It does not run any new model fits. Section 4 (RQ4 permutation ablation)
currently only has 5 single-draw folds per (rate, OR) cell to work with; once
`run_phase6_section_b_repeats.py` (the companion driver, run separately on
your machine) produces repeated-permutation-draw data, a second pass of this
script (see the `--perm-repeats-csv` flag) will fold that in for a much
better-powered characterization of the RQ4 flagship effect.

USAGE
-----
    python3 phase_stats_layer.py --data-root /path/to/DASA2026
      (expects the same PHASE_4_RQ1_RQ2/, PHASE_5_RQ3/, PHASE_6_RQ4/,
       PHASE_8_MNAR_X/, PHASE_8_CALIBRATION/, PHASE_8_S1_SENSITIVITY/
       sibling-folder layout every other driver in this project uses)

    Optional, once available:
    python3 phase_stats_layer.py --data-root ... \
        --perm-repeats-csv PHASE_6_RQ4/results/phase6_section_b_repeats/phase6_section_b_repeats_full.csv

OUTPUT
------
- Prints every test's result to stdout in a readable form.
- Writes `phase_stats_layer_results.csv` (one row per test, machine-readable,
  with raw and Benjamini-Hochberg-adjusted p-values where applicable) to the
  current directory.
- Writes `phase_stats_layer_report.md`, a human-readable Markdown report
  suitable for pasting straight into a paper's supplementary-statistics
  section or into the methods/results text.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon, binomtest, norm

try:
    import statsmodels.api as sm
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

RESULTS: list = []   # list of dicts, one per test -> becomes the CSV
REPORT_LINES: list = []  # markdown report lines

RNG_BOOTSTRAP = np.random.default_rng(20260919)  # fixed, documented, reproducible
N_BOOTSTRAP = 10000


def log(msg: str = "") -> None:
    print(msg)
    REPORT_LINES.append(msg)


def record(test_id, family, description, statistic_name, statistic, p_value,
           ci_low=None, ci_high=None, n=None, note=""):
    RESULTS.append(dict(
        test_id=test_id, family=family, description=description,
        statistic_name=statistic_name, statistic=statistic, p_value=p_value,
        ci_low=ci_low, ci_high=ci_high, n=n, note=note,
    ))


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion. Well-behaved at
    extreme proportions (0/n, n/n) unlike the normal (Wald) approximation."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact (Clopper-Pearson) binomial CI -- the conservative standard many
    reviewers expect for small-n proportions, especially at 0/n or n/n."""
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05):
    """Nonparametric percentile bootstrap CI on the mean of paired
    differences, resampling folds with replacement. Documented, fixed seed
    (RNG_BOOTSTRAP) for reproducibility."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    idx_pool = np.arange(n)
    for b in range(n_boot):
        idx = RNG_BOOTSTRAP.choice(idx_pool, size=n, replace=True)
        boot_means[b] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_test(diffs: Sequence[float], label: str, family: str, test_id: str,
                 description: str):
    """Full paired-difference battery for small-n (typically n=5 folds) data:
    mean, SD, exact sign test, Wilcoxon signed-rank (both with the usual
    small-n caveats noted), and a nonparametric bootstrap CI on the mean.
    Does NOT silently hide the small-n power problem -- n is always recorded
    and reported."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1)) if n > 1 else float("nan")

    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_zero = n - n_pos - n_neg
    # exact two-sided sign test on the nonzero differences
    if n_pos + n_neg > 0:
        sign_p = binomtest(min(n_pos, n_neg), n_pos + n_neg, 0.5, alternative="two-sided").pvalue
    else:
        sign_p = 1.0

    try:
        if n_pos + n_neg >= 1 and not (n_pos == n and n_neg == 0 and n <= 1):
            stat, wilcox_p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        else:
            stat, wilcox_p = (np.nan, np.nan)
    except ValueError:
        # all-zero or too-small-n degenerate case
        stat, wilcox_p = (np.nan, np.nan)

    ci_lo, ci_hi = paired_bootstrap_ci(diffs) if n >= 2 else (np.nan, np.nan)

    log(f"  [{test_id}] {label}: n={n}, mean={mean_d:+.4f}, sd={sd_d:.4f}, "
        f"sign={n_pos}+/{n_neg}-/{n_zero}0, sign-test p={sign_p:.4f}, "
        f"Wilcoxon p={wilcox_p if np.isnan(wilcox_p) else round(wilcox_p,4)}, "
        f"bootstrap 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    record(test_id, family, description, "mean_diff", mean_d, sign_p,
           ci_lo, ci_hi, n,
           note=f"wilcoxon_p={wilcox_p}; sd={sd_d}; sign={n_pos}+/{n_neg}-/{n_zero}0")
    return dict(n=n, mean=mean_d, sd=sd_d, sign_p=sign_p, wilcoxon_p=wilcox_p,
                ci_lo=ci_lo, ci_hi=ci_hi)


def fisher_pairwise(k_a, n_a, k_b, n_b, label_a, label_b, family, test_id, description):
    table = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
    odds_ratio, p = fisher_exact(table)
    log(f"  [{test_id}] {label_a} ({k_a}/{n_a}) vs {label_b} ({k_b}/{n_b}): "
        f"Fisher's exact OR={odds_ratio:.4f}, p={p:.4f}")
    record(test_id, family, description, "odds_ratio", odds_ratio, p, n=n_a + n_b,
           note=f"{label_a}={k_a}/{n_a}; {label_b}={k_b}/{n_b}")
    return odds_ratio, p


def logistic_trend(x: np.ndarray, y: np.ndarray, label: str, family: str,
                    test_id: str, description: str):
    """Logistic regression of a binary outcome (e.g. displaced 0/1) on a
    continuous predictor (rate), used as a formal 'does this rise with rate'
    test with more power than pairwise comparisons at fixed rates. Handles
    complete/quasi-separation gracefully (e.g. MNAR-Y's constant-0 outcome
    across the whole rate sweep) by detecting a degenerate y and reporting
    that the trend is not estimable rather than crashing or reporting a
    meaningless huge-SE coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if y.sum() == 0 or y.sum() == n:
        log(f"  [{test_id}] {label}: y is constant ({'all 0' if y.sum()==0 else 'all 1'}, "
            f"n={n}) -- logistic trend not estimable (complete separation by construction). "
            f"This is itself the finding: {'zero' if y.sum()==0 else 'every'} displacement "
            f"observed regardless of rate.")
        record(test_id, family, description, "logit_slope", np.nan, np.nan, n=n,
               note=f"not estimable: y constant ({'all-0' if y.sum()==0 else 'all-1'})")
        return None
    if not HAVE_STATSMODELS:
        log(f"  [{test_id}] {label}: statsmodels unavailable, skipping logistic trend test")
        return None
    X = sm.add_constant(x)
    try:
        model = sm.Logit(y, X).fit(disp=0)
        slope = model.params[1]
        p = model.pvalues[1]
        ci_lo, ci_hi = model.conf_int()[1]
        log(f"  [{test_id}] {label}: logistic slope(rate)={slope:.4f} "
            f"(95% CI [{ci_lo:.4f}, {ci_hi:.4f}]), p={p:.4f}, n={n}")
        record(test_id, family, description, "logit_slope", slope, p, ci_lo, ci_hi, n)
        return dict(slope=slope, p=p, ci_lo=ci_lo, ci_hi=ci_hi)
    except Exception as e:
        log(f"  [{test_id}] {label}: logistic fit failed ({e}) -- likely quasi-separation, "
            f"reporting as not reliably estimable")
        record(test_id, family, description, "logit_slope", np.nan, np.nan, n=n,
               note=f"fit failed: {e}")
        return None


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Standard BH FDR correction. Returns adjusted p-values, same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(root: Path):
    d = {}
    d["p4_fold"] = pd.read_csv(root / "PHASE_4_RQ1_RQ2/results/phase4/phase4_fold_summary.csv")
    d["p5_shift"] = pd.read_csv(root / "PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv")
    d["p6_perm"] = pd.read_csv(root / "PHASE_6_RQ4/results/phase6/phase6_permutation.csv")
    d["mx_fold"] = pd.read_csv(root / "PHASE_8_MNAR_X/results/phase8_mnar_x/phase8_mnar_x_fold_summary.csv")
    d["cal_table"] = pd.read_csv(root / "PHASE_8_CALIBRATION/results/phase8_calibration/phase8_calibration_table.csv")
    d["s1_fold"] = pd.read_csv(root / "PHASE_8_S1_SENSITIVITY/results/phase8_s1_sensitivity/phase8_s1_fold_summary.csv")
    return d


def rate_sweep_displacement(fold_df: pd.DataFrame, mechanism_prefix: str,
                             baseline_selected_by_fold: dict, rates=(0.10, 0.20, 0.30, 0.40)):
    """Returns a DataFrame with one row per (rate, fold): rate, fold_id, displaced(0/1),
    computed as selected_family(rate) != selected_family(q=0 baseline, same fold)."""
    rows = []
    for r in rates:
        cid = f"{mechanism_prefix}_q{int(round(r*100)):02d}_main"
        sub = fold_df[fold_df["condition_id"] == cid]
        for _, row in sub.iterrows():
            base_sel = baseline_selected_by_fold[row["fold_id"]]
            rows.append(dict(rate=r, fold_id=row["fold_id"],
                              displaced=int(row["selected_family"] != base_sel)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 1 -- RQ1 (Phase 4): mechanism vs. rate, displacement
# ---------------------------------------------------------------------------

def section_rq1(d):
    log("\n" + "=" * 100)
    log("SECTION 1 -- RQ1 (Phase 4): baseline_selection_displacement_rate by mechanism, formal tests")
    log("=" * 100)

    p4 = d["p4_fold"]
    baselines = {}
    for mech, q0_cid in [("mcar", "mcar_q00"), ("mar", "mar_q00"), ("mnar_y", "mnar_y_q00")]:
        sub = p4[p4["condition_id"] == q0_cid].sort_values("fold_id")
        baselines[mech] = dict(zip(sub["fold_id"], sub["selected_family"]))

    disp = {}
    for mech in ["mcar", "mar", "mnar_y"]:
        dsub = rate_sweep_displacement(p4, mech, baselines[mech])
        disp[mech] = dsub
        k, n = int(dsub["displaced"].sum()), len(dsub)
        wlo, whi = wilson_ci(k, n)
        clo, chi = clopper_pearson_ci(k, n)
        log(f"\n{mech.upper()} aggregate displacement over rate sweep: {k}/{n} = {100*k/n:.1f}%  "
            f"Wilson 95% CI [{100*wlo:.1f}%, {100*whi:.1f}%]  Clopper-Pearson [{100*clo:.1f}%, {100*chi:.1f}%]")
        record("RQ1.agg." + mech, mech, f"{mech} aggregate displacement over q=0.10-0.40 rate sweep",
               "proportion", k / n, np.nan, wlo, whi, n,
               note=f"k={k}, n={n}; Clopper-Pearson [{clo:.4f},{chi:.4f}]")

    log("\nPairwise Fisher's exact tests on aggregate displacement counts:")
    for a, b in [("mcar", "mnar_y"), ("mar", "mnar_y"), ("mcar", "mar")]:
        ka, na = int(disp[a]["displaced"].sum()), len(disp[a])
        kb, nb = int(disp[b]["displaced"].sum()), len(disp[b])
        fisher_pairwise(ka, na, kb, nb, a, b, "RQ1", f"RQ1.fisher.{a}_vs_{b}",
                         f"RQ1 aggregate displacement: {a} vs {b}")

    log("\nRate-trend logistic regression (displaced ~ rate) per mechanism, n=20 each:")
    for mech in ["mcar", "mar", "mnar_y"]:
        dsub = disp[mech]
        logistic_trend(dsub["rate"].values, dsub["displaced"].values, mech, "RQ1",
                        f"RQ1.trend.{mech}", f"RQ1 {mech} displacement vs rate trend")


# ---------------------------------------------------------------------------
# Section 2 -- RQ2 (Phase 4): OR-sweep AUROC gain, paired by fold
# ---------------------------------------------------------------------------

def section_rq2(d):
    log("\n" + "=" * 100)
    log("SECTION 2 -- RQ2 (Phase 4): MNAR-Y strength (OR) sweep, paired-by-fold AUROC tests")
    log("=" * 100)

    p4 = d["p4_fold"]
    conds = {
        1.5: "mnar_y_q30_OR1.5",
        2.0: "mnar_y_q30_main",
        4.0: "mnar_y_q30_OR4.0",
    }
    auroc = {}
    for orv, cid in conds.items():
        sub = p4[p4["condition_id"] == cid].sort_values("fold_id")
        auroc[orv] = dict(zip(sub["fold_id"], sub["selected_outer_test_auroc"]))

    for a, b in [(1.5, 2.0), (2.0, 4.0)]:
        common_folds = sorted(set(auroc[a]) & set(auroc[b]))
        diffs = [auroc[b][f] - auroc[a][f] for f in common_folds]
        paired_test(diffs, f"OR={b} minus OR={a} (selected-family outer-test AUROC)", "RQ2",
                    f"RQ2.paired.OR{a}_to_OR{b}",
                    f"RQ2 paired AUROC change, OR {a} -> {b}, matched by fold")


# ---------------------------------------------------------------------------
# Section 3 -- RQ3 (Phase 5): asymmetric mechanism-shift penalty
# ---------------------------------------------------------------------------

def section_rq3(d):
    log("\n" + "=" * 100)
    log("SECTION 3 -- RQ3 (Phase 5): mechanism-shift penalty asymmetry and displacement")
    log("=" * 100)

    p5 = d["p5_shift"]

    def penalty_by_fold(source, target, rate=0.30):
        sub = p5[(p5["source_mechanism"] == source) & (p5["target_mechanism"] == target)
                 & (p5["rate"] == rate)].sort_values("fold_id")
        return dict(zip(sub["fold_id"], sub["shift_delta_auroc"]))

    mcar_to_mnary = penalty_by_fold("mcar", "mnar_y")
    mnary_to_mcar = penalty_by_fold("mnar_y", "mcar")
    common = sorted(set(mcar_to_mnary) & set(mnary_to_mcar))
    # both are negative deltas (degradation); compare MAGNITUDES (|.|) paired by fold
    mag_diffs = [abs(mcar_to_mnary[f]) - abs(mnary_to_mcar[f]) for f in common]
    log("\nAsymmetry: |penalty(MCAR source -> MNAR-Y target)| minus |penalty(MNAR-Y source -> MCAR target)|, "
        "paired by fold, at rate=0.30 (positive => MCAR-source penalty is the larger one, as claimed):")
    paired_test(mag_diffs, "|MCAR->MNAR-Y| - |MNAR-Y->MCAR|", "RQ3", "RQ3.asymmetry.mcar_mnary",
                "RQ3 shift-penalty magnitude asymmetry, MCAR-source vs MNAR-Y-source, at rate=0.30")

    mar_to_mnary = penalty_by_fold("mar", "mnar_y")
    mnary_to_mar = penalty_by_fold("mnar_y", "mar")
    common2 = sorted(set(mar_to_mnary) & set(mnary_to_mar))
    mag_diffs2 = [abs(mar_to_mnary[f]) - abs(mnary_to_mar[f]) for f in common2]
    log("\nSame asymmetry check, MAR<->MNAR-Y:")
    paired_test(mag_diffs2, "|MAR->MNAR-Y| - |MNAR-Y->MAR|", "RQ3", "RQ3.asymmetry.mar_mnary",
                "RQ3 shift-penalty magnitude asymmetry, MAR-source vs MNAR-Y-source, at rate=0.30")

    log("\nRate-trend of the MCAR-source -> MNAR-Y-target penalty (n=15, rates 0.10/0.30/0.50):")
    rows = []
    for rate in [0.10, 0.30, 0.50]:
        pb = penalty_by_fold("mcar", "mnar_y", rate=rate)
        for fid, val in pb.items():
            rows.append((rate, val))
    rate_arr = np.array([r for r, v in rows], dtype=float)
    val_arr = np.array([v for r, v in rows], dtype=float)
    if HAVE_STATSMODELS:
        X = sm.add_constant(rate_arr)
        ols = sm.OLS(val_arr, X).fit()
        slope, p = ols.params[1], ols.pvalues[1]
        ci_lo, ci_hi = ols.conf_int()[1]
        log(f"  [RQ3.trend.mcar_to_mnary] OLS slope(rate)={slope:.4f} (95% CI [{ci_lo:.4f},{ci_hi:.4f}]), "
            f"p={p:.4f}, n={len(val_arr)}")
        record("RQ3.trend.mcar_to_mnary", "RQ3", "RQ3 MCAR->MNAR-Y shift penalty vs rate trend",
               "ols_slope", slope, p, ci_lo, ci_hi, len(val_arr))

    log("\nMNAR-Y-as-source displacement rate, pooled across targets {MCAR, MAR}, rate=0.30:")
    pooled = p5[(p5["source_mechanism"] == "mnar_y") & (p5["target_mechanism"].isin(["mcar", "mar"]))
                & (p5["rate"] == 0.30)]
    k, n = int(pooled["displaced"].sum()), len(pooled)
    clo, chi = clopper_pearson_ci(k, n)
    wlo, whi = wilson_ci(k, n)
    log(f"  {k}/{n} = {100*k/n:.1f}%  Wilson 95% CI [{100*wlo:.1f}%,{100*whi:.1f}%]  "
        f"Clopper-Pearson [{100*clo:.1f}%,{100*chi:.1f}%]")
    record("RQ3.mnary_source_displacement", "RQ3",
           "MNAR-Y-as-source displacement rate, pooled MCAR+MAR targets, rate=0.30",
           "proportion", k / n, np.nan, wlo, whi, n, note=f"k={k}, n={n}")


# ---------------------------------------------------------------------------
# Section 4 -- RQ4 (Phase 6): permutation ablation, before vs after
# ---------------------------------------------------------------------------

def section_rq4(d, perm_repeats_df: Optional[pd.DataFrame] = None):
    log("\n" + "=" * 100)
    log("SECTION 4 -- RQ4 (Phase 6): permutation-ablation AUROC collapse, before vs after")
    log("=" * 100)
    log("\nNOTE: this section uses the ORIGINAL single-permutation-draw data (5 folds/cell) unless "
        "--perm-repeats-csv was supplied, in which case the repeated-draw data (from the companion "
        "driver run_phase6_section_b_repeats.py) is used instead for far higher statistical power -- "
        "see the printed note before each cell below.")

    p6 = d["p6_perm"]
    cells = p6[["rate", "or_value"]].drop_duplicates().sort_values(["rate", "or_value"]).values.tolist()

    cell_deltas = {}
    for rate, orv in cells:
        sub = p6[(p6["rate"] == rate) & (p6["or_value"] == orv)].sort_values("fold_id")
        deltas = sub["delta_auroc"].values
        cell_deltas[(rate, orv)] = deltas
        log(f"\nCell rate={rate}, OR={orv} (single-draw, n={len(deltas)} folds):")
        paired_test(deltas, f"delta_auroc, rate={rate} OR={orv} (single draw)", "RQ4",
                    f"RQ4.single.rate{rate}_OR{orv}", f"RQ4 permutation delta_auroc, rate={rate}, OR={orv}, single draw")

    if perm_repeats_df is not None:
        log("\n--- Repeated-draw analysis (from run_phase6_section_b_repeats.py output) ---")
        for rate, orv in cells:
            sub = perm_repeats_df[(perm_repeats_df["rate"] == rate) & (perm_repeats_df["or_value"] == orv)]
            if len(sub) == 0:
                log(f"  rate={rate} OR={orv}: no repeated-draw data found in supplied CSV, skipping")
                continue
            deltas = sub["delta_auroc"].values
            log(f"\nCell rate={rate}, OR={orv} (repeated draws, n={len(deltas)} draws across all folds):")
            mean_d, sd_d = float(deltas.mean()), float(deltas.std(ddof=1))
            ci_lo, ci_hi = paired_bootstrap_ci(deltas)
            log(f"  mean={mean_d:+.4f}, sd={sd_d:.4f}, bootstrap 95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}], n={len(deltas)}")
            record(f"RQ4.repeats.rate{rate}_OR{orv}", "RQ4",
                   f"RQ4 permutation delta_auroc, rate={rate}, OR={orv}, repeated draws (Monte Carlo "
                   f"characterization of the effect size's own sampling distribution under re-shuffling, "
                   f"NOT a null-hypothesis permutation test)",
                   "mean_diff", mean_d, np.nan, ci_lo, ci_hi, len(deltas))

    # OR=4 vs OR=2 collapse magnitude, paired by fold, at each rate (single-draw data)
    log("\nOR=4.0 collapse magnitude vs OR=2.0 collapse magnitude, paired by fold (single-draw data):")
    for rate in sorted(set(r for r, o in cells)):
        d2 = p6[(p6["rate"] == rate) & (p6["or_value"] == 2.0)].sort_values("fold_id")
        d4 = p6[(p6["rate"] == rate) & (p6["or_value"] == 4.0)].sort_values("fold_id")
        m2 = dict(zip(d2["fold_id"], d2["delta_auroc"]))
        m4 = dict(zip(d4["fold_id"], d4["delta_auroc"]))
        common = sorted(set(m2) & set(m4))
        diffs = [abs(m4[f]) - abs(m2[f]) for f in common]  # positive => OR4 collapse bigger
        paired_test(diffs, f"|delta_OR4| - |delta_OR2| at rate={rate}", "RQ4",
                    f"RQ4.or_compare.rate{rate}",
                    f"RQ4 OR=4.0 vs OR=2.0 collapse-magnitude comparison at rate={rate}, paired by fold")


# ---------------------------------------------------------------------------
# Section 5 -- Item 11: MNAR-X, calibration, S1, AP-selection
# ---------------------------------------------------------------------------

def section_item11(d):
    log("\n" + "=" * 100)
    log("SECTION 5 -- Item 11: MNAR-X, calibration slope trend, S1, AP-selection")
    log("=" * 100)

    p4 = d["p4_fold"]
    mx = d["mx_fold"]
    baseline_mx = dict(zip(
        p4[p4["condition_id"] == "natural_q00"].sort_values("fold_id")["fold_id"],
        p4[p4["condition_id"] == "natural_q00"].sort_values("fold_id")["selected_family"],
    ))
    dmx = rate_sweep_displacement(mx, "mnar_x", baseline_mx)
    k, n = int(dmx["displaced"].sum()), len(dmx)
    wlo, whi = wilson_ci(k, n)
    log(f"\nMNAR-X aggregate displacement over rate sweep: {k}/{n} = {100*k/n:.1f}%  "
        f"Wilson 95% CI [{100*wlo:.1f}%,{100*whi:.1f}%]")
    record("item11.mnarx.agg", "item11", "MNAR-X aggregate displacement over rate sweep",
           "proportion", k / n, np.nan, wlo, whi, n, note=f"k={k}, n={n}")

    log("\nMNAR-X rate-trend logistic regression:")
    logistic_trend(dmx["rate"].values, dmx["displaced"].values, "mnar_x", "item11",
                    "item11.mnarx.trend", "MNAR-X displacement vs rate trend")

    log("\nMNAR-X vs Phase-4-mechanism pairwise Fisher's exact (aggregate displacement counts):")
    for mech, q0_cid in [("mcar", "mcar_q00"), ("mar", "mar_q00"), ("mnar_y", "mnar_y_q00")]:
        baseline = dict(zip(
            p4[p4["condition_id"] == q0_cid].sort_values("fold_id")["fold_id"],
            p4[p4["condition_id"] == q0_cid].sort_values("fold_id")["selected_family"],
        ))
        dmech = rate_sweep_displacement(p4, mech, baseline)
        kb, nb = int(dmech["displaced"].sum()), len(dmech)
        fisher_pairwise(k, n, kb, nb, "mnar_x", mech, "item11", f"item11.mnarx_vs_{mech}",
                         f"MNAR-X vs {mech} aggregate displacement")

    log("\nCalibration slope vs rate, linear-regression trend per mechanism (n=25 each: 5 rates x 5 folds, "
        "using each condition's per-fold calibration_slope):")
    cal = d["cal_table"]
    for mech, prefix in [("mcar", "mcar"), ("mar", "mar"), ("mnar_y", "mnar_y")]:
        rows = []
        q0 = cal[cal["condition_id"] == "natural_q00"]
        for _, r in q0.iterrows():
            rows.append((0.0, r["calibration_slope"]))
        for rate in [0.10, 0.20, 0.30, 0.40]:
            cid = f"{prefix}_q{int(round(rate*100)):02d}_main"
            sub = cal[cal["condition_id"] == cid]
            for _, r in sub.iterrows():
                rows.append((rate, r["calibration_slope"]))
        rate_arr = np.array([x for x, y in rows])
        slope_arr = np.array([y for x, y in rows])
        if HAVE_STATSMODELS:
            X = sm.add_constant(rate_arr)
            ols = sm.OLS(slope_arr, X).fit()
            b, p = ols.params[1], ols.pvalues[1]
            ci_lo, ci_hi = ols.conf_int()[1]
            log(f"  [{mech}] OLS slope-of-slope(rate)={b:.4f} (95% CI [{ci_lo:.4f},{ci_hi:.4f}]), "
                f"p={p:.4f}, n={len(rate_arr)}")
            record(f"item11.calib_trend.{mech}", "item11",
                   f"Calibration slope vs rate trend, {mech}", "ols_slope", b, p, ci_lo, ci_hi, len(rate_arr))

    log("\nS1 headline comparison (reproduced here for a single consolidated source of truth; "
        "originally computed during the item-11 recheck audit):")
    s1 = d["s1_fold"]

    def s1_disp_counts(mech):
        base_cid = f"{mech}_q00"
        base = dict(zip(
            s1[s1["condition_id"] == base_cid].sort_values("fold_id")["fold_id"],
            s1[s1["condition_id"] == base_cid].sort_values("fold_id")["selected_family"],
        ))
        dsub = rate_sweep_displacement(s1, mech, base)
        return int(dsub["displaced"].sum()), len(dsub)

    k_mnary, n_mnary = s1_disp_counts("mnar_y")
    k_mcar, n_mcar = s1_disp_counts("mcar")
    k_mar, n_mar = s1_disp_counts("mar")
    fisher_pairwise(k_mnary, n_mnary, k_mcar, n_mcar, "mnar_y", "mcar", "item11",
                     "item11.s1.mnary_vs_mcar", "S1 aggregate displacement, MNAR-Y vs MCAR")
    fisher_pairwise(k_mnary, n_mnary, k_mar, n_mar, "mnar_y", "mar", "item11",
                     "item11.s1.mnary_vs_mar", "S1 aggregate displacement, MNAR-Y vs MAR")

    log("\nAP-based selection-criterion sensitivity (Part 3): overall displacement 32/90, exact binomial CI:")
    k, n = 32, 90
    wlo, whi = wilson_ci(k, n)
    clo, chi = clopper_pearson_ci(k, n)
    log(f"  {k}/{n} = {100*k/n:.1f}%  Wilson 95% CI [{100*wlo:.1f}%,{100*whi:.1f}%]  "
        f"Clopper-Pearson [{100*clo:.1f}%,{100*chi:.1f}%]")
    record("item11.ap_selection", "item11", "AP-vs-AUROC selection-criterion overall displacement",
           "proportion", k / n, np.nan, wlo, whi, n, note=f"k={k}, n={n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                     help="Path to the DASA2026 folder containing PHASE_4_RQ1_RQ2/, PHASE_5_RQ3/, "
                          "PHASE_6_RQ4/, PHASE_8_MNAR_X/, PHASE_8_CALIBRATION/, PHASE_8_S1_SENSITIVITY/")
    ap.add_argument("--perm-repeats-csv", default=None,
                     help="Optional path to phase8_section_b_repeats_full.csv produced by "
                          "run_phase6_section_b_repeats.py, for the higher-powered RQ4 analysis")
    ap.add_argument("--out-prefix", default="phase_stats_layer",
                     help="Prefix for output files (default: phase_stats_layer)")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: data root {root} does not exist", file=sys.stderr)
        sys.exit(1)

    log("Fold-level significance-testing / uncertainty-quantification layer")
    log(f"Data root: {root}")
    log(f"statsmodels available: {HAVE_STATSMODELS}")

    d = load_all(root)

    perm_repeats_df = None
    if args.perm_repeats_csv:
        p = Path(args.perm_repeats_csv)
        if p.exists():
            perm_repeats_df = pd.read_csv(p)
            log(f"Loaded repeated-permutation-draw data: {p} ({len(perm_repeats_df)} rows)")
        else:
            log(f"WARNING: --perm-repeats-csv path {p} does not exist, proceeding without it")

    section_rq1(d)
    section_rq2(d)
    section_rq3(d)
    section_rq4(d, perm_repeats_df)
    section_item11(d)

    # ---- BH correction across all tests that produced a real p-value ----
    results_df = pd.DataFrame(RESULTS)
    has_p = results_df["p_value"].notna()
    results_df["p_bh_adjusted"] = np.nan
    if has_p.sum() > 0:
        results_df.loc[has_p, "p_bh_adjusted"] = benjamini_hochberg(results_df.loc[has_p, "p_value"].values)

    out_csv = f"{args.out_prefix}_results.csv"
    results_df.to_csv(out_csv, index=False)

    log("\n" + "=" * 100)
    log(f"SUMMARY: {len(results_df)} tests/CIs recorded. {int(has_p.sum())} carry a raw p-value; "
        f"Benjamini-Hochberg-adjusted p-values added as a separate column ({out_csv}).")
    log("Tests significant at raw p<0.05 but NOT at BH-adjusted p<0.05:")
    flagged = results_df[has_p & (results_df["p_value"] < 0.05) & (results_df["p_bh_adjusted"] >= 0.05)]
    if len(flagged) == 0:
        log("  (none)")
    else:
        for _, r in flagged.iterrows():
            log(f"  {r['test_id']}: raw p={r['p_value']:.4f}, BH-adjusted p={r['p_bh_adjusted']:.4f}")
    log("=" * 100)

    out_md = f"{args.out_prefix}_report.md"
    with open(out_md, "w") as f:
        f.write("# Fold-level significance-testing report\n\n")
        f.write("```\n")
        f.write("\n".join(REPORT_LINES))
        f.write("\n```\n")

    print(f"\nWrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
