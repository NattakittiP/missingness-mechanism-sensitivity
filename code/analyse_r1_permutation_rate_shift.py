"""Analysis of Results/R1_permutation_rate_shift (read-only); prints every number used in the write-up.

Run from Revision_R1/Code:
    python analyse_r1_permutation_rate_shift.py > ../Results/R1_permutation_rate_shift/r1_perm_rate_analysis.txt
The OR 2 and OR 4 permutation cells and the Phase 5 shift cells are checked against the main results (Phase 4 fold summary, Phase 5 full table, Phase 6
baselines, permutation draws); the script stops if a check fails.
Intervals: permutation draws, percentile bootstrap of the mean (10,000 resamples, seed 20260919, as the
statistics layer); pooled case-level AUROC, stratified bootstrap (2,000 resamples, seed 20260920, the
functions of case_level_bootstrap_ci.py); rate trends, OLS on fold-level shift penalties with
Benjamini-Hochberg adjustment across the six off-diagonal shifts.
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "main_study/CASE_LEVEL_BOOTSTRAP_CI")
import case_level_bootstrap_ci as cl  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else "../Results/R1_permutation_rate_shift"
MS = "main_study"
REF = dict(p4=f"{MS}/PHASE_4_RQ1_RQ2/results/phase4/phase4_fold_summary.csv",
           p5=f"{MS}/PHASE_5_RQ3/results/phase5/phase5_full_table.csv",
           p6b=f"{MS}/PHASE_6_RQ4/results/phase6/phase6_baselines.csv",
           p6i=f"{MS}/PHASE_6_RQ4/results/phase6/phase6_indicator_with.csv",
           p6o=f"{MS}/PHASE_6_RQ4/results/phase6/phase6_indicator_without.csv",
           p6r=f"{MS}/PHASE_6_SECTION_B_REPEATS/results/phase6_section_b_repeats/phase8_section_b_repeats_full.csv")
files = [f"{R}/{n}.csv" for n in ("perm_cells", "perm_draws", "perm_predictions", "rate_shift_eval")] + list(REF.values())
for f in files:
    print(f"input {f} sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}")
cells, draws = pd.read_csv(f"{R}/perm_cells.csv"), pd.read_csv(f"{R}/perm_draws.csv")
preds, rs = pd.read_csv(f"{R}/perm_predictions.csv"), pd.read_csv(f"{R}/rate_shift_eval.csv")
ORDER = ["mnar_y_q30_OR1.5", "mnar_y_q30_OR2", "mnar_y_q30_OR4", "mnar_y_q40_OR2", "mnar_y_q40_OR4"]
TOL_XGB = 0.03  # cross-environment xgboost band used by the Phase 6/8 drivers


def boot_ci(x, B=10000, seed=20260919):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (B, len(x)))].mean(1), [2.5, 97.5])


print("\n== 1. checks against the main results")
c0, d0 = cells[cells.repeat == 0], draws[draws.repeat == 0]
p4 = pd.read_csv(REF["p4"]); p4 = p4[(p4.mechanism == "mnar_y") & (p4.rate == 0.30) & (p4.or_value == 1.5)]
m = c0[c0.cell == "mnar_y_q30_OR1.5"].merge(p4, on="fold_id", suffixes=("", "_p4"))
dd = (m.auroc_before - m.selected_outer_test_auroc).abs().max()
print(f"OR 1.5 q 0.30 before vs Phase 4: max abs diff {dd:.2e}, family match {(m.selected_family == m.selected_family_p4).sum()}/{len(m)}")
assert len(m) == 5 and dd < TOL_XGB and (m.selected_family == m.selected_family_p4).all()
p6r = pd.read_csv(REF["p6r"])
p6r["cell"] = p6r.apply(lambda r: f"mnar_y_q{int(round(r.rate * 100))}_OR{int(r.or_value)}", axis=1)
m = d0.merge(p6r, on=["cell", "fold_id", "draw_id"], suffixes=("", "_ref"))
dd = (m.delta_auroc - m.delta_auroc_ref).abs().max()
print(f"permutation draws vs main repeats: {len(m)} draws compared, max abs diff of delta {dd:.2e}")
assert len(m) == 600 and dd < TOL_XGB
p6b = pd.read_csv(REF["p6b"])
p6b["cell"] = p6b.condition_id.map({"mnar_y_q30_OR1.5": "mnar_y_q30_OR1.5", "mnar_y_q30_main": "mnar_y_q30_OR2",
                                    "mnar_y_q30_OR4.0": "mnar_y_q30_OR4", "mnar_y_q40_main": "mnar_y_q40_OR2"})
m = c0.merge(p6b.dropna(subset=["cell"])[["cell", "fold_id", "synth_only_mask_auroc"]], on=["cell", "fold_id"])
dd = (m.mask_only_auroc - m.synth_only_mask_auroc).abs().max()
print(f"injected-mask-only AUROC vs main baselines: {len(m)} cells, max abs diff {dd:.2e}")
assert len(m) == 20 and dd < 1e-9

print("\n== 2. selected AUROC, pooled case-level AUROC, mask-only AUROC, permutation")
rng = np.random.default_rng(cl.CASE_BOOTSTRAP_SEED)
for cid in ORDER:
    g = c0[c0.cell == cid]
    pr = preds[(preds.repeat == 0) & (preds.cell == cid)]
    assert len(pr) == 14081 and pr.row_index.is_unique
    ci = cl.case_bootstrap_ci(pr.y.to_numpy(), pr.p.to_numpy(), rng)
    dl = d0[d0.cell == cid].delta_auroc
    lo, hi = boot_ci(dl.values)
    print(f"{cid:17s} AUROC {g.auroc_before.mean():.4f} ± {g.auroc_before.std(ddof=1):.4f} sel {list(g.selected_family)} | "
          f"pooled {ci['auroc']:.4f} ({ci['auroc_ci_lo']:.4f}-{ci['auroc_ci_hi']:.4f}) | mask-only {g.mask_only_auroc.mean():.4f} | "
          f"perm n={len(dl)} mean {dl.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    fm = d0[d0.cell == cid].groupby("fold_id").delta_auroc.mean()
    print(f"{'':17s} permutation fold means {[round(v, 4) for v in fm]} folds negative {(fm < 0).sum()}/5 | "
          f"AUROC after (mean over draws) {g.auroc_before.mean() + dl.mean():.4f}")
    x0 = d0[(d0.cell == cid) & (d0.draw_id == 0)].sort_values("fold_id").delta_auroc
    print(f"{'':17s} single draw 0: mean {x0.mean():+.4f}, after {g.auroc_before.mean() + x0.mean():.4f}, folds negative {(x0 < 0).sum()}/5")

print("\n== 3. source -> target shift at q = 0.10, 0.30, 0.50 (main folds)")
fams = [c[6:] for c in rs.columns if c.startswith("auroc_")]
rs["sel"] = rs.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
rs["best"] = rs[[f"auroc_{f}" for f in fams]].max(axis=1)
rs["mismatch"] = rs.apply(lambda r: r[f"auroc_{r.selected_family}"] < r.best, axis=1)
rs["src"], rs["tgt"] = rs.source.str.rsplit("_q", n=1).str[0], rs.target.str.rsplit("_q", n=1).str[0]
own = rs[rs.source == rs.target].set_index(["rate", "src", "fold_id"]).sel
rs["delta"] = rs.sel.values - own.loc[list(zip(rs.rate, rs.src, rs.fold_id))].values
p5 = pd.read_csv(REF["p5"]); p5 = p5[p5.model_key == p5.selected_family]
m = rs.merge(p5, left_on=["rate", "fold_id", "src", "tgt"], right_on=["rate", "fold_id", "source_mechanism", "target_mechanism"])
dd = (m.sel - m.outer_test_auroc).abs().max()
print(f"reproduction of Phase 5: {len(m)} cells compared, max abs diff {dd:.2e}, family match {(m.selected_family_x == m.selected_family_y).sum()}/{len(m)}")
assert len(m) == 75 and dd < TOL_XGB
trend = []
for (src, tgt), g in rs[rs.src != rs.tgt].groupby(["src", "tgt"]):
    for q, gq in g.groupby("rate"):
        print(f"{src:6s}->{tgt:6s} q {q:.2f} delta {gq.delta.mean():+.4f} negative {int((gq.delta < 0).sum())}/5 "
              f"oracle mismatch {int(gq.mismatch.sum())}/5 regret {(gq.best - gq.sel).mean():.4f}")
    fit = sm.OLS(g.delta.values, sm.add_constant(g.rate.values)).fit()
    trend.append(dict(shift=f"{src}->{tgt}", slope=fit.params[1], p=fit.pvalues[1], lo=fit.conf_int()[1][0], hi=fit.conf_int()[1][1], n=len(g)))
tr = pd.DataFrame(trend)
o = np.argsort(tr.p.values); adj = tr.p.values[o] * len(tr) / np.arange(1, len(tr) + 1)
tr.loc[tr.index[o], "p_bh"] = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1)
print("\nOLS slope of the shift penalty on rate (n = 15 each), BH across the six shifts:")
print(tr.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
for q in sorted(rs.rate.unique()):
    g = rs[(rs.rate == q) & (rs.src == "mnar_y") & (rs.tgt != "mnar_y")]
    print(f"q {q:.2f}: MNAR-Y-selected family not the oracle on MCAR/MAR targets {int(g.mismatch.sum())}/{len(g)}, mean regret {(g.best - g.sel).mean():.4f}")

print("\n== 4. indicator ablation (main folds): with minus without indicators, fold-paired")
ind = pd.read_csv(f"{MS}/PHASE_6_RQ4/results/phase6/phase6_indicator_with.csv").merge(
    pd.read_csv(f"{MS}/PHASE_6_RQ4/results/phase6/phase6_indicator_without.csv"), on=["mechanism", "rate", "fold_id"])
for (mech, q), g in ind.groupby(["mechanism", "rate"]):
    d = (g.with_indicator_auroc - g.without_indicator_auroc).values; lo, hi = boot_ci(d)
    print(f"{mech:7s} q {q:.2f} diff {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] positive {int((d > 0).sum())}/{len(d)}")
