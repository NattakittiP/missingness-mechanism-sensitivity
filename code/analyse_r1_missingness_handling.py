"""Analysis of Results/R1_missingness_handling (read-only); prints every number used in the write-up.

Run from Revision_R1/Code:
    python analyse_r1_missingness_handling.py > ../Results/R1_missingness_handling/r1_handling_analysis.txt
When ../Results/R1_multiple_imputation holds the multiple-imputation tables, the mi5 arm is analysed
alongside the three single-imputation arms (same folds, masks and permutation draws); when
../Results/R1_multiple_imputation_m1 holds the one-draw arm (mi1), section 7 also separates the posterior
draw (mi1 - mice) from the averaging over imputations (mi5 - mi1).
Fold-level intervals: percentile bootstrap over the 5 folds (10,000 resamples, seed 20260919), as in the
statistics layer; permutation intervals: the same bootstrap over the 150 draws and over the 5 fold means.
"""
import hashlib, os, sys, numpy as np, pandas as pd
from scipy import stats
R = "../Results/R1_missingness_handling"
P4 = "main_study/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv"
P5 = "main_study/PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv"
RMI = "../Results/R1_multiple_imputation"
if len(sys.argv) >= 4:
    R, P4, P5 = sys.argv[1:4]
if len(sys.argv) == 5:
    RMI = sys.argv[4]
MI = os.path.exists(f"{RMI}/r1_mi_eval_table.csv")
RMI1 = RMI + "_m1"
MI1 = MI and os.path.exists(f"{RMI1}/r1_mi_eval_table.csv")
inputs = [f"{R}/r1_handling_eval_table.csv", f"{R}/r1_handling_permutation_table.csv", P4, P5]
if MI:
    inputs += [f"{RMI}/r1_mi_eval_table.csv", f"{RMI}/r1_mi_permutation_table.csv"]
if MI1:
    inputs += [f"{RMI1}/r1_mi_eval_table.csv", f"{RMI1}/r1_mi_permutation_table.csv"]
for f in inputs:
    print(f"input {f} sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}")
ev = pd.read_csv(f"{R}/r1_handling_eval_table.csv"); pm = pd.read_csv(f"{R}/r1_handling_permutation_table.csv")
STRATS = ["median", "median_ind", "mice"]
if MI:
    ev = pd.concat([ev, pd.read_csv(f"{RMI}/r1_mi_eval_table.csv")], ignore_index=True)
    pm = pd.concat([pm, pd.read_csv(f"{RMI}/r1_mi_permutation_table.csv")], ignore_index=True)
    STRATS.append("mi5")
if MI1:
    ev = pd.concat([ev, pd.read_csv(f"{RMI1}/r1_mi_eval_table.csv")], ignore_index=True)
    pm = pd.concat([pm, pd.read_csv(f"{RMI1}/r1_mi_permutation_table.csv")], ignore_index=True)
    STRATS.append("mi1")
fams = [c[6:] for c in ev.columns if c.startswith("auroc_")]
ev["sel_auroc"] = ev.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
ev["sel_ap"] = ev.apply(lambda r: r[f"ap_{r.selected_family}"], axis=1)
ev["best_auroc"] = ev[[f"auroc_{f}" for f in fams]].max(axis=1)
ev["best_fam"] = ev[[f"auroc_{f}" for f in fams]].idxmax(axis=1).str[6:]
diag = ev[ev.source == ev.target].copy()
assert len(ev) == 70 * len(STRATS) and len(pm) == 300 * len(STRATS) and len(diag) == 25 * len(STRATS), "unexpected table sizes"
rng_seed = 20260919
def boot_ci(x, B=10000, seed=rng_seed):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])
def sign_p(x):
    x = np.asarray(x); x = x[x != 0]; k = (x > 0).sum()
    return stats.binomtest(k, len(x), 0.5).pvalue if len(x) else np.nan

print("== 1. reproduction of the main study by the median arm")
p4 = pd.read_csv(P4)
p4map = {"natural_q00": "natural_q00", "mcar_q30": "mcar_q30_main", "mar_q30": "mar_q30_main",
         "mnar_y_q30_OR2": "mnar_y_q30_main", "mnar_y_q30_OR4": "mnar_y_q30_OR4.0"}
n = mx = 0; selmatch = 0
for _, r in diag[diag.strategy == "median"].iterrows():
    sub = p4[(p4.condition_id == p4map[r.source]) & (p4.fold_id == r.fold_id)]
    for f in fams:
        v = sub[sub.model_key == f].outer_test_auroc
        if len(v) != 1: print("missing", r.source, r.fold_id, f); continue
        n += 1; mx = max(mx, abs(v.iloc[0] - r[f"auroc_{f}"]))
    selmatch += int(sub.selected_family.iloc[0] == r.selected_family)
print(f"family-fold AUROC cells compared {n}, max abs diff {mx:.2e}, selected-family matches {selmatch}/25")
assert n == 125 and mx < 1e-12 and selmatch == 25, "median arm does not reproduce the main grid"
p5 = pd.read_csv(P5); p5 = p5[p5.rate == 0.3]
off = ev[(ev.strategy == "median") & (ev.source != ev.target) & ev.source.isin(["mcar_q30", "mar_q30", "mnar_y_q30_OR2"])]
m5 = {"mcar_q30": "mcar", "mar_q30": "mar", "mnar_y_q30_OR2": "mnar_y"}
d5 = []
for _, r in off.iterrows():
    v = p5[(p5.source_mechanism == m5[r.source]) & (p5.target_mechanism == m5[r.target]) & (p5.fold_id == r.fold_id)]
    col = ["selected_outer_test_auroc"]
    d5.append(abs(v[col[0]].iloc[0] - r.sel_auroc))
print(f"shift cells vs Phase 5: {len(d5)}, max abs diff {max(d5):.2e} (column {col[0]})")

print("\n== 2. per strategy x source (own mechanism)")
base = diag[diag.source == "natural_q00"].set_index(["strategy", "fold_id"]).selected_family
order = ["natural_q00", "mcar_q30", "mar_q30", "mnar_y_q30_OR2", "mnar_y_q30_OR4"]
for s in STRATS:
    for src in order:
        g = diag[(diag.strategy == s) & (diag.source == src)].sort_values("fold_id")
        disp = sum(g.selected_family.values != base.loc[s].loc[g.fold_id].values)
        mism = (g.selected_family != g.best_fam).sum()
        print(f"{s:10s} {src:15s} AUROC {g.sel_auroc.mean():.4f} ± {g.sel_auroc.std(ddof=1):.4f}  AP {g.sel_ap.mean():.4f}  sel {list(g.selected_family)}  disp {disp}/5  oracle-mismatch {mism}/5  regret {(g.best_auroc-g.sel_auroc).mean():.4f}")

print("\n== 3. fold-paired difference vs median arm (own mechanism)")
for s in STRATS[1:]:
    for src in order:
        a = diag[(diag.strategy == s) & (diag.source == src)].sort_values("fold_id").sel_auroc.values
        b = diag[(diag.strategy == "median") & (diag.source == src)].sort_values("fold_id").sel_auroc.values
        d = a - b; lo, hi = boot_ci(d)
        print(f"{s:10s} {src:15s} diff {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] positive {int((d>0).sum())}/5 sign p {sign_p(d):.4f} range [{d.min():+.4f}, {d.max():+.4f}]")

print("\n== 3b. MNAR-Y OR4 minus OR2 and MNAR-Y minus MCAR, within strategy")
for s in STRATS:
    get = lambda src: diag[(diag.strategy == s) & (diag.source == src)].sort_values("fold_id").sel_auroc.values
    for a_, b_ in [("mnar_y_q30_OR4", "mnar_y_q30_OR2"), ("mnar_y_q30_OR2", "mcar_q30"), ("mcar_q30", "natural_q00")]:
        d = get(a_) - get(b_); lo, hi = boot_ci(d)
        print(f"{s:10s} {a_} - {b_}: {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] positive {int((d>0).sum())}/5")

print("\n== 4. frozen source -> target shift (selected family), q = 0.30")
for s in STRATS:
    for src in ["mcar_q30", "mar_q30", "mnar_y_q30_OR2", "mnar_y_q30_OR4"]:
        own = diag[(diag.strategy == s) & (diag.source == src)].set_index("fold_id").sel_auroc
        for tgt in ["mcar_q30", "mar_q30", "mnar_y_q30_OR2"]:
            if tgt == src: continue
            g = ev[(ev.strategy == s) & (ev.source == src) & (ev.target == tgt)].set_index("fold_id").sel_auroc
            d = (g - own.loc[g.index]).values; lo, hi = boot_ci(d)
            print(f"{s:10s} {src:15s}->{tgt:15s} {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] negative {int((d<0).sum())}/5 range [{d.min():+.4f}, {d.max():+.4f}]")

print("\n== 4b. shift penalty difference vs median arm (fold-paired)")
for s in STRATS[1:]:
    for src, tgt in [("mcar_q30", "mnar_y_q30_OR2"), ("mar_q30", "mnar_y_q30_OR2"), ("mnar_y_q30_OR2", "mcar_q30"), ("mnar_y_q30_OR2", "mar_q30"), ("mnar_y_q30_OR4", "mcar_q30"), ("mnar_y_q30_OR4", "mar_q30")]:
        def pen(st):
            own = diag[(diag.strategy == st) & (diag.source == src)].set_index("fold_id").sel_auroc
            g = ev[(ev.strategy == st) & (ev.source == src) & (ev.target == tgt)].set_index("fold_id").sel_auroc
            return (g - own).sort_index().values
        d = pen(s) - pen("median"); lo, hi = boot_ci(d)
        print(f"{s:10s} {src}->{tgt}: penalty diff {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] smaller-loss folds {int((d>0).sum())}/5")

print("\n== 5. test-time permutation of the injected mask (after - before)")
pm["delta"] = pm.auroc_after - pm.auroc_before
for s in STRATS:
    for src in ["mnar_y_q30_OR2", "mnar_y_q30_OR4"]:
        g = pm[(pm.strategy == s) & (pm.source == src)]
        fm = g.groupby("fold_id").delta.mean(); lo, hi = boot_ci(g.delta.values)
        flo, fhi = boot_ci(fm.values)
        print(f"{s:10s} {src:15s} n={len(g)} mean {g.delta.mean():+.4f} draw-CI [{lo:+.4f}, {hi:+.4f}] fold-CI [{flo:+.4f}, {fhi:+.4f}] folds negative {int((fm<0).sum())}/5 fold means {[round(v,4) for v in fm]} before {g.groupby('fold_id').auroc_before.first().mean():.4f}")

print("\n== 6. families selected (count over the 25 own-mechanism cells)")
print(diag.groupby("strategy").selected_family.value_counts().to_string())
disp = {}
for s_ in STRATS:
    g = diag[(diag.strategy == s_) & (diag.source != "natural_q00")]
    disp[s_] = int(sum(g.selected_family.values != base.loc[s_].loc[g.fold_id].values))
print("displacement vs q = 0 (k/20):", disp)
pairs = [("median_ind", "mice"), ("median_ind", "median"), ("median", "mice")] + ([("mi5", "median"), ("mi5", "mice")] if MI else []) \
    + ([("mi1", "mice"), ("mi5", "mi1")] if MI1 else [])
for a_, b_ in pairs:
    p_ = stats.fisher_exact([[disp[a_], 20 - disp[a_]], [disp[b_], 20 - disp[b_]]])[1]
    print(f"Fisher {a_} vs {b_}: p = {p_:.4f}")

for A, B in ([("mi5", "mice")] if MI else []) + ([("mi1", "mice"), ("mi5", "mi1")] if MI1 else []):
    print(f"\n== 7. {A} minus {B}, fold-paired")
    for src in order:
        a = diag[(diag.strategy == A) & (diag.source == src)].sort_values("fold_id").sel_auroc.values
        b = diag[(diag.strategy == B) & (diag.source == src)].sort_values("fold_id").sel_auroc.values
        d = a - b; lo, hi = boot_ci(d)
        print(f"own AUROC {src:15s} diff {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] positive {int((d>0).sum())}/5 range [{d.min():+.4f}, {d.max():+.4f}]")
    for src, tgt in [("mcar_q30", "mnar_y_q30_OR2"), ("mar_q30", "mnar_y_q30_OR2"), ("mnar_y_q30_OR2", "mcar_q30"), ("mnar_y_q30_OR2", "mar_q30"), ("mnar_y_q30_OR4", "mcar_q30"), ("mnar_y_q30_OR4", "mar_q30")]:
        def pen(st):
            own = diag[(diag.strategy == st) & (diag.source == src)].set_index("fold_id").sel_auroc
            g = ev[(ev.strategy == st) & (ev.source == src) & (ev.target == tgt)].set_index("fold_id").sel_auroc
            return (g - own).sort_index().values
        d = pen(A) - pen(B); lo, hi = boot_ci(d)
        print(f"shift penalty {src}->{tgt}: diff {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}] smaller-loss folds {int((d>0).sum())}/5")
    for src in ["mnar_y_q30_OR2", "mnar_y_q30_OR4"]:
        fa = pm[(pm.strategy == A) & (pm.source == src)].groupby("fold_id").delta.mean().sort_index()
        fb = pm[(pm.strategy == B) & (pm.source == src)].groupby("fold_id").delta.mean().sort_index()
        d = (fa - fb).values; lo, hi = boot_ci(d)
        print(f"permutation delta {src}: fold-mean diff {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}]")
for A in [a for a in ("mi5", "mi1") if a in STRATS]:
    # the permutation "before" value must equal the evaluation AUROC of the same frozen model
    chk = pm[pm.strategy == A].groupby(["source", "fold_id"]).auroc_before.first()
    own = diag[diag.strategy == A].set_index(["source", "fold_id"]).sel_auroc
    print(f"{A} permutation-before vs evaluation AUROC: max abs diff {max(abs(chk - own.loc[chk.index])):.2e} over {len(chk)} cells")
