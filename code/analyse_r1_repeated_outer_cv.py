"""Analysis of Results/R1_repeated_outer_cv (read-only); prints every number used in the write-up.

Run from Revision_R1/Code:
    python analyse_r1_repeated_outer_cv.py > ../Results/R1_repeated_outer_cv/r1_repeated_cv_analysis.txt
Repeated-CV contrasts use the exact sign test over the 25 folds and the Nadeau-Bengio corrected t test
(folds of different repeats share training rows); the corrected test is the one to read.
"""
import hashlib, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

R = "../Results/R1_repeated_outer_cv"; P4 = "main_study/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv"
P5 = "main_study/PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv"; DATA = "../Dataset/full_analytic_dataset_mortality_all_admissions.csv"
if len(sys.argv) == 5:
    R, P4, P5, DATA = sys.argv[1:5]
for f in [f"{R}/r1_repeated_cv_eval_table.csv", P4, P5]:
    print(f"input {f} sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}")
ev = pd.read_csv(f"{R}/r1_repeated_cv_eval_table.csv")
fams = [c[6:] for c in ev.columns if c.startswith("auroc_")]
ev["sel"] = ev.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
ev["best"] = ev[[f"auroc_{f}" for f in fams]].max(axis=1)
assert len(ev) == 525 and sorted(ev.repeat.unique()) == [0, 1, 2, 3, 4], "unexpected table"
diag = ev[ev.source == ev.target].copy()
base = diag[diag.source == "natural_q00"].set_index(["repeat", "fold_id"]).selected_family
diag["displaced"] = [s != base.loc[(r, k)] for r, k, s in zip(diag.repeat, diag.fold_id, diag.selected_family)]
sel = ev.set_index(["repeat", "fold_id", "source", "target"]).sel

# fold sizes of every repeat, for the Nadeau-Bengio correction
df = pd.read_csv(DATA); y = df.label_mortality.to_numpy(); g = df.subject_id.to_numpy()
ntr, nte = [], []
for r in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=2026 + r).split(df, y, g):
        ntr.append(len(tr)); nte.append(len(te))
ratio = np.mean(nte) / np.mean(ntr)
print(f"mean n_train {np.mean(ntr):.1f}, mean n_test {np.mean(nte):.1f}, ratio {ratio:.4f}")

def nb(d):
    d = np.asarray(d, float); t = d.mean() / np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), len(d) - 1)
def sign(d):
    d = np.asarray(d); d = d[d != 0]; k = int((d > 0).sum()); return k, len(d), stats.binomtest(k, len(d), 0.5).pvalue

print("\n== 1. repeat 0 reproduces the main grid")
p4 = pd.read_csv(P4); mp = {"natural_q00": "natural_q00"}
for s in diag.source.unique():
    if s != "natural_q00":
        mp[s] = s.replace("OR4", "OR4.0") if "_OR" in s else s + "_main"
n = mx = sm = 0
for _, r in diag[diag.repeat == 0].iterrows():
    sub = p4[(p4.condition_id == mp[r.source]) & (p4.fold_id == r.fold_id)]
    sm += int(sub.selected_family.iloc[0] == r.selected_family)
    for f in fams:
        n += 1; mx = max(mx, abs(sub[sub.model_key == f].outer_test_auroc.iloc[0] - r[f"auroc_{f}"]))
p5 = pd.read_csv(P5); p5 = p5[p5.rate == 0.3]; m5 = {"mcar_q30": "mcar", "mar_q30": "mar", "mnar_y_q30": "mnar_y"}
off = ev[(ev.repeat == 0) & (ev.source != ev.target)]; d5 = []
for _, r in off.iterrows():
    v = p5[(p5.source_mechanism == m5[r.source]) & (p5.target_mechanism == m5[r.target]) & (p5.fold_id == r.fold_id)]
    d5.append(abs(v.selected_outer_test_auroc.iloc[0] - r.sel))
print(f"family-fold AUROCs {n}, max abs diff {mx:.2e}; selections matching {sm}/75; shift cells {len(d5)}, max abs diff {max(d5):.2e}")

print("\n== 2. selected-family AUROC and selection, 25 folds per condition")
order = ["natural_q00"] + [f"{m}_q{q}" for m in ["mcar", "mar", "mnar_y"] for q in ["10", "20", "30", "40"]] + ["mnar_y_q30_OR1.5", "mnar_y_q30_OR4"]
for s in order:
    gg = diag[diag.source == s]; rep = gg.groupby("repeat").sel.mean()
    print(f"{s:18s} AUROC {gg.sel.mean():.4f} ± {gg.sel.std(ddof=1):.4f} (25 folds); repeat means {rep.min():.4f}-{rep.max():.4f}; "
          f"XGB {int((gg.selected_family == 'xgb').sum())}/25; displaced {int(gg.displaced.sum())}/25; mismatch {int((gg.sel < gg.best).sum())}/25")
print("non-XGB selections:", diag[diag.selected_family != "xgb"][["repeat", "fold_id", "source", "selected_family"]].to_dict("records"))
for m in ["mcar", "mar", "mnar_y"]:
    gg = diag[diag.source.str.match(fr"^{m}_q\d\d$")]
    print(f"displacement {m}: {int(gg.displaced.sum())}/{len(gg)}")
for a, b in [("mcar", "mnar_y"), ("mar", "mnar_y"), ("mcar", "mar")]:
    ka = int(diag[diag.source.str.match(fr"^{a}_q\d\d$")].displaced.sum()); kb = int(diag[diag.source.str.match(fr"^{b}_q\d\d$")].displaced.sum())
    print(f"Fisher {a} vs {b}: {ka}/100 vs {kb}/100, p = {stats.fisher_exact([[ka, 100 - ka], [kb, 100 - kb]])[1]:.4f}")
print("injected cells selecting XGBoost:", int((diag[diag.source != "natural_q00"].selected_family == "xgb").sum()), "/", len(diag[diag.source != "natural_q00"]))

print("\n== 3. paired contrasts over 25 folds")
folds = diag[["repeat", "fold_id"]].drop_duplicates().itertuples(index=False)
folds = list(folds)
def show(name, d):
    k, nn, ps = sign(d); t, pn = nb(d); rm = pd.Series(d, index=pd.MultiIndex.from_tuples(folds)).groupby(level=0).mean()
    print(f"{name}: mean {np.mean(d):+.4f}; folds > 0 {k}/{nn}, sign p {ps:.4g}; NB t {t:.3f}, p {pn:.4g}; repeat means {rm.min():+.4f} to {rm.max():+.4f} (> 0 in {(rm > 0).sum()}/5)")
for hi, lo in [("mnar_y_q30_OR4", "mnar_y_q30"), ("mnar_y_q30", "mnar_y_q30_OR1.5"), ("mcar_q30", "natural_q00"), ("mnar_y_q30", "mcar_q30")]:
    show(f"AUROC {hi} - {lo}", [sel[(r, k, hi, hi)] - sel[(r, k, lo, lo)] for r, k in folds])
print()
for s1 in ["mcar_q30", "mar_q30", "mnar_y_q30"]:
    for s2 in ["mcar_q30", "mar_q30", "mnar_y_q30"]:
        if s1 != s2:
            show(f"shift {s1} -> {s2}", [sel[(r, k, s1, s2)] - sel[(r, k, s1, s1)] for r, k in folds])
print()
for a in ["mcar_q30", "mar_q30"]:
    show(f"|shift {a}->mnar_y| - |shift mnar_y->{a}|",
         [abs(sel[(r, k, a, "mnar_y_q30")] - sel[(r, k, a, a)]) - abs(sel[(r, k, "mnar_y_q30", a)] - sel[(r, k, "mnar_y_q30", "mnar_y_q30")]) for r, k in folds])
print()
for s in ["mcar_q30", "mar_q30", "mnar_y_q30"]:
    gg = ev[(ev.source == s) & (ev.target != s)]
    print(f"oracle mismatch on other q = 0.30 targets, source {s}: {int((gg.sel < gg.best).sum())}/{len(gg)}, mean regret {(gg.best - gg.sel).mean():.4f}, max {(gg.best - gg.sel).max():.4f}")
