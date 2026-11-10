"""Repeated outer cross-validation for the fold-level contrasts (Checklist R2.1).

Repeats the main study's design (median imputation pipeline, five families, nested
selection, StratifiedGroupKFold on subject_id, K = 5) R = 5 times with a different outer
split seed AND a different mask seed per repeat, giving 25 outer folds per condition and
five mask realisations per condition. Repeat 0 uses the main study's seeds, so it also
checks that this code reproduces the main study.

Grid = the main RQ1/RQ2 grid: natural (q = 0); MCAR, MAR (OR 2), MNAR-Y (OR 2) at
q in {0.10, 0.20, 0.30, 0.40}; MNAR-Y at q = 0.30 with OR 1.5 and OR 4.
Frozen q = 0.30 sources are also scored on the q = 0.30 targets (MCAR, MAR, MNAR-Y OR 2)
of the same fold for the shift-asymmetry contrast.

Statistics (written to r1_repeated_cv_tests.csv):
  * displacement pooled over rates and folds, Fisher exact test MCAR vs MNAR-Y and MAR vs MNAR-Y;
  * paired fold differences (shift asymmetry; OR 4 - OR 2; OR 2 - OR 1.5): exact sign test over the
    25 folds AND the Nadeau-Bengio corrected resampled t-test, because folds from repeated CV share
    training data and are not independent (an uncorrected test over 25 folds would be anti-conservative).

Run from this folder (Revision_R1/Code):
    python run_r1_repeated_outer_cv.py               # full run (about 7-8 h on the main-study machine)
    python run_r1_repeated_outer_cv.py --smoke-test  # quick end-to-end check
Re-running resumes from the per-cell checkpoints.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

import run_phase4_rq1_rq2 as p4
import run_r1_missingness_handling as h
import selection_v2 as selv2

N_REPEATS = 5
SEEDS = [(p4.OUTER_SEED + r, p4.MASK_SEED + r) for r in range(N_REPEATS)]  # repeat 0 = main study
CONDITIONS = {"natural_q00": ("natural", 0.0, 0.0)}
for _m, _th in [("mcar", 0.0), ("mar", p4.THETA_MAIN), ("mnar_y", p4.THETA_MAIN)]:
    for _q in p4.RATES_MAIN:
        CONDITIONS[f"{_m}_q{int(round(_q * 100)):02d}"] = (_m, _q, _th)
CONDITIONS["mnar_y_q30_OR1.5"] = ("mnar_y", 0.30, float(np.log(1.5)))
CONDITIONS["mnar_y_q30_OR4"] = ("mnar_y", 0.30, float(np.log(4.0)))
SHIFT_SET = ["mcar_q30", "mar_q30", "mnar_y_q30"]


def run_cell(cid, masks, y_tr, g_tr, y_te, num_cols, cat_cols, seed):
    t0 = time.time()
    fit = selv2.fit_and_select_all_families(masks[cid]["X_tr"], y_tr, g_tr, num_cols, cat_cols, seed=seed)
    fit_s = time.time() - t0
    rows = []
    for tgt in (SHIFT_SET if cid in SHIFT_SET else [cid]):
        row = dict(source=cid, target=tgt, selected_family=fit.selected_family, fit_seconds=fit_s)
        for fam, model in fit.final_models.items():
            row[f"auroc_{fam}"] = selv2.evaluate_frozen_model(model, masks[tgt]["X_te"], y_te)[0]
        rows.append(row)
    return rows


def nadeau_bengio(d: np.ndarray, n_train: float, n_test: float):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for J correlated CV differences."""
    j = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1.0 / j + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def sign_test(d: np.ndarray):
    d = d[d != 0]
    return int((d > 0).sum()), len(d), float(stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue) if len(d) else float("nan")


def analyse(ev: pd.DataFrame, n_train: float, n_test: float):
    fams = [c[6:] for c in ev.columns if c.startswith("auroc_")]
    ev = ev.copy()
    ev["selected_auroc"] = ev.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
    ev["regret"] = ev[[f"auroc_{f}" for f in fams]].max(axis=1) - ev["selected_auroc"]
    diag = ev[ev.source == ev.target].set_index(["repeat", "fold_id", "source"])
    base = diag.xs("natural_q00", level="source")["selected_family"]
    diag = diag.reset_index()
    diag["displaced"] = [s != base.loc[(r, k)] for r, k, s in zip(diag.repeat, diag.fold_id, diag.selected_family)]
    summ = diag.groupby("source").agg(n_folds=("fold_id", "size"), mean_auroc=("selected_auroc", "mean"),
                                      sd_auroc=("selected_auroc", "std"), displacement=("displaced", "mean"),
                                      mean_regret=("regret", "mean")).reset_index()
    tests = []
    pooled = {m: diag[diag.source.str.match(fr"^{m}_q\d\d$") & (diag.source != "natural_q00")] for m in ["mcar", "mar", "mnar_y"]}
    for a in ["mcar", "mar"]:
        ka, na = int(pooled[a].displaced.sum()), len(pooled[a])
        kb, nb = int(pooled["mnar_y"].displaced.sum()), len(pooled["mnar_y"])
        p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        tests.append(dict(contrast=f"displacement {a} vs mnar_y (pooled rates)", estimate=ka / na - kb / nb,
                          detail=f"{ka}/{na} vs {kb}/{nb}", test="Fisher exact", p_value=p))
    key = ["repeat", "fold_id"]
    sel = ev.set_index(key + ["source", "target"])["selected_auroc"]

    def paired(name, d):
        k, n, p_sign = sign_test(d)
        t, p_nb = nadeau_bengio(d, n_train, n_test)
        tests.append(dict(contrast=name, estimate=float(d.mean()), detail=f"{k}/{n} folds positive", test="exact sign test",
                          p_value=p_sign))
        tests.append(dict(contrast=name, estimate=float(d.mean()), detail=f"t={t:.3f}, J={len(d)}",
                          test="Nadeau-Bengio corrected t", p_value=p_nb))

    folds = diag[key].drop_duplicates()
    for a in ["mcar", "mar"]:
        s1, s2 = f"{a}_q30", "mnar_y_q30"
        d = []
        for r, k in folds.itertuples(index=False):
            fwd = sel[(r, k, s1, s2)] - sel[(r, k, s1, s1)]
            back = sel[(r, k, s2, s1)] - sel[(r, k, s2, s2)]
            d.append(abs(fwd) - abs(back))
        paired(f"|shift {a}->mnar_y| - |shift mnar_y->{a}| (q=0.30)", np.array(d))
    for hi, lo in [("mnar_y_q30_OR4", "mnar_y_q30"), ("mnar_y_q30", "mnar_y_q30_OR1.5")]:
        d = np.array([sel[(r, k, hi, hi)] - sel[(r, k, lo, lo)] for r, k in folds.itertuples(index=False)])
        paired(f"selected AUROC {hi} - {lo}", d)
    return summ, pd.DataFrame(tests)


def main(smoke: bool, data_path: str | None):
    out_dir = Path("../Results/R1_repeated_outer_cv" + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    df = h.load_df(data_path)
    seeds, conds = SEEDS, CONDITIONS
    if smoke:
        # shortcut: smoke test subsamples subjects, 2 repeats x 2 folds, 6 conditions; code-path check only
        keep = pd.Series(df.subject_id.unique()).sample(frac=0.25, random_state=0)
        df = df[df.subject_id.isin(keep)].reset_index(drop=True)
        seeds = SEEDS[:2]
        conds = {c: CONDITIONS[c] for c in ["natural_q00", "mcar_q30", "mar_q30", "mnar_y_q30",
                                            "mnar_y_q30_OR1.5", "mnar_y_q30_OR4"]}
    n_outer = 2 if smoke else p4.N_OUTER
    log(f"=== R1 repeated outer CV start (smoke={smoke}, rows={len(df)}, repeats={len(seeds)}, n_outer={n_outer}) ===")

    log(h.versions())
    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    h.check_columns(num_cols, cat_cols)
    rows, n_train, n_test = [], [], []
    for rep, (outer_seed, mask_seed) in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=outer_seed)
        for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
            assert set(groups[tr]).isdisjoint(set(groups[te])), "subject_id leaked across outer train/test"
            n_train.append(len(tr)); n_test.append(len(te))
            X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
            masks = h.fold_masks(df_tr, y[tr], df_te, y[te], X_tr, X_te, conditions=conds, mask_seed=mask_seed)
            for cid in conds:
                ckpt = out_dir / f"rep{rep}__{cid}__fold{fold_id}.pkl"
                if ckpt.exists():
                    cell = pickle.loads(ckpt.read_bytes())
                else:
                    cell = run_cell(cid, masks, y[tr], groups[tr], y[te], num_cols, cat_cols, seed=outer_seed)
                    ckpt.write_bytes(pickle.dumps(cell))
                    log(f"  [rep {rep} {cid} fold {fold_id}] selected={cell[0]['selected_family']} "
                        f"auroc={cell[0]['auroc_' + cell[0]['selected_family']]:.4f} ({cell[0]['fit_seconds']:.0f}s)")
                for r in cell:
                    rows.append(dict(repeat=rep, outer_seed=outer_seed, mask_seed=mask_seed, fold_id=fold_id, **r))

    ev = pd.DataFrame(rows)
    ev.to_csv(out_dir / "r1_repeated_cv_eval_table.csv", index=False)
    summ, tests = analyse(ev, float(np.mean(n_train)), float(np.mean(n_test)))
    summ.to_csv(out_dir / "r1_repeated_cv_summary.csv", index=False)
    tests.to_csv(out_dir / "r1_repeated_cv_tests.csv", index=False)
    log(f"Wrote {len(ev)} evaluation rows, summary and tests to {out_dir}")
    log("=== R1 repeated outer CV finished ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-path", default=None)
    a = ap.parse_args()
    main(a.smoke_test, a.data_path)
