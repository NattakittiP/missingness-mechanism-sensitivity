"""Missingness-handling comparison (Checklist R1.1a, R1.1b, R1.2a).

Re-runs the decision-robustness measurements under three ways of handling missing
laboratory values, with everything else held identical to the main study
(same outer folds, same synthetic masks, same five families, same nested selection):

  median      within-fold median imputation + scaling (the main study's pipeline)
  median_ind  as `median`, plus one explicit missingness indicator per eligible lab
              (same construction as the indicator ablation of run_phase6_rq4.py)
  mice        chained-equation regression imputation (scikit-learn IterativeImputer,
              BayesianRidge, single imputation, outcome NOT used), fitted inside every
              training fold, followed by the same scaling

Conditions (training mechanism at q = 0.30 unless natural): natural (q = 0), MCAR,
MAR (OR 2), MNAR-Y (OR 2), MNAR-Y (OR 4).

Per strategy, condition and outer fold it records:
  * the selected family and the outer-test AUROC/AP/Brier of all five frozen families
    on every target mechanism at q = 0.30 (MCAR, MAR, MNAR-Y OR 2) -> displacement,
    regret and frozen source-to-target shift are computed from this table;
  * for MNAR-Y sources, test-time permutation of the synthetic mask rows (R draws).

Run from this folder (Revision_R1/Code):
    python run_r1_missingness_handling.py               # full run
    python run_r1_missingness_handling.py --smoke-test  # quick end-to-end check
Re-running resumes from the per-cell checkpoints in the results folder.
"""
from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import missingness_generator_v2 as genmod
import run_phase4_rq1_rq2 as p4
import selection_v2 as selv2

br = selv2.base_runner
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*[Ee]arly stopping criterion not reached.*")

OUTER_SEED, MASK_SEED, N_OUTER = p4.OUTER_SEED, p4.MASK_SEED, p4.N_OUTER
RATE = 0.30
STRATEGIES = ["median", "median_ind", "mice"]
# condition_id -> (mechanism, rate, theta)
CONDITIONS = {
    "natural_q00": ("natural", 0.0, 0.0),
    "mcar_q30": ("mcar", RATE, 0.0),
    "mar_q30": ("mar", RATE, p4.THETA_MAIN),
    "mnar_y_q30_OR2": ("mnar_y", RATE, p4.THETA_MAIN),
    "mnar_y_q30_OR4": ("mnar_y", RATE, float(np.log(4.0))),
}
TARGETS = ["mcar_q30", "mar_q30", "mnar_y_q30_OR2"]
PERM_SOURCES = ["mnar_y_q30_OR2", "mnar_y_q30_OR4"]
N_PERM = 30
IND_COLS = [f"missing_{c}" for c in genmod.ELIGIBLE_LAB_COLS]
EXPECTED_CAT = ["admission_location", "admission_type", "anchor_year_group", "gender", "insurance",
                "marital_status", "race"]


def load_df(data_path):
    df = pd.read_csv(p4._resolve_data_path(data_path))
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    # pandas >= 3 reads text columns as dtype "str", which split_columns_A does not treat as
    # categorical (they would be coerced to NaN and dropped); object dtype restores pandas-2 behaviour.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object)
    return df


def check_columns(num_cols, cat_cols):
    assert sorted(cat_cols) == EXPECTED_CAT, f"categorical columns differ from the main study: {cat_cols}"
    assert set(genmod.ELIGIBLE_LAB_COLS) <= set(num_cols), "eligible lab columns missing from numeric features"


def versions():
    import sklearn, xgboost, platform
    return (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}")


def _to_float_copy(X):
    # writable float copy: IterativeImputer writes into its input buffer, which pandas may expose read-only
    return np.array(X, dtype=float, copy=True)


def build_preprocessor_mice(num_cols, cat_cols, *, include_imputer=True, include_scaler=True):
    """Same ColumnTransformer as br.build_preprocessor, with IterativeImputer for numbers."""
    num = Pipeline([
        ("to_float", FunctionTransformer(_to_float_copy)),
        ("imputer", IterativeImputer(max_iter=10, random_state=OUTER_SEED, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("sanitize", FunctionTransformer(br._final_sanitize_numeric, feature_names_out="one-to-one")),
    ])
    cat = Pipeline([
        ("imputer", br.SafeMostFrequentImputer(fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)],
                             remainder="drop", sparse_threshold=0.0)


def with_strategy(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy != "median_ind":
        return X
    X = X.copy()
    for c in genmod.ELIGIBLE_LAB_COLS:
        X[f"missing_{c}"] = X[c].isna().astype(float)
    return X


def fit_select(strategy, X_tr, y_tr, g_tr, num_cols, cat_cols):
    cols = num_cols + IND_COLS if strategy == "median_ind" else num_cols
    original = br.build_preprocessor
    if strategy == "mice":
        br.build_preprocessor = build_preprocessor_mice
    try:
        return selv2.fit_and_select_all_families(with_strategy(X_tr, strategy), y_tr, g_tr, cols, cat_cols,
                                                 seed=OUTER_SEED)
    finally:
        br.build_preprocessor = original


def apply_mask(X: pd.DataFrame, final_mask: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in genmod.ELIGIBLE_LAB_COLS:
        X.loc[final_mask[c].to_numpy(), c] = np.nan
    return X


def fold_masks(df_tr, y_tr, df_te, y_te, X_tr, X_te, conditions=None, mask_seed=MASK_SEED):
    """Masked train/test X for every condition; RNG fresh per (condition, fold), train then
    test, exactly as run_phase4_rq1_rq2.run_condition_with_injection does."""
    nat_tr, nat_te = genmod.compute_natural_mask(df_tr), genmod.compute_natural_mask(df_te)
    out = {}
    for cid, (mech, rate, theta) in (conditions or CONDITIONS).items():
        if mech == "natural":
            out[cid] = dict(X_tr=apply_mask(X_tr, nat_tr), X_te=apply_mask(X_te, nat_te), syn_te=None, nat_te=nat_te)
            continue
        gen, d_tr, d_te = p4._fit_generator_and_drivers(mech, rate, theta, df_tr, y_tr, nat_tr, df_te, y_te)
        rng = np.random.default_rng(mask_seed)
        syn_tr = genmod.apply_synthetic_mask(nat_tr, d_tr, gen, rng)
        syn_te = genmod.apply_synthetic_mask(nat_te, d_te, gen, rng)
        out[cid] = dict(X_tr=apply_mask(X_tr, genmod.compute_final_mask(nat_tr, syn_tr)),
                        X_te=apply_mask(X_te, genmod.compute_final_mask(nat_te, syn_te)),
                        syn_te=syn_te, nat_te=nat_te, syn_tr=syn_tr)
    return out


def run_cell(strategy, cid, fold_id, masks, X_te_raw, y_tr, g_tr, y_te, num_cols, cat_cols, n_perm):
    t0 = time.time()
    fit = fit_select(strategy, masks[cid]["X_tr"], y_tr, g_tr, num_cols, cat_cols)
    fit_s = time.time() - t0
    eval_rows = []
    targets = [cid] if cid == "natural_q00" else sorted(set(TARGETS) | {cid})
    for tgt in targets:
        X_eval = with_strategy(masks[tgt]["X_te"], strategy)
        row = dict(strategy=strategy, source=cid, target=tgt, fold_id=fold_id, selected_family=fit.selected_family,
                   inner_auroc_selected=fit.inner_scores[fit.selected_family][0], fit_seconds=fit_s)
        for fam, model in fit.final_models.items():
            row[f"auroc_{fam}"], row[f"ap_{fam}"], row[f"brier_{fam}"] = selv2.evaluate_frozen_model(model, X_eval, y_te)
        eval_rows.append(row)
    perm_rows = []
    if cid in PERM_SOURCES:
        sel = fit.final_models[fit.selected_family]
        before = selv2.evaluate_frozen_model(sel, with_strategy(masks[cid]["X_te"], strategy), y_te)[0]
        syn, nat = masks[cid]["syn_te"], masks[cid]["nat_te"]
        for draw in range(n_perm):
            perm = np.random.default_rng([MASK_SEED, fold_id, draw]).permutation(len(syn))
            syn_p = pd.DataFrame(syn.to_numpy()[perm], columns=syn.columns)
            X_p = with_strategy(apply_mask(X_te_raw, genmod.compute_final_mask(nat, syn_p)), strategy)
            after = selv2.evaluate_frozen_model(sel, X_p, y_te)[0]
            perm_rows.append(dict(strategy=strategy, source=cid, fold_id=fold_id, draw=draw,
                                  selected_family=fit.selected_family, auroc_before=before, auroc_after=after))
    return eval_rows, perm_rows


def summarise(ev: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    fams = [c[6:] for c in ev.columns if c.startswith("auroc_")]
    ev = ev.copy()
    ev["selected_auroc"] = ev.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
    ev["oracle_auroc"] = ev[[f"auroc_{f}" for f in fams]].max(axis=1)
    ev["regret"] = ev["oracle_auroc"] - ev["selected_auroc"]
    diag = ev[ev.source == ev.target].set_index(["strategy", "source", "fold_id"])
    base_sel = diag.xs("natural_q00", level="source")["selected_family"]
    rows = []
    for (s, src), g in diag.groupby(level=["strategy", "source"]):
        disp = [g.loc[(s, src, k), "selected_family"] != base_sel.loc[(s, k)] for k in g.index.get_level_values("fold_id")]
        rows.append(dict(strategy=s, source=src, target=src, n_folds=len(g), mean_selected_auroc=g.selected_auroc.mean(),
                         displacement=float(np.mean(disp)), mean_regret=g.regret.mean(), shift_delta=0.0))
    off = ev[(ev.source != ev.target)]
    for (s, src, tgt), g in off.groupby(["strategy", "source", "target"]):
        d = diag.loc[[(s, src, k) for k in g.fold_id], "selected_auroc"].to_numpy()
        rows.append(dict(strategy=s, source=src, target=tgt, n_folds=len(g), mean_selected_auroc=g.selected_auroc.mean(),
                         displacement=np.nan, mean_regret=g.regret.mean(),
                         shift_delta=float(np.mean(g.selected_auroc.to_numpy() - d))))
    out = pd.DataFrame(rows)
    if len(pm):
        coll = pm.assign(collapse=pm.auroc_before - pm.auroc_after).groupby(["strategy", "source"]).collapse.mean()
        out = out.merge(coll.rename("mean_perm_collapse").reset_index(), on=["strategy", "source"], how="left")
    return out.sort_values(["strategy", "source", "target"])


def main(smoke: bool, data_path: str | None):
    out_dir = Path("../Results/R1_missingness_handling" + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    df = load_df(data_path)
    n_outer, n_perm, strategies, conds = N_OUTER, N_PERM, STRATEGIES, list(CONDITIONS)
    if smoke:
        # shortcut: smoke test subsamples subjects and folds; only checks the code path end to end
        keep = pd.Series(df.subject_id.unique()).sample(frac=0.25, random_state=0)
        df = df[df.subject_id.isin(keep)].reset_index(drop=True)
        n_outer, n_perm = 2, 2
    log(f"=== R1 missingness-handling driver start (smoke={smoke}, rows={len(df)}, n_outer={n_outer}) ===")
    log(versions())

    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    check_columns(num_cols, cat_cols)
    splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    eval_rows, perm_rows = [], []
    for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        assert set(groups[tr]).isdisjoint(set(groups[te])), "subject_id leaked across outer train/test"
        if smoke and fold_id > 1:
            break
        X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        masks = fold_masks(df_tr, y[tr], df_te, y[te], X_tr, X_te)
        log(f"fold {fold_id}: n_train={len(tr)} n_test={len(te)} masks ready")
        for s in strategies:
            for cid in conds:
                ckpt = out_dir / f"{s}__{cid}__fold{fold_id}.pkl"
                if ckpt.exists():
                    ev, pm = pickle.loads(ckpt.read_bytes())
                    log(f"  [{s} {cid} fold {fold_id}] checkpoint found, skipping")
                else:
                    ev, pm = run_cell(s, cid, fold_id, masks, X_te, y[tr], groups[tr], y[te], num_cols, cat_cols, n_perm)
                    ckpt.write_bytes(pickle.dumps((ev, pm)))
                    diag = [r for r in ev if r["target"] == cid][0]
                    log(f"  [{s} {cid} fold {fold_id}] selected={diag['selected_family']} "
                        f"auroc={diag['auroc_' + diag['selected_family']]:.4f} ({diag['fit_seconds']:.0f}s)")
                eval_rows += ev
                perm_rows += pm

    ev_df, pm_df = pd.DataFrame(eval_rows), pd.DataFrame(perm_rows)
    ev_df.to_csv(out_dir / "r1_handling_eval_table.csv", index=False)
    pm_df.to_csv(out_dir / "r1_handling_permutation_table.csv", index=False)
    summarise(ev_df, pm_df).to_csv(out_dir / "r1_handling_summary.csv", index=False)
    log(f"Wrote {len(ev_df)} evaluation rows, {len(pm_df)} permutation rows and the summary to {out_dir}")
    log("=== R1 missingness-handling driver finished ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-path", default=None)
    a = ap.parse_args()
    main(a.smoke_test, a.data_path)
