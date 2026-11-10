"""Multiple-imputation arm (m = 5) for the missingness-handling comparison.

Same folds, masks, families, grids, selection rule, conditions, targets and permutation draws as
run_r1_missingness_handling.py; only the handling of missing laboratory values differs:

  mi5   M = 5 chained-equation imputations (scikit-learn IterativeImputer, BayesianRidge,
        sample_posterior=True, max_iter 10, random_state 2026 + m, outcome NOT used), each fitted
        inside every training pipeline. For every imputation m the five families are tuned and
        scored by inner CV exactly as in the main study. The family is selected on the inner AUROC
        (tie-break inner AP) averaged over the five imputations, and each family's prediction is the
        mean of its five imputation-specific predicted risks (pooling of predictions, not of
        coefficients).

Output: ../Results/R1_multiple_imputation/ with the same table layout as the handling driver
(strategy = "mi5"), so the two can be analysed together.

With --m 1 the arm keeps one posterior draw (seed 2026, the first imputation of mi5) and no
averaging (strategy "mi1", output ../Results/R1_multiple_imputation_m1/); comparing mi1 with the
single chained-equation arm isolates the posterior draw, and mi5 with mi1 isolates the averaging.

Run from this folder (Revision_R1/Code):
    python run_r1_multiple_imputation.py               # full run (about five times the mice arm)
    python run_r1_multiple_imputation.py --m 1         # one-draw arm (about the cost of the mice arm)
    python run_r1_multiple_imputation.py --smoke-test  # quick end-to-end check
Re-running resumes from the per-cell checkpoints in the results folder.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import run_phase4_rq1_rq2 as p4
import run_r1_missingness_handling as h
import selection_v2 as selv2

br = h.br
M_IMPUTATIONS = 5
STRATEGY = "mi5"


def build_preprocessor_mi(seed_m):
    """h.build_preprocessor_mice with posterior sampling and imputation-specific seed."""
    def build(num_cols, cat_cols, *, include_imputer=True, include_scaler=True):
        num = Pipeline([
            ("to_float", FunctionTransformer(h._to_float_copy)),
            ("imputer", IterativeImputer(max_iter=10, sample_posterior=True, random_state=seed_m,
                                         keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("sanitize", FunctionTransformer(br._final_sanitize_numeric, feature_names_out="one-to-one")),
        ])
        cat = Pipeline([
            ("imputer", br.SafeMostFrequentImputer(fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)],
                                 remainder="drop", sparse_threshold=0.0)
    return build


def _fitted_imputers(model):
    """The fitted IterativeImputer objects inside a (possibly calibrated) pipeline."""
    pipe = getattr(model, "base_estimator", model)  # PrefitCalibrator wraps the fitted pipeline
    found = []
    for _, step in getattr(pipe, "steps", []):
        for _, trans, _ in getattr(step, "transformers_", []):
            for _, sub in getattr(trans, "steps", []):
                if isinstance(sub, IterativeImputer):
                    found.append(sub)
    return found


class MIAverage:
    """Frozen ensemble over imputations: mean predicted risk of the M imputation-specific models.

    With sample_posterior=True, IterativeImputer.transform draws new values on every call; its RNG is
    reset before each prediction so the same test matrix always gets the same imputations."""

    def __init__(self, members):
        self.members = members
        self._imputers = [_fitted_imputers(m) for m in members]
        assert all(self._imputers), "no IterativeImputer found in an imputation-specific model"

    def predict_proba(self, X):
        ps = []
        for m, imps in zip(self.members, self._imputers):
            for imp in imps:
                imp.random_state_ = np.random.RandomState(imp.random_state)
            ps.append(br.predict_proba_safe(m, X))
        p = np.mean(ps, axis=0)
        return np.column_stack([1.0 - p, p])


def fit_select_mi(strategy, X_tr, y_tr, g_tr, num_cols, cat_cols):
    assert strategy == STRATEGY
    fits, original = [], br.build_preprocessor
    try:
        for m in range(M_IMPUTATIONS):
            br.build_preprocessor = build_preprocessor_mi(h.OUTER_SEED + m)
            fits.append(selv2.fit_and_select_all_families(X_tr, y_tr, g_tr, num_cols, cat_cols, seed=h.OUTER_SEED))
    finally:
        br.build_preprocessor = original
    fams = list(fits[0].inner_scores)
    inner = {f: (float(np.mean([r.inner_scores[f][0] for r in fits])),
                 float(np.mean([r.inner_scores[f][1] for r in fits])), 0.0) for f in fams}
    return selv2.FitAndSelectResult(
        selected_family=selv2.select_family(inner),
        final_models={f: MIAverage([r.final_models[f] for r in fits]) for f in fams},
        best_params={f: [r.best_params[f] for r in fits] for f in fams},
        inner_scores=inner,
        inner_metrics=inner,
    )


def main(smoke: bool, data_path: str | None, m: int = 5):
    global M_IMPUTATIONS, STRATEGY
    M_IMPUTATIONS, STRATEGY = m, f"mi{m}"
    out_dir = Path("../Results/R1_multiple_imputation" + ("" if m == 5 else f"_m{m}") + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    h.fit_select = fit_select_mi  # run_cell looks fit_select up in its own module at call time
    df = h.load_df(data_path)
    n_outer, n_perm, conds = h.N_OUTER, h.N_PERM, list(h.CONDITIONS)
    if smoke:
        # shortcut: smoke test subsamples subjects and folds; only checks the code path end to end
        keep = pd.Series(df.subject_id.unique()).sample(frac=0.25, random_state=0)
        df = df[df.subject_id.isin(keep)].reset_index(drop=True)
        n_outer, n_perm, conds = 2, 2, ["natural_q00", "mnar_y_q30_OR2"]
        M_IMPUTATIONS = min(m, 2)  # smoke only: at most two imputations, first fold
    log(f"=== multiple-imputation driver start (smoke={smoke}, rows={len(df)}, n_outer={n_outer}, M={M_IMPUTATIONS}) ===")
    log(h.versions())

    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    h.check_columns(num_cols, cat_cols)
    splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=h.OUTER_SEED)
    eval_rows, perm_rows = [], []
    for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        assert set(groups[tr]).isdisjoint(set(groups[te])), "subject_id leaked across outer train/test"
        if smoke and fold_id > 1:
            break
        X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        masks = h.fold_masks(df_tr, y[tr], df_te, y[te], X_tr, X_te)
        log(f"fold {fold_id}: n_train={len(tr)} n_test={len(te)} masks ready")
        for cid in conds:
            ckpt = out_dir / f"{STRATEGY}__{cid}__fold{fold_id}.pkl"
            if ckpt.exists():
                ev, pm = pickle.loads(ckpt.read_bytes())
                log(f"  [{STRATEGY} {cid} fold {fold_id}] checkpoint found, skipping")
            else:
                ev, pm = h.run_cell(STRATEGY, cid, fold_id, masks, X_te, y[tr], groups[tr], y[te],
                                    num_cols, cat_cols, n_perm)
                ckpt.write_bytes(pickle.dumps((ev, pm)))
                diag = [r for r in ev if r["target"] == cid][0]
                log(f"  [{STRATEGY} {cid} fold {fold_id}] selected={diag['selected_family']} "
                    f"auroc={diag['auroc_' + diag['selected_family']]:.4f} ({diag['fit_seconds']:.0f}s)")
            eval_rows += ev
            perm_rows += pm

    ev_df, pm_df = pd.DataFrame(eval_rows), pd.DataFrame(perm_rows)
    ev_df.to_csv(out_dir / "r1_mi_eval_table.csv", index=False)
    pm_df.to_csv(out_dir / "r1_mi_permutation_table.csv", index=False)
    log(f"Wrote {len(ev_df)} evaluation rows and {len(pm_df)} permutation rows to {out_dir}")
    log("=== multiple-imputation driver finished ===")
    log_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--m", type=int, default=5, help="number of imputations (5 = main arm, 1 = one-draw arm)")
    a = ap.parse_args()
    main(a.smoke_test, a.data_path, a.m)
