"""Test-time mask permutation for every MNAR-Y strength, and the full source-to-target shift matrix over rates.

Part "perm": MNAR-Y cells (q, OR) = (0.30, 1.5), (0.30, 2), (0.30, 4), (0.40, 2), (0.40, 4) on the
main-study folds and masks (outer-split seed 2026, mask seed 20260917). Per cell and fold: nested selection on the masked training rows (same code as
the main study), outer-test AUROC and predictions of the frozen selected model, 30 test-time row
permutations of the injected test mask (seed tuple of
run_phase6_section_b_repeats.py), and the AUROC of a logistic regression on the injected mask alone (same
construction as the mask-only baseline of run_phase6_rq4.py). The OR 2 and OR 4 cells reproduce the
main permutation results and serve as a check; the predictions give the pooled case-level AUROC.

Part "rate": MCAR, MAR (OR 2) and MNAR-Y (OR 2) sources at q in {0.10, 0.30, 0.50}, main-study seeds
(5 folds); every source is frozen and scored on the three targets at the same q. The q = 0.30 cells
and the MCAR source at q = 0.10 and 0.50 reproduce the main shift results.

Run from this folder (Revision_R1/Code):
    python run_r1_permutation_rate_shift.py --part perm     # about 35 min
    python run_r1_permutation_rate_shift.py --part rate     # about 1 h
    python run_r1_permutation_rate_shift.py --smoke-test    # quick end-to-end check of both parts
Re-running resumes from the per-cell checkpoints.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as genmod
import run_phase4_rq1_rq2 as p4
import run_r1_missingness_handling as h
import selection_v2 as selv2

N_PERM = 30
PERM_SEED_OFFSET = 555  # same constant as run_phase6_rq4.py / run_phase6_section_b_repeats.py
PERM_CELLS = {
    "mnar_y_q30_OR1.5": ("mnar_y", 0.30, float(np.log(1.5))),
    "mnar_y_q30_OR2": ("mnar_y", 0.30, p4.THETA_MAIN),
    "mnar_y_q30_OR4": ("mnar_y", 0.30, float(np.log(4.0))),
    "mnar_y_q40_OR2": ("mnar_y", 0.40, p4.THETA_MAIN),
    "mnar_y_q40_OR4": ("mnar_y", 0.40, float(np.log(4.0))),
}
OR_OF = {"mnar_y_q30_OR1.5": 1.5, "mnar_y_q30_OR2": 2.0, "mnar_y_q30_OR4": 4.0, "mnar_y_q40_OR2": 2.0,
         "mnar_y_q40_OR4": 4.0}
RATE_GRID = [0.10, 0.30, 0.50]
RATE_CELLS = {f"{m}_q{int(round(q * 100)):02d}": (m, q, th) for q in RATE_GRID
              for m, th in [("mcar", 0.0), ("mar", p4.THETA_MAIN), ("mnar_y", p4.THETA_MAIN)]}


def perm_entropy(mask_seed, fold_id, rate, or_val, draw_id):
    """Seed tuple of run_phase6_section_b_repeats._perm_seed_entropy."""
    base = (mask_seed, PERM_SEED_OFFSET, fold_id, int(round(rate * 100)), int(round(or_val * 10)))
    return base if draw_id == 0 else base + (draw_id,)


def mask_only_auroc(syn_tr, syn_te, y_tr, y_te):
    """run_phase6_rq4._fit_eval_baseline, mask part, on the injected mask alone."""
    cols = genmod.ELIGIBLE_LAB_COLS
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(syn_tr[cols].to_numpy().astype(float), y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(syn_te[cols].to_numpy().astype(float))[:, 1]))


def perm_cell(cid, masks, X_te, y_tr, g_tr, y_te, te_idx, num_cols, cat_cols, outer_seed, mask_seed, fold_id, n_perm):
    _, rate, _ = PERM_CELLS[cid]
    t0 = time.time()
    fit = selv2.fit_and_select_all_families(masks[cid]["X_tr"], y_tr, g_tr, num_cols, cat_cols, seed=outer_seed)
    fit_s = time.time() - t0
    model = fit.final_models[fit.selected_family]
    before, _, _, _, p_te = selv2.evaluate_frozen_model_with_predictions(model, masks[cid]["X_te"], y_te)
    syn, nat = masks[cid]["syn_te"], masks[cid]["nat_te"]
    draws = []
    for d in range(n_perm):
        perm = np.random.default_rng(perm_entropy(mask_seed, fold_id, rate, OR_OF[cid], d)).permutation(len(syn))
        syn_p = syn.iloc[perm].reset_index(drop=True)
        after = selv2.evaluate_frozen_model(model, h.apply_mask(X_te, genmod.compute_final_mask(nat, syn_p)), y_te)[0]
        draws.append(dict(draw_id=d, auroc_after=after, delta_auroc=after - before))
    info = dict(selected_family=fit.selected_family, auroc_before=before, fit_seconds=fit_s,
                mask_only_auroc=mask_only_auroc(masks[cid]["syn_tr"], syn, y_tr, y_te))
    preds = pd.DataFrame(dict(row_index=te_idx, y=y_te, p=p_te))
    return info, draws, preds


def rate_cell(cid, masks, y_tr, g_tr, y_te, num_cols, cat_cols, seed):
    q = RATE_CELLS[cid][1]
    t0 = time.time()
    fit = selv2.fit_and_select_all_families(masks[cid]["X_tr"], y_tr, g_tr, num_cols, cat_cols, seed=seed)
    fit_s = time.time() - t0
    rows = []
    for tgt in [c for c, (_, qq, _) in RATE_CELLS.items() if qq == q]:
        row = dict(source=cid, target=tgt, rate=q, selected_family=fit.selected_family, fit_seconds=fit_s)
        for fam, model in fit.final_models.items():
            row[f"auroc_{fam}"] = selv2.evaluate_frozen_model(model, masks[tgt]["X_te"], y_te)[0]
        rows.append(row)
    return rows


def main(parts, smoke: bool, data_path: str | None):
    out_dir = Path("../Results/R1_permutation_rate_shift" + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    df = h.load_df(data_path)
    seeds, n_outer, n_perm = [(p4.OUTER_SEED, p4.MASK_SEED)], p4.N_OUTER, N_PERM
    perm_cells, rate_cells = list(PERM_CELLS), list(RATE_CELLS)
    if smoke:
        # shortcut: smoke test subsamples subjects, 2 folds, 2 draws; code-path check only
        keep = pd.Series(df.subject_id.unique()).sample(frac=0.25, random_state=0)
        df = df[df.subject_id.isin(keep)].reset_index(drop=True)
        n_outer, n_perm = 2, 2
        perm_cells, rate_cells = ["mnar_y_q30_OR1.5", "mnar_y_q40_OR4"], ["mcar_q10", "mar_q10", "mnar_y_q10"]
    log(f"=== permutation / rate-shift driver start (parts={parts}, smoke={smoke}, rows={len(df)}) ===")
    log(h.versions())
    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    h.check_columns(num_cols, cat_cols)

    if "perm" in parts:
        cells, draws, preds = [], [], []
        for rep, (outer_seed, mask_seed) in enumerate(seeds):
            splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=outer_seed)
            for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
                assert set(groups[tr]).isdisjoint(set(groups[te])), "subject_id leaked across outer train/test"
                X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
                df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
                masks = h.fold_masks(df_tr, y[tr], df_te, y[te], X_tr, X_te,
                                     conditions={c: PERM_CELLS[c] for c in perm_cells}, mask_seed=mask_seed)
                for cid in perm_cells:
                    ckpt = out_dir / f"perm__rep{rep}__{cid}__fold{fold_id}.pkl"
                    if ckpt.exists():
                        info, dr, pr = pickle.loads(ckpt.read_bytes())
                    else:
                        info, dr, pr = perm_cell(cid, masks, X_te, y[tr], groups[tr], y[te], te, num_cols, cat_cols,
                                                 outer_seed, mask_seed, fold_id, n_perm)
                        ckpt.write_bytes(pickle.dumps((info, dr, pr)))
                        log(f"  [perm rep {rep} {cid} fold {fold_id}] selected={info['selected_family']} "
                            f"before={info['auroc_before']:.4f} mean_delta={np.mean([d['delta_auroc'] for d in dr]):+.4f} "
                            f"mask_only={info['mask_only_auroc']:.4f} ({info['fit_seconds']:.0f}s)")
                    key = dict(repeat=rep, outer_seed=outer_seed, mask_seed=mask_seed, fold_id=fold_id, cell=cid,
                               rate=PERM_CELLS[cid][1], or_value=OR_OF[cid])
                    cells.append(dict(**key, n_train=len(tr), n_test=len(te), **info))
                    draws += [dict(**key, **d) for d in dr]
                    preds.append(pr.assign(**key))
        pd.DataFrame(cells).to_csv(out_dir / "perm_cells.csv", index=False)
        pd.DataFrame(draws).to_csv(out_dir / "perm_draws.csv", index=False)
        pd.concat(preds).to_csv(out_dir / "perm_predictions.csv", index=False)
        log(f"Wrote perm_cells.csv ({len(cells)} rows), perm_draws.csv ({len(draws)} rows), perm_predictions.csv")

    if "rate" in parts:
        outer_seed, mask_seed = seeds[0]
        rows = []
        splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=outer_seed)
        for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
            assert set(groups[tr]).isdisjoint(set(groups[te])), "subject_id leaked across outer train/test"
            X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
            masks = h.fold_masks(df_tr, y[tr], df_te, y[te], X_tr, X_te,
                                 conditions={c: RATE_CELLS[c] for c in rate_cells}, mask_seed=mask_seed)
            for cid in rate_cells:
                ckpt = out_dir / f"rate__{cid}__fold{fold_id}.pkl"
                if ckpt.exists():
                    cell = pickle.loads(ckpt.read_bytes())
                else:
                    cell = rate_cell(cid, masks, y[tr], groups[tr], y[te], num_cols, cat_cols, outer_seed)
                    ckpt.write_bytes(pickle.dumps(cell))
                    log(f"  [rate {cid} fold {fold_id}] selected={cell[0]['selected_family']} ({cell[0]['fit_seconds']:.0f}s)")
                rows += [dict(fold_id=fold_id, n_train=len(tr), n_test=len(te), **r) for r in cell]
        pd.DataFrame(rows).to_csv(out_dir / "rate_shift_eval.csv", index=False)
        log(f"Wrote rate_shift_eval.csv ({len(rows)} rows)")
    log("=== permutation / rate-shift driver finished ===")
    log_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["perm", "rate", "all"], default="all")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-path", default=None)
    a = ap.parse_args()
    main(["perm", "rate"] if a.part == "all" else [a.part], a.smoke_test, a.data_path)
