"""Phase 6 driver: RQ4 shortcut attribution via mask-only/count-only baselines, permutation ablation, and indicator-concat ablation."""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as genmod
import selection_v2 as selv2
import metrics_v2 as met

import run_phase4_rq1_rq2 as p4

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

OUTER_SEED = p4.OUTER_SEED
MASK_SEED = p4.MASK_SEED
N_OUTER = 5

RESULTS_DIR = Path("results/phase6")

PERMUTATION_RATES = [0.30, 0.40]
PERMUTATION_ORS = {2.0: p4.THETA_MAIN, 4.0: float(np.log(4.0))}
PERMUTATION_SEED_OFFSET = 555

INDICATOR_MECHANISMS = ["mcar", "mnar_y"]
INDICATOR_RATES = [0.0, 0.30, 0.40]

_PHASE4_CSV_CONTAINER = "/mnt/user-data/uploads/DASA2026/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv"

def _phase4_csv_candidates() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [script_dir.parent, script_dir, cwd.parent, cwd]
    suffixes = [
        Path("PHASE_4_RQ1_RQ2") / "results" / "phase4" / "phase4_full_table.csv",
        Path("results") / "phase4" / "phase4_full_table.csv",
    ]
    seen: List[Path] = []
    for root in roots:
        for suf in suffixes:
            p = (root / suf).resolve()
            if p not in seen:
                seen.append(p)
    return seen


def _resolve_phase4_csv_path(cli_arg: Optional[str], log=None) -> Optional[str]:
    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    if cli_arg:
        if Path(cli_arg).exists():
            return cli_arg
        _say(f"  WARNING: --phase4-csv {cli_arg!r} does not exist.")
        return None
    candidates = _phase4_csv_candidates()
    for p in candidates:
        if p.exists():
            _say(f"  Resolved Phase 4 CSV via candidate search: {p}")
            return str(p)
    if Path(_PHASE4_CSV_CONTAINER).exists():
        _say(f"  Resolved Phase 4 CSV via cloud-sandbox fallback path: {_PHASE4_CSV_CONTAINER}")
        return _PHASE4_CSV_CONTAINER
    _say("  Phase 4 CSV not found. Tried candidates:\n" + "\n".join(f"    {p}" for p in candidates)
         + f"\n    {_PHASE4_CSV_CONTAINER}")
    return None



def _fit_eval_baseline(mask_train_df: pd.DataFrame, mask_test_df: pd.DataFrame,
                        y_train: np.ndarray, y_test: np.ndarray, cols: List[str]) -> Dict[str, float]:
    """Fits mask-only and count-only LogisticRegression baselines and scores them, guarding against degenerate folds."""
    mask_train = mask_train_df[cols].to_numpy().astype(float)
    mask_test = mask_test_df[cols].to_numpy().astype(float)
    count_train = mask_train.sum(axis=1, keepdims=True)
    count_test = mask_test.sum(axis=1, keepdims=True)

    y_degenerate = len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2

    out: Dict[str, float] = {}
    if mask_train.std() == 0 or y_degenerate:
        out["mask_auroc"] = float("nan")
        out["mask_ap"] = float("nan")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(mask_train, y_train)
        p = clf.predict_proba(mask_test)[:, 1]
        out["mask_auroc"] = float(roc_auc_score(y_test, p))
        out["mask_ap"] = float(average_precision_score(y_test, p))

    if count_train.std() == 0 or y_degenerate:
        out["count_auroc"] = float("nan")
        out["count_ap"] = float("nan")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(count_train, y_train)
        p = clf.predict_proba(count_test)[:, 1]
        out["count_auroc"] = float(roc_auc_score(y_test, p))
        out["count_ap"] = float(average_precision_score(y_test, p))
    return out


def _mask_one_condition_for_fold(df_tr, y_tr, df_te, y_te, mechanism, rate, theta, mask_seed):
    """Fold-safe masking for one (mechanism, rate, theta) condition, mirroring Phase 4's per-fold masking."""
    nat_mask_tr = genmod.compute_natural_mask(df_tr)
    nat_mask_te = genmod.compute_natural_mask(df_te)
    gen, driver_tr, driver_te = p4._fit_generator_and_drivers(
        mechanism, rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
    )
    rng = np.random.default_rng(mask_seed)
    syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
    syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
    final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
    final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)
    return dict(nat_tr=nat_mask_tr, nat_te=nat_mask_te, syn_tr=syn_mask_tr, syn_te=syn_mask_te,
                final_tr=final_mask_tr, final_te=final_mask_te)


def run_baseline_grid(df: pd.DataFrame, log, n_outer: int, results_dir: Path, smoke_test: bool) -> pd.DataFrame:
    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    fold_splits = list(enumerate(outer_splitter.split(X, y, groups), start=1))

    conditions = [dict(mechanism="natural", rate=0.0, theta=0.0, or_value=None, condition_id="natural_q00")]
    conditions += p4.build_condition_grid()
    if smoke_test:
        conditions = [c for c in conditions if c["condition_id"] in ("natural_q00", "mcar_q30_main")]
    log(f"Section A (baselines): {len(conditions)} conditions x {n_outer} folds")

    rows: List[Dict[str, Any]] = []
    for c in conditions:
        cid = c["condition_id"]
        for fold_id, (tr_idx, te_idx) in fold_splits:
            ckpt = results_dir / f"baseline_{cid}_fold{fold_id}.pkl"
            if ckpt.exists():
                with open(ckpt, "rb") as f:
                    row = pickle.load(f)
                rows.append(row)
                continue

            X_tr_df = df.iloc[tr_idx].reset_index(drop=True)
            X_te_df = df.iloc[te_idx].reset_index(drop=True)
            y_tr = y[tr_idx]
            y_te = y[te_idx]
            assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"

            masks = _mask_one_condition_for_fold(
                X_tr_df, y_tr, X_te_df, y_te,
                mechanism=("mcar" if c["mechanism"] == "natural" else c["mechanism"]),
                rate=c["rate"], theta=c["theta"], mask_seed=MASK_SEED,
            )
            final_res = _fit_eval_baseline(masks["final_tr"], masks["final_te"], y_tr, y_te, genmod.ELIGIBLE_LAB_COLS)
            if c["rate"] > 0.0:
                synth_res = _fit_eval_baseline(masks["syn_tr"], masks["syn_te"], y_tr, y_te, genmod.ELIGIBLE_LAB_COLS)
            else:
                synth_res = {"mask_auroc": float("nan"), "mask_ap": float("nan"),
                             "count_auroc": float("nan"), "count_ap": float("nan")}

            row = dict(
                condition_id=cid, mechanism=c["mechanism"], rate=c["rate"], or_value=c["or_value"], fold_id=fold_id,
                mask_only_auroc=final_res["mask_auroc"], mask_only_ap=final_res["mask_ap"],
                count_only_auroc=final_res["count_auroc"], count_only_ap=final_res["count_ap"],
                synth_only_mask_auroc=synth_res["mask_auroc"], synth_only_mask_ap=synth_res["mask_ap"],
                synth_only_count_auroc=synth_res["count_auroc"], synth_only_count_ap=synth_res["count_ap"],
            )
            with open(ckpt, "wb") as f:
                pickle.dump(row, f)
            rows.append(row)
            log(f"  [baseline {cid} fold {fold_id}] mask_only={final_res['mask_auroc']:.4f} "
                f"count_only={final_res['count_auroc']:.4f} synth_only_mask={synth_res['mask_auroc']:.4f}")
    return pd.DataFrame(rows)



def run_permutation_ablation(df: pd.DataFrame, log, n_outer: int, results_dir: Path, smoke_test: bool) -> pd.DataFrame:
    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    fold_splits = list(enumerate(outer_splitter.split(X, y, groups), start=1))

    rates = [0.30] if smoke_test else PERMUTATION_RATES
    ors = {2.0: PERMUTATION_ORS[2.0]} if smoke_test else PERMUTATION_ORS
    log(f"Section B (permutation ablation): rates={rates}, ORs={list(ors.keys())}, {n_outer} folds each")

    rows: List[Dict[str, Any]] = []
    for rate in rates:
        for or_val, theta in ors.items():
            cond_id = f"mnar_y_q{int(round(rate * 100)):02d}_OR{or_val}_permtest"
            for fold_id, (tr_idx, te_idx) in fold_splits:
                ckpt = results_dir / f"perm_{cond_id}_fold{fold_id}.pkl"
                if ckpt.exists():
                    with open(ckpt, "rb") as f:
                        row = pickle.load(f)
                    rows.append(row)
                    log(f"  [perm {cond_id} fold {fold_id}] checkpoint already exists, skipping")
                    continue

                X_tr = X.iloc[tr_idx].reset_index(drop=True)
                y_tr = y[tr_idx]
                g_tr = groups[tr_idx]
                X_te = X.iloc[te_idx].reset_index(drop=True)
                y_te = y[te_idx]
                assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"
                df_tr = df.iloc[tr_idx].reset_index(drop=True)
                df_te = df.iloc[te_idx].reset_index(drop=True)

                nat_mask_tr = genmod.compute_natural_mask(df_tr)
                nat_mask_te = genmod.compute_natural_mask(df_te)
                gen, driver_tr, driver_te = p4._fit_generator_and_drivers(
                    "mnar_y", rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
                )
                rng = np.random.default_rng(MASK_SEED)
                syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
                syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
                final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
                final_mask_te_orig = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

                X_tr_masked = X_tr.copy()
                for col in genmod.ELIGIBLE_LAB_COLS:
                    X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
                X_te_masked_orig = X_te.copy()
                for col in genmod.ELIGIBLE_LAB_COLS:
                    X_te_masked_orig.loc[final_mask_te_orig[col].to_numpy(), col] = np.nan

                perm_seed_entropy = (
                    MASK_SEED, PERMUTATION_SEED_OFFSET, fold_id,
                    int(round(rate * 100)), int(round(or_val * 10)),
                )
                perm_rng = np.random.default_rng(perm_seed_entropy)
                perm_idx = perm_rng.permutation(len(syn_mask_te))
                syn_mask_te_permuted = syn_mask_te.iloc[perm_idx].reset_index(drop=True)
                final_mask_te_permuted = genmod.compute_final_mask(nat_mask_te, syn_mask_te_permuted)
                X_te_masked_permuted = X_te.copy()
                for col in genmod.ELIGIBLE_LAB_COLS:
                    X_te_masked_permuted.loc[final_mask_te_permuted[col].to_numpy(), col] = np.nan

                t0 = time.time()
                fit_result = selv2.fit_and_select_all_families(
                    X_tr_masked, y_tr, g_tr, num_cols, cat_cols, seed=OUTER_SEED
                )
                fit_elapsed = time.time() - t0
                selected_family = fit_result.selected_family
                final_model = fit_result.final_models[selected_family]

                auroc_before, ap_before, brier_before = selv2.evaluate_frozen_model(final_model, X_te_masked_orig, y_te)
                auroc_after, ap_after, brier_after = selv2.evaluate_frozen_model(final_model, X_te_masked_permuted, y_te)

                row = dict(
                    condition_id=cond_id, rate=rate, or_value=or_val, fold_id=fold_id,
                    selected_family=selected_family,
                    auroc_before=auroc_before, ap_before=ap_before, brier_before=brier_before,
                    auroc_after=auroc_after, ap_after=ap_after, brier_after=brier_after,
                    delta_auroc=auroc_after - auroc_before,
                    fit_seconds=fit_elapsed,
                )
                with open(ckpt, "wb") as f:
                    pickle.dump(row, f)
                rows.append(row)
                log(f"  [perm {cond_id} fold {fold_id}] selected={selected_family} "
                    f"auroc_before={auroc_before:.4f} auroc_after={auroc_after:.4f} "
                    f"delta={row['delta_auroc']:+.4f} ({fit_elapsed:.1f}s)")
    return pd.DataFrame(rows)



def run_indicator_ablation(df: pd.DataFrame, log, n_outer: int, results_dir: Path, smoke_test: bool,
                            phase4_csv_arg: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y, groups, num_cols, cat_cols = p4._prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    fold_splits = list(enumerate(outer_splitter.split(X, y, groups), start=1))

    indicator_cols = [f"missing_{c}" for c in genmod.ELIGIBLE_LAB_COLS]
    num_cols_with_ind = num_cols + indicator_cols

    mechanisms = ["mnar_y"] if smoke_test else INDICATOR_MECHANISMS
    rates = [0.30] if smoke_test else INDICATOR_RATES
    log(f"Section C (indicator-concat, WITH-indicator side): mechanisms={mechanisms}, rates={rates}, {n_outer} folds each")

    rows: List[Dict[str, Any]] = []
    for mech in mechanisms:
        theta = 0.0 if mech == "mcar" else p4.THETA_MAIN
        for rate in rates:
            cond_id = f"{mech}_q{int(round(rate * 100)):02d}_withind"
            for fold_id, (tr_idx, te_idx) in fold_splits:
                ckpt = results_dir / f"ind_{cond_id}_fold{fold_id}.pkl"
                if ckpt.exists():
                    with open(ckpt, "rb") as f:
                        row = pickle.load(f)
                    rows.append(row)
                    log(f"  [ind {cond_id} fold {fold_id}] checkpoint already exists, skipping")
                    continue

                X_tr = X.iloc[tr_idx].reset_index(drop=True)
                y_tr = y[tr_idx]
                g_tr = groups[tr_idx]
                X_te = X.iloc[te_idx].reset_index(drop=True)
                y_te = y[te_idx]
                assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "subject_id leaked across outer train/test"
                df_tr = df.iloc[tr_idx].reset_index(drop=True)
                df_te = df.iloc[te_idx].reset_index(drop=True)

                nat_mask_tr = genmod.compute_natural_mask(df_tr)
                nat_mask_te = genmod.compute_natural_mask(df_te)
                gen, driver_tr, driver_te = p4._fit_generator_and_drivers(
                    mech, rate, theta, df_tr, y_tr, nat_mask_tr, df_te, y_te
                )
                rng = np.random.default_rng(MASK_SEED)
                syn_mask_tr = genmod.apply_synthetic_mask(nat_mask_tr, driver_tr, gen, rng)
                syn_mask_te = genmod.apply_synthetic_mask(nat_mask_te, driver_te, gen, rng)
                final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
                final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

                X_tr_masked = X_tr.copy()
                X_te_masked = X_te.copy()
                for col in genmod.ELIGIBLE_LAB_COLS:
                    X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
                    X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan

                X_tr_with_ind = X_tr_masked.copy()
                X_te_with_ind = X_te_masked.copy()
                for col in genmod.ELIGIBLE_LAB_COLS:
                    X_tr_with_ind[f"missing_{col}"] = X_tr_masked[col].isna().astype(float)
                    X_te_with_ind[f"missing_{col}"] = X_te_masked[col].isna().astype(float)

                t0 = time.time()
                fit_result = selv2.fit_and_select_all_families(
                    X_tr_with_ind, y_tr, g_tr, num_cols_with_ind, cat_cols, seed=OUTER_SEED
                )
                fit_elapsed = time.time() - t0
                selected_family = fit_result.selected_family
                auroc, ap, brier = selv2.evaluate_frozen_model(
                    fit_result.final_models[selected_family], X_te_with_ind, y_te
                )

                row = dict(
                    condition_id=cond_id, mechanism=mech, rate=rate, fold_id=fold_id,
                    selected_family=selected_family,
                    with_indicator_auroc=auroc, with_indicator_ap=ap, with_indicator_brier=brier,
                    fit_seconds=fit_elapsed,
                )
                with open(ckpt, "wb") as f:
                    pickle.dump(row, f)
                rows.append(row)
                log(f"  [ind {cond_id} fold {fold_id}] selected={selected_family} auroc={auroc:.4f} ({fit_elapsed:.1f}s)")

    with_ind_df = pd.DataFrame(rows)

    phase4_csv = _resolve_phase4_csv_path(phase4_csv_arg, log=log)
    without_rows: List[Dict[str, Any]] = []
    if phase4_csv is None:
        log("  WARNING: Phase 4's phase4_full_table.csv was not found anywhere searched (see "
            "candidate list logged above). Skipping the without-indicator comparison -- the "
            "with-indicator results above are still valid on their own.")
    else:
        log(f"  Reusing Phase 4's real results for the without-indicator comparison: {phase4_csv}")
        phase4_full = pd.read_csv(phase4_csv)
        for mech in mechanisms:
            for rate in rates:
                p4_cond_id = f"{mech}_q00" if rate == 0.0 else f"{mech}_q{int(round(rate * 100)):02d}_main"
                sub = phase4_full[phase4_full.condition_id == p4_cond_id]
                if sub.empty:
                    log(f"  WARNING: condition {p4_cond_id!r} not found in {phase4_csv} -- "
                        "without-indicator comparison missing for this cell")
                    continue
                for fold_id, fg in sub.groupby("fold_id"):
                    sel_fam = fg["selected_family"].iloc[0]
                    sel_row = fg[fg.model_key == sel_fam].iloc[0]
                    without_rows.append(dict(
                        condition_id=f"{mech}_q{int(round(rate * 100)):02d}_noind", mechanism=mech, rate=rate,
                        fold_id=int(fold_id), selected_family=sel_fam,
                        without_indicator_auroc=float(sel_row["outer_test_auroc"]),
                        without_indicator_ap=float(sel_row["outer_test_ap"]),
                        without_indicator_brier=float(sel_row["outer_test_brier"]),
                        source="phase4_reuse",
                    ))
    without_ind_df = pd.DataFrame(without_rows)
    return with_ind_df, without_ind_df



def make_logger(log_path: Path):
    return p4.make_logger(log_path)


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None, phase4_csv_arg: Optional[str] = None):
    results_dir = Path("results/phase6_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    log(f"=== Phase 6 driver start (smoke_test={smoke_test}, n_outer={n_outer}) ===")

    data_path = p4._resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    log("--- Section A: mask-only / count-only / synth-only baselines ---")
    t0 = time.time()
    baseline_df = run_baseline_grid(df, log, n_outer, results_dir, smoke_test)
    log(f"Section A done in {time.time() - t0:.1f}s")

    log("--- Section B: test-only permutation ablation ---")
    t0 = time.time()
    perm_df = run_permutation_ablation(df, log, n_outer, results_dir, smoke_test)
    log(f"Section B done in {time.time() - t0:.1f}s")

    log("--- Section C: explicit-indicator-concat ablation ---")
    t0 = time.time()
    with_ind_df, without_ind_df = run_indicator_ablation(df, log, n_outer, results_dir, smoke_test, phase4_csv_arg)
    log(f"Section C done in {time.time() - t0:.1f}s")

    log("All sections complete. Writing outputs...")
    baseline_df.to_csv(results_dir / "phase6_baselines.csv", index=False)
    log(f"Wrote {results_dir / 'phase6_baselines.csv'} ({len(baseline_df)} rows)")
    perm_df.to_csv(results_dir / "phase6_permutation.csv", index=False)
    log(f"Wrote {results_dir / 'phase6_permutation.csv'} ({len(perm_df)} rows)")
    with_ind_df.to_csv(results_dir / "phase6_indicator_with.csv", index=False)
    log(f"Wrote {results_dir / 'phase6_indicator_with.csv'} ({len(with_ind_df)} rows)")
    without_ind_df.to_csv(results_dir / "phase6_indicator_without.csv", index=False)
    log(f"Wrote {results_dir / 'phase6_indicator_without.csv'} ({len(without_ind_df)} rows)")

    with open(results_dir / "phase6_all_results.pkl", "wb") as f:
        pickle.dump({
            "baseline_df": baseline_df, "perm_df": perm_df,
            "with_ind_df": with_ind_df, "without_ind_df": without_ind_df,
        }, f)
    log(f"Wrote {results_dir / 'phase6_all_results.pkl'}")
    log("=== Phase 6 driver finished ===")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="2 outer folds; Section A reduced to 2 conditions; Section B reduced to "
                              "1 (rate,OR) cell; Section C reduced to 1 (mechanism,rate) cell. "
                              "Writes to results/phase6_smoke/.")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv.")
    parser.add_argument("--phase4-csv", default=None,
                         help="Explicit path to Phase 4's phase4_full_table.csv, used only for Section C's "
                              "without-indicator comparison. If omitted, tries "
                              "../PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv relative to this "
                              "script, then the cloud-sandbox path. Missing this file is non-fatal.")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path, phase4_csv_arg=args.phase4_csv)
