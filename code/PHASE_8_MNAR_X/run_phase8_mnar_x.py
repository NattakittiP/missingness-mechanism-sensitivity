"""Phase 8 driver, part 1 of 3: the MNAR-X rate and strength sweep, mirroring Phase 4's structure and conventions."""

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
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as genmod
import selection_v2 as selv2
import metrics_v2 as met

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

_CONTAINER_DATA_PATH = "/mnt/user-data/uploads/DASA2026/Dataset/full_analytic_dataset_mortality_all_admissions.csv"
_RELATIVE_DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "full_analytic_dataset_mortality_all_admissions.csv"


def _resolve_data_path(cli_arg: Optional[str]) -> str:
    if cli_arg:
        if not Path(cli_arg).exists():
            raise FileNotFoundError(f"--data-path was given as {cli_arg!r} but that file does not exist")
        return cli_arg
    if _RELATIVE_DATA_PATH.exists():
        return str(_RELATIVE_DATA_PATH)
    if Path(_CONTAINER_DATA_PATH).exists():
        return _CONTAINER_DATA_PATH
    raise FileNotFoundError(
        "Could not find full_analytic_dataset_mortality_all_admissions.csv. Tried "
        f"{_RELATIVE_DATA_PATH} (relative to this script) and {_CONTAINER_DATA_PATH!r} "
        "(cloud-sandbox path). Pass --data-path /full/path/to/the/csv explicitly."
    )



def _natural_q0_candidates() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    roots = [script_dir.parent, script_dir, cwd.parent, cwd]
    suffixes = [
        Path("PHASE_4_RQ1_RQ2") / "results" / "phase4" / "natural_q0.pkl",
        Path("results") / "phase4" / "natural_q0.pkl",
        Path("phase3_validation_natural_results.pkl"),
    ]
    seen: List[Path] = []
    for root in roots:
        for suf in suffixes:
            p = (root / suf).resolve()
            if p not in seen:
                seen.append(p)
    return seen


OUTER_SEED = 2026
MASK_SEED = 20260917
N_OUTER = 5

Z_CLIP = 5.0
GAMMA_MAIN = float(np.log(2.0) / 2.0)
GAMMA_SENS = {1.5: float(np.log(1.5) / 2.0), 4.0: float(np.log(4.0) / 2.0)}
RATES_MAIN = [0.10, 0.20, 0.30, 0.40]
RQ2_RATE = 0.30

RESULTS_DIR = Path("results/phase8_mnar_x")



def build_condition_grid() -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for r in RATES_MAIN:
        conditions.append(dict(
            rq="RQ1", mechanism="mnar_x", rate=r, gamma=GAMMA_MAIN, or_value=2.0,
            condition_id=f"mnar_x_q{int(round(r * 100)):02d}_main",
        ))
    for or_val, gamma in GAMMA_SENS.items():
        conditions.append(dict(
            rq="RQ2", mechanism="mnar_x", rate=RQ2_RATE, gamma=gamma, or_value=or_val,
            condition_id=f"mnar_x_q{int(round(RQ2_RATE * 100)):02d}_OR{or_val}",
        ))
    return conditions



def _prepare_X_y_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    y = df[genmod.LABEL_COL].astype(int).to_numpy()
    groups = df[genmod.GROUP_COL].to_numpy()
    X = df.drop(columns=[genmod.LABEL_COL], errors="ignore").copy()
    for c in genmod.EXCLUDED_FROM_MODEL_ENTIRELY:
        if c in X.columns:
            X = X.drop(columns=[c], errors="ignore")
    num_cols, cat_cols = selv2.base_runner.split_columns_A(df)
    present = set(X.columns)
    num_cols = [c for c in num_cols if c in present]
    cat_cols = [c for c in cat_cols if c in present]
    return X, y, groups, num_cols, cat_cols


def run_condition_with_injection(
    df: pd.DataFrame,
    rate: float,
    gamma: float,
    seed: int = OUTER_SEED,
    mask_seed: int = MASK_SEED,
    n_outer: int = N_OUTER,
) -> Tuple[List[Any], List[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Fold-safe: calibrates the MNAR-X generator per outer fold, then runs family selection/evaluation and captures predictions."""
    X, y, groups, num_cols, cat_cols = _prepare_X_y_groups(df)
    outer_splitter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    results = []
    diagnostics = []
    predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for fold_id, (tr_idx, te_idx) in enumerate(outer_splitter.split(X, y, groups), start=1):
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

        if rate == 0.0:
            gen = genmod.MnarXGenerator(
                gamma=gamma, target_rate=0.0, z_clip=Z_CLIP,
                alpha={c: -np.inf for c in genmod.ELIGIBLE_LAB_COLS},
                mu={c: float("nan") for c in genmod.ELIGIBLE_LAB_COLS},
                sigma={c: float("nan") for c in genmod.ELIGIBLE_LAB_COLS},
            )
        else:
            gen = genmod.fit_mnar_x(df_tr, nat_mask_tr, gamma=gamma, target_rate=rate, z_clip=Z_CLIP)

        rng = np.random.default_rng(mask_seed)
        syn_mask_tr = genmod.apply_mnar_x_mask(df_tr, nat_mask_tr, gen, rng)
        syn_mask_te = genmod.apply_mnar_x_mask(df_te, nat_mask_te, gen, rng)

        final_mask_tr = genmod.compute_final_mask(nat_mask_tr, syn_mask_tr)
        final_mask_te = genmod.compute_final_mask(nat_mask_te, syn_mask_te)

        X_tr_masked = X_tr.copy()
        X_te_masked = X_te.copy()
        for col in genmod.ELIGIBLE_LAB_COLS:
            X_tr_masked.loc[final_mask_tr[col].to_numpy(), col] = np.nan
            X_te_masked.loc[final_mask_te[col].to_numpy(), col] = np.nan

        rates_tr = genmod.compute_rates(nat_mask_tr, syn_mask_tr)
        rates_te = genmod.compute_rates(nat_mask_te, syn_mask_te)

        diag = {
            "fold_id": fold_id, "mechanism": "mnar_x", "target_rate": rate, "gamma": gamma,
            "r_inject_train": rates_tr["r_inject"], "r_total_train": rates_tr["r_total"],
            "r_inject_test": rates_te["r_inject"], "r_total_test": rates_te["r_total"],
        }
        if rate > 0.0:
            elig_te = genmod.compute_eligible_mask(nat_mask_te)
            z_list, m_list = [], []
            for col in genmod.ELIGIBLE_LAB_COLS:
                e = elig_te[col].to_numpy()
                if e.sum() == 0 or not np.isfinite(gen.sigma[col]) or gen.sigma[col] <= 0:
                    continue
                x = df_te[col].to_numpy(dtype=float)[e]
                z = np.clip(np.abs((x - gen.mu[col]) / gen.sigma[col]), 0.0, Z_CLIP)
                m = syn_mask_te[col].to_numpy()[e].astype(float)
                z_list.append(z)
                m_list.append(m)
            if z_list:
                z_all = np.concatenate(z_list)
                m_all = np.concatenate(m_list)
                if np.std(z_all) > 0 and np.std(m_all) > 0:
                    diag["corr_mask_vs_abs_z"] = float(np.corrcoef(z_all, m_all)[0, 1])
        diagnostics.append(diag)

        result, y_sel, p_sel = selv2.run_outer_fold_with_predictions(
            X_tr_masked, y_tr, g_tr, X_te_masked, y_te, num_cols, cat_cols, seed=seed, fold_id=fold_id
        )
        results.append(result)
        predictions[fold_id] = (y_sel, p_sel)

    return results, diagnostics, predictions



def load_or_compute_natural_q0(df: pd.DataFrame, log, n_outer: int = N_OUTER) -> Tuple[List[Any], List[Dict[str, Any]], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    for p in _natural_q0_candidates():
        if p.exists():
            log(f"q=0: reusing existing Phase 3/4 natural_q0 checkpoint from {p}")
            with open(p, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                results, diag = loaded
            else:
                results = loaded
                diag = [{"fold_id": r.fold_id, "mechanism": "mnar_x", "target_rate": 0.0, "gamma": 0.0,
                          "r_inject_train": 0.0, "r_total_train": None, "r_inject_test": 0.0, "r_total_test": None}
                         for r in results]
            log(f"q=0: no calibration predictions available for the reused q=0 point "
                f"(natural_q0 predates prediction-capture) -- calibration analysis for "
                f"mnar_x will start from q>0 conditions only, consistent with q=0 carrying "
                f"no injected missingness to calibrate against in the first place.")
            return results, diag, {}
    log("q=0: no existing natural_q0 checkpoint found anywhere searched, computing fresh")
    return run_condition_with_injection(df, rate=0.0, gamma=GAMMA_MAIN, n_outer=n_outer)



def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    log.close = f.close
    return log


def run_all(smoke_test: bool = False, data_path_arg: Optional[str] = None):
    results_dir = Path("results/phase8_mnar_x_smoke") if smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    n_outer = 2 if smoke_test else N_OUTER
    log(f"=== Phase 8 (MNAR-X) driver start (smoke_test={smoke_test}, n_outer={n_outer}) ===")
    log(f"GAMMA_MAIN={GAMMA_MAIN:.6f} (OR=2.0 @ |z|=2), GAMMA_SENS={GAMMA_SENS}, Z_CLIP={Z_CLIP}")

    data_path = _resolve_data_path(data_path_arg)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    assert df.shape[0] == 14081, f"unexpected row count: {df.shape[0]}"
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    conditions = build_condition_grid()
    if smoke_test:
        conditions = [c for c in conditions if c["condition_id"] == "mnar_x_q10_main"]
    log(f"Condition grid: {len(conditions)} fresh conditions")
    for c in conditions:
        log(f"  {c['condition_id']}: rq={c['rq']} rate={c['rate']} gamma={c['gamma']:.6f} OR={c['or_value']}")

    all_results: Dict[str, List[Any]] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    all_predictions: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    condition_meta: Dict[str, Dict[str, Any]] = {}

    q0_ckpt = results_dir / "natural_q0.pkl"
    if q0_ckpt.exists():
        log("q=0: checkpoint already exists, loading")
        with open(q0_ckpt, "rb") as f:
            q0_results, q0_diag, q0_pred = pickle.load(f)
    else:
        t0 = time.time()
        q0_results, q0_diag, q0_pred = load_or_compute_natural_q0(df, log, n_outer=n_outer)
        with open(q0_ckpt, "wb") as f:
            pickle.dump((q0_results, q0_diag, q0_pred), f)
        log(f"q=0: done in {time.time() - t0:.1f}s, checkpointed to {q0_ckpt}")
    all_results["mnar_x_q00"] = q0_results
    all_diagnostics.extend(q0_diag)
    all_predictions["mnar_x_q00"] = q0_pred
    condition_meta["mnar_x_q00"] = dict(rq="RQ1", mechanism="mnar_x", rate=0.0, gamma=0.0, or_value=None)

    for i, c in enumerate(conditions, start=1):
        cid = c["condition_id"]
        ckpt = results_dir / f"{cid}.pkl"
        condition_meta[cid] = c
        if ckpt.exists():
            log(f"[{i}/{len(conditions)}] {cid}: checkpoint already exists, skipping")
            with open(ckpt, "rb") as f:
                r, d, p = pickle.load(f)
            all_results[cid] = r
            all_diagnostics.extend(d)
            all_predictions[cid] = p
            continue

        log(f"[{i}/{len(conditions)}] {cid}: starting (rq={c['rq']} rate={c['rate']} "
            f"gamma={c['gamma']:.6f} OR={c['or_value']})")
        t0 = time.time()
        r, d, p = run_condition_with_injection(df, rate=c["rate"], gamma=c["gamma"], n_outer=n_outer)
        elapsed = time.time() - t0
        with open(ckpt, "wb") as f:
            pickle.dump((r, d, p), f)
        all_results[cid] = r
        all_diagnostics.extend(d)
        all_predictions[cid] = p
        for fold_result in r:
            log(f"    fold {fold_result.fold_id}: selected={fold_result.selected_family} "
                f"outer_test_auroc={fold_result.selected_outer_test_auroc:.4f} "
                f"oracle={fold_result.oracle_family} regret={fold_result.selection_regret:.4f} "
                f"displaced={fold_result.displaced}")
        log(f"[{i}/{len(conditions)}] {cid}: done in {elapsed:.1f}s, checkpointed to {ckpt}")

    log("All conditions complete. Aggregating...")
    aggregate_and_save(all_results, all_diagnostics, all_predictions, condition_meta, results_dir, log)
    log("=== Phase 8 (MNAR-X) driver finished ===")
    log.close()



def aggregate_and_save(all_results, all_diagnostics, all_predictions, condition_meta, results_dir: Path, log):
    rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            for fam, fres in fr.families.items():
                rows.append({
                    "condition_id": cid, "rq": meta["rq"], "mechanism": "mnar_x",
                    "rate": meta["rate"], "gamma": meta["gamma"], "or_value": meta["or_value"],
                    "fold_id": fr.fold_id, "model_key": fam,
                    "inner_auroc_mean": fres.inner_auroc_mean, "inner_ap_mean": fres.inner_ap_mean,
                    "outer_test_auroc": fres.outer_test_auroc, "outer_test_ap": fres.outer_test_ap,
                    "outer_test_brier": fres.outer_test_brier,
                    "selected_family": fr.selected_family, "oracle_family": fr.oracle_family,
                    "selection_regret": fr.selection_regret, "displaced": fr.displaced,
                })
    full_table = pd.DataFrame(rows)
    full_table.to_csv(results_dir / "phase8_mnar_x_full_table.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_full_table.csv'} ({len(full_table)} rows)")

    summary_rows = []
    for cid, fold_results in all_results.items():
        meta = condition_meta[cid]
        for fr in fold_results:
            summary_rows.append({
                "condition_id": cid, "rq": meta["rq"], "rate": meta["rate"], "or_value": meta["or_value"],
                "fold_id": fr.fold_id, "selected_family": fr.selected_family,
                "selected_outer_test_auroc": fr.selected_outer_test_auroc,
                "oracle_family": fr.oracle_family, "oracle_outer_test_auroc": fr.oracle_outer_test_auroc,
                "selection_regret": fr.selection_regret, "displaced": fr.displaced,
            })
    fold_summary = pd.DataFrame(summary_rows)
    fold_summary.to_csv(results_dir / "phase8_mnar_x_fold_summary.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_fold_summary.csv'}")

    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(results_dir / "phase8_mnar_x_manipulation_check.csv", index=False)
    log(f"Wrote {results_dir / 'phase8_mnar_x_manipulation_check.csv'}")

    metrics_summary: Dict[str, Any] = {}
    for cid, fold_results in all_results.items():
        selected = met.selected_family_by_fold(fold_results)
        entropy = met.within_condition_selection_entropy(list(selected.values()), selv2.MODELS)
        scores_by_repeat = [met.outer_test_auroc_by_family(r) for r in fold_results]
        kendall = met.kendall_tau_b_rank_stability(scores_by_repeat) if len(scores_by_repeat) >= 2 else None
        sel_margins = [met.top1_minus_top2_margin(met.inner_auroc_by_family(r)) for r in fold_results]
        eval_margins = [met.top1_minus_top2_margin(met.outer_test_auroc_by_family(r)) for r in fold_results]
        metrics_summary[cid] = dict(
            entropy_normalized=entropy.normalized_entropy, modal_family=entropy.modal_family,
            modal_frequency=entropy.modal_frequency,
            kendall_tau_b_mean=(kendall.mean_tau_b if kendall else None),
            kendall_tau_b_min=(kendall.min_tau_b if kendall else None),
            kendall_tau_b_max=(kendall.max_tau_b if kendall else None),
            selection_margin_mean=float(np.mean(sel_margins)),
            evaluation_margin_mean=float(np.mean(eval_margins)),
        )

    by_fold_and_rate: Dict[int, Dict[float, str]] = {}
    for r in [0.0] + RATES_MAIN:
        cid = "mnar_x_q00" if r == 0.0 else f"mnar_x_q{int(round(r * 100)):02d}_main"
        if cid not in all_results:
            continue
        for fr in all_results[cid]:
            by_fold_and_rate.setdefault(fr.fold_id, {})[r] = fr.selected_family
    disp = met.baseline_selection_displacement_rate(by_fold_and_rate, baseline_rate=0.0)
    displacement = {
        r: dict(displacement_rate=v.displacement_rate, n_folds=v.n_folds, n_displaced=v.n_displaced,
                 displaced_fold_ids=v.displaced_fold_ids)
        for r, v in disp.items()
    }

    with open(results_dir / "phase8_mnar_x_metrics_summary.json", "w") as f:
        json.dump({"per_condition": metrics_summary, "displacement_mnar_x": displacement}, f, indent=2, default=str)
    log(f"Wrote {results_dir / 'phase8_mnar_x_metrics_summary.json'}")

    with open(results_dir / "phase8_mnar_x_all_results.pkl", "wb") as f:
        pickle.dump({"all_results": all_results, "condition_meta": condition_meta, "all_diagnostics": all_diagnostics}, f)
    log(f"Wrote {results_dir / 'phase8_mnar_x_all_results.pkl'}")

    with open(results_dir / "phase8_mnar_x_predictions.pkl", "wb") as f:
        pickle.dump({"predictions": all_predictions, "condition_meta": condition_meta}, f)
    log(f"Wrote {results_dir / 'phase8_mnar_x_predictions.pkl'} "
        f"({sum(len(v) for v in all_predictions.values())} (condition, fold) prediction arrays)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run 1 condition x 2 outer folds only. Writes to results/phase8_mnar_x_smoke/.")
    parser.add_argument("--data-path", default=None,
                         help="Explicit path to full_analytic_dataset_mortality_all_admissions.csv.")
    args = parser.parse_args()
    run_all(smoke_test=args.smoke_test, data_path_arg=args.data_path)
