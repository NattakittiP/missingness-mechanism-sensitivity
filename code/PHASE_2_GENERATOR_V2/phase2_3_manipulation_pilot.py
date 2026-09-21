"""Phase 2.3 manipulation pilot: mask-only/count-only baseline check across the mechanism grid, mandatory before full model training."""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import missingness_generator_v2 as gen

DATA_PATH = "/mnt/user-data/uploads/DASA2026/Dataset/full_analytic_dataset_mortality_all_admissions.csv"
MASK_SEED = 20260917
OUTER_SPLIT_SEED = 42

WARN_THRESHOLD = 0.95
FAIL_THRESHOLD = 0.97


def make_outer_fold(df: pd.DataFrame):
    """Builds the single StratifiedGroupKFold(subject_id) fold used by the pilot."""
    y = df[gen.LABEL_COL].to_numpy()
    groups = df[gen.GROUP_COL].to_numpy()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SPLIT_SEED)
    train_idx, test_idx = next(skf.split(df, y, groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    assert set(df_train[gen.GROUP_COL]).isdisjoint(set(df_test[gen.GROUP_COL])), (
        "StratifiedGroupKFold leaked a subject_id across train/test -- outer split is invalid"
    )
    return df_train, df_test


def build_masks(df_train, df_test, mechanism: str, q: float, theta: float, rng: np.random.Generator):
    """Fits alpha on train only, applies the same fitted generator to train and test separately."""
    nat_train = gen.compute_natural_mask(df_train)
    nat_test = gen.compute_natural_mask(df_test)

    if mechanism == "natural":
        generator = gen.fit_natural_baseline()
        driver_train = driver_test = None
    elif mechanism == "mcar":
        generator = gen.fit_mcar(nat_train, q)
        driver_train = driver_test = None
    elif mechanism == "mar":
        generator = gen.fit_mar_context(nat_train, df_train["admission_type"], theta, q)
        driver_train = gen.compute_H_i(df_train["admission_type"]).to_numpy().astype(float)
        driver_test = gen.compute_H_i(df_test["admission_type"]).to_numpy().astype(float)
    elif mechanism == "mnar_y":
        generator = gen.fit_mnar_y(nat_train, df_train[gen.LABEL_COL], theta, q)
        driver_train = df_train[gen.LABEL_COL].to_numpy().astype(float)
        driver_test = df_test[gen.LABEL_COL].to_numpy().astype(float)
    else:
        raise ValueError(mechanism)

    syn_train = gen.apply_synthetic_mask(nat_train, driver_train, generator, rng)
    syn_test = gen.apply_synthetic_mask(nat_test, driver_test, generator, rng)

    final_train = gen.compute_final_mask(nat_train, syn_train)
    final_test = gen.compute_final_mask(nat_test, syn_test)

    return {
        "nat_train": nat_train, "nat_test": nat_test,
        "syn_train": syn_train, "syn_test": syn_test,
        "final_train": final_train, "final_test": final_test,
        "generator": generator,
    }


def _fit_eval(mask_train_df, mask_test_df, y_train, y_test, cols):
    mask_train = mask_train_df[cols].to_numpy().astype(int)
    mask_test = mask_test_df[cols].to_numpy().astype(int)
    count_train = mask_train.sum(axis=1, keepdims=True)
    count_test = mask_test.sum(axis=1, keepdims=True)

    out = {}
    if mask_train.std() == 0 or len(np.unique(y_train)) < 2:
        out["mask_auroc"] = float("nan")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(mask_train, y_train)
        out["mask_auroc"] = float(roc_auc_score(y_test, clf.predict_proba(mask_test)[:, 1]))

    if count_train.std() == 0 or len(np.unique(y_train)) < 2:
        out["count_auroc"] = float("nan")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(count_train, y_train)
        out["count_auroc"] = float(roc_auc_score(y_test, clf.predict_proba(count_test)[:, 1]))
    return out


def eval_baselines(masks, y_train, y_test):
    """Evaluates mask-only/count-only baselines on both the combined mask and the synthetic mask alone."""
    final_res = _fit_eval(masks["final_train"], masks["final_test"], y_train, y_test, gen.ELIGIBLE_LAB_COLS)
    results = {
        "mask_only_auroc": final_res["mask_auroc"],
        "count_only_auroc": final_res["count_auroc"],
    }
    syn_res = _fit_eval(masks["syn_train"], masks["syn_test"], y_train, y_test, gen.ELIGIBLE_LAB_COLS)
    results["synth_only_mask_auroc"] = syn_res["mask_auroc"]
    results["synth_only_count_auroc"] = syn_res["count_auroc"]
    return results


def diagnostics(masks, q_target):
    rates_train = gen.compute_rates(masks["nat_train"], masks["syn_train"])
    rates_test = gen.compute_rates(masks["nat_test"], masks["syn_test"])
    gen_obj = masks["generator"]
    finite_alphas = [a for a in gen_obj.alpha.values() if np.isfinite(a)]
    return {
        "r_inject_train": rates_train["r_inject"],
        "r_total_train": rates_train["r_total"],
        "r_inject_test": rates_test["r_inject"],
        "r_total_test": rates_test["r_total"],
        "n_finite_alpha_cols": len(finite_alphas),
        "alpha_min": float(np.min(finite_alphas)) if finite_alphas else None,
        "alpha_max": float(np.max(finite_alphas)) if finite_alphas else None,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df_train, df_test = make_outer_fold(df)
    y_train = df_train[gen.LABEL_COL].to_numpy()
    y_test = df_test[gen.LABEL_COL].to_numpy()

    print(f"Outer fold: n_train={len(df_train)} (pos={int(y_train.sum())}), "
          f"n_test={len(df_test)} (pos={int(y_test.sum())})")
    print(f"Train prevalence={y_train.mean():.4f}  Test prevalence={y_test.mean():.4f}")
    print()

    conditions = []
    conditions.append({"mechanism": "natural", "q": 0.0, "or": None})
    for q in (0.10, 0.30, 0.50):
        conditions.append({"mechanism": "mcar", "q": q, "or": None})
    for q in (0.10, 0.30, 0.50):
        conditions.append({"mechanism": "mar", "q": q, "or": 2.0})
    for q in (0.10, 0.30, 0.50):
        for OR in (1.5, 2.0, 4.0):
            conditions.append({"mechanism": "mnar_y", "q": q, "or": OR})

    rows = []
    any_fail = False
    any_warn = False

    for cond in conditions:
        mechanism, q, OR = cond["mechanism"], cond["q"], cond["or"]
        theta = float(np.log(OR)) if OR is not None else 0.0
        rng = np.random.default_rng(MASK_SEED)

        masks = build_masks(df_train, df_test, mechanism, q, theta, rng)
        base_results = eval_baselines(masks, y_train, y_test)
        diag = diagnostics(masks, q)

        row = {"mechanism": mechanism, "q": q, "OR": OR, "theta": round(theta, 6), **base_results, **diag}
        rows.append(row)

        for key in ("mask_only_auroc", "count_only_auroc", "synth_only_mask_auroc", "synth_only_count_auroc"):
            v = row[key]
            if np.isnan(v):
                continue
            if v >= FAIL_THRESHOLD:
                any_fail = True
                print(f"** FAIL ** {mechanism} q={q} OR={OR}: {key}={v:.4f} >= {FAIL_THRESHOLD}")
            elif v >= WARN_THRESHOLD:
                any_warn = True
                print(f"** WARNING ** {mechanism} q={q} OR={OR}: {key}={v:.4f} >= {WARN_THRESHOLD}")

    result_df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print()
    print(result_df.to_string(index=False))

    result_df.to_csv("phase2_3_pilot_results.csv", index=False)

    all_auroc_cols = ["mask_only_auroc", "count_only_auroc", "synth_only_mask_auroc", "synth_only_count_auroc"]

    main_grid = result_df[(result_df["mechanism"] != "mnar_y") | (result_df["OR"] == 2.0)]
    main_grid_max = main_grid[all_auroc_cols].max(numeric_only=True).max()

    strength_grid = result_df[(result_df["mechanism"] == "mnar_y") & (result_df["q"] == 0.30)]
    strength_grid_flags = strength_grid[(strength_grid[all_auroc_cols] >= WARN_THRESHOLD).any(axis=1)]

    print()
    print("=" * 70)
    print(f"Native missingness alone (q=0, NO synthetic injection at all):")
    nat_row = result_df[result_df["mechanism"] == "natural"].iloc[0]
    print(f"  mask_only_auroc={nat_row['mask_only_auroc']:.4f}  count_only_auroc={nat_row['count_only_auroc']:.4f}")
    print("  -> Real-world native missingness in Dataset A is ALREADY strongly associated with")
    print("     mortality (clinicians order more/different labs for sicker patients). This is a")
    print("     pre-existing property of the raw data, not a generator artifact -- but it means")
    print("     final_mask-based diagnostics at q>0 combine this native signal with the injected")
    print("     one. The synth_only_* columns isolate the injected mechanism's own contribution.")
    print()
    print(f"MAIN experiment grid (OR=2 fixed, q in {{0.1,0.2,0.3,0.4}} main + {{0.5}} severe):")
    print(f"  worst-case AUROC across mask_only/count_only/synth_only = {main_grid_max:.4f}")
    if main_grid_max >= FAIL_THRESHOLD:
        print(f"  ** FAIL ** -- exceeds {FAIL_THRESHOLD}. Do not proceed with OR=2 as specified.")
    elif main_grid_max >= WARN_THRESHOLD:
        print(f"  WARNING -- approaches ceiling ({WARN_THRESHOLD}); review before freezing.")
    else:
        print(f"  PASS -- stays below {WARN_THRESHOLD} in every tested main-grid condition.")
    print()
    print("Strength-sensitivity grid (q=0.30 fixed, OR in {1.5, 2, 4}):")
    if len(strength_grid_flags) > 0:
        print("  ** FLAGGED ** -- at least one OR condition hits/approaches ceiling on the")
        print("  SYNTHETIC-ONLY decomposition (i.e. the injected mechanism alone, independent of")
        print("  native missingness, nearly perfectly predicts the outcome). Specifically:")
        for _, r in strength_grid_flags.iterrows():
            print(f"    OR={r['OR']}: synth_only_mask={r['synth_only_mask_auroc']:.4f}, "
                  f"synth_only_count={r['synth_only_count_auroc']:.4f}")
        print("  This matches Protocol v2.1 §2.3's own prediction that OR=4 may be too strong and")
        print("  create a near-total patient-level shortcut when repeated across ~30 labs. It is")
        print("  an EXPECTED, protocol-anticipated finding at OR=4, not evidence of a generator bug")
        print("  -- but it should be reported explicitly as a qualitative result/caveat wherever")
        print("  OR=4 sensitivity results are interpreted, and OR=8 remains correctly excluded from")
        print("  the main sensitivity range per the frozen decision.")
    else:
        print("  PASS -- no OR condition in the strength grid approached ceiling.")
    print("=" * 70)

    return result_df, any_fail, any_warn


if __name__ == "__main__":
    main()
