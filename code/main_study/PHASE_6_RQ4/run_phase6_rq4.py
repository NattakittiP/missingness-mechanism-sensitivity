"""
Phase 6 driver -- RQ4 (shortcut attribution: mask-only / count-only baselines,
test-only permutation ablation, explicit-indicator-concat ablation).

RQ4 (methodology-redesign-spec.md Sec.0): "In outcome-linked MNAR-Y, how much
of the apparent AUROC gain is attributable to the missingness PATTERN itself
vs genuine signal?" -- this is the direct mechanistic follow-up to RQ2's
shortcut-learning caveat (Phase 4) and to Phase 5's finding that a
shortcut-reliant model degrades gracefully rather than catastrophically under
mechanism shift: this phase asks how much of its apparent skill was ever
"genuine" in the first place.

Primary protocol document, Phase 6 list (verbatim, item order preserved):
  - mask-only baseline
  - count-only baseline
  - test-only permutation ablation
  - indicator-concat ablation
  - MNAR-X, calibration sensitivity, AP-based family-selection sensitivity,
    alternative split sensitivity   <- OUT OF SCOPE here, protocol_v2_1_frozen.md
    item 11, strictly after RQ4 (item 10). This driver implements ONLY the
    first four (RQ4 proper).

methodology-redesign-spec.md Sec.10 (verbatim spec for the four techniques):
  1. Mask-only model: LogisticRegression on M (indicator vector) alone, no
     feature values. Report AUROC/AP.
  2. Mask-count-only: single score C_i = sum_j M^syn_ij, AUROC/AP directly.
  3. Permutation ablation: shuffle M^syn across patients (preserve per-feature
     marginal rate, destroy Y-association), rerun full model, compare AUROC
     before/after -- run at MNAR-Y only, at r in {30%, 40%} for the main and
     one sensitivity OR.
  4. Explicit-indicator ablation: [X_imputed] vs [X_imputed, M] as model
     input, at baseline / r=30% / r=40%, MCAR vs MNAR-Y only (not all 4xN
     cells).

THREE SECTIONS, each independently checkpointed:

SECTION A -- mask-only / count-only / synth-only baselines (methodology
spec items 1+2, generalized across the whole grid, not just MNAR-Y -- see
"scope decision" below). Cheap: plain LogisticRegression on <=30 features or
1 scalar, no GridSearchCV. Grid: q=0 (Natural) + run_phase4_rq1_rq2's exact
14-condition RQ1+RQ2 grid (mechanism x rate for RQ1, MNAR-Y OR-sweep for
RQ2), using the SAME masking code, seeds, and per-fold splits as Phase 4 --
so these baseline numbers sit directly alongside Phase 4's already-reported
full-model AUROC for the identical conditions, which is exactly what an
attribution analysis needs (full-model AUROC vs. mask-only/count-only AUROC,
same condition, same fold).

  SCOPE DECISION (documented, not silently assumed): methodology-redesign-
  spec.md's RQ4 row scopes this to "mechanism = MNAR-Y" only. This driver
  runs it across ALL THREE mechanisms (matching what phase2_3_manipulation_
  pilot.py already did "beyond the Protocol's literal minimum" for its own
  1-fold pilot -- see that file's own docstring), at negligible extra
  compute (LogisticRegression baselines, not GridSearchCV) -- the same
  design choice the pilot already made and that phase4-results-analysis.md
  relied on ("the pilot's synthetic-mask-only baseline already hit
  AUROC=0.97 at OR=4").

  IMPORTANT CORRECTION vs. this driver's own first-draft assumption, found
  during its own smoke test (2026-09-18) -- read before interpreting Section
  A's MCAR/MAR numbers: MCAR/MAR are NOT a clean "no Y-signal" negative
  control for the synth-only baselines, even though their injection
  PROBABILITY model has zero Y-dependence built in. Because synthetic
  injection can only occur on cells that are natively OBSERVED (eligible_
  mask = ~natural_mask -- see missingness_generator_v2.compute_eligible_
  mask/apply_synthetic_mask), and native missingness already correlates
  with Y (the pre-existing "floor" this project established back in the
  Phase 2.3 pilot: native mask_only/count_only AUROC approx 0.86/0.84), the
  REALIZED synthetic mask's row-level statistics (how many cells even had a
  CHANCE to be injected) inherit a nontrivial residual Y-correlation purely
  through this eligibility-count channel -- confirmed numerically on real
  Dataset A (fold 1, q=0.30): corr(eligible_count, synthetic_count)=0.68,
  and a raw (unfit) synthetic-injection-count score alone already reaches
  AUROC~0.63 (MCAR) / ~0.64 (MAR) vs. Y, purely from this structural
  artifact -- MNAR-Y's own raw synthetic-count score reaches ~0.91 by the
  same measure, i.e. its INTENTIONAL outcome-dependence sits on TOP of this
  same ~0.63 baseline, not instead of it. The write-up must report this
  ~0.63 eligibility-confound floor explicitly and read MNAR-Y's synth-only
  numbers as (confound + genuine outcome-driven signal), not attribute the
  full gap over 0.5 to intentional MNAR-Y design.

SECTION B -- test-only permutation ablation (spec item 3). MNAR-Y only, rate
in {0.30, 0.40} (both in the frozen main rate grid), OR in {2.0 (main),
4.0}. OR=4.0 is this driver's choice of "one sensitivity OR" (not itself
frozen anywhere) -- justified because OR=4 is the specific condition RQ2's
own write-up (phase4-results-analysis.md Sec.4) already flagged as the
"near-ceiling shortcut" case, making it the single most informative
sensitivity point to test whether permuting the synthetic mask collapses
that near-ceiling effect. Requires a genuinely fresh full-model fit per
(rate, OR, fold) cell -- Phase 4's own results for these exact conditions
have scores but not frozen model objects (same limitation Phase 5 already
worked around for its own purposes), and the permutation test specifically
needs the frozen model to evaluate against a second, permuted-mask copy of
the SAME outer-test fold without retraining -- reuses selection_v2.
fit_and_select_all_families()/evaluate_frozen_model() exactly as Phase 5
does.

  "Test-only" (source protocol doc's own wording, Phase 6 list) means: only
  the synthetic mask of the OUTER-TEST fold is permuted; the model is never
  retrained, retuned, or recalibrated -- structurally identical discipline
  to Phase 5's "freeze everything" rule, just swapping a permuted mask in
  place of a different mechanism's mask.

SECTION C -- explicit-indicator-concat ablation (spec item 4). MCAR vs
MNAR-Y only, rate in {0.0 (baseline), 0.30, 0.40}. The WITH-indicator side
(model input = [X_imputed, M], M = one binary "is this lab missing"
column per eligible lab, appended as ordinary numeric features so they flow
through the SAME impute/scale pipeline as everything else) is computed
FRESH here (6 conditions x 5 folds = 30 fold-fits). The WITHOUT-indicator
side (model input = [X_imputed] only) is NOT recomputed -- it is read
directly from Phase 4's own real results for the IDENTICAL conditions
(mcar_q00/q30_main/q40_main, mnar_y_q00/q30_main/q40_main), which this
project has already established (Phase 5's driver summary Sec.2, and this
session's Phase-5 recheck) reproduce bit-for-bit under these exact seeds --
recomputing them here would be pure duplicate compute for a value already
known exactly. If Phase 4's results file isn't found at the expected path,
this comparison is skipped with a clear warning (non-fatal) rather than
failing the whole driver -- the WITH-indicator numbers are still useful on
their own.

FOLD-SAFETY: identical convention to Phase 4/5 in every section -- for every
(mechanism, rate[, OR], fold) cell, alpha_j is calibrated on that fold's own
outer-train partition only, then the SAME fitted generator is applied to
that fold's outer-train and outer-test. Outer-test rows never influence any
alpha_j, in any section.

SEED REUSE: OUTER_SEED/MASK_SEED are imported (not redefined) from
run_phase4_rq1_rq2, exactly as Phase 5 did -- so Section A's and Section
B/C's diagonal-equivalent cells (e.g. mcar_q30_main's masks) are, by
construction, the SAME masks Phase 4 and Phase 5 already used.

CHECKPOINTING: one checkpoint file per (section, condition, fold) cell under
results/phase6/ -- baseline_<condition_id>_fold<F>.pkl, perm_<condition_id>_
fold<F>.pkl, ind_<condition_id>_fold<F>.pkl. Re-running this script skips
any cell whose checkpoint already exists.

Run with:   python3 run_phase6_rq4.py
Smoke-test: python3 run_phase6_rq4.py --smoke-test
            (2 outer folds instead of 5; Section A reduced to 2 conditions
            (Natural q=0, mcar_q30_main); Section B reduced to 1 (rate, OR)
            cell (r=0.30, OR=2.0 main only); Section C reduced to 1
            (mechanism, rate) cell (MNAR-Y, r=0.30) -- exercises every code
            path in all three sections at a scale that finishes in a few
            minutes. Writes to results/phase6_smoke/, never touches
            results/phase6/.)
"""

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

# Reuses Phase 4's driver for: _prepare_X_y_groups, _fit_generator_and_drivers,
# _resolve_data_path, make_logger, OUTER_SEED, MASK_SEED, THETA_MAIN,
# build_condition_grid -- see module docstring's "SEED REUSE" note.
import run_phase4_rq1_rq2 as p4

warnings.resetwarnings()
warnings.simplefilter("once")
warnings.filterwarnings("ignore", message=r".*'penalty' was deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*'n_jobs' has no effect.*", category=FutureWarning)

OUTER_SEED = p4.OUTER_SEED       # 2026 -- same as Phase 4/5
MASK_SEED = p4.MASK_SEED         # 20260917 -- same as Phase 4/5
N_OUTER = 5

RESULTS_DIR = Path("results/phase6")

# Section B
PERMUTATION_RATES = [0.30, 0.40]
PERMUTATION_ORS = {2.0: p4.THETA_MAIN, 4.0: float(np.log(4.0))}  # main + one sensitivity OR -- see module docstring
PERMUTATION_SEED_OFFSET = 555  # documented, distinct from MASK_SEED's own stream -- never reused for anything else

# Section C
INDICATOR_MECHANISMS = ["mcar", "mnar_y"]
INDICATOR_RATES = [0.0, 0.30, 0.40]

_PHASE4_CSV_CONTAINER = "/mnt/user-data/uploads/DASA2026/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv"

# FIX (post-build adversarial review, Finding 2, 2026-09-18): a single
# hardcoded relative path (script's grandparent / PHASE_4_RQ1_RQ2 / results /
# phase4 / phase4_full_table.csv) is correct ONLY under the exact documented
# sibling layout (DASA2026/PHASE_6_RQ4/ next to DASA2026/PHASE_4_RQ1_RQ2/,
# this script run from within PHASE_6_RQ4/) -- and gives no signal at all if
# the user's real layout differs even slightly (nested one level deeper/
# shallower, a renamed folder, or a copy of this script run from elsewhere).
# It also silently prefers a STALE phase4_full_table.csv over saying nothing,
# if one happens to exist at a guessed path that isn't actually the fresh
# run's output. Replaced with a short list of plausible candidate locations,
# tried in order, each checked for existence before use, and the path that
# was actually chosen (or the full candidate list, if none matched) is always
# logged -- so a wrong guess is visible in run.log rather than silent.
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


# ---------------------------------------------------------------------------
# Section A -- mask-only / count-only / synth-only baselines
# ---------------------------------------------------------------------------

def _fit_eval_baseline(mask_train_df: pd.DataFrame, mask_test_df: pd.DataFrame,
                        y_train: np.ndarray, y_test: np.ndarray, cols: List[str]) -> Dict[str, float]:
    """Ported from phase2_3_manipulation_pilot.py's _fit_eval (the pilot's
    1-fold version of the exact same computation), extended with average
    precision. LogisticRegression on the binary mask vector -> mask_auroc/ap;
    LogisticRegression on the scalar per-row missing-count -> count_auroc/ap.
    NaN out (not an error) on a degenerate cell (constant mask, a single
    class in y_train, OR a single class in y_test -- matches the pilot's own
    convention; the y_test guard was added post-build, adversarial review
    Finding 3: roc_auc_score/average_precision_score both raise on a
    single-class y_test, which y_train's guard alone does not catch)."""
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
    """Fold-safe masking for ONE (mechanism, rate, theta) condition -- mirrors
    run_phase4_rq1_rq2.run_condition_with_injection's per-fold body exactly
    (same helper calls, same RNG convention), factored out here so Section A
    can call it once per condition per fold without duplicating that logic."""
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


# ---------------------------------------------------------------------------
# Section B -- test-only permutation ablation
# ---------------------------------------------------------------------------

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

                # Test-only permutation: shuffle syn_mask_te's ROWS across
                # patients -- a row permutation preserves every column's sum
                # (per-feature marginal injection rate) exactly, while
                # destroying row-to-row correlation with y_te (y_te itself is
                # never permuted). nat_mask_te is left untouched -- only the
                # MNAR-Y-injected component of the mask is permuted. Distinct,
                # documented seed offset -- never reuses the masking stream.
                #
                # FIX (post-build adversarial review, 2026-09-18): the seed
                # must vary by fold_id/rate/or_val, not just be a single fixed
                # MASK_SEED+offset -- otherwise two outer folds with the same
                # test-set SIZE (this grid has two such pairs: folds 1&2 both
                # n=2817, folds 4&5 both n=2816) draw byte-identical
                # np.random.default_rng(...).permutation(n) arrays, and all 4
                # (rate,OR) cells within one fold reused that same single
                # permutation pattern too. Mixing in fold_id/rate/or_val via a
                # SeedSequence-style entropy tuple gives every (fold, rate,
                # OR) cell its own independent permutation draw. Still never
                # touches the masking stream (MASK_SEED itself, used above),
                # and is fully reproducible from these four integers alone.
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


# ---------------------------------------------------------------------------
# Section C -- explicit-indicator-concat ablation
# ---------------------------------------------------------------------------

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

                # Explicit-indicator ablation: append one binary "is this lab
                # missing" column per eligible lab (computed from the ALREADY-
                # masked X, so it exactly encodes final_mask -- native +
                # synthetic combined). Model input becomes [X_imputed, M]
                # instead of just [X_imputed]; these flow through the SAME
                # impute/scale/sanitize pipeline as every other numeric
                # feature (they have no missing values themselves).
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

    # WITHOUT-indicator side: reuse Phase 4's real results for the identical
    # conditions (see module docstring, Section C) rather than recomputing.
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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
