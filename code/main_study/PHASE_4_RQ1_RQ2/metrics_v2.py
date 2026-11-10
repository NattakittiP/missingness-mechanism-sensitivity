"""
Phase 7 — Metric Layer (Protocol v2.1 §"Phase 7"; formalized in
methodology-redesign-spec.md §6 "Model-selection integrity" and §11 "Metric
cleanup"). Four metrics, exactly as frozen there:

1. selection_regret
       Regret_{A->B} = S(m*_B, B) - S(m_hat_A, B)
   where m_hat_A is the family SELECTED (via inner-CV, never outer-test) when
   trained/tuned on source condition A, m*_B is the ORACLE family (argmax of
   outer-test score) for target condition B, and S(.,B) is that family's
   outer-test score under condition B. In the single-condition case used by
   Phase 3/4 (A == B, no deployment shift), this is exactly
   `OuterFoldResult.selection_regret` already computed per-fold by
   selection_v2.py. The general two-condition form (A != B) is what RQ3
   (Phase 5, source->target mechanism shift) will need; `selection_regret()`
   below implements the general formula so both cases share one function.

2. baseline_selection_displacement_rate
       D_{m,r} = (1/K) * sum_k I(m_hat_{k,r} != m_hat_{k,0})
   For a fixed mechanism m and rate r, the fraction of outer folds/repeats k
   whose SELECTED family at rate r differs from that SAME fold's selected
   family at the r=0 baseline. Replaces the old "winner-flip rate."

   NAMING NOTE (added during the Phase 0-3 recheck audit -- read this before
   using either quantity): selection_v2.OuterFoldResult ALSO has a field
   called `displaced`, which means something different: WITHIN one fold,
   whether the inner-CV-selected family differs from THAT SAME FOLD's own
   outer-test oracle. That is a legitimate, useful per-fold diagnostic (it is
   what phase3-selection-integrity-fix.md reported for fold 1 of the Natural
   condition), but it is NOT this module's `baseline_selection_displacement_rate`
   metric, which instead compares the selected family ACROSS RATES (r vs the
   r=0 baseline) within the same fold. One is a same-condition selected-vs-
   oracle check; the other is a cross-rate selected-vs-selected-at-baseline
   check. Do not conflate the two when writing up results.

3. within_condition_selection_entropy
       H = -sum_g p_g * log(p_g) / log(n_families)
   Normalized Shannon entropy of the distribution of which family gets
   selected across repeats (folds / mask seeds) within ONE fixed
   (mechanism, rate) condition. H=0 means the same family is selected every
   time (maximally stable); H=1 means selection is uniformly spread across
   all n_families (maximally unstable). Also reports the modal family and its
   selection frequency.

4. Kendall's tau_b rank-stability
       pairwise tau_b between the 5-family score ranking in different
   repeats of the SAME condition, summarized (mean/min/max). Spearman is
   dropped per §11 ("Kendall's tau_b only... 5 candidates is small enough").

Bonus (also specified in §11, cheap to include alongside the above):
selection_margin (inner-CV top1-top2) and evaluation_margin (outer-test
top1-top2), reported SEPARATELY per §11 ("not conflated").
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau

# selection_v2 types are used only for the convenience extractor functions at
# the bottom; the core metric functions below take plain dicts/floats so they
# have no hard dependency on selection_v2's internals and can be unit-tested
# in isolation.


# ---------------------------------------------------------------------------
# 1. Selection regret
# ---------------------------------------------------------------------------

def selection_regret(oracle_score_on_target: float, selected_score_on_target: float) -> float:
    """Regret_{A->B} = S(m*_B, B) - S(m_hat_A, B).

    Both scores must already be evaluated ON THE SAME target condition B
    (oracle_score_on_target = that condition's own best outer-test performer;
    selected_score_on_target = the source-selected family's score when
    evaluated on B). Non-negative by construction whenever the oracle really
    is the argmax over the same candidate set the selected family was drawn
    from (oracle_score_on_target >= selected_score_on_target); this function
    does not itself enforce that -- callers computing both scores over the
    same family set get non-negativity for free (see selection_v2.py, which
    does exactly that for the single-condition case).
    """
    return float(oracle_score_on_target) - float(selected_score_on_target)


# ---------------------------------------------------------------------------
# 2. Baseline-selection displacement rate
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DisplacementResult:
    rate: float
    baseline_rate: float
    n_folds: int
    n_displaced: int
    displacement_rate: float
    displaced_fold_ids: List[Any]


def baseline_selection_displacement_rate(
    selected_family_by_fold_and_rate: Dict[Any, Dict[float, str]],
    baseline_rate: float = 0.0,
) -> Dict[float, DisplacementResult]:
    """D_{m,r} for every rate r present (other than baseline_rate), for one
    fixed mechanism m.

    selected_family_by_fold_and_rate: {fold_id: {rate: selected_family_at_that_rate}}
    -- i.e. one fold-keyed dict per rate, collected from running the SAME set
    of outer folds at every rate in a rate sweep (RQ1), all with the SAME
    mechanism (this function computes D for a single mechanism; call it once
    per mechanism to get the full table).

    A fold missing its baseline_rate entry is skipped (with a note in the
    returned dict's rate>=0 sanity), since D_{m,r} is only defined relative to
    that fold's own baseline selection.
    """
    if not selected_family_by_fold_and_rate:
        raise ValueError("selected_family_by_fold_and_rate is empty")

    all_rates = set()
    for per_rate in selected_family_by_fold_and_rate.values():
        all_rates.update(per_rate.keys())
    other_rates = sorted(r for r in all_rates if r != baseline_rate)

    results: Dict[float, DisplacementResult] = {}
    for r in other_rates:
        displaced_folds = []
        n_considered = 0
        for fold_id, per_rate in selected_family_by_fold_and_rate.items():
            if baseline_rate not in per_rate or r not in per_rate:
                continue  # this fold doesn't have both rates -- can't compare
            n_considered += 1
            if per_rate[r] != per_rate[baseline_rate]:
                displaced_folds.append(fold_id)
        if n_considered == 0:
            continue
        results[r] = DisplacementResult(
            rate=r,
            baseline_rate=baseline_rate,
            n_folds=n_considered,
            n_displaced=len(displaced_folds),
            displacement_rate=len(displaced_folds) / n_considered,
            displaced_fold_ids=displaced_folds,
        )
    return results


# ---------------------------------------------------------------------------
# 3. Within-condition selection entropy
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EntropyResult:
    entropy_nats: float
    normalized_entropy: float
    modal_family: str
    modal_frequency: float
    distribution: Dict[str, float]
    n_repeats: int


def within_condition_selection_entropy(
    selected_families: Sequence[str],
    all_families: Sequence[str],
) -> EntropyResult:
    """Normalized Shannon entropy of which family gets selected across
    repeats within ONE fixed (mechanism, rate) condition.

    selected_families: one entry per repeat (e.g. one per outer fold, or one
        per mask seed) -- the family selected via inner-CV in that repeat.
    all_families: the full candidate roster (e.g. the 5 MODELS), used as the
        support for the distribution/normalization even if some families were
        never selected in any repeat (they just get probability 0).
    """
    n = len(selected_families)
    if n == 0:
        raise ValueError("selected_families is empty")
    if len(all_families) < 1:
        raise ValueError("all_families must be non-empty")
    if len(set(all_families)) != len(all_families):
        # Found during the independent Phase 4 pre-flight review: a duplicate
        # entry in all_families silently inflates the log(len(all_families))
        # normalization denominator (distribution is keyed by value, so it's
        # still correct; only the normalized_entropy scale would be wrong).
        raise ValueError(f"all_families contains duplicate entries: {all_families}")
    unknown = set(selected_families) - set(all_families)
    if unknown:
        raise ValueError(f"selected_families contains entries not in all_families: {unknown}")

    counts = Counter(selected_families)
    probs = {f: counts.get(f, 0) / n for f in all_families}
    nonzero_probs = [p for p in probs.values() if p > 0]

    if len(all_families) == 1:
        # Degenerate: only one possible family, entropy is trivially 0 by
        # convention (no uncertainty possible), avoid a log(1)=0 division.
        H = 0.0
        H_norm = 0.0
    else:
        H = float(-sum(p * np.log(p) for p in nonzero_probs))
        H_norm = H / np.log(len(all_families))

    modal_family, modal_count = counts.most_common(1)[0]
    return EntropyResult(
        entropy_nats=H,
        normalized_entropy=H_norm,
        modal_family=modal_family,
        modal_frequency=modal_count / n,
        distribution=probs,
        n_repeats=n,
    )


# ---------------------------------------------------------------------------
# 4. Kendall's tau_b rank-stability across repeats
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class KendallStabilityResult:
    mean_tau_b: float
    min_tau_b: float
    max_tau_b: float
    n_pairs: int
    n_repeats: int
    families: List[str]
    pairwise_tau_b: List[Tuple[int, int, float, float]]  # (repeat_i, repeat_j, tau, pvalue)


def kendall_tau_b_rank_stability(scores_by_repeat: Sequence[Dict[str, float]]) -> KendallStabilityResult:
    """Pairwise Kendall's tau_b between the family ranking (by score) in every
    pair of repeats of the SAME (mechanism, rate) condition. Spearman is
    deliberately not implemented here (dropped per methodology-redesign-spec
    §11: "5 candidates is small enough that one rank metric is enough").

    scores_by_repeat: list of {family: score} dicts, one per repeat (e.g. one
        per outer fold), all sharing the identical set of family keys (e.g.
        outer-test AUROC per family in each fold -- the natural choice for
        asking "how stable is the true generalization ranking across folds").
    """
    if len(scores_by_repeat) < 2:
        raise ValueError("need at least 2 repeats to assess rank stability")
    families = sorted(scores_by_repeat[0].keys())
    for i, d in enumerate(scores_by_repeat):
        if set(d.keys()) != set(families):
            raise ValueError(f"repeat {i} has a different family set than repeat 0: {set(d.keys())} vs {set(families)}")

    pairwise = []
    for i in range(len(scores_by_repeat)):
        for j in range(i + 1, len(scores_by_repeat)):
            xi = [scores_by_repeat[i][f] for f in families]
            xj = [scores_by_repeat[j][f] for f in families]
            tau, pvalue = kendalltau(xi, xj, variant="b")
            pairwise.append((i, j, float(tau) if not np.isnan(tau) else float("nan"), float(pvalue) if not np.isnan(pvalue) else float("nan")))

    taus = [t for (_, _, t, _) in pairwise if not np.isnan(t)]
    if not taus:
        raise ValueError("all pairwise tau_b values were NaN (likely a constant ranking in every repeat)")

    return KendallStabilityResult(
        mean_tau_b=float(np.mean(taus)),
        min_tau_b=float(np.min(taus)),
        max_tau_b=float(np.max(taus)),
        n_pairs=len(pairwise),
        n_repeats=len(scores_by_repeat),
        families=families,
        pairwise_tau_b=pairwise,
    )


# ---------------------------------------------------------------------------
# Bonus (§11): selection margin vs evaluation margin, reported separately
# ---------------------------------------------------------------------------

def top1_minus_top2_margin(scores: Dict[str, float]) -> float:
    """top1 - top2 among the given family scores. Used for BOTH:
    - selection margin: pass inner-CV scores (the margin the selection
      decision itself was made on)
    - evaluation margin: pass outer-test scores (the margin actually realized
      at evaluation time)
    Per §11 these must be reported as two separate numbers, never conflated
    into one "margin" -- this function is intentionally generic so callers
    are responsible for correctly labeling which one they computed.
    """
    if len(scores) < 2:
        return float("nan")
    vals = sorted(scores.values(), reverse=True)
    return float(vals[0] - vals[1])


# ---------------------------------------------------------------------------
# Convenience extractors from selection_v2.OuterFoldResult (duck-typed: works
# on anything with the same attribute names, so no hard import dependency).
# ---------------------------------------------------------------------------

def selected_family_by_fold(results: Sequence[Any]) -> Dict[Any, str]:
    """{fold_id: selected_family} from a list of OuterFoldResult."""
    return {r.fold_id: r.selected_family for r in results}


def outer_test_auroc_by_family(result: Any) -> Dict[str, float]:
    return {k: v.outer_test_auroc for k, v in result.families.items()}


def inner_auroc_by_family(result: Any) -> Dict[str, float]:
    return {k: v.inner_auroc_mean for k, v in result.families.items()}
