"""Phase 7 metric layer: selection regret, displacement rate, selection entropy, and Kendall's tau_b rank stability."""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau




def selection_regret(oracle_score_on_target: float, selected_score_on_target: float) -> float:
    """Regret of a source-selected family relative to the target condition's oracle family."""
    return float(oracle_score_on_target) - float(selected_score_on_target)



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
    """Fraction of outer folds whose selected family at rate r differs from that fold's own r=0 baseline selection."""
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
                continue
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
    """Normalized Shannon entropy of which family gets selected across repeats within one condition."""
    n = len(selected_families)
    if n == 0:
        raise ValueError("selected_families is empty")
    if len(all_families) < 1:
        raise ValueError("all_families must be non-empty")
    if len(set(all_families)) != len(all_families):
        raise ValueError(f"all_families contains duplicate entries: {all_families}")
    unknown = set(selected_families) - set(all_families)
    if unknown:
        raise ValueError(f"selected_families contains entries not in all_families: {unknown}")

    counts = Counter(selected_families)
    probs = {f: counts.get(f, 0) / n for f in all_families}
    nonzero_probs = [p for p in probs.values() if p > 0]

    if len(all_families) == 1:
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



@dataclasses.dataclass
class KendallStabilityResult:
    mean_tau_b: float
    min_tau_b: float
    max_tau_b: float
    n_pairs: int
    n_repeats: int
    families: List[str]
    pairwise_tau_b: List[Tuple[int, int, float, float]]


def kendall_tau_b_rank_stability(scores_by_repeat: Sequence[Dict[str, float]]) -> KendallStabilityResult:
    """Pairwise Kendall's tau_b between the family score ranking across repeats of the same condition."""
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



def top1_minus_top2_margin(scores: Dict[str, float]) -> float:
    """Margin between the top-ranked and second-ranked family's scores."""
    if len(scores) < 2:
        return float("nan")
    vals = sorted(scores.values(), reverse=True)
    return float(vals[0] - vals[1])



def selected_family_by_fold(results: Sequence[Any]) -> Dict[Any, str]:
    """Maps fold id to its selected family from a list of OuterFoldResult."""
    return {r.fold_id: r.selected_family for r in results}


def outer_test_auroc_by_family(result: Any) -> Dict[str, float]:
    return {k: v.outer_test_auroc for k, v in result.families.items()}


def inner_auroc_by_family(result: Any) -> Dict[str, float]:
    return {k: v.inner_auroc_mean for k, v in result.families.items()}
