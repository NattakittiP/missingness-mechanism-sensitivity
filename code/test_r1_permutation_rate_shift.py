"""Known-answer checks for run_r1_permutation_rate_shift.py (python -m pytest test_r1_permutation_rate_shift.py)."""
import numpy as np

import run_r1_permutation_rate_shift as d


def test_perm_seed_matches_main_permutation_driver():
    # draw 0 = the 5-tuple of run_phase6_rq4.py; later draws append the draw id
    assert d.perm_entropy(20260917, 1, 0.30, 2.0, 0) == (20260917, 555, 1, 30, 20)
    assert d.perm_entropy(20260917, 3, 0.30, 1.5, 7) == (20260917, 555, 3, 30, 15, 7)
    p0 = np.random.default_rng(d.perm_entropy(20260917, 1, 0.40, 4.0, 0)).permutation(50)
    p1 = np.random.default_rng(d.perm_entropy(20260917, 1, 0.40, 4.0, 1)).permutation(50)
    assert not np.array_equal(p0, p1)


def test_rate_cells_cover_the_full_matrix():
    assert sorted(d.RATE_CELLS) == sorted(f"{m}_q{q}" for m in ("mcar", "mar", "mnar_y") for q in ("10", "30", "50"))
