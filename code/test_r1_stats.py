"""Known-answer tests for the statistics and summaries of the R1 drivers.
Run from Revision_R1/Code:  python -m pytest test_r1_stats.py -q
"""
import numpy as np
import pandas as pd

import run_r1_missingness_handling as h
import run_r1_repeated_outer_cv as rcv


def test_nadeau_bengio_known_answer():
    d = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    t, p = rcv.nadeau_bengio(d, n_train=80.0, n_test=20.0)
    # var = 2.5e-4; se = sqrt((1/5 + 0.25) * 2.5e-4) = 0.0106066; t = 0.03 / se
    assert abs(t - 2.8284271) < 1e-6
    assert 0.04 < p < 0.05


def test_sign_test_ignores_ties():
    k, n, p = rcv.sign_test(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    assert (k, n) == (5, 5) and abs(p - 0.0625) < 1e-12


def test_handling_summary_displacement_and_shift():
    def row(s, src, tgt, k, fam, auc):
        r = dict(strategy=s, source=src, target=tgt, fold_id=k, selected_family=fam,
                 inner_auroc_selected=0.0, fit_seconds=0.0)
        for f in ("a", "b"):
            r[f"auroc_{f}"], r[f"ap_{f}"], r[f"brier_{f}"] = (auc if f == fam else auc - 0.01), 0.0, 0.0
        return r
    ev = pd.DataFrame([
        row("median", "natural_q00", "natural_q00", 1, "a", 0.90),
        row("median", "natural_q00", "natural_q00", 2, "a", 0.90),
        row("median", "mcar_q30", "mcar_q30", 1, "a", 0.88),
        row("median", "mcar_q30", "mcar_q30", 2, "b", 0.86),
        row("median", "mcar_q30", "mnar_y_q30_OR2", 1, "a", 0.85),
        row("median", "mcar_q30", "mnar_y_q30_OR2", 2, "b", 0.84),
    ])
    s = h.summarise(ev, pd.DataFrame()).set_index(["source", "target"])
    assert s.loc[("mcar_q30", "mcar_q30"), "displacement"] == 0.5  # fold 2 moved from a to b
    assert abs(s.loc[("mcar_q30", "mnar_y_q30_OR2"), "shift_delta"] - (-0.025)) < 1e-12  # mean(-0.03, -0.02)
