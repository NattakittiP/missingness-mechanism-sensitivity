"""Known-answer checks for run_r1_multiple_imputation.py (python -m pytest test_r1_multiple_imputation.py)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import run_r1_multiple_imputation as mi


def _pipe(seed, X, y):
    pre = ColumnTransformer([("num", Pipeline([("imputer", IterativeImputer(sample_posterior=True, random_state=seed))]),
                              ["a", "b", "c"])])
    return Pipeline([("pre", pre), ("model", LogisticRegression())]).fit(X, y)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X.a + rng.normal(size=n) > 0).astype(int).to_numpy()
    X.loc[rng.random(n) < 0.3, "b"] = np.nan
    return X, y


def test_mi_average_is_mean_of_member_risks_and_reproducible():
    X, y = _data()
    members = [_pipe(s, X, y) for s in (1, 2, 3)]
    avg = mi.MIAverage(members)
    p1, p2 = avg.predict_proba(X), avg.predict_proba(X)
    assert np.array_equal(p1, p2)                     # same test matrix -> same imputations
    assert np.allclose(p1.sum(axis=1), 1.0)
    single = []
    for m, imps in zip(members, avg._imputers):       # each member, with its RNG at the start state
        imps[0].random_state_ = np.random.RandomState(imps[0].random_state)
        single.append(m.predict_proba(X)[:, 1])
    assert np.allclose(p1[:, 1], np.mean(single, axis=0))
