"""Synthetic check with 'sure things'.

Users with f1 > 1 have a high baseline visit rate (30%) and ZERO uplift; uplift (+15 pp) exists only
for f0 > 0 among the rest. Uplift learners should rank the f0 > 0 group on top; the response model
should rank the sure things on top even though treating them gains nothing.
"""
import numpy as np
import pandas as pd
import pytest

from src.common import FEATURES
from src.uplift_models import PropensityModel, ResponseModel, SLearner, TLearner, XLearner

PARAMS = {"num_leaves": 15, "min_child_samples": 50}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 60_000
    X = pd.DataFrame(rng.normal(size=(n, 12)).astype(np.float32), columns=FEATURES)
    t = (rng.random(n) < 0.85).astype(np.int8)
    base = 0.05 + 0.25 * (X["f1"] > 1)
    y = (rng.random(n) < base + 0.15 * t * ((X["f0"] > 0) & (X["f1"] <= 1))).astype(np.int8)
    k = int(n * 0.8)
    return X[:k], t[:k], y[:k], X[k:], t[k:], y[k:]


def _groups(Xv):
    f0, f1 = Xv["f0"].to_numpy(), Xv["f1"].to_numpy()
    return (f0 > 0) & (f1 <= 1), f1 > 1  # persuadables, sure things


def _gap(model, Xv):
    s = model.predict(Xv)
    persuadable, sure = _groups(Xv)
    return s[persuadable].mean() - s[sure].mean()


@pytest.mark.parametrize("cls", [TLearner, SLearner])
def test_uplift_learners_find_heterogeneity(data, cls):
    X, t, y, Xv, tv, yv = data
    assert _gap(cls(PARAMS).fit(X, t, y, Xv, tv, yv), Xv) > 0.08


def test_x_learner_finds_heterogeneity(data):
    X, t, y, Xv, tv, yv = data
    tl = TLearner(PARAMS).fit(X, t, y, Xv, tv, yv)
    prop = PropensityModel(PARAMS).fit(X, t, Xv, tv)
    assert _gap(XLearner(PARAMS, t_learner=tl, propensity=prop).fit(X, t, y, Xv, tv, yv), Xv) > 0.08


def test_response_model_follows_baseline_not_uplift(data):
    X, t, y, Xv, tv, yv = data
    m = ResponseModel(PARAMS).fit(X, t, y, Xv, tv, yv)
    assert _gap(m, Xv) < -0.08  # sure things ranked ABOVE persuadables: the wrong targets


def test_exposure_is_rejected(data):
    X, t, y, Xv, tv, yv = data
    Xe, Xve = X.assign(exposure=1.0), Xv.assign(exposure=1.0)
    with pytest.raises(AssertionError, match="Leakage"):
        TLearner(PARAMS).fit(Xe, t, y, Xve, tv, yv)
