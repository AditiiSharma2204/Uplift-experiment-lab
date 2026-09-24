import numpy as np
from statsmodels.stats.proportion import proportions_ztest

from src.ate import diff_in_proportions
from src.common import assert_no_leakage


def test_diff_matches_statsmodels():
    rng = np.random.default_rng(0)
    t = rng.random(200_000) < 0.85
    y = (rng.random(200_000) < np.where(t, 0.05, 0.04)).astype(int)
    r = diff_in_proportions(y, t)
    _, p_sm = proportions_ztest([y[t].sum(), y[~t].sum()], [t.sum(), (~t).sum()])
    assert np.isclose(r["p_value"], p_sm)
    assert r["ci"][0] < 0.01 < r["ci"][1]


def test_leakage_guard():
    assert_no_leakage([f"f{i}" for i in range(12)])
    try:
        assert_no_leakage(["f0", "exposure"])
    except AssertionError:
        return
    raise AssertionError("exposure should be rejected")
