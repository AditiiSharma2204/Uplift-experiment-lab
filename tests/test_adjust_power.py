import numpy as np
import statsmodels.api as sm

from src.adjust import ols_hc1
from src.power import analytical_power, mde_abs


def test_chunked_ols_matches_statsmodels_hc1():
    rng = np.random.default_rng(1)
    n = 20_000
    X = rng.normal(size=(n, 3))
    t = (rng.random(n) < 0.85).astype(float)
    y = (rng.random(n) < 0.05 + 0.01 * t + 0.02 * (X[:, 0] > 0)).astype(float)
    xc = X - X.mean(0)
    ours = ols_hc1(y, t, xc, chunk=3_001)
    D = np.column_stack([np.ones(n), t, xc, xc * t[:, None]])
    sm_fit = sm.OLS(y, D).fit(cov_type="HC1")
    assert np.isclose(ours["ate"], sm_fit.params[1])
    assert np.isclose(ours["se"], sm_fit.bse[1])


def test_power_at_mde_is_80pct():
    n, p, share = 1_000_000, 0.04, 0.85
    m = mde_abs(n, p, share)
    assert abs(analytical_power(n, p, m, share) - 0.80) < 0.01
