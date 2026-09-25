import numpy as np

from src.evaluate import GRID, Ranked, auuc


def _data(n=200_000, seed=0, confounded=False):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    tau = 0.10 * (x > 0.8)                       # only the top 20% by x respond
    e = 0.85 + (0.1 * (x - 0.5) if confounded else 0.0)
    t = (rng.random(n) < e).astype(int)
    base = 0.05 + 0.3 * x                         # baseline rises with x -> confounding if e depends on x
    y = (rng.random(n) < base + tau * t).astype(int)
    return x, t, y, np.full(n, e) if np.isscalar(e) else e, tau


def test_full_population_gain_equals_difference_in_means():
    x, t, y, e, _ = _data()
    _, gain = Ranked(x, t, y, None).curve()
    assert np.isclose(gain[-1], y[t == 1].mean() - y[t == 0].mean())


def test_oracle_beats_random_and_random_is_null():
    x, t, y, e, tau = _data()
    g_oracle = Ranked(tau + 1e-9 * x, t, y, e).curve()[1]
    g_random = Ranked(np.random.default_rng(1).random(len(x)), t, y, e).curve()[1]
    assert auuc(g_oracle) > 0.005
    assert abs(auuc(g_random)) < 0.002
    # the oracle's top 20% captures (almost) the whole effect
    j = int(np.argmin(np.abs(GRID - 0.2)))
    assert abs(g_oracle[j] - 0.02) < 0.004


def test_ipw_removes_confounding_bias():
    x, t, y, e, tau = _data(n=400_000, confounded=True)
    true_ate = tau.mean()
    naive = Ranked(x, t, y, None).curve()[1][-1]
    ipw = Ranked(x, t, y, e).curve()[1][-1]
    assert abs(ipw - true_ate) < abs(naive - true_ate)
    assert abs(ipw - true_ate) < 0.004
