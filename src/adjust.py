"""Covariate-adjusted ATE (Lin 2013 estimator) and how much it shrinks the standard error.

Lin estimator:  y ~ 1 + t + Xc + t:Xc,  Xc = X - mean(X) over the whole sample, HC1 robust SE.
Centring makes the coefficient on t the ATE; the interactions let each arm have its own slopes,
which guarantees the adjusted estimator is no less precise asymptotically than the unadjusted one.

Wording matters: CUPED proper uses the *pre-period value of the same metric*. Criteo gives us only
anonymized pre-treatment features, so this is regression adjustment on features, not CUPED.

Two covariate sets:
  * linear   - the 12 raw features (the textbook Lin estimator)
  * ml_score - one covariate: a cross-fitted LightGBM prediction of the outcome from f0-f11
               (CUPAC-style). Treatment is not an input, and each row's score comes from a model that
               never saw that row, so the score is a valid pre-treatment covariate.

OLS is computed in row chunks (X'X, X'y, then the HC1 "meat"), so 14M rows x 26 columns never
needs more than a few hundred MB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import FEATURES, FIGURES_DIR, OUTCOMES, PROCESSED_DIR, SEED, TREATMENT, assert_no_leakage
from .plotting import NEUTRAL, SERIES, apply_style, plt

CHUNK = 1_000_000
SCORES_PARQUET = PROCESSED_DIR / "outcome_scores_crossfit.parquet"
N_STRATA = 20
Z95 = stats.norm.ppf(0.975)


def _design(t: np.ndarray, xc: np.ndarray | None) -> np.ndarray:
    cols = [np.ones_like(t), t]
    if xc is not None:
        cols += [xc, xc * t[:, None]]
    return np.column_stack(cols)


def ols_hc1(y: np.ndarray, t: np.ndarray, xc: np.ndarray | None, chunk: int = CHUNK) -> dict:
    """OLS of y on [1, t, (xc, t*xc)] with HC1 robust covariance, computed chunk by chunk.

    Returns the coefficient and robust SE of t (index 1), plus residual sum of squares.
    """
    n = len(y)
    t = t.astype(np.float64)
    k = 2 + (0 if xc is None else 2 * xc.shape[1])
    xtx, xty = np.zeros((k, k)), np.zeros(k)
    for s in range(0, n, chunk):
        X = _design(t[s:s + chunk], None if xc is None else xc[s:s + chunk].astype(np.float64))
        xtx += X.T @ X
        xty += X.T @ y[s:s + chunk]
    beta = np.linalg.solve(xtx, xty)
    meat, ssr = np.zeros((k, k)), 0.0
    for s in range(0, n, chunk):
        X = _design(t[s:s + chunk], None if xc is None else xc[s:s + chunk].astype(np.float64))
        e = y[s:s + chunk] - X @ beta
        meat += (X * (e ** 2)[:, None]).T @ X
        ssr += float(e @ e)
    bread = np.linalg.inv(xtx)
    cov = n / (n - k) * bread @ meat @ bread
    return {"ate": float(beta[1]), "se": float(np.sqrt(cov[1, 1])), "ssr": ssr, "k": k, "n": n}


def crossfit_score(X: pd.DataFrame, y: np.ndarray, n_folds: int = 2, seed: int = SEED) -> np.ndarray:
    """Out-of-fold LightGBM prediction of y from features only (no treatment)."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    assert_no_leakage(X.columns)
    score = np.zeros(len(y), dtype=np.float64)
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                               min_child_samples=500, subsample=0.5, subsample_freq=1,
                               random_state=seed, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        score[te] = m.predict_proba(X.iloc[te])[:, 1]
    return score


def outcome_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-fitted outcome scores for every row, cached to parquet (row order = full parquet)."""
    if SCORES_PARQUET.exists():
        cached = pd.read_parquet(SCORES_PARQUET)
        if len(cached) == len(df):
            return cached
    scores = pd.DataFrame({f"score_{o}": crossfit_score(df[FEATURES], df[o].to_numpy()).astype(np.float32)
                           for o in OUTCOMES})
    scores.to_parquet(SCORES_PARQUET, index=False)
    return scores


def score_strata(df: pd.DataFrame, score: np.ndarray, outcome: str, n_strata: int = N_STRATA) -> dict:
    """Diagnostic + post-stratified ATE: bin users by predicted outcome, compare arms within each bin.

    Under clean randomization the treated share is ~flat across bins. The post-stratified ATE
    (bin-size-weighted mean of within-bin differences) is a transparent, model-light check on the
    regression-adjusted estimate.
    """
    t = df[TREATMENT].to_numpy().astype(bool)
    y = df[outcome].to_numpy(dtype=np.float64)
    edges = np.unique(np.quantile(score, np.linspace(0, 1, n_strata + 1)))
    b = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    rows, est, var = [], 0.0, 0.0
    n = len(y)
    for k in range(len(edges) - 1):
        m = b == k
        mt, mc = m & t, m & ~t
        if mt.sum() == 0 or mc.sum() == 0:
            continue
        pt, pc = y[mt].mean(), y[mc].mean()
        w = m.sum() / n
        est += w * (pt - pc)
        var += w ** 2 * (pt * (1 - pt) / mt.sum() + pc * (1 - pc) / mc.sum())
        rows.append({"stratum": k, "n": int(m.sum()), "treat_share": float(t[m].mean()),
                     "rate_treated": pt, "rate_control": pc, "diff": pt - pc,
                     "mean_score": float(score[m].mean())})
    se = np.sqrt(var)
    tab = pd.DataFrame(rows)
    rho = stats.spearmanr(tab["mean_score"], tab["treat_share"])
    return {"ate_poststratified": est, "se": se, "ci": [est - Z95 * se, est + Z95 * se],
            "n_strata": len(rows), "treat_share_min": float(tab["treat_share"].min()),
            "treat_share_max": float(tab["treat_share"].max()),
            "spearman_score_vs_treat_share": float(rho.statistic), "table": rows}


def plot_strata(strata: dict, outcome: str, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / f"strata_{outcome}.png"
    tab = pd.DataFrame(strata["table"])
    x = np.arange(1, len(tab) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(x, 100 * tab["treat_share"], "o-", color=SERIES[0])
    ax.axhline(85, color=NEUTRAL, ls="--", lw=1)
    ax.text(1, 85.05, "85% design ratio", color=NEUTRAL, fontsize=8, va="bottom")
    ax.set_xlabel(f"Bin of predicted {outcome} probability (low to high)")
    ax.set_ylabel("Treated share (%)")
    ax.set_title("Treated share by predicted-outcome bin")
    ax = axes[1]
    ax.plot(x, 100 * tab["rate_control"], "o-", color=SERIES[1], label="Control")
    ax.plot(x, 100 * tab["rate_treated"], "o-", color=SERIES[0], label="Treated")
    ax.set_yscale("log")
    ax.set_xlabel(f"Bin of predicted {outcome} probability (low to high)")
    ax.set_ylabel(f"{outcome} rate (%, log scale)")
    ax.set_title(f"{outcome} rate by arm within bin")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _summary(base: dict, adj: dict) -> dict:
    vr = 1 - (adj["se"] / base["se"]) ** 2
    return {
        "ate_adjusted": adj["ate"], "se_adjusted": adj["se"],
        "ci_adjusted": [adj["ate"] - Z95 * adj["se"], adj["ate"] + Z95 * adj["se"]],
        "variance_reduction_pct": 100 * vr,
        "se_reduction_pct": 100 * (1 - adj["se"] / base["se"]),
        "equivalent_sample_size_multiplier": 1 / (1 - vr) if vr < 1 else None,
        "r2_outcome_explained": 1 - adj["ssr"] / base["ssr"],
        "ate_shift_vs_unadjusted_in_se": (adj["ate"] - base["ate"]) / base["se"],
    }


def run(df: pd.DataFrame, with_ml_score: bool = True) -> dict:
    assert_no_leakage(FEATURES)
    t = df[TREATMENT].to_numpy()
    X = df[FEATURES].to_numpy(dtype=np.float64)
    xc = X - X.mean(axis=0)
    del X
    out: dict = {"method": "Lin (2013): y ~ t + Xc + t:Xc, HC1 robust SE", "n": len(df)}
    for o in OUTCOMES:
        y = df[o].to_numpy(dtype=np.float64)
        base = ols_hc1(y, t, None)
        lin = ols_hc1(y, t, xc)
        res = {"ate_unadjusted": base["ate"], "se_unadjusted": base["se"],
               "ci_unadjusted": [base["ate"] - Z95 * base["se"], base["ate"] + Z95 * base["se"]],
               "linear": _summary(base, lin)}
        if with_ml_score:
            s = outcome_scores(df)[f"score_{o}"].to_numpy(dtype=np.float64)
            sc = (s - s.mean())[:, None]
            res["ml_score"] = _summary(base, ols_hc1(y, t, sc))
            res["ml_score"]["score_auc"] = _auc(df[o].to_numpy(), s)
            res["poststratified"] = score_strata(df, s, o)
            plot_strata(res["poststratified"], o)
        out[o] = res
    v = out["visit"]
    out["variance_reduction_pct"] = v["linear"]["variance_reduction_pct"]
    if with_ml_score:
        out["variance_reduction_pct_ml_score"] = v["ml_score"]["variance_reduction_pct"]
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))
