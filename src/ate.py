"""Average treatment effect: difference in proportions with CI, p-value and relative lift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import FIGURES_DIR, OUTCOMES, TREATMENT
from .plotting import SERIES, apply_style, plt

Z95 = stats.norm.ppf(0.975)


def diff_in_proportions(y: np.ndarray, t: np.ndarray) -> dict:
    """ATE = P(y|t=1) - P(y|t=0). Wald CI (unpooled SE); two-sided z-test p-value (pooled SE).

    Relative lift CI uses the delta method on log(p_t / p_c), which stays sensible when p_c is tiny.
    """
    y, t = np.asarray(y, dtype=np.float64), np.asarray(t).astype(bool)
    n_t, n_c = int(t.sum()), int((~t).sum())
    x_t, x_c = float(y[t].sum()), float(y[~t].sum())
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c

    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))

    rel = diff / p_c
    se_log = np.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    log_rr = np.log(p_t / p_c)
    rel_ci = [np.exp(log_rr - Z95 * se_log) - 1, np.exp(log_rr + Z95 * se_log) - 1]

    return {
        "n_treated": n_t, "n_control": n_c,
        "events_treated": int(x_t), "events_control": int(x_c),
        "rate_treated": p_t, "rate_control": p_c,
        "abs": diff, "se": se, "ci": [diff - Z95 * se, diff + Z95 * se],
        "z": z, "p_value": p_value,
        "rel": rel, "rel_ci": rel_ci,
        "incremental_per_1m_treated": diff * 1e6,
        "incremental_per_1m_treated_ci": [(diff - Z95 * se) * 1e6, (diff + Z95 * se) * 1e6],
    }


def plain_english(outcome: str, r: dict) -> str:
    pp = 100 * r["abs"]
    lo, hi = 100 * r["ci"][0], 100 * r["ci"][1]
    sig = "statistically significant" if r["p_value"] < 0.05 else "not statistically significant"
    p_txt = "p < 1e-300" if r["p_value"] < 1e-300 else f"p = {r['p_value']:.2g}"
    return (
        f"The campaign changed the {outcome} rate by {pp:+.3f} percentage points "
        f"(95% CI [{lo:+.3f}, {hi:+.3f}]), from {100 * r['rate_control']:.3f}% in control to "
        f"{100 * r['rate_treated']:.3f}% in treated - a relative lift of {100 * r['rel']:+.1f}% "
        f"(95% CI [{100 * r['rel_ci'][0]:+.1f}%, {100 * r['rel_ci'][1]:+.1f}%]). "
        f"That is about {r['incremental_per_1m_treated']:,.0f} extra {outcome}s per 1M treated users "
        f"(95% CI [{r['incremental_per_1m_treated_ci'][0]:,.0f}, {r['incremental_per_1m_treated_ci'][1]:,.0f}]); "
        f"{sig} ({p_txt})."
    )


def plot_rates(res: dict, path=None) -> None:
    """One small panel per outcome (different scales - never a dual axis)."""
    apply_style()
    path = path or FIGURES_DIR / "ate_rates.png"
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6))
    for ax, (outcome, r) in zip(np.atleast_1d(axes), res.items()):
        rates = [r["rate_control"], r["rate_treated"]]
        ses = [np.sqrt(p * (1 - p) / n) for p, n in zip(rates, [r["n_control"], r["n_treated"]])]
        ax.bar(["Control", "Treated"], [100 * v for v in rates], width=0.55,
               color=[SERIES[1], SERIES[0]], yerr=[100 * Z95 * s for s in ses],
               capsize=4, error_kw={"lw": 1.2})
        for i, v in enumerate(rates):
            ax.annotate(f"{100 * v:.3f}%", (i, 100 * v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9)
        ax.set_ylabel(f"{outcome} rate (%)")
        ax.set_title(f"{outcome}: {100 * r['rel']:+.1f}% relative lift")
        ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    t = df[TREATMENT].to_numpy()
    res = {o: diff_in_proportions(df[o].to_numpy(), t) for o in OUTCOMES}
    plot_rates(res)
    out = {}
    for o, r in res.items():
        out[o] = {**r, "summary": plain_english(o, r)}
        out[f"{o}_abs"] = r["abs"]
        out[f"{o}_ci"] = r["ci"]
    return out
