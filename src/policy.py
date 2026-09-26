"""Budget-constrained targeting policy and illustrative economics.

Policy: treat the top k of users by the best model's predicted uplift (selected in S4 on validation).
Everything is per 1,000,000 users of the population and reuses S4's 200 paired bootstrap replicates
of the propensity-weighted gain curve, so every number here has a 95% interval.

    incremental(k) = extra visits if the top k are treated (vs treating nobody)
    profit(k)      = incremental(k) * v  -  treated(k) * c        treated(k) = k * 1,000,000

c (cost per treatment) and v (value per visit) are ILLUSTRATIVE ASSUMPTIONS, not Criteo numbers.
Dividing by v shows the optimal k depends only on the ratio r = c / v:
    argmax_k  incremental(k) - k * 1e6 * r
so the sensitivity analysis is a sweep over r, and the app only needs to look up r = c / v.

Regimes: treat everyone when r is below the payoff of the least-promising users, treat nobody when
r exceeds the best achievable uplift rate in the very top slice, target a fraction in between.
"""
from __future__ import annotations

import numpy as np

from .common import FIGURES_DIR
from .evaluate import boot_path
from .plotting import NEUTRAL, SERIES, apply_style, plt

PER = 1_000_000
ASSUMPTIONS = {
    "label": "ILLUSTRATIVE cost/value assumptions - not Criteo figures",
    "value_per_visit_range_usd": [0.10, 2.00],
    "cost_to_value_ratio_range": [0.0, 0.2],
    "default_value_per_visit_usd": 1.00,
    "default_cost_per_treatment_usd": 0.005,
}
RATIO_TABLE = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2]
RATIO_GRID = np.r_[0.0, np.geomspace(1e-5, 0.3, 240)]
K_REPORT = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]


def load_curves(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(k grid incl. 0, point gain per user, bootstrap gains per user [reps, k])."""
    z = np.load(boot_path("full"))
    k = np.r_[0.0, z["grid"]]
    point = np.r_[0.0, z[f"point_{model}"]]
    boot = np.c_[np.zeros(len(z[f"gain_{model}"])), z[f"gain_{model}"]]
    return k, point, boot


def ci(a, axis=0):
    lo, hi = np.percentile(a, [2.5, 97.5], axis=axis)
    return lo, hi


def optimal_k(gain: np.ndarray, k: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """argmax_k gain(k) - k * r for each r. gain: (..., K) -> result (..., R)."""
    obj = gain[..., None, :] - ratios[:, None] * k[None, :]  # (..., R, K)
    return k[np.argmax(obj, axis=-1)]


def regime(k_opt: float) -> str:
    """k* >= 95% is reported as 'treat (almost) all': the last few % of the curve is flat within noise,
    so the argmax wanders between 90% and 100% (the treat-all threshold's CI straddles 0)."""
    if k_opt == 0:
        return "treat none"
    return "treat (almost) all" if k_opt >= 0.95 else "target top k"


def economics(model: str) -> tuple[dict, dict]:
    k, point, boot = load_curves(model)
    ate, ate_boot = point[-1], boot[:, -1]

    # --- incremental visits by k, vs nobody and vs random targeting of the same k
    by_k = {}
    for kk in K_REPORT:
        j = int(np.argmin(np.abs(k - kk)))
        vs_random = boot[:, j] - k[j] * ate_boot
        by_k[f"{int(round(kk * 100))}"] = {
            "treated_per_1m": kk * PER,
            "incremental_per_1m": point[j] * PER, "incremental_per_1m_ci": [v * PER for v in ci(boot[:, j])],
            "vs_random_per_1m": (point[j] - k[j] * ate) * PER, "vs_random_per_1m_ci": [v * PER for v in ci(vs_random)],
            "share_of_all_incremental": point[j] / ate,
            "incremental_per_treated": point[j] / kk,
        }

    # --- optimal k across cost/value ratios, with bootstrap intervals
    k_point = optimal_k(point, k, RATIO_GRID)
    k_boot = optimal_k(boot, k, RATIO_GRID)  # (reps, R)
    k_lo, k_hi = ci(k_boot)
    profit_opt = point[np.searchsorted(k, k_point)] - k_point * RATIO_GRID  # per user, in units of v
    idx = np.searchsorted(k, k_point)
    profit_opt_boot = boot[:, idx] - k_point[None, :] * RATIO_GRID[None, :]  # profit of the chosen k, resampled
    p_lo, p_hi = ci(profit_opt_boot)

    table = []
    for r in RATIO_TABLE:
        i = int(np.argmin(np.abs(RATIO_GRID - r)))
        j_all = len(k) - 1
        table.append({
            "cost_to_value": r,
            "optimal_k": float(k_point[i]), "optimal_k_ci": [float(k_lo[i]), float(k_hi[i])],
            "regime": regime(k_point[i]),
            "profit_per_1m_at_optimum_in_v": float(profit_opt[i] * PER),
            "profit_per_1m_at_optimum_in_v_ci": [float(p_lo[i] * PER), float(p_hi[i] * PER)],
            "profit_per_1m_treat_all_in_v": float((point[j_all] - r) * PER),
            "profit_per_1m_treat_none_in_v": 0.0,
        })

    # --- break-even ratios (each with bootstrap CI)
    def treat_all_threshold(g):  # largest r at which treating everyone is still optimal
        return np.min((g[..., -1:] - g[..., :-1]) / (1 - k[None, :-1] if g.ndim > 1 else 1 - k[:-1]), axis=-1)

    def treat_none_threshold(g):  # smallest r at which treating nobody is optimal = best average uplift of a top slice
        return np.max(g[..., 1:] / k[1:], axis=-1)

    breakeven = {
        "blanket_campaign_breakeven_c_over_v": float(ate),
        "blanket_campaign_breakeven_ci": [float(v) for v in ci(ate_boot)],
        "treat_all_optimal_below_c_over_v": float(treat_all_threshold(point)),
        "treat_all_optimal_below_ci": [float(v) for v in ci(treat_all_threshold(boot))],
        "treat_none_optimal_above_c_over_v": float(treat_none_threshold(point)),
        "treat_none_optimal_above_ci": [float(v) for v in ci(treat_none_threshold(boot))],
        "note": "blanket = treat everyone; profitable only while c/v < overall uplift per user",
    }

    d = ASSUMPTIONS
    r0 = d["default_cost_per_treatment_usd"] / d["default_value_per_visit_usd"]
    i0 = int(np.argmin(np.abs(RATIO_GRID - r0)))
    default = {
        "cost_per_treatment_usd": d["default_cost_per_treatment_usd"],
        "value_per_visit_usd": d["default_value_per_visit_usd"], "cost_to_value": r0,
        "optimal_k": float(k_point[i0]), "optimal_k_ci": [float(k_lo[i0]), float(k_hi[i0])],
        "profit_per_1m_usd": float(profit_opt[i0] * PER * d["default_value_per_visit_usd"]),
        "profit_per_1m_usd_ci": [float(v * PER * d["default_value_per_visit_usd"]) for v in (p_lo[i0], p_hi[i0])],
        "profit_treat_all_per_1m_usd": float((ate - r0) * PER * d["default_value_per_visit_usd"]),
    }

    result = {
        "model": model, "assumptions": ASSUMPTIONS, "units": "per 1,000,000 users; profit in units of v unless *_usd",
        "incremental_by_k": by_k,
        "incremental_per_1m_at_30pct": by_k["30"]["incremental_per_1m"],
        "incremental_per_1m_at_30pct_ci": by_k["30"]["incremental_per_1m_ci"],
        "sensitivity_table": table, "breakeven": breakeven, "default_scenario": default,
        "optimal_k": default["optimal_k"], "optimal_k_ci": default["optimal_k_ci"],
    }
    curves = {
        "model": model, "k": k.round(4).tolist(),
        "incremental_per_1m": (point * PER).round(1).tolist(),
        "incremental_per_1m_lo": (ci(boot)[0] * PER).round(1).tolist(),
        "incremental_per_1m_hi": (ci(boot)[1] * PER).round(1).tolist(),
        "random_per_1m": (k * ate * PER).round(1).tolist(),
        "ratio_grid": RATIO_GRID.round(7).tolist(),
        "optimal_k": k_point.round(4).tolist(), "optimal_k_lo": k_lo.round(4).tolist(),
        "optimal_k_hi": k_hi.round(4).tolist(),
        "assumptions": ASSUMPTIONS,
    }
    return result, curves


def compare_models(models: list[str], ratios=(0.001, 0.005, 0.02, 0.05)) -> dict:
    """Profit at each model's own optimal k (per 1M users, in units of v) - does the ranking matter for money?"""
    out = {}
    for m in models:
        k, point, boot = load_curves(m)
        rs = np.array(ratios)
        kp = optimal_k(point, k, rs)
        idx = np.searchsorted(k, kp)
        prof = point[idx] - kp * rs
        prof_boot = boot[:, idx] - kp[None, :] * rs[None, :]
        lo, hi = ci(prof_boot)
        out[m] = {str(r): {"optimal_k": float(kp[i]), "profit_per_1m_in_v": float(prof[i] * PER),
                           "profit_ci": [float(lo[i] * PER), float(hi[i] * PER)]} for i, r in enumerate(ratios)}
    return out


# ----------------------------------------------------------------------------- figures
def plot_policy(res: dict, curves: dict, path=None) -> None:
    """Left: profit vs share treated for a few cost levels (v = $1). Right: best share vs cost/value."""
    apply_style()
    path = path or FIGURES_DIR / "policy.png"
    kk = np.array(curves["k"])
    x = kk * 100
    inc = np.array(curves["incremental_per_1m"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))

    top = 0.0
    for i, c in enumerate([0.001, 0.005, 0.02, 0.05]):
        prof = inc - kk * PER * c  # value per visit = $1
        j = int(np.argmax(prof))
        top = max(top, prof[j])
        ax1.plot(x, prof, color=SERIES[i], label=f"cost ${c:g} per user (best: treat {x[j]:.1f}%)")
        ax1.plot(x[j], prof[j], "o", color=SERIES[i])
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylim(-0.45 * top, 1.2 * top)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of users treated (top by X-learner uplift)")
    ax1.set_ylabel("Profit per 1M users ($)")
    ax1.set_title("Profit vs share treated (assumes $1 per visit)")
    ax1.legend(fontsize=8, loc="lower right")

    rg = np.array(curves["ratio_grid"])[1:]
    ax2.fill_between(rg, np.array(curves["optimal_k_lo"])[1:] * 100, np.array(curves["optimal_k_hi"])[1:] * 100,
                     color=SERIES[0], alpha=0.2, lw=0, step="mid", label="95% bootstrap CI")
    ax2.step(rg, np.array(curves["optimal_k"])[1:] * 100, where="mid", color=SERIES[0], label="best share")
    b = res["breakeven"]
    ax2.axvline(b["blanket_campaign_breakeven_c_over_v"], color=NEUTRAL, ls="--", lw=1)
    ax2.axvline(b["treat_none_optimal_above_c_over_v"], color=NEUTRAL, ls=":", lw=1)
    ax2.text(b["blanket_campaign_breakeven_c_over_v"] * 1.1, 75, "treating everyone\nstops paying", fontsize=8)
    ax2.text(b["treat_none_optimal_above_c_over_v"] * 0.9, 45, "treat\nnobody", fontsize=8, ha="right")
    ax2.set_xscale("log")
    ax2.set_ylim(-2, 102)
    ax2.set_xlabel("cost per treatment / value per visit")
    ax2.set_ylabel("Profit-maximizing % treated")
    ax2.set_title("Best share to treat vs cost/value ratio")
    ax2.legend(fontsize=8, loc="upper right")

    fig.text(0.01, -0.03, "Costs and values are illustrative assumptions.", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(model: str) -> tuple[dict, dict]:
    res, curves = economics(model)
    res["model_comparison_profit"] = compare_models([model, "response", "t_learner", "s_learner"])
    plot_policy(res, curves)
    return res, curves
