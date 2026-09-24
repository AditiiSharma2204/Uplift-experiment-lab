"""Power and minimum detectable effect (MDE) for the visit and conversion ATE.

Analytical: two-sided two-proportion z-test at alpha = 0.05, 80% power, 85/15 allocation.
Simulation: draw random subsamples of size n from the full data and record how often the
effect is significant. A simple random subsample only changes the counts in the
(treatment x outcome) cells, so we draw those counts exactly from a multivariate
hypergeometric distribution - statistically identical to sampling rows, but instant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from .common import FIGURES_DIR, OUTCOMES, SEED, TREATMENT
from .plotting import NEUTRAL, SERIES, apply_style, plt

ALPHA = 0.05
POWER = 0.80
SIM_REPS = 200
Z_A = stats.norm.ppf(1 - ALPHA / 2)


def default_sizes(n_full: int) -> list[int]:
    return [10_000, 20_000, 50_000, 100_000, 200_000, 1_000_000, 5_000_000, n_full]


def analytical_power(n: float, p_c: float, effect: float, treat_share: float) -> float:
    """P(two-sided pooled z-test rejects AND the estimate has the right sign).

    Uses the pooled SE under H0 for the critical value and the unpooled SE under H1, i.e. exactly
    the test we run. With an 85/15 split the pooled rate sits near the treated rate, so the
    null SE is larger than the naive baseline-variance formula assumes - ignoring that overstates
    power. The wrong-sign rejection tail is negligible and is excluded, matching the simulation.
    """
    p_t = p_c + effect
    n_t, n_c = n * treat_share, n * (1 - treat_share)
    p_bar = treat_share * p_t + (1 - treat_share) * p_c
    se0 = np.sqrt(p_bar * (1 - p_bar) * (1 / n_t + 1 / n_c))
    se1 = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    return float(stats.norm.sf((Z_A * se0 - abs(effect)) / se1))


def mde_abs(n: float, p_c: float, treat_share: float) -> float:
    """Smallest positive absolute effect detected with 80% power by the test above (solved numerically)."""
    return float(brentq(lambda d: analytical_power(n, p_c, d, treat_share) - POWER, 1e-9, 1 - p_c - 1e-9))


def n_for_power(p_c: float, effect: float, treat_share: float) -> int:
    """Total sample size giving 80% power at `effect`."""
    return int(np.ceil(brentq(lambda n: analytical_power(n, p_c, effect, treat_share) - POWER, 10, 1e10)))


def _z_pvalues(x_t, n_t, x_c, n_c) -> np.ndarray:
    p_t, p_c = x_t / n_t, x_c / n_c
    p_pool = (x_t + x_c) / (n_t + n_c)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p_t - p_c) / se, 0.0)
    return 2 * stats.norm.sf(np.abs(z))


def simulate_power(cell_counts: np.ndarray, n: int, reps: int, rng: np.random.Generator,
                   replace: bool) -> dict:
    """cell_counts: shape (4,) = [treated & y=1, treated & y=0, control & y=1, control & y=0]."""
    if replace:  # bootstrap at the full size (sampling without replacement would return the same data)
        draws = rng.multinomial(n, cell_counts / cell_counts.sum(), size=reps)
    else:
        draws = rng.multivariate_hypergeometric(cell_counts, n, size=reps)
    x_t, n_t = draws[:, 0], draws[:, 0] + draws[:, 1]
    x_c, n_c = draws[:, 2], draws[:, 2] + draws[:, 3]
    sig = (_z_pvalues(x_t, n_t, x_c, n_c) < ALPHA) & (x_t / n_t > x_c / n_c)
    share = sig.mean()
    return {"detected_share": float(share),
            "detected_share_mc_se": float(np.sqrt(share * (1 - share) / reps)),
            "sampling": "bootstrap (with replacement)" if replace else "subsample without replacement"}


def run(df: pd.DataFrame, sizes: list[int] | None = None, seed: int = SEED) -> dict:
    n_full = len(df)
    sizes = sizes or default_sizes(n_full)
    t = df[TREATMENT].to_numpy().astype(bool)
    treat_share = float(t.mean())
    rng = np.random.default_rng(seed)
    out: dict = {"alpha": ALPHA, "power": POWER, "sim_reps": SIM_REPS, "seed": seed,
                 "treat_share": treat_share, "assumed_true_effect": "full-data ATE from stage S1"}
    for o in OUTCOMES:
        y = df[o].to_numpy().astype(bool)
        cells = np.array([(t & y).sum(), (t & ~y).sum(), (~t & y).sum(), (~t & ~y).sum()], dtype=np.int64)
        p_c = cells[2] / (cells[2] + cells[3])
        p_t = cells[0] / (cells[0] + cells[1])
        effect = p_t - p_c
        rows = []
        for n in sizes:
            mde = mde_abs(n, p_c, treat_share)
            sim = simulate_power(cells, n, SIM_REPS, rng, replace=(n >= n_full))
            rows.append({
                "n": n, "mde_abs": mde, "mde_rel": mde / p_c,
                "expected_events_control": n * (1 - treat_share) * p_c,
                "analytical_power_at_true_effect": analytical_power(n, p_c, effect, treat_share),
                "simulated_detection_share": sim["detected_share"],
                "simulated_mc_se": sim["detected_share_mc_se"],
                "simulation_sampling": sim["sampling"],
            })
        out[o] = {"baseline_rate_control": p_c, "true_effect_abs": effect,
                  "n_for_80pct_power_at_true_effect": n_for_power(p_c, effect, treat_share), "table": rows}
    plot_power(out)
    plot_mde(out)
    return out


def plot_power(res: dict, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / "power_curves.png"
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(10, 3.8), sharey=True)
    for ax, o, color in zip(axes, OUTCOMES, SERIES):
        tab = pd.DataFrame(res[o]["table"])
        grid = np.geomspace(tab["n"].min(), tab["n"].max(), 200)
        p_c, eff = res[o]["baseline_rate_control"], res[o]["true_effect_abs"]
        ax.plot(grid, [analytical_power(n, p_c, eff, res["treat_share"]) for n in grid],
                color=color, label="Analytical")
        ax.errorbar(tab["n"], tab["simulated_detection_share"], yerr=1.96 * tab["simulated_mc_se"],
                    fmt="o", color=color, markerfacecolor="white", markeredgewidth=2, markersize=7,
                    capsize=3, lw=1, label=f"Simulated ({res['sim_reps']} subsamples)")
        ax.axhline(POWER, color=NEUTRAL, ls="--", lw=1)
        ax.text(grid[0], POWER + 0.02, "80% power", color=NEUTRAL, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Total sample size (log scale)")
        ax.set_title(f"{o}: effect {100 * eff:+.3f} pp")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Probability effect is detected")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_mde(res: dict, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / "mde_relative.png"
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for o, color in zip(OUTCOMES, SERIES):
        tab = pd.DataFrame(res[o]["table"])
        ax.plot(tab["n"], 100 * tab["mde_rel"], "o-", color=color, markeredgecolor="white",
                markeredgewidth=1.5, markersize=7, label=o)
        ax.axhline(100 * res[o]["true_effect_abs"] / res[o]["baseline_rate_control"], color=color,
                   ls=":", lw=1.2)
        last = tab.iloc[0]
        ax.annotate(o, (last["n"], 100 * last["mde_rel"]), xytext=(6, 0), textcoords="offset points",
                    color="#52514e", fontsize=9, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total sample size (log scale)")
    ax.set_ylabel("MDE, relative lift (%)")
    ax.set_title("Minimum detectable relative lift at 80% power (dotted: observed lift)")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)
