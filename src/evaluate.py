"""Honest evaluation of uplift rankings on the untouched test set.

Curve. Rank users by score, take the top fraction k, and estimate
    uplift@k = P(y | t=1, top k) - P(y | t=0, top k)
    gain(k)  = (n_k / N) * uplift@k      -> incremental outcomes per user of the whole population
                                            if exactly the top k were treated (and nobody else).
This is the "uplift curve" form of the Qini curve: the same ranking signal, but in units the policy
stage can use directly (x 1e6 = incremental visits per 1M users). Random targeting is the straight
line k * gain(1).  AUUC = area between the curve and that line; > 0 means better than random.

Propensity correction (D-16, D-22). The pooled data is randomized only conditional on x, so within a
top-k group the treated and control users can differ. Each arm's rate is an inverse-propensity weighted
(Hajek) mean: treated weight 1/e(x), control weight 1/(1 - e(x)). With e constant this reduces to plain
means, which we also report ("unweighted").

Uncertainty. 200 Poisson(1) bootstrap replicates of the test rows (each row gets a random count; at
n = 4.2M this is equivalent to the multinomial bootstrap and lets us sort each model once). All models
share each replicate's weights, so model-vs-model differences get paired intervals.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .common import FIGURES_DIR, OUTCOMES, PRIMARY_OUTCOME, RESULTS_DIR, SEED, TREATMENT
from .plotting import NEUTRAL, SERIES, TEXT_SECONDARY, apply_style, plt
from .uplift_models import LEARNERS, scores_path

GRID = np.round(np.arange(0.005, 1.0 + 1e-9, 0.005), 3)  # targeting fractions 0.5% .. 100%
K_REPORT = [0.05, 0.10, 0.20, 0.30, 0.50]
N_BOOT = 200
E_CLIP = (0.01, 0.99)
MODELS = [*LEARNERS, "random"]
LABELS = {"response": "Response model", "t_learner": "T-learner", "s_learner": "S-learner",
          "x_learner": "X-learner", "random": "Random score"}
COLORS = {"response": SERIES[1], "t_learner": SERIES[2], "s_learner": SERIES[3], "x_learner": SERIES[0],
          "random": NEUTRAL}
BOOT_PATH = RESULTS_DIR / "scores" / "eval_bootstrap.npz"


# ----------------------------------------------------------------------------- data
def load_test(tag: str = "full") -> tuple[pd.DataFrame, list[int]]:
    test = pd.read_parquet(scores_path(tag))
    seeds = sorted(int(c.rsplit("_s", 1)[1]) for c in test.columns if c.startswith("x_learner_s"))
    return test, seeds


def model_scores(test: pd.DataFrame, seeds: list[int], seed: int = SEED) -> dict[str, np.ndarray]:
    """Seed-averaged score per learner, plus a random score as a pipeline sanity check."""
    out = {n: test[[f"{n}_s{s}" for s in seeds]].to_numpy(dtype=np.float64).mean(axis=1) for n in LEARNERS}
    out["random"] = np.random.default_rng(seed + 1000).random(len(test))
    return out


def propensity(test: pd.DataFrame, seeds: list[int]) -> np.ndarray:
    e = test[[f"propensity_s{s}" for s in seeds]].to_numpy(dtype=np.float64).mean(axis=1)
    return np.clip(e, *E_CLIP)


# ----------------------------------------------------------------------------- core curve maths
class Ranked:
    """One model's ranking, pre-sorted once; evaluates curves for any row weights."""

    def __init__(self, score: np.ndarray, t: np.ndarray, y: np.ndarray, e: np.ndarray | None,
                 grid: np.ndarray = GRID, seed: int = SEED):
        n = len(score)
        # random tie-break so ties do not follow file order
        tie = np.random.default_rng(seed).random(n)
        self.order = np.lexsort((tie, -score))
        tt = t[self.order].astype(np.float64)
        yy = y[self.order].astype(np.float64)
        if e is None:
            wt, wc = tt, 1 - tt
        else:
            ee = e[self.order]
            wt, wc = tt / ee, (1 - tt) / (1 - ee)
        self.parts = np.stack([wt * yy, wt, wc * yy, wc, np.ones(n)])  # per-row contributions
        self.bounds = np.r_[0, np.ceil(grid * n).astype(np.int64)]
        self.bounds[-1] = n
        self.grid = grid

    def curve(self, b: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (uplift@k, gain(k)) on the grid. `b` = bootstrap counts in ORIGINAL row order."""
        p = self.parts if b is None else self.parts * b[self.order]
        seg = np.add.reduceat(p, self.bounds[:-1], axis=1)
        cum = np.cumsum(seg, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            uplift = cum[0] / cum[1] - cum[2] / cum[3]
        gain = cum[4] / cum[4][-1] * uplift
        return uplift, gain


def auuc(gain: np.ndarray, grid: np.ndarray = GRID) -> float:
    """Area between gain(k) and the random line k * gain(1), over k in [0, 1]."""
    k = np.r_[0.0, grid]
    g = np.r_[0.0, gain]
    return float(np.trapezoid(g - k * gain[-1], k))


def k_index(k: float, grid: np.ndarray = GRID) -> int:
    return int(np.argmin(np.abs(grid - k)))


# ----------------------------------------------------------------------------- evaluation
def evaluate(test: pd.DataFrame, seeds: list[int], outcome: str = PRIMARY_OUTCOME, weighted: bool = True,
             n_boot: int = N_BOOT, seed: int = SEED, keep_replicates: bool = False) -> dict:
    t = test[TREATMENT].to_numpy()
    y = test[outcome].to_numpy()
    e = propensity(test, seeds) if weighted else None
    scores = model_scores(test, seeds)
    ranked = {m: Ranked(s, t, y, e) for m, s in scores.items()}

    point = {m: r.curve() for m, r in ranked.items()}
    rng = np.random.default_rng(seed)
    boot_gain = {m: np.empty((n_boot, len(GRID))) for m in MODELS}
    boot_uplift = {m: np.empty((n_boot, len(GRID))) for m in MODELS}
    for i in range(n_boot):
        b = rng.poisson(1.0, len(t)).astype(np.float64)
        for m, r in ranked.items():
            boot_uplift[m][i], boot_gain[m][i] = r.curve(b)

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    ate_point = float(point["random"][1][-1])  # gain(1) = overall uplift, identical for every model
    ate_boot = boot_gain["random"][:, -1]
    res: dict = {"outcome": outcome, "weighted": weighted, "n_test": len(t), "n_boot": n_boot,
                 "seeds_averaged": seeds, "ate_test": ate_point, "ate_test_ci": ci(ate_boot), "models": {}}
    auuc_boot = {m: np.array([auuc(g) for g in boot_gain[m]]) for m in MODELS}
    for m in MODELS:
        up, gain = point[m]
        a = auuc(gain)
        entry = {"auuc": a, "auuc_ci": ci(auuc_boot[m]), "auuc_se": float(auuc_boot[m].std(ddof=1)),
                 "auuc_per_1m": a * 1e6, "auuc_per_1m_ci": [v * 1e6 for v in ci(auuc_boot[m])],
                 "better_than_random": bool(np.percentile(auuc_boot[m], 2.5) > 0), "uplift_at_k": {}}
        for k in K_REPORT:
            j = k_index(k)
            diff_vs_random = boot_uplift[m][:, j] - ate_boot
            entry["uplift_at_k"][f"{int(k * 100)}"] = {
                "uplift": float(up[j]), "uplift_ci": ci(boot_uplift[m][:, j]),
                "incremental_per_1m": float(gain[j] * 1e6), "incremental_per_1m_ci": [v * 1e6 for v in
                                                                                      ci(boot_gain[m][:, j])],
                "uplift_minus_random": float(up[j] - ate_point), "uplift_minus_random_ci": ci(diff_vs_random),
            }
        res["models"][m] = entry

    # paired model-vs-model AUUC differences; conservative Bonferroni across the learner pairs
    pairs = list(itertools.combinations(LEARNERS, 2))
    z_bonf = stats.norm.ppf(1 - 0.05 / (2 * len(pairs)))
    res["pairwise_auuc"] = {}
    for a_, b_ in pairs:
        d = auuc_boot[a_] - auuc_boot[b_]
        est = res["models"][a_]["auuc"] - res["models"][b_]["auuc"]
        se = float(d.std(ddof=1))
        lo, hi = est - z_bonf * se, est + z_bonf * se
        res["pairwise_auuc"][f"{a_}_vs_{b_}"] = {
            "diff": est, "ci95_percentile": ci(d), "ci_bonferroni": [lo, hi],
            "distinguishable_bonferroni": bool(lo > 0 or hi < 0),
        }
    if keep_replicates:
        res["_replicates"] = {"gain": boot_gain, "uplift": boot_uplift, "point": point}
    return res


def top_k_profile(test: pd.DataFrame, seeds: list[int], k: float = 0.10) -> dict:
    """Who does each model put in its top k? Baseline (control) visit rate vs incremental share."""
    t = test[TREATMENT].to_numpy()
    y = test[PRIMARY_OUTCOME].to_numpy()
    e = propensity(test, seeds)
    out = {}
    for m, s in model_scores(test, seeds).items():
        top = s >= np.quantile(s, 1 - k)
        tt, yy, ee = t[top], y[top], e[top]
        pt = np.sum(tt * yy / ee) / np.sum(tt / ee)
        pc = np.sum((1 - tt) * yy / (1 - ee)) / np.sum((1 - tt) / (1 - ee))
        out[m] = {"rate_treated": float(pt), "rate_control": float(pc), "uplift": float(pt - pc),
                  "share_of_treated_visits_incremental": float((pt - pc) / pt) if pt > 0 else None}
    return out


def per_seed_auuc(test: pd.DataFrame, seeds: list[int]) -> dict:
    t, y = test[TREATMENT].to_numpy(), test[PRIMARY_OUTCOME].to_numpy()
    e = propensity(test, seeds)
    return {n: {str(s): auuc(Ranked(test[f"{n}_s{s}"].to_numpy(dtype=np.float64), t, y, e).curve()[1])
                for s in seeds} for n in LEARNERS}


# ----------------------------------------------------------------------------- figures
def plot_curves(res: dict, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / "qini_curves.png"
    reps = res["_replicates"]
    k = np.r_[0, GRID] * 100
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, relative in zip(axes, [False, True]):
        for m in MODELS:
            g = reps["point"][m][1] * 1e6
            bg = reps["gain"][m] * 1e6
            if relative:  # distance above the random line, per replicate
                g = g - GRID * g[-1]
                bg = bg - GRID[None, :] * bg[:, [-1]]
            lo, hi = np.percentile(bg, [2.5, 97.5], axis=0)
            style = dict(color=COLORS[m], lw=2 if m != "random" else 1.2, ls="-" if m != "random" else ":")
            ax.plot(k, np.r_[0, g], label=LABELS[m], **style)
            ax.fill_between(k, np.r_[0, lo], np.r_[0, hi], color=COLORS[m], alpha=0.15, lw=0)
        if not relative:
            ate = reps["point"]["random"][1][-1] * 1e6
            ax.plot([0, 100], [0, ate], color=NEUTRAL, lw=1, ls="--")
            ax.text(62, ate * 0.55, "random targeting", color=TEXT_SECONDARY, fontsize=8, rotation=0)
            ax.set_ylabel("Incremental visits per 1M users")
            ax.set_title("Qini (uplift) curves, 95% bootstrap bands")
        else:
            ax.axhline(0, color=NEUTRAL, lw=1, ls="--")
            ax.set_ylabel("Incremental visits above random, per 1M")
            ax.set_title("Same curves minus the random line")
        ax.set_xlabel("Users targeted, ranked by model score (%)")
        ax.set_xlim(0, 100)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_auuc(res: dict, res_unw: dict, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / "auuc_forest.png"
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ms = MODELS[::-1]
    for i, m in enumerate(ms):
        for dy, r, filled in [(0.12, res, True), (-0.12, res_unw, False)]:
            a = r["models"][m]
            lo, hi = a["auuc_per_1m_ci"]
            ax.hlines(i + dy, lo, hi, color=COLORS[m], lw=2)
            ax.plot(a["auuc_per_1m"], i + dy, "o", color=COLORS[m], markersize=8,
                    markerfacecolor=COLORS[m] if filled else "white", markeredgewidth=2)
    ax.axvline(0, color=NEUTRAL, lw=1, ls="--")
    ax.set_yticks(range(len(ms)), [LABELS[m] for m in ms])
    ax.set_xlabel("AUUC: area above random, incremental visits per 1M users")
    ax.set_title("AUUC with 95% CI (filled: propensity-weighted, hollow: unweighted)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- stage runner
def curves_payload(res: dict) -> dict:
    """Compact curves for the app / policy stage (per 1M users)."""
    reps = res["_replicates"]
    out = {"k": GRID.tolist(), "units": "incremental visits per 1M users if the top k are treated",
           "weighted": res["weighted"], "models": {}}
    for m in MODELS:
        g = reps["gain"][m] * 1e6
        lo, hi = np.percentile(g, [2.5, 97.5], axis=0)
        out["models"][m] = {"gain": (reps["point"][m][1] * 1e6).round(2).tolist(),
                            "gain_lo": lo.round(2).tolist(), "gain_hi": hi.round(2).tolist()}
    return out


def run(tag: str = "full") -> tuple[dict, dict]:
    test, seeds = load_test(tag)
    res = evaluate(test, seeds, weighted=True, keep_replicates=True)
    res_unw = evaluate(test, seeds, weighted=False)
    res_conv = evaluate(test, seeds, outcome="conversion", weighted=True)
    plot_curves(res)
    plot_auuc(res, res_unw)
    reps = res.pop("_replicates")
    BOOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BOOT_PATH, grid=GRID, **{f"gain_{m}": reps["gain"][m] for m in MODELS},
                        **{f"point_{m}": reps["point"][m][1] for m in MODELS})
    res["_replicates"] = reps
    curves = curves_payload(res)
    res.pop("_replicates")

    learners = {m: res["models"][m] for m in LEARNERS}
    best = max(learners, key=lambda m: learners[m]["auuc"])
    out = {
        "primary": res,
        "unweighted": {m: {k: v for k, v in r.items() if k in ("auuc", "auuc_ci", "better_than_random")}
                       for m, r in res_unw["models"].items()},
        "unweighted_pairwise": res_unw["pairwise_auuc"],
        "conversion": {m: {k: v for k, v in r.items() if k in ("auuc", "auuc_ci", "better_than_random")}
                       for m, r in res_conv["models"].items()},
        "top10_profile": top_k_profile(test, seeds, 0.10),
        "per_seed_auuc": per_seed_auuc(test, seeds),
        "best_model": best,
    }
    return out, curves
