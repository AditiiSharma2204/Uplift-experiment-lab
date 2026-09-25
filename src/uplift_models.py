"""Uplift learners written out by hand on top of LightGBM.

Every learner exposes  fit(X, t, y, X_val, t_val, y_val)  and  predict(X) -> score per user.
For the uplift learners the score is the estimated uplift tau(x) = P(y=1 | x, t=1) - P(y=1 | x, t=0).
For the response model it is P(y=1 | x) - a ranking by "likely to visit anyway", NOT by effect.

  ResponseModel  one classifier on everyone, treatment ignored.         -> the baseline that should fail
  TLearner       mu1 on treated, mu0 on control; tau = mu1 - mu0.
  SLearner       one classifier with t as a feature; tau = f(x,1) - f(x,0).
  XLearner       T-learner first stage; impute individual effects in each arm with the *other*
                 arm's model; regress them on x; blend the two effect models with the propensity.

Early stopping uses the validation split only. The test split is never seen here except for
the final `predict` whose output is written to disk for stage S4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import FEATURES, OUTCOMES, PROCESSED_DIR, SEED, TREATMENT, assert_no_leakage

SPLIT_FRACTIONS = {"train": 0.6, "val": 0.1, "test": 0.3}
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

# Fixed parts of every LightGBM model. Tunable parts come from `tune_base_params`.
BASE = dict(learning_rate=0.1, n_estimators=1000, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, n_jobs=-1)
TUNING_GRID = [
    {"num_leaves": 31, "min_child_samples": 200},
    {"num_leaves": 127, "min_child_samples": 200},
    {"num_leaves": 31, "min_child_samples": 2000},
    {"num_leaves": 127, "min_child_samples": 2000},
]
EARLY_STOP = 50


# ----------------------------------------------------------------------------- split
def split_path(tag: str):
    return PROCESSED_DIR / f"split_{tag}.parquet"


def make_split(df: pd.DataFrame, tag: str, seed: int = SEED) -> np.ndarray:
    """Stratified (treatment x visit x conversion) train/val/test labels, cached so the split never changes."""
    path = split_path(tag)
    if path.exists():
        s = pd.read_parquet(path)["split"].to_numpy()
        assert len(s) == len(df), "cached split does not match the data"
        return s
    rng = np.random.default_rng(seed)
    labels = np.empty(len(df), dtype=np.int8)
    cuts = np.cumsum([SPLIT_FRACTIONS["train"], SPLIT_FRACTIONS["val"]])
    for _, idx in df.groupby([TREATMENT, *OUTCOMES], observed=True).indices.items():
        idx = rng.permutation(idx)
        a, b = (cuts * len(idx)).round().astype(int)
        labels[idx[:a]], labels[idx[a:b]], labels[idx[b:]] = 0, 1, 2
    pd.DataFrame({"split": labels}).to_parquet(path, index=False)
    return labels


# ----------------------------------------------------------------------------- LightGBM helpers
def _fit(kind: str, X, y, Xv, yv, params: dict, seed: int):
    assert_no_leakage(X.columns)
    cls = lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor
    m = cls(**BASE, **params, random_state=seed)
    m.fit(X, y, eval_X=Xv, eval_y=yv, callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return m


def _p1(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1]


def tune_base_params(X, y, Xv, yv, seed: int = SEED, max_rows: int = 2_000_000) -> dict:
    """Light tuning: pick num_leaves / min_child_samples by validation log-loss of an outcome model.

    One shared setting for every base learner keeps the comparison between learners fair.
    """
    from sklearn.metrics import log_loss

    if len(X) > max_rows:
        take = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[take], y[take]
    log = []
    for g in TUNING_GRID:
        t0 = time.time()
        m = _fit("clf", X, y, Xv, yv, g, seed)
        log.append({**g, "val_logloss": float(log_loss(yv, _p1(m, Xv))),
                    "best_iteration": int(m.best_iteration_), "seconds": round(time.time() - t0, 1)})
    best = min(log, key=lambda r: r["val_logloss"])
    return {"chosen": {k: best[k] for k in TUNING_GRID[0]}, "grid": log, "tuned_on_rows": len(X)}


# ----------------------------------------------------------------------------- learners
@dataclass
class Learner:
    params: dict
    seed: int = SEED
    info: dict = field(default_factory=dict)

    def _log(self, name: str, m) -> None:
        self.info[f"{name}_best_iteration"] = int(m.best_iteration_)


class ResponseModel(Learner):
    """P(y | x), ignoring treatment. Ranks 'sure things' (users who would visit anyway) on top."""

    def fit(self, X, t, y, Xv, tv, yv):
        self.m = _fit("clf", X, y, Xv, yv, self.params, self.seed)
        self._log("m", self.m)
        return self

    def predict(self, X):
        return _p1(self.m, X)


class TLearner(Learner):
    """Two separate outcome models; uplift is the difference of their predictions."""

    def fit(self, X, t, y, Xv, tv, yv):
        tr, va = t == 1, tv == 1
        self.mu1 = _fit("clf", X[tr], y[tr], Xv[va], yv[va], self.params, self.seed)
        self.mu0 = _fit("clf", X[~tr], y[~tr], Xv[~va], yv[~va], self.params, self.seed)
        self._log("mu1", self.mu1)
        self._log("mu0", self.mu0)
        return self

    def predict(self, X):
        return _p1(self.mu1, X) - _p1(self.mu0, X)


class SLearner(Learner):
    """One model with treatment as an extra feature; uplift = f(x, t=1) - f(x, t=0)."""

    @staticmethod
    def _with_t(X, t):
        Z = X.copy()
        Z[TREATMENT] = np.asarray(t, dtype=np.float32)
        return Z

    def fit(self, X, t, y, Xv, tv, yv):
        assert_no_leakage(X.columns)  # treatment is added deliberately below, after the guard
        self.m = _fit("clf", self._with_t(X, t), y, self._with_t(Xv, tv), yv, self.params, self.seed)
        self._log("m", self.m)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return _p1(self.m, self._with_t(X, ones)) - _p1(self.m, self._with_t(X, zeros))


class XLearner(Learner):
    """Kunzel et al. (2019).

    Stage 1  mu1, mu0 as in the T-learner (reused if a fitted T-learner is passed in).
    Stage 2  imputed effects:  treated  D1 = y - mu0(x)   control  D0 = mu1(x) - y.
             Each uses the model of the OTHER arm, so no row is predicted by a model trained on it.
             Regress D1 on x (tau1) and D0 on x (tau0).
    Stage 3  tau(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x),  e(x) = P(t=1 | x).
             With e ~ 0.85 most weight goes to tau0, whose imputations come from mu1 - the model fitted
             on the big treated arm - which is why the X-learner suits unbalanced designs.
    """

    def __init__(self, params, seed=SEED, t_learner: TLearner | None = None,
                 propensity: "PropensityModel | None" = None):
        super().__init__(params, seed)
        self.t_learner, self.propensity = t_learner, propensity

    def fit(self, X, t, y, Xv, tv, yv):
        if self.t_learner is None:
            self.t_learner = TLearner(self.params, self.seed).fit(X, t, y, Xv, tv, yv)
        if self.propensity is None:
            self.propensity = PropensityModel(self.params, self.seed).fit(X, t, Xv, tv)
        mu1, mu0 = self.t_learner.mu1, self.t_learner.mu0
        tr, va = t == 1, tv == 1
        d1, d1v = y[tr] - _p1(mu0, X[tr]), yv[va] - _p1(mu0, Xv[va])
        d0, d0v = _p1(mu1, X[~tr]) - y[~tr], _p1(mu1, Xv[~va]) - yv[~va]
        self.tau1 = _fit("reg", X[tr], d1, Xv[va], d1v, self.params, self.seed)
        self.tau0 = _fit("reg", X[~tr], d0, Xv[~va], d0v, self.params, self.seed)
        self._log("tau1", self.tau1)
        self._log("tau0", self.tau0)
        return self

    def predict(self, X):
        e = self.propensity.predict(X)
        return e * self.tau0.predict(X) + (1 - e) * self.tau1.predict(X)


class PropensityModel(Learner):
    """e(x) = P(t=1 | x). Needed because the pooled data is not uniformly randomized (D-16)."""

    def fit(self, X, t, Xv, tv):
        self.m = _fit("clf", X, t, Xv, tv, self.params, self.seed)
        self._log("m", self.m)
        return self

    def predict(self, X):
        return _p1(self.m, X)


LEARNERS = ["response", "t_learner", "s_learner", "x_learner"]


# ----------------------------------------------------------------------------- stage runner
def scores_path(tag: str):
    from .common import RESULTS_DIR
    return RESULTS_DIR / "scores" / f"test_scores_{tag}.parquet"


def tuning_path(tag: str):
    return PROCESSED_DIR / f"tuning_{tag}.json"


def run(df: pd.DataFrame, tag: str, seeds=(42, 43, 44), outcome: str = "visit",
        previous: dict | None = None) -> dict:
    """Fit all learners for each seed; append each seed's test scores to disk as soon as it finishes.

    Seeds already present in the scores file are skipped, so extra seeds can be added in a later run.
    `previous` is the existing results.json section, whose per-seed logs are carried over.
    """
    import json
    import warnings

    from sklift.metrics import qini_auc_score

    # sklift 0.5.1 calls a helper that scikit-learn 1.8 deprecated; harmless, but it floods the log
    warnings.filterwarnings("ignore", message=".*stable_cumsum is deprecated.*", category=FutureWarning)

    assert_no_leakage(FEATURES)
    split = make_split(df, tag)
    X = df[FEATURES]
    t = df[TREATMENT].to_numpy().astype(np.int8)
    y = df[outcome].to_numpy().astype(np.int8)
    tr, va, te = split == 0, split == 1, split == 2
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ttr, tva, tte = t[tr], t[va], t[te]
    ytr, yva = y[tr], y[va]

    tp = tuning_path(tag)
    if tp.exists():
        tuning = json.loads(tp.read_text())
    else:
        t0 = time.time()
        tuning = tune_base_params(Xtr, ytr, Xva, yva)
        tp.write_text(json.dumps(tuning, indent=2))
        print(f"tuned {tuning['chosen']} in {time.time() - t0:.0f}s")
    params = tuning["chosen"]

    path = scores_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        test = pd.read_parquet(path)
        assert np.array_equal(test["row_id"].to_numpy(), np.flatnonzero(te)), "scores file does not match split"
    else:
        test = pd.DataFrame({"row_id": np.flatnonzero(te), TREATMENT: tte,
                             **{o: df.loc[te, o].to_numpy() for o in OUTCOMES}})
    per_seed = {int(k): v for k, v in (previous or {}).get("per_seed", {}).items()}

    for seed in seeds:
        if f"x_learner_s{seed}" in test.columns:
            print(f"seed {seed} already done - skipping")
            continue
        info, val_scores, timings = {}, {}, {}
        s0 = time.time()
        prop = PropensityModel(params, seed).fit(Xtr, ttr, Xva, tva)
        timings["propensity"] = time.time() - s0
        models = {}
        for name, make in [("response", lambda: ResponseModel(params, seed)),
                           ("t_learner", lambda: TLearner(params, seed)),
                           ("s_learner", lambda: SLearner(params, seed)),
                           ("x_learner", lambda: XLearner(params, seed, models["t_learner"], prop))]:
            s0 = time.time()
            models[name] = make().fit(Xtr, ttr, ytr, Xva, tva, yva)
            timings[name] = time.time() - s0
            info[name] = models[name].info
            val_scores[name] = models[name].predict(Xva)
            test[f"{name}_s{seed}"] = models[name].predict(Xte).astype(np.float32)
            print(f"  seed {seed} {name:10s} {timings[name]:6.0f}s  {models[name].info}", flush=True)
        test[f"propensity_s{seed}"] = prop.predict(Xte).astype(np.float32)
        test.to_parquet(path, index=False)  # checkpoint after every seed
        per_seed[seed] = {
            "best_iterations": {**info, "propensity": prop.info},
            "fit_seconds": {k: round(v, 1) for k, v in timings.items()},
            "val_qini_auc": {k: float(qini_auc_score(yva, v, tva)) for k, v in val_scores.items()},
            "val_mean_score": {k: float(np.mean(v)) for k, v in val_scores.items()},
            "val_propensity_auc": _auc(tva, prop.predict(Xva)),
        }

    return {
        "outcome": outcome, "data_tag": tag, "n_rows": len(df),
        "split_sizes": {k: int((split == v).sum()) for k, v in SPLIT_CODES.items()},
        "split": {"fractions": SPLIT_FRACTIONS, "stratified_by": [TREATMENT, *OUTCOMES], "seed": SEED},
        "lightgbm_fixed": BASE, "tuning": tuning, **summarize_scores(test),
        "per_seed": {str(k): v for k, v in sorted(per_seed.items())},
        "scores_file": str(path.relative_to(path.parents[2])),
    }


def summarize_scores(test: pd.DataFrame) -> dict:
    """Test-set summaries that use features/predictions only (no outcomes): rank agreement and seed stability."""
    import itertools

    seeds = sorted(int(c.rsplit("_s", 1)[1]) for c in test.columns if c.startswith("x_learner_s"))
    s = seeds[0]
    rank_corr = test[[f"{n}_s{s}" for n in LEARNERS]].corr(method="spearman")
    rank_corr.index = rank_corr.columns = LEARNERS
    stability = None
    if len(seeds) > 1:
        stability = {n: float(np.mean([test[f"{n}_s{a}"].corr(test[f"{n}_s{b}"], method="spearman")
                                       for a, b in itertools.combinations(seeds, 2)])) for n in LEARNERS}
    return {
        "seeds": seeds,
        "test_rank_correlation_first_seed": rank_corr.round(3).to_dict(),
        "test_seed_stability_spearman": stability,
        "test_mean_predicted_uplift": {n: float(test[[f"{n}_s{x}" for x in seeds]].to_numpy().mean())
                                       for n in LEARNERS if n != "response"},
    }


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))
