"""Experiment integrity checks: SRM, covariate balance, treatment predictability, missing, duplicates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import FEATURES, FIGURES_DIR, OUTCOMES, RESULTS_DIR, SEED, TREATMENT, assert_no_leakage
from .plotting import NEUTRAL, SERIES, apply_style, plt

SRM_ALPHA = 0.001  # conventional SRM threshold: strict, because a real SRM invalidates everything
SMD_THRESHOLD = 0.1  # common "negligible imbalance" rule of thumb


def srm_test(treatment: pd.Series, expected_treat_share: float) -> dict:
    """Chi-square goodness-of-fit of observed arm counts against the design ratio."""
    n = len(treatment)
    n_t = int(treatment.sum())
    n_c = n - n_t
    expected = np.array([n * expected_treat_share, n * (1 - expected_treat_share)])
    chi2, p = stats.chisquare([n_t, n_c], f_exp=expected)
    share = n_t / n
    se = np.sqrt(share * (1 - share) / n)
    return {
        "n": n, "n_treated": n_t, "n_control": n_c,
        "observed_treat_share": share,
        "observed_treat_share_ci95": [share - 1.96 * se, share + 1.96 * se],
        "expected_treat_share": expected_treat_share,
        "expected_ratio_source": "dataset card (Criteo AI Lab, Hugging Face): 'Treatment Ratio: 0.85' - a documented "
                                 "summary of a pooled, subsampled dataset, not a per-test design spec",
        "chi2": float(chi2), "p_value": float(p), "alpha": SRM_ALPHA,
        "srm_detected": bool(p < SRM_ALPHA),
    }


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference and variance ratio per feature (treated vs control)."""
    t = df[TREATMENT].to_numpy() == 1
    rows = []
    for f in FEATURES:
        x = df[f].to_numpy(dtype=np.float64)
        xt, xc = x[t], x[~t]
        mt, mc, vt, vc = xt.mean(), xc.mean(), xt.var(ddof=1), xc.var(ddof=1)
        pooled_sd = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_sd if pooled_sd > 0 else 0.0
        ks = stats.ks_2samp(xt, xc, method="asymp")
        rows.append({
            "feature": f, "mean_treated": mt, "mean_control": mc,
            "sd_treated": np.sqrt(vt), "sd_control": np.sqrt(vc),
            "smd": smd, "variance_ratio": vt / vc if vc > 0 else np.nan,
            "ks_stat": ks.statistic, "n_unique": int(pd.Series(x).nunique()),
        })
    return pd.DataFrame(rows)


def treatment_predictability(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Classifier two-sample test: can f0-f11 predict assignment? Under randomization AUC ~ 0.5."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    assert_no_leakage(FEATURES)
    X, y = df[FEATURES], df[TREATMENT].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               min_child_samples=200, random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    # Hanley-McNeil standard error of AUC
    n1, n0 = int(y_te.sum()), int(len(y_te) - y_te.sum())
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return {"auc": auc, "auc_ci95": [auc - 1.96 * se, auc + 1.96 * se],
            "n_train": len(y_tr), "n_test": len(y_te),
            "z_vs_0_5": (auc - 0.5) / se}


def missing_and_duplicates(df: pd.DataFrame) -> dict:
    missing = {c: int(v) for c, v in df.isna().sum().items()}
    dup_all = int(df.duplicated().sum())
    dup_feat = int(df.duplicated(subset=FEATURES).sum())
    return {
        "missing_by_column": missing, "total_missing": int(sum(missing.values())),
        "duplicate_rows_all_columns": dup_all,
        "duplicate_rows_all_columns_pct": 100 * dup_all / len(df),
        "duplicate_rows_features_only": dup_feat,
        "duplicate_rows_features_only_pct": 100 * dup_feat / len(df),
    }


def outcome_shares(df: pd.DataFrame) -> dict:
    out = {"n_rows": len(df), "treatment_share": float(df[TREATMENT].mean())}
    for c in OUTCOMES + ["exposure"]:
        out[f"{c}_rate"] = float(df[c].mean())
        out[f"{c}_count"] = int(df[c].sum())
    out["exposure_rate_treated"] = float(df.loc[df[TREATMENT] == 1, "exposure"].mean())
    out["exposure_rate_control"] = float(df.loc[df[TREATMENT] == 0, "exposure"].mean())
    return out


def plot_balance(bal: pd.DataFrame, path=None) -> None:
    apply_style()
    path = path or FIGURES_DIR / "balance_smd.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    b = bal.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axvline(0, color="black", lw=0.8)
    for x in (-SMD_THRESHOLD, SMD_THRESHOLD):
        ax.axvline(x, color=NEUTRAL, lw=1, ls="--")
    ax.plot(b["smd"], b["feature"], "o", color=SERIES[0])
    lim = max(0.12, float(b["smd"].abs().max()) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardized mean difference (treated - control)")
    ax.set_title(f"Standardized mean difference, treated vs control (dashed: +/-{SMD_THRESHOLD})")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def run(df: pd.DataFrame, expected_treat_share: float = 0.85) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bal = balance_table(df)
    bal.to_csv(RESULTS_DIR / "balance_table.csv", index=False, float_format="%.6g")
    plot_balance(bal)
    return {
        "srm": srm_test(df[TREATMENT], expected_treat_share),
        "balance": {
            "max_abs_smd": float(bal["smd"].abs().max()),
            "features_over_threshold": bal.loc[bal["smd"].abs() > SMD_THRESHOLD, "feature"].tolist(),
            "threshold": SMD_THRESHOLD,
            "table": bal[["feature", "mean_treated", "mean_control", "smd", "variance_ratio", "ks_stat"]]
                .round(6).to_dict(orient="records"),
        },
        "data_quality": missing_and_duplicates(df),
    }
