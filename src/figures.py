"""Redraw every figure from saved results (results.json, balance table, bootstrap replicates).

No data loading or model fitting: useful after a style change.  python -c "from src import figures; figures.run()"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import adjust, ate, evaluate, integrity, policy, power
from .common import OUTCOMES, RESULTS_DIR, load_results


def run() -> None:
    r = load_results()
    integrity.plot_balance(pd.read_csv(RESULTS_DIR / "balance_table.csv"))
    ate.plot_rates({o: r["ate"][o] for o in OUTCOMES})
    power.plot_power(r["power"])
    power.plot_mde(r["power"])
    for o in OUTCOMES:
        adjust.plot_strata(r["adjust"][o]["poststratified"], o)

    ev = r["evaluation"]
    z = np.load(evaluate.boot_path("full"))
    reps = {"point": {m: (None, z[f"point_{m}"]) for m in evaluate.MODELS},
            "gain": {m: z[f"gain_{m}"] for m in evaluate.MODELS}}
    evaluate.plot_curves({**ev["primary"], "_replicates": reps})
    evaluate.plot_auuc(ev["primary"], {"models": ev["unweighted"]})

    res, curves = policy.economics(r["policy"]["model"])
    policy.plot_policy(res, curves)
