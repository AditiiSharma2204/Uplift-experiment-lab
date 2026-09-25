"""Run the pipeline stage by stage.  Usage:  python run_all.py --stage s01  (or: all)"""
from __future__ import annotations

import argparse
import time

import psutil

from src.common import load_results, update_results


def stage_s01(dev_n: int) -> None:
    from src import ate, integrity, load

    proc = psutil.Process()
    t0 = time.time()
    full = load.load_full()
    shares = integrity.outcome_shares(full)
    update_results("data", {
        "source": "Criteo Uplift Prediction Dataset v2.1",
        "n_rows_total": len(full),
        "n_rows_used": len(full),
        "memory_mb_in_pandas": round(load.memory_mb(full), 1),
        "process_rss_mb_after_load": round(proc.memory_info().rss / 2**20, 1),
        "dtypes": {c: str(t) for c, t in full.dtypes.items()},
        **shares,
    })
    print(f"Loaded {len(full):,} rows, {load.memory_mb(full):.0f} MB in pandas ({time.time() - t0:.0f}s)")

    dev = load.build_dev_sample(dev_n)
    update_results("data", {"dev_sample": {
        "n_rows": len(dev), "seed": 42, "stratified_by": ["treatment", "visit", "conversion"],
        **{k: v for k, v in integrity.outcome_shares(dev).items() if k != "n_rows"},
    }})

    integ = integrity.run(full, expected_treat_share=0.85)
    integ["treatment_predictability"] = {**integrity.treatment_predictability(dev), "data": "dev sample"}
    update_results("integrity", integ)
    print("SRM:", {k: integ["srm"][k] for k in ("observed_treat_share", "chi2", "p_value", "srm_detected")})
    print("Max |SMD|:", integ["balance"]["max_abs_smd"])
    print("Treatment predictability AUC:", integ["treatment_predictability"]["auc_ci95"])

    update_results("ate", ate.run(full))
    del full
    for o in ("visit", "conversion"):
        from src.common import load_results
        print(load_results()["ate"][o]["summary"])


def stage_s2() -> None:
    from src import adjust, load, power

    full = load.load_full()
    update_results("data", {"n_rows_used": len(full)})
    pw = power.run(full)
    update_results("power", pw)
    for o in ("visit", "conversion"):
        print(f"\n{o}: n for 80% power at observed effect = {pw[o]['n_for_80pct_power_at_true_effect']:,}")
        for r in pw[o]["table"]:
            print(f"  n={r['n']:>10,}  MDE={100 * r['mde_abs']:.4f}pp ({100 * r['mde_rel']:.1f}% rel)  "
                  f"power={r['analytical_power_at_true_effect']:.3f}  sim={r['simulated_detection_share']:.3f}")
    adj = adjust.run(full)
    update_results("adjust", adj)
    for o in ("visit", "conversion"):
        a = adj[o]
        print(f"\n{o}: unadjusted {a['ate_unadjusted']:.6f} (SE {a['se_unadjusted']:.3g})")
        for k in ("linear", "ml_score"):
            r = a[k]
            print(f"  {k:9s} ATE {r['ate_adjusted']:.6f} (SE {r['se_adjusted']:.3g})  "
                  f"VR {r['variance_reduction_pct']:.2f}%  shift {r['ate_shift_vs_unadjusted_in_se']:+.2f} SE"
                  + (f"  score AUC {r['score_auc']:.3f}" if "score_auc" in r else ""))
    set_headline_ate(adj)


def set_headline_ate(adj: dict) -> None:
    """Headline ATE = ML-score Lin estimate (decision D-18); the unadjusted S1 number is kept as `*_naive`."""
    headline = {"headline_method": "Lin estimator with cross-fitted LightGBM outcome score (see D-16, D-18)"}
    for o in ("visit", "conversion"):
        a = adj[o]
        headline.update({
            f"{o}_abs": a["ml_score"]["ate_adjusted"], f"{o}_ci": a["ml_score"]["ci_adjusted"],
            f"{o}_abs_naive": a["ate_unadjusted"], f"{o}_ci_naive": a["ci_unadjusted"],
            f"{o}_rel_adjusted": a["ml_score"]["ate_adjusted"] / load_results()["ate"][o]["rate_control"],
        })
    update_results("ate", headline)


def stage_s3(dev: bool, seeds: list[int]) -> None:
    from src import load, uplift_models

    df = load.load_dev() if dev else load.load_full()
    tag, section = ("dev", "uplift_dev") if dev else ("full", "uplift")
    res = uplift_models.run(df, tag, seeds=seeds, previous=load_results().get(section))
    update_results(section, res)
    if not dev:
        update_results("data", {"n_rows_used": len(df)})
    for seed, r in res["per_seed"].items():
        print(seed, "val Qini AUC:", {k: round(v, 5) for k, v in r["val_qini_auc"].items()})
    print("mean predicted uplift (test):", res["test_mean_predicted_uplift"])
    print("seed stability:", res["test_seed_stability_spearman"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["s01", "s2", "s3", "all"])
    ap.add_argument("--dev-n", type=int, default=1_000_000)
    ap.add_argument("--dev", action="store_true", help="stage s3: run on the 1M dev sample")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="stage s3: model seeds; seeds already in the scores file are skipped")
    args = ap.parse_args()
    if args.stage in ("s01", "all"):
        stage_s01(args.dev_n)
    if args.stage in ("s2", "all"):
        stage_s2()
    if args.stage in ("s3", "all"):
        stage_s3(args.dev, args.seeds)


if __name__ == "__main__":
    main()
