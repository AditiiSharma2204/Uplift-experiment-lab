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


def stage_s4(dev: bool) -> None:
    import json

    from src import evaluate
    from src.common import CURVES_JSON

    ev, curves = evaluate.run("dev" if dev else "full")
    update_results("evaluation_dev" if dev else "evaluation", ev)
    if not dev:
        m = ev["primary"]["models"][ev["best_model"]]
        update_results("uplift", {"best_model": ev["best_model"], "auuc": m["auuc"], "auuc_ci": m["auuc_ci"],
                                  "auuc_per_1m": m["auuc_per_1m"], "auuc_per_1m_ci": m["auuc_per_1m_ci"]})
        data = json.loads(CURVES_JSON.read_text()) if CURVES_JSON.exists() else {}
        data["qini"] = curves
        CURVES_JSON.write_text(json.dumps(data))
    p = ev["primary"]
    print(f"test ATE (IPW): {p['ate_test']:.5f} {p['ate_test_ci']}")
    for m, r in p["models"].items():
        print(f"{m:10s} AUUC/1M {r['auuc_per_1m']:8.1f} [{r['auuc_per_1m_ci'][0]:8.1f}, {r['auuc_per_1m_ci'][1]:8.1f}]"
              f"  unweighted {ev['unweighted'][m]['auuc'] * 1e6:8.1f}  conv {ev['conversion'][m]['auuc'] * 1e6:7.1f}"
              f"  up@10 {r['uplift_at_k']['10']['uplift']:.4f}")
    for k, v in p["pairwise_auuc"].items():
        print(f"  {k:26s} diff/1M {v['diff'] * 1e6:8.1f}  bonf [{v['ci_bonferroni'][0] * 1e6:8.1f}, "
              f"{v['ci_bonferroni'][1] * 1e6:8.1f}]  {'DIFFERENT' if v['distinguishable_bonferroni'] else 'within noise'}")
    print("top10 profile:", json.dumps(ev["top10_profile"], indent=0))
    print("per-seed AUUC:", ev["per_seed_auuc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["s01", "s2", "s3", "s4", "all"])
    ap.add_argument("--dev-n", type=int, default=1_000_000)
    ap.add_argument("--dev", action="store_true", help="stages s3/s4: run on the 1M dev sample")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="stage s3: model seeds; seeds already in the scores file are skipped")
    args = ap.parse_args()
    if args.stage in ("s01", "all"):
        stage_s01(args.dev_n)
    if args.stage in ("s2", "all"):
        stage_s2()
    if args.stage in ("s3", "all"):
        stage_s3(args.dev, args.seeds)
    if args.stage in ("s4", "all"):
        stage_s4(args.dev)


if __name__ == "__main__":
    main()
