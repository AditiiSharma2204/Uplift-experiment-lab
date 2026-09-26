"""Render README.md and docs/memo.md from results/results.json so every headline number is traceable.

Run via:  python run_all.py --stage s6
"""
from __future__ import annotations

from .common import ROOT, load_results

LABEL = {"x_learner": "X-learner", "s_learner": "S-learner", "t_learner": "T-learner",
         "response": "Response model", "random": "Random score"}


def pp(x: float, d: int = 3) -> str:
    return f"{100 * x:+.{d}f} pp"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def n0(x: float) -> str:
    return f"{x:,.0f}"


def ci_pp(c, d=3) -> str:
    return f"[{100 * c[0]:+.{d}f}, {100 * c[1]:+.{d}f}]"


def ci_n(c) -> str:
    return f"[{c[0]:,.0f}, {c[1]:,.0f}]"


def ci_pct(c, d=1) -> str:
    return f"[{100 * c[0]:.{d}f}%, {100 * c[1]:.{d}f}%]"


def facts(r: dict) -> dict:
    d, ate, adj, pw = r["data"], r["ate"], r["adjust"], r["power"]
    ev, pol, up = r["evaluation"], r["policy"], r["uplift"]
    prim = ev["primary"]
    best = ev["best_model"]
    ds = pol["default_scenario"]
    be = pol["breakeven"]
    comp = pol["model_comparison_profit"]
    return dict(
        n_rows=n0(d["n_rows_used"]), treat_share=pct(d["treatment_share"]), visit_rate=pct(d["visit_rate"], 2),
        conv_rate=pct(d["conversion_rate"], 2), n_conv=n0(d["conversion_count"]), n_visit=n0(d["visit_count"]),
        exposure_t=pct(d["exposure_rate_treated"]),
        srm_share=f"{r['integrity']['srm']['observed_treat_share']:.7f}",
        srm_p=f"{r['integrity']['srm']['p_value']:.3f}",
        max_smd=f"{r['integrity']['balance']['max_abs_smd']:.3f}",
        prop_auc=f"{r['integrity']['treatment_predictability']['auc']:.3f}",
        prop_auc_ci="[{:.3f}, {:.3f}]".format(*r["integrity"]["treatment_predictability"]["auc_ci95"]),
        ate_visit=pp(ate["visit_abs"]), ate_visit_ci=ci_pp(ate["visit_ci"]),
        ate_visit_rel=pct(ate["visit_rel_adjusted"]),
        ate_visit_naive=pp(ate["visit_abs_naive"]), ate_visit_naive_ci=ci_pp(ate["visit_ci_naive"]),
        ate_conv=pp(ate["conversion_abs"], 3), ate_conv_ci=ci_pp(ate["conversion_ci"], 3),
        ate_conv_rel=pct(ate["conversion_rel_adjusted"]),
        ate_conv_naive=pp(ate["conversion_abs_naive"]), ate_conv_naive_ci=ci_pp(ate["conversion_ci_naive"]),
        poststrat=pp(adj["visit"]["poststratified"]["ate_poststratified"]),
        poststrat_ci=ci_pp(adj["visit"]["poststratified"]["ci"]),
        top_bin_share=pct(adj["visit"]["poststratified"]["treat_share_max"]),
        vr_visit=f"{adj['visit']['linear']['variance_reduction_pct']:.1f}%",
        vr_visit_ml=f"{adj['visit']['ml_score']['variance_reduction_pct']:.1f}%",
        vr_conv=f"{adj['conversion']['linear']['variance_reduction_pct']:.1f}%",
        mde_visit_1m=pct(next(t["mde_rel"] for t in pw["visit"]["table"] if t["n"] == 1_000_000)),
        mde_conv_1m=pct(next(t["mde_rel"] for t in pw["conversion"]["table"] if t["n"] == 1_000_000)),
        n80_visit=n0(pw["visit"]["n_for_80pct_power_at_true_effect"]),
        n80_conv=n0(pw["conversion"]["n_for_80pct_power_at_true_effect"]),
        n_test=n0(prim["n_test"]), best=LABEL[best], best_key=best,
        auuc=n0(up["auuc_per_1m"]), auuc_ci=ci_n(up["auuc_per_1m_ci"]),
        test_ate=pp(prim["ate_test"]), test_ate_ci=ci_pp(prim["ate_test_ci"]),
        top10_resp_ctrl=pct(ev["top10_profile"]["response"]["rate_control"]),
        top10_best_ctrl=pct(ev["top10_profile"][best]["rate_control"]),
        top10_resp_inc=pct(ev["top10_profile"]["response"]["share_of_treated_visits_incremental"], 0),
        top10_best_inc=pct(ev["top10_profile"][best]["share_of_treated_visits_incremental"], 0),
        up5_best=pp(prim["models"][best]["uplift_at_k"]["5"]["uplift"], 1),
        up5_best_ci=ci_pp(prim["models"][best]["uplift_at_k"]["5"]["uplift_ci"], 1),
        up5_resp=pp(prim["models"]["response"]["uplift_at_k"]["5"]["uplift"], 1),
        up5_resp_ci=ci_pp(prim["models"]["response"]["uplift_at_k"]["5"]["uplift_ci"], 1),
        inc30=n0(pol["incremental_per_1m_at_30pct"]), inc30_ci=ci_n(pol["incremental_per_1m_at_30pct_ci"]),
        share30=pct(pol["incremental_by_k"]["30"]["share_of_all_incremental"], 0),
        inc100=n0(pol["incremental_by_k"]["100"]["incremental_per_1m"]),
        c=f"${ds['cost_per_treatment_usd']:.3f}", v=f"${ds['value_per_visit_usd']:.2f}",
        k_opt=pct(ds["optimal_k"]), k_opt_ci=ci_pct(ds["optimal_k_ci"]),
        profit=f"${ds['profit_per_1m_usd']:,.0f}", profit_ci="[${:,.0f}, ${:,.0f}]".format(*ds["profit_per_1m_usd_ci"]),
        profit_all=f"${ds['profit_treat_all_per_1m_usd']:,.0f}",
        profit_mult=f"{ds['profit_per_1m_usd'] / ds['profit_treat_all_per_1m_usd']:.1f}x",
        be_blanket=f"{be['blanket_campaign_breakeven_c_over_v']:.4f}",
        be_none=f"{be['treat_none_optimal_above_c_over_v']:.3f}",
        be_none_ci="[{:.3f}, {:.3f}]".format(*be["treat_none_optimal_above_ci"]),
        be_mult=f"{be['treat_none_optimal_above_c_over_v'] / be['blanket_campaign_breakeven_c_over_v']:.0f}x",
        p05_best=n0(comp[best]["0.05"]["profit_per_1m_in_v"]), p05_best_ci=ci_n(comp[best]["0.05"]["profit_ci"]),
        p05_resp=n0(comp["response"]["0.05"]["profit_per_1m_in_v"]),
        p05_resp_ci=ci_n(comp["response"]["0.05"]["profit_ci"]),
    )


def auuc_table(r: dict) -> str:
    prim, ev = r["evaluation"]["primary"], r["evaluation"]
    rows = ["| Model | AUUC per 1M users | 95% CI | Uplift in top 5% | Uplift in top 10% |", "|---|---|---|---|---|"]
    order = sorted(prim["models"], key=lambda m: -prim["models"][m]["auuc"])
    for m in order:
        a = prim["models"][m]
        name = LABEL[m] + (" (selected)" if m == ev["best_model"] else "")
        rows.append(f"| {name} | {a['auuc_per_1m']:,.0f} | {ci_n(a['auuc_per_1m_ci'])} | "
                    f"{100 * a['uplift_at_k']['5']['uplift']:.1f} pp | {100 * a['uplift_at_k']['10']['uplift']:.1f} pp |")
    return "\n".join(rows)


def sensitivity_table(r: dict) -> str:
    rows = ["| Cost / value | Best share to treat | 95% CI | Profit at best k (units of v) | Treat everyone |",
            "|---|---|---|---|---|"]
    for t in r["policy"]["sensitivity_table"]:
        if t["cost_to_value"] not in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            continue
        rows.append(f"| {t['cost_to_value']:g} | {pct(t['optimal_k'])} | {ci_pct(t['optimal_k_ci'])} | "
                    f"{t['profit_per_1m_at_optimum_in_v']:,.0f} | {t['profit_per_1m_treat_all_in_v']:,.0f} |")
    return "\n".join(rows)


README = """# Uplift and Experimentation Lab (Criteo)

**Business question.** A randomized ad campaign ran on millions of users. Did it work on average, and can we do better
than treating everyone by choosing *whom* to target, given a cost per treatment?

**Short answer.**
- Yes, it worked. Being assigned to the campaign raised the visit rate by **{ate_visit}** (95% CI {ate_visit_ci}),
  a {ate_visit_rel} relative lift, after correcting a bias in the pooled data that inflates the naive estimate to {ate_visit_naive}.
- Yes, targeting does better. Ranking users by predicted *uplift* ({best}) and treating the top 30% captures
  **{inc30} of the {inc100} incremental visits per 1M users** ({share30} of the total; 95% CI {inc30_ci}).
- Under illustrative costs ({c} per treatment, {v} per visit), the profit-maximizing policy treats the top **{k_opt}**
  (95% CI {k_opt_ci}) and earns **{profit}** per 1M users, 95% CI {profit_ci}, vs {profit_all} for treating everyone
  ({profit_mult}).

![Qini curves](results/figures/qini_curves.png)

## Design

- **Data:** Criteo Uplift Prediction Dataset v2.1. **{n_rows} rows**, 12 anonymized features (f0-f11), `treatment`
  ({treat_share} treated), outcomes `visit` ({visit_rate}, primary) and `conversion` ({conv_rate}, {n_conv} events, secondary).
- **A randomized experiment, checked rather than assumed.** No sample-ratio mismatch (treated share {srm_share} vs the
  documented 0.85, p = {srm_p}). All features are balanced (max |SMD| {max_smd}). But a classifier predicts treatment from features
  slightly better than chance (AUC {prop_auc}, 95% CI {prop_auc_ci}). The data pools several tests, so assignment is random
  only *conditional on features*. Among the users most likely to visit anyway, {top_bin_share} were treated rather than 85%, which inflates
  the simple difference in means. Headline effects are therefore covariate-adjusted, and model evaluation is propensity-weighted.
- **`exposure` is post-treatment** (only {exposure_t} of treated users actually saw an ad). It is never used as a feature, and the
  code asserts this. Effects are intent-to-treat.
- **Uplift models** (LightGBM, learner logic written by hand): response model (baseline), T-learner, S-learner, X-learner.
  Stratified 60/10/30 train/validation/test split; light tuning and model selection on validation only; 3 seeds.
- **Evaluation** on {n_test} held-out users: propensity-weighted Qini/uplift curves, AUUC and uplift@k, with 200 paired
  bootstrap resamples. A random score is run through the same pipeline as a sanity check.

## Results

### 1. Average effect (full data, intent-to-treat)

| Outcome | Effect (adjusted, headline) | 95% CI | Relative | Naive difference in means |
|---|---|---|---|---|
| Visit | {ate_visit} | {ate_visit_ci} | {ate_visit_rel} | {ate_visit_naive} {ate_visit_naive_ci} |
| Conversion | {ate_conv} | {ate_conv_ci} | {ate_conv_rel} | {ate_conv_naive} {ate_conv_naive_ci} |

A model-light cross-check (compare the arms within 20 bins of predicted visit probability) gives {poststrat} {poststrat_ci}.

### 2. Power and variance reduction

- Minimum detectable effect at 80% power with 1M users: **{mde_visit_1m}** relative for visit, **{mde_conv_1m}** for conversion.
  80% power at the observed effect needs ~{n80_visit} users for visit and ~{n80_conv} for conversion. Analytical power matches a
  200-subsample simulation.
- Covariate adjustment (Lin estimator, robust SE) cuts the variance of the visit effect by **{vr_visit}** ({vr_visit_ml} with
  an ML outcome score) and conversion by only **{vr_conv}**. A 0.3% outcome is mostly irreducible noise. The features are
  anonymized pre-treatment covariates, not a pre-period metric, so this is regression adjustment, not CUPED.

### 3. Who to target: uplift models on the test set

{auuc_table}

AUUC is the area between the uplift curve and random targeting, in incremental visits per 1M users. The model was selected on
validation; on test the X- and S-learners are statistically tied (paired bootstrap, Bonferroni-corrected).

**Why not just target likely visitors?** The response model's whole-curve AUUC is close to the uplift learners', but at the top
of the ranking, where budgets bind, it falls behind. Uplift in the top 5% is {up5_resp} {up5_resp_ci} for the response model vs
{up5_best} {up5_best_ci} for the {best}. The response model's top 10% already visit {top10_resp_ctrl} of the time without the ad
(only {top10_resp_inc} of their visits are incremental). The {best}'s top 10% visit {top10_best_ctrl} of the time without it
({top10_best_inc} incremental).

### 4. Policy and economics (illustrative costs)

![Policy](results/figures/policy.png)

{sensitivity_table}

- Treating everyone breaks even at cost/value = **{be_blanket}**. Targeting stays profitable up to **{be_none}** {be_none_ci},
  a {be_mult} wider range. Beyond that, treat nobody.
- The ranking model matters most when treatment is expensive. At cost/value 0.05, ranking by the {best} earns {p05_best}
  {p05_best_ci} per 1M users (in units of v) vs {p05_resp} {p05_resp_ci} for the response model.

## Simulator

`app/streamlit_app.py` reads only `results/curves.json` (never the raw data). It has sliders for the targeting share, cost per
treatment and value per visit, and shows incremental visits with the CI band, profit, the profit-maximizing share, and an
optional response-model comparison. Cost/value assumptions are labelled as illustrative in the app.

## Limitations

- **Rare outcome.** Conversion is 0.29% of rows, so intervals are wide. We model visit and report conversion as secondary.
- **Anonymized, randomly projected features.** There is no way to interpret *who* the high-uplift users are, and no pre-period metric
  for CUPED.
- **One pooled experiment.** The public data combines several tests and was subsampled non-uniformly. Effects hold for this
  dataset, not as Criteo's real campaign lift. Adjustment and propensity weighting reduce, but cannot guarantee away, the
  resulting imbalance (IPW moves the test effect from ~1.03 pp to {test_ate}, not all the way to the adjusted {ate_visit}).
- **Short-term, single campaign.** No long-term effects, ad fatigue, carry-over or competition between campaigns. Costs and values are
  illustrative.
- **Optimistic optimum.** The profit-maximizing share is chosen and valued on the same test curve (a winner's curse). The bootstrap
  interval on k shows how stable the choice is.

## How to run

```bash
conda create -n uplift python=3.11 && conda activate uplift
pip install -r requirements.txt
python run_all.py --stage all        # downloads ~300 MB, then S0/S1 -> S6 (the full run takes ~1.5 h on a laptop CPU)
python run_all.py --stage s3 --dev   # quick run of the models on the 1M-row development sample
pytest -q                            # 17 tests: statistics, learners, evaluation, leakage guard, app
streamlit run app/streamlit_app.py   # the simulator
```

Stages: `s01` load, integrity and ATE; `s2` power and adjustment; `s3` uplift models; `s4` evaluation; `s5` policy;
`s6` this README and the memo. Every analysis decision (and every mistake caught along the way) is logged in
[docs/decisions.md](docs/decisions.md). The stakeholder summary is in [docs/memo.md](docs/memo.md).

```
src/     load  integrity  ate  power  adjust  uplift_models  evaluate  policy  report  plotting  common
app/     streamlit_app.py            results/  results.json  curves.json  figures/
docs/    decisions.md  memo.md       tests/    run_all.py  requirements.txt
```

## Data source and licence

Criteo Uplift Prediction Dataset v2.1, Criteo AI Lab: <https://huggingface.co/datasets/criteo/criteo-uplift>
(also <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>). Licence: **CC BY-NC-SA 4.0** (non-commercial,
attribution, share-alike). The raw data is not redistributed here. Reference: Diemert, Betlei, Renaudin, Amini, *A Large Scale
Benchmark for Uplift Modeling*, AdKDD 2018.

*README generated by `src/report.py` from `results/results.json`.*
"""

MEMO = """# Memo: targeting the display campaign by predicted uplift

**To:** Marketing / growth leadership   **Re:** Did the campaign work, and whom should we treat?

**Recommendation.** Stop treating everyone. Rank users by predicted *incremental* effect and treat only the top slice. Under
our illustrative costs ({c} per treatment, {v} per visit) that is the top **{k_opt}** of users (95% CI {k_opt_ci}). It keeps
most of the campaign's effect for a fraction of the cost: **{profit}** vs **{profit_all}** profit per 1M users.

**What we found**
1. **The campaign works.** Assignment raised the visit rate by {ate_visit} (95% CI {ate_visit_ci}), a {ate_visit_rel} relative
   lift. A naive read of the data says {ate_visit_naive}. That overstates it by about half, because heavy visitors were slightly
   over-represented in the treated group. We corrected for this.
2. **The effect is concentrated.** Treating the 30% of users with the highest predicted uplift gets {share30} of all the extra
   visits. The bottom half of users adds nothing measurable, so every dollar spent on them is wasted.
3. **Target by uplift, not by "likely to visit".** Picking people who are likely to visit anyway looks sensible but buys visits
   that would have happened regardless. When treatment is expensive (e.g. $0.05 per treatment at {v} per visit), that approach
   earns about ${p05_resp} per 1M users vs ${p05_best} for uplift targeting.
4. **When to switch strategy.** If cost per treatment is below ~{be_blanket} x the value of a visit, treating everyone is still
   profitable, but targeting is better. Above ~{be_none} x, no targeting pays for itself.

**Caveats.** Cost and value figures are illustrative placeholders. The data is a public, anonymized, pooled sample, so the lift
is not Criteo's real-world number. We measured short-term visits only. **Next step:** run the targeting policy against
treat-everyone as a live A/B test before rolling it out, and re-estimate costs with finance.
"""


def run() -> None:
    r = load_results()
    f = facts(r)
    readme = README.format(**f, auuc_table=auuc_table(r), sensitivity_table=sensitivity_table(r))
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "docs" / "memo.md").write_text(MEMO.format(**f), encoding="utf-8")
