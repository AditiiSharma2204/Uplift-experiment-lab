# Decisions log

Each entry: what we decided, why, and what it affects. Numbers come from `results/results.json`.

## Stage S0/S1: load, integrity, average effect

**D-01. Data source and licence.** Criteo Uplift Prediction Dataset v2.1, official Criteo AI Lab copy on Hugging Face
(`criteo/criteo-uplift`). The scikit-uplift S3 mirror now returns HTTP 403, so we download from Hugging Face instead. The MD5
(`d2236769...`) matches the hash scikit-uplift pins, so it is the same file. Licence: **CC BY-NC-SA 4.0**, which means
non-commercial use only, attribution required, and share-alike for derivatives. The raw data is never committed.

**D-02. Loading.** pyarrow reads the gzipped CSV directly with explicit dtypes (features float32, flags int8), so the ~3 GB
uncompressed CSV never hits disk. The data is cached as zstd parquet. Result: 13,979,592 rows, 693 MB in pandas. The
full data fits in memory, so S0 to S2 use all rows.

**D-03. Development sample.** 1,000,000 rows, stratified by treatment x visit x conversion, seed 42
(`data/processed/criteo_dev.parquet`). Used for fast iteration and for the treatment-predictability check. Headline
numbers come from the full data.

**D-04. Primary outcome = visit; conversion is secondary.** Conversions are 0.29% of rows (40,774 events) and visits are
4.70% (656,929). Visit has about 16x more events, so its intervals are much tighter and uplift rankings are less noisy.

**D-05. `exposure` is post-treatment and is never a feature.** It records whether the ad was actually shown, which happens
after assignment: 3.6% of treated users were exposed and 0% of control users. Using it as a feature is leakage.
`common.assert_no_leakage` enforces this and has a test. It also means our ATE is an **intent-to-treat** effect: the effect
of *being assigned* to the campaign, not of seeing an ad.

**D-06. SRM reference ratio = 0.85.** Source: the dataset card ("Treatment Ratio: 0.85"). The observed share is
0.8500001 (chi-square p = 0.999), so there is **no SRM**. *Caveat:* a share this close to 0.85 is not what randomization
produces on its own. Per the card, the data was "assembled from several incrementality tests" and "sub-sampled
non-uniformly", so the pooled ratio was very likely set during construction. The SRM check passes, but here it
cannot detect a broken assignment mechanism. We report it and do not lean on it.

**D-07. Balance is judged by standardized mean differences, not p-values.** With 14M rows even trivial differences are
"significant", so we use the conventional |SMD| < 0.1 threshold. Result: all 12 features are inside it (max |SMD| = 0.049,
f3). Some variance ratios are well above 1 (f1 1.32, f5 1.22, f3 1.19), still inside the usual 0.5 to 2 range.

**D-08. Treatment-predictability check (classifier two-sample test).** LightGBM trained on the dev sample to predict
treatment from f0-f11. Held-out AUC = 0.507, 95% CI [0.504, 0.510], which excludes 0.5. The features carry a
small but real signal about assignment. The likeliest explanation is the pooling in D-06: several tests with
different populations and treatment shares, pooled to 85/15, make the features weakly correlated with treatment
even though each test was randomized. Consequences:
- The simple difference in means is still our headline ATE, since the imbalance is tiny (AUC 0.507).
- Covariate adjustment in S2 (Lin estimator) doubles as a robustness check. If the adjusted ATE moves materially, the
  pooling matters.
- Uplift models (S3) condition on the features anyway, and evaluation uses the same randomized test split.

**D-09. Duplicates are kept.** 11.8% of rows duplicate another row on all columns, and 15.2% on features only.
The dataset has no user id, and the features are randomly projected, several of them with few distinct values (f1 has 60).
So identical rows can be different users with the same projected profile. Dropping them would change the design
without evidence of an error. No missing values in any column.

**D-10. ATE method.** Difference in proportions, with a Wald 95% CI (unpooled SE) and a two-sided z-test (pooled SE,
which matches statsmodels `proportions_ztest`; tested). The relative-lift CI uses the delta method on log(p_t/p_c),
which behaves well when p_c is small. Effects are also given per 1M treated users.

Headline (full data, intent-to-treat):
- **visit**: +1.034 pp, 95% CI [+1.006, +1.063]; 3.820% -> 4.854%; relative +27.1% [+26.2%, +28.0%].
- **conversion**: +0.115 pp, 95% CI [+0.108, +0.122]; 0.194% -> 0.309%; relative +59.4% [+54.4%, +64.7%].

*Caution when quoting these:* the card says the data was subsampled non-uniformly so that "the original incrementality
level cannot be deduced". The lifts are valid **for this dataset** but are not Criteo's real-world campaign lift.

## Stage S2: power and variance reduction

**D-11. The power test matches the test we actually run.** Two-sided pooled z-test, alpha 0.05, 80% power, 85/15 split. Power uses
the pooled SE under H0 for the critical value and the unpooled SE under H1. MDE and required n are solved numerically from that
same function. A first version used the textbook "baseline variance in both arms" formula and **overstated power** (conversion at 20k:
24% analytical vs 10% simulated). With an 85/15 split the pooled rate sits near the *treated* rate, so the real null SE is
larger. After the fix, analytical and simulated power agree within Monte Carlo error.

**D-12. Simulation by exact cell counts.** A random subsample of n rows only changes the counts in the 4
(treatment x outcome) cells, so we draw those counts from a multivariate hypergeometric (200 reps, seed 42). The result is
identical in distribution to sampling rows, and instant at 5M. At the full size we bootstrap, because a without-replacement
"subsample" of everything is the same data 200 times.

**D-13. Power findings** (using the unadjusted full-data effect as the "true" effect):
- visit: MDE is 1.1% relative (0.040 pp) at full size and 4.0% at 1M. 80% power needs ~25k users (57k at the adjusted effect).
- conversion: MDE is 4.8% relative at full size and 18.8% at 1M. 80% power needs ~123k users (153k at the adjusted effect).
- At 10k to 20k users, simulated conversion power falls *below* the analytical value (3.5% vs 8% at 10k). The control arm has only
  3 to 6 expected conversions there, so the normal approximation behind the z-test breaks down. With that few events you
  would use an exact test, or not run the test at all.

**D-14. Covariate adjustment = Lin estimator with HC1 SEs, computed in chunks.** We wrote our own chunked OLS because 14M x 26 in
statsmodels would need several GB for the robust covariance. It matches statsmodels HC1 to machine precision (test).
Wording: this is **regression adjustment on anonymized pre-treatment features, not CUPED.** CUPED uses the pre-period value of
the same metric, which this dataset does not have. Two covariate sets: (a) the 12 raw features, linear (textbook Lin);
(b) one cross-fitted LightGBM outcome score (CUPAC-style, 2 folds, no treatment input, out-of-fold predictions only).

**D-15. Variance reduction: moderate for visit, almost none for conversion.**

| outcome | linear Lin | ML-score Lin |
|---|---|---|
| visit | **19.5%** | 23.8% |
| conversion | 2.3% | 0.9% |

Why: adjustment can only remove variance the features explain. Visit is quite predictable from f0-f11 (out-of-fold AUC
0.947). Conversion is a 0.3% event, and most of its variance is irreducible Bernoulli noise: even among users with identical
features, which ones convert is close to a coin flip. So the SE barely moves. A 19.5% variance reduction is worth about
1.24x the sample size for visit.

**D-16. The biggest finding of S2: adjustment moves the point estimate, which should not happen in a clean randomized test.**

| visit ATE | estimate | 95% CI | shift vs unadjusted |
|---|---|---|---|
| unadjusted difference in means | 1.034 pp | [1.006, 1.063] | |
| Lin, linear features | 0.773 pp | [0.748, 0.799] | -18 SE |
| Lin, ML score | 0.665 pp | [0.640, 0.690] | -25 SE |
| post-stratified on 20 score bins | 0.691 pp | [0.666, 0.717] | |

Diagnosis (`results/figures/strata_visit.png`): the treated share is ~84.8% across most of the population but rises to
**86.5% among users most likely to visit anyway** (visit rate ~50%). Heavy visitors are over-represented in the treated arm, which
inflates the simple difference. This matches D-08 (the features predict treatment, AUC 0.507) and the dataset card
(several incrementality tests, pooled and non-uniformly subsampled). Each test may be randomized, but the **pooled data is
randomized only conditional on features**, roughly. Conversion shows the same pattern (0.115 pp unadjusted -> 0.100 pp linear, 0.102 pp ML score, 0.090 pp
post-stratified).

Consequences:
- The unadjusted ATE from S1 is **biased upward** (about +50% for visit) and should not be the headline. The adjusted
  estimates agree with each other (0.67 to 0.69 pp from the two flexible methods). The linear one sits in between because a
  linear model underfits a strongly nonlinear outcome.
- The variance-reduction percentages above are real, but the main value of adjustment here is **bias correction**.
- S3 (uplift models): T-, S- and X-learners condition on X, so they are consistent if X captures the pooling.
- S4/S5 (evaluation, policy): a plain Qini curve compares treated and control rates inside top-k groups, and inherits the
  same bias. **Plan: estimate a cross-fitted propensity e(x) and use inverse-propensity-weighted uplift@k / Qini,** with the
  plain version shown alongside as a sensitivity check.

**D-17. Oddity noted, not yet explained.** In conversion's lowest score bin the observed conversion rate (0.33%) is far above bins
2 to 10 (<0.02%). A cluster of rows seems to be mis-scored by the 2-fold model. It does not affect the visit results. We will check
whether the S3 models show the same thing.

## Stage S3: uplift models

**D-18. Headline ATE = ML-score Lin estimate.** `ate.visit_abs` = +0.665 pp [0.640, 0.690] (+17.4% relative). The unadjusted
+1.034 pp is kept as `ate.visit_abs_naive`, labelled as biased by the pooling (D-16). Reason: the two flexible adjustments
(ML-score Lin 0.665, post-stratified 0.691) agree, while the unadjusted number fails the "adjustment should not move it" check.

**D-19. Modelling outcome = visit.** Conversion has 40.8k events in total, about 12k of them in the test set and only about 1.2k in the test
control arm. Uplift rankings on that would be mostly noise (see the conversion MDE in D-13).

**D-20. Split.** 60/10/30 train/validation/test, stratified by treatment x visit x conversion, seed 42, cached in
`data/processed/split_full.parquet` so it never changes between runs. The test set is large on purpose (4.2M rows, about 197k
visits) because Qini on a rare outcome is noisy. The test outcomes are not read in S3. Only test *features* are scored and
written to `results/scores/`.

**D-21. Light tuning, one shared setting.** Grid over num_leaves {31, 127} x min_child_samples {200, 2000}, learning rate 0.1,
early stopping (50 rounds) on the validation log-loss of a response model trained on a 2M-row subsample of train. The best
setting is used for *every* base learner, so the learners differ only in their logic, not in how hard each was tuned.
Log: `data/processed/tuning_full.json` and `results.json -> uplift.tuning`. Each model's number of trees is chosen by early
stopping on the validation split (the right arm's rows for arm-specific models).

**D-22. Propensity model e(x) = P(t=1 | x).** In a clean randomized experiment e(x) would equal 0.85 for everyone. Because of the
pooling (D-16) it varies slightly, so we fit it (same LightGBM setting) on train and:
(a) use it as the blending weight in the X-learner, and (b) save it for the test set so S4 can weight by inverse propensity.

**D-23. Seeds.** The split is fixed (seed 42). "Seeds" change only the LightGBM randomness (row/column subsampling). The full data
was run with seed 42 first (about 16 min per seed on this laptop). Seeds 43 and 44 can be added with
`python run_all.py --stage s3 --seeds 43 44`. The runner appends them without refitting seed 42.

### The four learners in plain English

| learner | what it does | what it assumes | where it goes wrong |
|---|---|---|---|
| **Response model** | Predicts P(visit \| x) for everyone, ignoring treatment. Target the highest scores. | That people likely to visit are the people the ad *changes*. | Ranks "sure things" (would visit anyway) at the top. We pay to treat people whose behaviour we don't change. It answers "who will visit?", not "who will visit *because of* the ad?". |
| **T-learner** | Two models: one on treated users, one on control. Uplift = difference of their predictions. | Each arm's model is accurate enough that the *difference* is meaningful. | The control arm is small (15%), so its model is noisier. The difference of two noisy predictions amplifies noise, and the models may pick up unrelated patterns in each arm that show up as fake uplift. |
| **S-learner** | One model with treatment as an extra input. Uplift = prediction with t=1 minus t=0. | The model will learn how treatment interacts with the features. | Trees can largely ignore the treatment flag when it is a weak predictor next to strong features (regularisation bias). Uplift is then shrunk toward zero and heterogeneity is lost. |
| **X-learner** | Starts from the T-learner. Imputes each user's individual effect using the *other* arm's model (treated: y - mu0(x); control: mu1(x) - y), fits models to those imputed effects, and blends them with the propensity. | The first-stage models are good. Effects are smoother than outcomes, so they are easier to learn. | Errors in mu0/mu1 carry straight into the imputed effects, and it has more moving parts to validate. It is built for unbalanced designs like ours: most weight goes to the effect model built from control users, whose imputations use the model fitted on the large treated arm. |

### S3 results (full data, seed 42; seeds 43/44 appended later)

- Split: 8,387,756 train / 1,397,958 validation / 4,193,878 test.
- Tuning barely matters: all four grid settings are within 0.00005 validation log-loss (0.10358 to 0.10363). Chosen:
  num_leaves 31, min_child_samples 2000.
- Early stopping shows how much signal each model found: mu1 (treated) used 498 trees, **mu0 (control) only 83**, and the X-learner's
  tau0 only 28. The control arm has 6x fewer rows, so its model is coarser. That is the main weakness of the T-learner here.
- Propensity model validation AUC = 0.511, consistent with D-08: assignment is almost, but not quite, random.
- **Sanity check (no test outcomes used):** the mean predicted uplift on test is 0.72 to 0.74 pp for all three uplift learners. That is
  close to the *adjusted* ATE (0.665 to 0.691 pp) and far from the naive 1.034 pp. Because the learners condition on x, they are
  largely free of the pooling bias, as expected (D-16).
- Rank agreement on test (Spearman, seed 42): response vs S-learner 0.86, vs X-learner 0.83, vs T-learner 0.57. **On this
  data the uplift rankings largely track the baseline visit propensity.** The absolute effect grows with the baseline rate
  (D-16 strata: +0.01 pp in the lowest bin, +5.7 pp in the top). So a response model is a much stronger baseline here than
  the textbook "sure things" story suggests. S4 decides whether the uplift models beat it by more than noise.
- Validation Qini AUC (for information only; not used for selection, and not propensity-corrected): X 0.092, S 0.089,
  response 0.086, T 0.079.

## Stage S4: honest evaluation (test set, 4,193,878 users, first use of test outcomes)

**D-24. Metric definition.** Uplift-curve form of Qini: gain(k) = (n_k/N) x [P(visit | treated, top k) - P(visit | control, top k)],
i.e. the incremental visits (per user of the whole population, reported per 1M) if exactly the top k% were treated. It ranks
models the same way as the classic Qini (Y_T - Y_C x N_T/N_C) but is already in policy units for S5. AUUC = area between
gain(k) and the random line k x gain(100%). The grid runs from 0.5% to 100% in 0.5% steps. Scores are averaged over the 3 seeds.

**D-25. Propensity-weighted (primary) and unweighted (sensitivity).** Arm rates inside each top-k group are Hajek IPW means
with the seed-averaged e(x) clipped to [0.01, 0.99]. On synthetic data with planted confounding, IPW recovers the true effect and
the naive estimate does not (tests/test_evaluate.py). On the real test set IPW moves the overall effect from about 1.03 pp
(unweighted) to **0.759 pp [0.704, 0.817]**. That removes roughly 3/4 of the gap to the Lin/post-stratified estimates (0.665 to
0.691 pp), but not all of it. The propensity model is weak (AUC 0.511), and IPW can only correct what e(x) captures. Every
unweighted AUUC is ~15% higher than its weighted version: the naive evaluation flatters every model.

**D-26. Bootstrap.** 200 Poisson(1) replicates, seed 42. All models share each replicate, so model differences are paired.
Percentile 95% CIs. For "is model A better than model B" we use a Bonferroni-corrected interval over the 6 learner
pairs (normal approximation with the bootstrap SE), to be conservative.

**D-27. Sanity check passed.** A random score run through the same pipeline gets AUUC -132 per 1M, 95% CI [-294, +26], which covers 0.
(Its point estimate is slightly negative. That is sampling noise: the CI includes 0, and the uplift@k CIs vs random all include 0 too.)

**D-28. Results (visit, propensity-weighted, AUUC in incremental visits per 1M users above random):**

| model | AUUC per 1M | 95% CI | uplift@5% | uplift@10% | uplift@20% | uplift@30% |
|---|---|---|---|---|---|---|
| X-learner | **3,298** | [3,066, 3,549] | 9.6 pp | 6.2 pp | 3.5 pp | 2.4 pp |
| S-learner | 3,350 | [3,147, 3,580] | 9.4 pp | 6.1 pp | 3.5 pp | 2.5 pp |
| Response | 3,109 | [2,909, 3,367] | 6.2 pp | 5.2 pp | 3.4 pp | 2.4 pp |
| T-learner | 3,061 | [2,825, 3,301] | 9.0 pp | 5.8 pp | 3.4 pp | 2.4 pp |
| Random score | -132 | [-294, +26] | 0.6 pp | 0.6 pp | 0.7 pp | 0.7 pp |

(Overall uplift on the test set = 0.76 pp, so random targeting gets ~0.76 pp at every k.)

Pairwise, Bonferroni-corrected:
- **Distinguishable:** S > response; S > T; X > T.
- **Within noise:** S vs X; X vs response (the interval just includes 0: [-412, +34] per 1M); response vs T.

**D-29. Best model = X-learner, chosen on validation, not on test.** The selection rule was fixed before the test set was read:
the highest mean validation Qini across seeds. X won in every seed (mean 0.0914 vs S 0.0887, response 0.0856, T 0.0801). On test,
X and S are statistically tied (S's point estimate is 51 per 1M higher, CI [-109, +212]). We don't switch to S because of a
test-set win, since that would be selecting on the test set. X also has the most stable ranking across seeds (Spearman 0.95 vs S 0.92, T 0.81).
The first version of `evaluate.run` picked the best test AUUC; this was fixed before reporting.

**D-30. The response model is not the right tool, but the difference shows up at the top of the ranking, not in AUUC.**
- AUUC over the whole curve is similar (response 3,109 vs X 3,298), because on this data the absolute uplift grows with the
  baseline visit rate, so both rankings eventually find the same people.
- **Where budgets actually bind (small k), the gap is large and clearly outside noise:** uplift@5% is 6.2 pp [5.4, 6.8] for response
  vs 9.6 pp [9.0, 10.2] for X-learner, about 55% more incremental visits per treated user.
- Who each model picks (top 10%): response picks users with a **32% control visit rate**, and only **14%** of their visits under
  treatment are incremental. X picks users with a 22% control rate, and **22%** of their visits are incremental. That is the "sure
  things" effect: the response model spends budget on people who would have visited anyway.

**D-31. Where the value is.** All curves flatten after ~30 to 40% targeted. Treating the top ~35% by X-learner captures essentially all
of the incremental visits the campaign produces (gain ~7,600 per 1M, same as treating everyone). The bottom ~60% contributes
nothing measurable. This is the key input for the S5 cost/value analysis.

**D-32. Secondary outcome (conversion), same visit-trained rankings.** Every learner beats random on conversion AUUC (per 1M:
response 440 [381, 516], S 426, X 412, T 316; random -3 [-42, +34]). Differences between learners are within noise, except that T is
lowest. We did not train conversion-specific models (D-19).

**D-33. Seed averaging helps a little.** Single-seed AUUC for the S-learner ranges from 3,249 to 3,390 per 1M; the 3-seed average scores 3,350.
The T-learner varies most across seeds, consistent with its noisy control-arm model.

## Stage S5: targeting policy and economics

**D-34. Policy = treat the top k by X-learner uplift score** (the model selected on validation in D-29). All quantities are on the
test set, propensity-weighted, per 1,000,000 users, with 95% intervals from the same 200 paired bootstrap replicates as S4
(`results/scores/eval_bootstrap_full.npz`). *Bug caught here:* the replicate file used to be shared between dev and full runs,
and a dev check had overwritten it. The first S5 draft therefore showed dev numbers (the treat-everyone point was 8,212 instead of the
test ATE of 7,594). The file is now per dataset (`eval_bootstrap_{tag}.npz`) and S4 was rerun. The rerun reproduced S4 exactly.

**D-35. Cost and value are ILLUSTRATIVE assumptions** and are labelled as such in results, figures and the app. Range: value
per visit $0.10 to $2.00, cost-to-value ratio c/v from 0 to 0.2. Default scenario: v = $1.00, c = $0.005 (c/v = 0.005). The optimal k depends
only on c/v because profit/v = incremental(k) - 1e6 x k x (c/v). So the sensitivity analysis is a sweep over c/v (241 values,
0 to 0.3).

**D-36. Incremental visits by share treated (vs treating nobody):**

| treat top | incremental visits per 1M | 95% CI | vs random targeting | share of all incremental visits |
|---|---|---|---|---|
| 5% | 4,798 | [4,509, 5,088] | +4,418 | 63% |
| 10% | 6,160 | [5,791, 6,583] | +5,400 | 81% |
| 20% | 6,972 | [6,531, 7,455] | +5,453 | 92% |
| **30%** | **7,212** | **[6,732, 7,760]** | +4,934 | **95%** |
| 50% | 7,591 | [7,019, 8,153] | +3,794 | 100% |
| 100% | 7,594 | [7,042, 8,170] | 0 | 100% |

Treating 30% of users gets 95% of the campaign's incremental visits. The other half of the population contributes nothing
measurable.

**D-37. Sensitivity: profit-maximizing k across cost/value ratios** (profit per 1M users in units of v):

| c/v | optimal k | 95% CI | regime | profit at optimum | treat everyone |
|---|---|---|---|---|---|
| 0 | 99.5% | [42.5, 100] | treat (almost) all | 7,661 | 7,594 |
| 0.0005 | 45.0% | [40.5, 57.0] | target | 7,380 | 7,094 |
| 0.001 | 42.0% | [39.0, 45.5] | target | 7,158 | 6,594 |
| 0.002 | 40.5% | [22.5, 42.5] | target | 6,740 | 5,594 |
| **0.005** | **19.5%** | **[14.0, 26.5]** | target | **6,001 [5,554, 6,484]** | 2,594 |
| 0.0075 | 18.5% | [10.0, 20.5] | target | 5,554 | 94 |
| 0.01 | 14.0% | [8.5, 18.5] | target | 5,190 | -2,406 |
| 0.02 | 8.0% | [7.0, 10.0] | target | 4,253 | -12,406 |
| 0.05 | 3.5% | [3.5, 5.0] | target | 2,424 | -42,406 |
| 0.1 | 1.5% | [1.5, 2.0] | target | 968 | -92,406 |
| 0.2 | 0.5% | [0.0, 0.5] | ~treat none (profit CI [-78, +188] includes 0) | 57 | -192,406 |

Regimes and break-even ratios (with bootstrap CIs):
- **Treat everyone** is only optimal when treatment is essentially free: the threshold is c/v < ~0, CI [-0.040, +0.0002]. The bottom half of the
  population adds no visits, so any positive cost makes excluding them worthwhile. k >= 95% is labelled "treat (almost) all"
  because the flat tail makes the exact argmax wander between 90% and 100%.
- **A blanket campaign (treat everyone) breaks even at c/v = 0.0076 [0.0070, 0.0082]**, which is the overall uplift per user.
  Above that, treating everyone loses money.
- **Targeting stays profitable up to c/v = 0.215 [0.188, 0.241]**, the best average uplift of any top slice (the top 0.5%).
  Beyond it, **treat nobody**. So targeting widens the profitable cost range by about 28x.
- Default scenario (c = $0.005, v = $1): treat the top **19.5% [14.0%, 26.5%]**, profit **$6,001 [5,554, 6,484] per 1M users**, vs
  $2,594 for treating everyone. That is 2.3x the profit with a fifth of the treatments.

**D-38. Does the ranking model matter for money?** Profit at each model's own optimal k (per 1M, units of v):

| c/v | X-learner | S-learner | T-learner | Response model |
|---|---|---|---|---|
| 0.001 | 7,162 | 7,154 | 6,855 | 7,060 |
| 0.005 | 5,998 | 6,178 | 5,871 | 5,925 |
| 0.02 | **4,238** [3,872, 4,621] | 4,154 | 3,933 | **3,291** [2,853, 3,803] |
| 0.05 | **2,390** [2,146, 2,649] | 2,352 | 2,070 | **610** [324, 893] |

When treatment is cheap, every ranking is similar, because you treat nearly everyone who responds. **When treatment is expensive and only the top few %
can be treated, ranking by response loses most of the profit** (c/v = 0.05: 610 vs 2,390, about 4x less, and the CIs do not overlap). That is the business
version of D-30: the response model's top slice is full of people who would have visited anyway.

**D-39. Caveat: the chosen k is optimistic (winner's curse).** k is picked on the test curve and its profit is read off that same curve,
so the profit at the optimum is biased slightly upward. The optimal-k interval (argmax recomputed on each bootstrap replicate)
shows how unstable the choice is. In production, pick k on one sample and confirm it on a fresh holdout, or run the policy as an
A/B test against treat-all.
