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
