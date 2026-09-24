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
