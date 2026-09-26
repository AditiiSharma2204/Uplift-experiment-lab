# Memo: targeting the display campaign by predicted uplift

**To:** Marketing / growth leadership   **Re:** Did the campaign work, and whom should we treat?

**Recommendation.** Stop treating everyone. Rank users by predicted *incremental* effect and treat only the top slice. Under
our illustrative costs ($0.005 per treatment, $1.00 per visit) that is the top **19.5%** of users (95% CI [14.0%, 26.5%]). It keeps
most of the campaign's effect for a fraction of the cost: **$6,001** vs **$2,594** profit per 1M users.

**What we found**
1. **The campaign works.** Assignment raised the visit rate by +0.665 pp (95% CI [+0.640, +0.690]), a 17.4% relative
   lift. A naive read of the data says +1.034 pp. That overstates it by about half, because heavy visitors were slightly
   over-represented in the treated group. We corrected for this.
2. **The effect is concentrated.** Treating the 30% of users with the highest predicted uplift gets 95% of all the extra
   visits. The bottom half of users adds nothing measurable, so every dollar spent on them is wasted.
3. **Target by uplift, not by "likely to visit".** Picking people who are likely to visit anyway looks sensible but buys visits
   that would have happened regardless. When treatment is expensive (e.g. $0.05 per treatment at $1.00 per visit), that approach
   earns about $610 per 1M users vs $2,390 for uplift targeting.
4. **When to switch strategy.** If cost per treatment is below ~0.0076 x the value of a visit, treating everyone is still
   profitable, but targeting is better. Above ~0.215 x, no targeting pays for itself.

**Caveats.** Cost and value figures are illustrative placeholders. The data is a public, anonymized, pooled sample, so the lift
is not Criteo's real-world number. We measured short-term visits only. **Next step:** run the targeting policy against
treat-everyone as a live A/B test before rolling it out, and re-estimate costs with finance.
