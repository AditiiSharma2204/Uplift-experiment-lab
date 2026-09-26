"""Uplift targeting simulator.

Reads ONLY results/curves.json (precomputed by run_all.py stages s4/s5) - never the raw 14M rows.
Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

CURVES = Path(__file__).resolve().parents[1] / "results" / "curves.json"
PER = 1_000_000
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8d8c86"
LABELS = {"x_learner": "X-learner", "s_learner": "S-learner", "t_learner": "T-learner",
          "response": "Response model", "random": "Random score"}


@st.cache_data
def load() -> dict:
    return json.loads(CURVES.read_text())


st.set_page_config(page_title="Uplift targeting simulator", layout="wide")
data = load()
pol, qini, meta = data["policy"], data["qini"], data.get("meta", {})
best = pol["model"]

st.title("Who should see the ad? Uplift targeting simulator")
st.warning("**Illustrative cost/value assumptions.** Cost per treatment and value per visit are not Criteo figures. "
           "Move the sliders to see how the best policy changes.", icon="⚠️")
st.caption(f"Criteo Uplift v2.1 - held-out test set of {meta.get('n_test', 0):,} users - ranking by the "
           f"{LABELS[best]} (chosen on validation) - propensity-weighted - 95% bootstrap intervals "
           f"({meta.get('n_boot', 200)} resamples). All numbers are per 1,000,000 users.")

# ---------------------------------------------------------------- controls
with st.sidebar:
    st.header("Assumptions (illustrative)")
    v = st.slider("Value per incremental visit ($)", 0.10, 2.00, 1.00, 0.05)
    c = st.slider("Cost per treated user ($)", 0.0, 0.40, 0.005, 0.0005, format="%.4f")
    st.header("Policy")
    k_pct = st.slider("Share of users to treat (top k% by predicted uplift)", 0.5, 100.0, 30.0, 0.5)
    compare = st.checkbox("Compare with ranking by the response model", value=False)

r = c / v
k = np.array(pol["k"])
inc, lo, hi = (np.array(pol[x]) for x in ("incremental_per_1m", "incremental_per_1m_lo", "incremental_per_1m_hi"))
rand = np.array(pol["random_per_1m"])
treated = k * PER
profit, profit_lo, profit_hi = inc * v - treated * c, lo * v - treated * c, hi * v - treated * c

j = int(np.argmin(np.abs(k - k_pct / 100)))
j_opt = int(np.argmax(profit))
rg = np.array(pol["ratio_grid"])
i_r = int(np.argmin(np.abs(rg - min(r, rg[-1]))))
k_opt_ci = (pol["optimal_k_lo"][i_r], pol["optimal_k_hi"][i_r])

# ---------------------------------------------------------------- headline numbers
a, b, cc, d = st.columns(4)
a.metric(f"Incremental visits at top {k_pct:g}%", f"{inc[j]:,.0f}",
         help=f"95% CI {lo[j]:,.0f} to {hi[j]:,.0f}. Random targeting of the same share: {rand[j]:,.0f}.")
a.caption(f"95% CI [{lo[j]:,.0f}, {hi[j]:,.0f}] - random: {rand[j]:,.0f}")
b.metric("Profit at this k", f"${profit[j]:,.0f}")
b.caption(f"95% CI [${profit_lo[j]:,.0f}, ${profit_hi[j]:,.0f}] - cost ${treated[j] * c:,.0f}")
cc.metric("Profit-maximizing k", f"{100 * k[j_opt]:.1f}%")
cc.caption(f"95% CI [{100 * k_opt_ci[0]:.1f}%, {100 * k_opt_ci[1]:.1f}%] - profit ${profit[j_opt]:,.0f}")
d.metric("Treat everyone instead", f"${profit[-1]:,.0f}")
d.caption(f"Treat nobody: $0 - cost/value ratio c/v = {r:.4f}")

if k[j_opt] == 0:
    st.info("At this cost/value ratio no targeting fraction pays for itself: **treat nobody**.")
elif k[j_opt] >= 0.95:
    st.info("Treatment is so cheap relative to a visit that treating (almost) **everyone** is best.")

# ---------------------------------------------------------------- charts
df = pd.DataFrame({"k": 100 * k, "incremental": inc, "lo": lo, "hi": hi, "random": rand,
                   "profit": profit, "profit_lo": profit_lo, "profit_hi": profit_hi})
base = alt.Chart(df).encode(x=alt.X("k:Q", title="Users treated (%)", scale=alt.Scale(domain=[0, 100])))
tip = [alt.Tooltip("k:Q", title="treated %", format=".1f"),
       alt.Tooltip("incremental:Q", title="incremental visits", format=",.0f"),
       alt.Tooltip("lo:Q", title="95% CI low", format=",.0f"), alt.Tooltip("hi:Q", title="95% CI high", format=",.0f"),
       alt.Tooltip("random:Q", title="random targeting", format=",.0f"),
       alt.Tooltip("profit:Q", title="profit $", format=",.0f")]
hover = alt.selection_point(fields=["k"], nearest=True, on="pointerover", empty=False)
rule_k = alt.Chart(pd.DataFrame({"k": [k_pct]})).mark_rule(color=GREY, strokeDash=[4, 3]).encode(x="k:Q")

left, right = st.columns(2)
with left:
    st.subheader("Incremental visits per 1M users")
    band = base.mark_area(opacity=0.18, color=BLUE).encode(y=alt.Y("lo:Q", title="Incremental visits"), y2="hi:Q")
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="incremental:Q")
    rnd = base.mark_line(color=GREY, strokeDash=[6, 4], strokeWidth=1.2).encode(y="random:Q")
    layers = [band, line, rnd]
    if compare:
        rm = qini["models"]["response"]
        rdf = pd.DataFrame({"k": [0.0] + [100 * x for x in qini["k"]], "response": [0.0] + rm["gain"]})
        layers.append(alt.Chart(rdf).mark_line(color=ORANGE, strokeWidth=2).encode(x="k:Q", y="response:Q"))
    points = base.mark_point(size=60, filled=True, color=BLUE).encode(
        y="incremental:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tip).add_params(hover)
    st.altair_chart(alt.layer(*layers, rule_k, points).properties(height=340), use_container_width=True)
    st.caption(f"Blue: rank by {LABELS[best]} (band = 95% CI). Grey dashed: random targeting."
               + (" Orange: rank by response model." if compare else ""))

with right:
    st.subheader("Profit per 1M users (illustrative $)")
    pband = base.mark_area(opacity=0.18, color=BLUE).encode(y=alt.Y("profit_lo:Q", title="Profit ($)"), y2="profit_hi:Q")
    pline = base.mark_line(color=BLUE, strokeWidth=2).encode(y="profit:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GREY).encode(y="y:Q")
    opt = alt.Chart(pd.DataFrame({"k": [100 * k[j_opt]], "profit": [profit[j_opt]]})).mark_point(
        size=110, filled=True, color=BLUE, stroke="white", strokeWidth=2).encode(x="k:Q", y="profit:Q")
    ppoints = base.mark_point(size=60, filled=True, color=BLUE).encode(
        y="profit:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tip).add_params(hover)
    st.altair_chart(alt.layer(pband, pline, zero, rule_k, opt, ppoints).properties(height=340),
                    use_container_width=True)
    st.caption("Large dot: profit-maximizing k. Dashed line: your chosen k. "
               "Profit = incremental visits x value - treated users x cost.")

# ---------------------------------------------------------------- explanations
with st.expander("How to read this"):
    st.markdown(f"""
- **Incremental visits** are visits that happen *because* of the ad: the treated visit rate minus what the same
  users would have done untreated, estimated on a randomized hold-out and weighted by each user's treatment propensity
  (the public dataset pools several tests, so assignment is random only conditional on features).
- Users are ranked by the **{LABELS[best]}**, which predicts each user's *uplift*, not their chance to visit. A
  response model ranks people who would visit anyway at the top; tick the comparison box to see it lag at small k.
- The curve **flattens after ~30-40%**: the rest of the population adds almost no visits, so treating them is pure cost.
- The **profit-maximizing k** depends only on the ratio cost/value. Its 95% interval comes from repeating the choice
  on {meta.get('n_boot', 200)} bootstrap resamples of the test set.
- Limitations: one pooled experiment, anonymized features, rare outcome (wide intervals), short-term visits only - no
  long-term or cross-campaign effects. Cost and value are **illustrative**.
""")
