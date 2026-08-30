"""Overview page: headline KPIs and the project's key findings."""

from __future__ import annotations

import streamlit as st

from app_pages._helpers import (
    chart_card,
    empty_state,
    format_compact,
    format_currency,
    format_pct,
    metric_row,
    q,
    section,
)
from rapido import charts

st.title(":material/dashboard: Rapido Mobility Insights")
st.caption(
    "Ride patterns, cancellation drivers and fare forecasting across five cities, "
    "100,000 bookings, calendar year 2025."
)

filters = st.session_state.get("filters", {})
kpis = q("q_kpi_summary", filters)

if kpis.empty or not kpis["total_bookings"].iloc[0]:
    empty_state()
    st.stop()

row = kpis.iloc[0]

metric_row(
    [
        ("Total bookings", format_compact(row["total_bookings"]), None),
        ("Completion rate", format_pct(row["completion_rate"]), None),
        ("Cancellation rate", format_pct(row["cancel_rate"]), None),
        ("Revenue (completed)", format_currency(row["revenue"]), None),
    ]
)
metric_row(
    [
        ("Average fare", format_currency(row["avg_fare"]), None),
        ("Average distance", f"{row['avg_distance']:.1f} km", None),
        ("Average surge", f"{row['avg_surge']:.2f}x", None),
        ("Active customers", format_compact(row["active_customers"]), None),
    ]
)

st.divider()

left, right = st.columns(2)
with left:
    chart_card(charts.line_rides_by_hour(q("q_rides_by_hour", filters)))
with right:
    chart_card(charts.bar_rides_by_city(q("q_rides_by_city", filters)))

chart_card(charts.line_monthly_trend(q("q_monthly_trend", filters)))

st.divider()
section(
    "What the data says",
    "Findings verified in the analysis, each reproducible from the code in this repository.",
)

findings = [
    (
        ":material/traffic: Traffic and weather drive cancellations - not geography",
        "High traffic lifts cancellations to **33.5%** against ~18% otherwise, and heavy rain to "
        "**33.7%** against 10.0% in clear conditions. City and vehicle type are **not statistically "
        "significant** (chi-square p = 0.40 and p = 0.70). Cancellation rate varies only between "
        "22.95% and 23.78% across the five cities.",
    ),
    (
        ":material/bolt: Surge pricing is the strongest single lever",
        "Cancellations climb from **5.3%** at no surge to **35.3%** above 2.0x - a near sevenfold "
        "increase. Surge is set by the platform, which makes this the most actionable finding here.",
    ),
    (
        ":material/rainy: Weather affects cancellations but not incompletions",
        "Incomplete-ride rates stay flat at 8.3-8.4% across all weather conditions, while traffic "
        "moves them from 5.1% to **14.8%**. Riders abandon bookings in bad weather; drivers fail to "
        "complete them in bad traffic. These are two different operational problems.",
    ),
    (
        ":material/functions: Fare follows a fixed tariff",
        "`booking_value = base_fare x surge x (1 +/- 5%)`, and base fare is exactly "
        "**flagfall + rate x distance** (Bike ₹20 + ₹8/km, Auto ₹40 + ₹12/km, Cab ₹80 + ₹18/km, "
        "R² = 1.000000). The fare model reaches 2.76% MAPE against a **2.50% theoretical floor** set "
        "by the noise term - it is effectively optimal.",
    ),
]

for title, body in findings:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)

st.divider()
section("Recommended operational actions")

actions = [
    (
        "Cap surge during adverse conditions",
        "The heavy-rain and high-traffic combination is where surge and cancellation risk compound. "
        "Capping the multiplier in those windows attacks the single largest driver directly.",
    ),
    (
        "Pre-position drivers on traffic, not on city",
        "Since cancellation rates are effectively equal across cities, allocation should be driven "
        "by live traffic and demand-pressure signals per zone rather than by city-level targets.",
    ),
    (
        "Flag high-risk bookings at request time",
        "The cancellation model reaches ROC-AUC 0.851. Use the Live Prediction page to score a "
        "booking before dispatch and hold the driver assignment for high-risk requests.",
    ),
    (
        "Separate the two failure modes",
        "Weather-driven cancellations need rider-side intervention (fare guarantees, wait-time "
        "transparency). Traffic-driven incompletions need driver-side routing support.",
    ),
]

for title, body in actions:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)
