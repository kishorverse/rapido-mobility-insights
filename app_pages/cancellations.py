"""Cancellation analysis: where, when and why bookings fail."""

from __future__ import annotations

import streamlit as st

from app_pages._helpers import (
    chart_card,
    dataframe_card,
    download_button,
    empty_state,
    format_pct,
    q,
    section,
)
from rapido import charts

st.title(":material/cancel: Cancellation Analysis")
st.caption("Which conditions cause bookings to fail, and by how much.")

filters = st.session_state.get("filters", {})

kpis = q("q_kpi_summary", filters)
if kpis.empty or not kpis["total_bookings"].iloc[0]:
    empty_state()
    st.stop()

row = kpis.iloc[0]
incomplete_rate = 100 * row["incomplete"] / row["total_bookings"]

with st.container(horizontal=True):
    st.metric("Cancellation rate", format_pct(row["cancel_rate"]), border=True)
    st.metric("Incomplete rate", format_pct(incomplete_rate), border=True)
    st.metric("Cancelled bookings", f"{int(row['cancelled']):,}", border=True)
    st.metric("Incomplete bookings", f"{int(row['incomplete']):,}", border=True)

st.divider()

tab_drivers, tab_windows, tab_reasons = st.tabs(
    ["Condition drivers", "Peak windows", "Stated reasons"]
)

with tab_drivers:
    section(
        "What actually drives cancellations",
        "Traffic and weather move the rate sharply. City and vehicle type do not - "
        "a chi-square test returns p = 0.40 and p = 0.70 respectively.",
    )

    left, right = st.columns(2)
    with left:
        chart_card(
            charts.bar_cancellation_rate(
                q("q_cancellation_by_category", filters, category="traffic_level"),
                "traffic_level",
            )
        )
    with right:
        chart_card(
            charts.bar_cancellation_rate(
                q("q_cancellation_by_category", filters, category="weather_condition"),
                "weather_condition",
            )
        )

    left, right = st.columns(2)
    with left:
        chart_card(
            charts.stacked_status_by_category(
                q("q_status_split_by_category", filters, category="traffic_level"),
                "traffic_level",
            )
        )
    with right:
        chart_card(
            charts.stacked_status_by_category(
                q("q_status_split_by_category", filters, category="weather_condition"),
                "weather_condition",
            )
        )

    surge = q("q_cancellation_by_surge", filters)
    chart_card(charts.bar_cancellation_rate(surge, "surge_band"))
    st.info(
        "Cancellations rise from about 5% with no surge to about 35% above 2.0x. "
        "Surge is platform-controlled, which makes it the most actionable lever available.",
        icon=":material/lightbulb:",
    )

    conditions = q("q_fare_by_conditions", filters)
    dataframe_card(conditions, "Traffic x Weather Combinations")

with tab_windows:
    section(
        "Peak cancellation windows",
        "City-hour combinations with at least 50 bookings, ranked by cancellation rate.",
    )
    chart_card(charts.heatmap_cancellation(q("q_cancellation_rate_by_city_hour", filters)))

    windows = q("q_peak_cancellation_windows", filters, limit=15)
    left, right = st.columns([3, 2])
    with left:
        chart_card(charts.bar_peak_cancellation_windows(windows))
    with right:
        dataframe_card(windows, "Worst Windows", height=460)
        download_button(windows, "peak_cancellation_windows.csv")

with tab_reasons:
    section(
        "Stated incomplete-ride reasons",
        "Recorded only for the 8,370 rides that ended Incomplete. Cancelled bookings "
        "carry no reason code in the source data.",
    )
    reasons = q("q_cancellation_reasons", filters)
    left, right = st.columns(2)
    with left:
        chart_card(charts.pie_cancellation_reasons(reasons))
    with right:
        dataframe_card(reasons, "Reason Counts")
        st.caption(
            "Driver Delay dominates at roughly 4,700 of 8,370 incomplete rides, which is "
            "why the driver-risk model targets incompletion specifically."
        )

    section(
        "Customer vs driver accountability",
        "The source records what went wrong, not who is answerable for it. Reasons are "
        "attributed here so the two failure types can be costed separately: no-shows are "
        "a demand-side problem, delays and vehicle issues are a supply-side one.",
    )
    by_party = q("q_cancellation_reasons_by_party", filters)
    left, right = st.columns([3, 2])
    with left:
        chart_card(charts.bar_reasons_by_party(by_party))
    with right:
        chart_card(charts.pie_party_share(by_party))
    download_button(by_party, "reasons_by_party.csv")
