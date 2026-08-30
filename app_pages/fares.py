"""Fare and revenue analysis."""

from __future__ import annotations

import streamlit as st

from app_pages._helpers import (
    chart_card,
    dataframe_card,
    download_button,
    empty_state,
    format_compact,
    format_currency,
    q,
    section,
)
from rapido import charts
from rapido.models import serve

st.title(":material/payments: Fares & Revenue")
st.caption("How fare is formed, how it varies, and where revenue comes from.")

filters = st.session_state.get("filters", {})

kpis = q("q_kpi_summary", filters)
if kpis.empty or not kpis["total_bookings"].iloc[0]:
    empty_state()
    st.stop()

row = kpis.iloc[0]

with st.container(horizontal=True):
    st.metric("Revenue (completed)", format_currency(row["revenue"]), border=True)
    st.metric("Average fare", format_currency(row["avg_fare"]), border=True)
    st.metric("Average distance", f"{row['avg_distance']:.1f} km", border=True)
    st.metric("Average surge", f"{row['avg_surge']:.2f}x", border=True)

st.divider()

tab_structure, tab_variation, tab_revenue = st.tabs(
    ["Fare structure", "Variation", "Revenue mix"]
)

with tab_structure:
    section(
        "The pricing formula",
        "Recovered directly from the data by linear fit, R² = 1.000000 for every vehicle type.",
    )

    with st.container(border=True):
        st.markdown("**base_fare = flagfall + rate per km x distance**")
        st.dataframe(
            {
                "Vehicle": ["Bike", "Auto", "Cab"],
                "Flagfall": ["₹20", "₹40", "₹80"],
                "Per km": ["₹8", "₹12", "₹18"],
                "Fit R²": ["1.000000", "1.000000", "1.000000"],
            },
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            "Final value is then `base_fare x surge_multiplier x (1 ± 5% noise)`. "
            "That ±5% uniform noise is the only irreducible error, which puts a "
            "**2.50% floor on achievable MAPE** - the trained model reaches 2.76%."
        )

    section("Try the tariff")
    left, middle, right = st.columns(3)
    with left:
        vehicle = st.selectbox("Vehicle", ["Bike", "Auto", "Cab"], index=1)
    with middle:
        distance = st.slider("Distance (km)", 1.0, 25.0, 8.0, 0.5)
    with right:
        surge = st.slider("Surge", 1.0, 2.5, 1.0, 0.1)

    base = serve.estimate_base_fare(vehicle, distance)
    with st.container(horizontal=True):
        st.metric("Base fare", format_currency(base), border=True)
        st.metric("Surge applied", f"{surge:.1f}x", border=True)
        st.metric("Expected value", format_currency(base * surge), border=True)

with tab_variation:
    scatter = q("q_distance_vs_fare", filters, sample=4000)
    chart_card(charts.scatter_distance_fare(scatter))
    st.caption(
        "The three clean bands are the three vehicle tariffs; the spread within each "
        "band is surge."
    )

    left, right = st.columns(2)
    with left:
        chart_card(charts.box_fare_by_vehicle(scatter))
    with right:
        chart_card(charts.line_surge_by_hour(q("q_surge_by_hour", filters)))

    fare_table = q("q_fare_by_vehicle_city", filters)
    chart_card(charts.bar_fare_per_km(fare_table))
    dataframe_card(fare_table, "Fare by City and Vehicle Type")
    download_button(fare_table, "fare_by_city_vehicle.csv")

with tab_revenue:
    revenue = q("q_revenue_by_city_vehicle", filters)
    chart_card(charts.treemap_revenue(revenue))
    dataframe_card(revenue, "Completed-Ride Revenue")
    download_button(revenue, "revenue_by_city_vehicle.csv")

    st.info(
        "Revenue here counts completed rides only. Cancelled bookings carry a quoted "
        "value in the source data but never convert.",
        icon=":material/info:",
    )
