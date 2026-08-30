"""Demand and volume page: when and where rides are requested."""

from __future__ import annotations

import streamlit as st

from app_pages._helpers import (
    chart_card,
    dataframe_card,
    download_button,
    empty_state,
    q,
    section,
)
from rapido import charts

st.title(":material/trending_up: Demand & Volume")
st.caption("Ride volume across time, city and pickup zone.")

filters = st.session_state.get("filters", {})

hourly = q("q_rides_by_hour", filters)
if hourly.empty:
    empty_state()
    st.stop()

peak = hourly.loc[hourly["rides"].idxmax()]
trough = hourly.loc[hourly["rides"].idxmin()]

with st.container(horizontal=True):
    st.metric("Busiest hour", f"{int(peak['hour_of_day']):02d}:00", f"{int(peak['rides']):,} rides", border=True)
    st.metric("Quietest hour", f"{int(trough['hour_of_day']):02d}:00", f"{int(trough['rides']):,} rides", border=True)
    st.metric("Total bookings", f"{int(hourly['rides'].sum()):,}", border=True)

tab_time, tab_places, tab_zones = st.tabs(
    ["By time", "By city & route", "Zone demand"]
)

with tab_time:
    left, right = st.columns(2)
    with left:
        chart_card(charts.line_rides_by_hour(hourly))
    with right:
        chart_card(charts.bar_rides_by_weekday(q("q_rides_by_weekday", filters)))

    chart_card(charts.heatmap_demand(q("q_demand_by_day_hour", filters)))
    chart_card(charts.line_monthly_trend(q("q_monthly_trend", filters)))

with tab_places:
    chart_card(charts.bar_rides_by_city(q("q_rides_by_city", filters)))

    left, right = st.columns(2)
    with left:
        top_zones = q("q_top_pickup_locations", filters, limit=15)
        chart_card(charts.bar_top_locations(top_zones))
    with right:
        routes = q("q_busiest_routes", filters, limit=15)
        dataframe_card(routes, "Busiest Routes", height=520)
        download_button(routes, "busiest_routes.csv")

with tab_zones:
    section(
        "Zone-level demand",
        "Aggregated from location_demand.csv: requests, wait time and surge by zone, "
        "hour and vehicle type.",
    )

    wait = q("q_wait_time_by_hour")
    if wait.empty:
        empty_state()
    else:
        left, right = st.columns(2)
        with left:
            figure = charts.px.line(
                wait,
                x="hour_of_day",
                y="avg_wait_min",
                markers=True,
                title="Average Zone Wait Time by Hour",
                labels={"hour_of_day": "Hour", "avg_wait_min": "Wait (min)"},
            )
            figure.update_traces(line_color="#E8A33D", line_width=2.5)
            chart_card(charts.apply_theme(figure))
        with right:
            chart_card(charts.line_surge_by_hour(q("q_surge_by_hour", filters)))

        dataframe_card(q("q_demand_level_distribution"), "Demand Level Distribution")
        st.caption(
            "The source data contains only Low and Medium demand levels - no High "
            "level is present, so this dimension is effectively binary."
        )
