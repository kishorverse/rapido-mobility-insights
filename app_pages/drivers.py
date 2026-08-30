"""Driver performance, reliability scoring and delay analysis."""

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

st.title(":material/two_wheeler: Driver Analysis")
st.caption("Reliability scoring, delay behaviour and allocation guidance.")

tab_reliability, tab_risk, tab_best = st.tabs(
    ["Reliability", "Delay risk", "Top performers"]
)

with tab_reliability:
    section(
        "Reliability score",
        "0-100, weighting acceptance rate 35%, punctuality 35% and rating 30%. "
        "Higher is better.",
    )

    scatter = q("q_driver_scatter", limit=3000)
    if scatter.empty:
        empty_state()
    else:
        with st.container(horizontal=True):
            st.metric(
                "Mean reliability",
                f"{scatter['driver_reliability_score'].mean():.1f}",
                border=True,
            )
            st.metric(
                "Mean pickup delay",
                f"{scatter['avg_pickup_delay_min'].mean():.1f} min",
                border=True,
            )
            st.metric("Mean rating", f"{scatter['avg_driver_rating'].mean():.2f}", border=True)

        chart_card(charts.scatter_reliability_vs_delay(scatter))
        st.caption(
            "Pickup delay is only loosely related to the composite score, because "
            "acceptance rate and rating carry most of the weight."
        )

with tab_risk:
    section(
        "Drivers with the highest observed delay rate",
        "Ranked on historical delay rate, restricted to drivers with at least five assigned rides.",
    )

    limit = st.slider("Rows", 10, 200, 50, 10, key="driver_rows")
    risky = q("q_unreliable_drivers", limit=limit)

    if risky.empty:
        empty_state()
    else:
        dataframe_card(
            risky,
            "Least Reliable Drivers",
            height=520,
            column_config={
                "delay_rate": st.column_config.ProgressColumn(
                    "Delay rate (%)", min_value=0, max_value=50, format="%.1f%%"
                ),
                "acceptance_rate": st.column_config.ProgressColumn(
                    "Acceptance (%)", min_value=0, max_value=100, format="%.0f%%"
                ),
            },
        )
        download_button(risky, "unreliable_drivers.csv")

        st.info(
            "Driver Delay accounts for roughly 4,700 of the 8,370 incomplete rides - the "
            "largest single reason. Traffic exposure raises the incompletion rate from 5.1% "
            "to 14.8%, so routing support matters more here than driver replacement.",
            icon=":material/lightbulb:",
        )

with tab_best:
    section(
        "Highest-scoring drivers",
        "Candidates for priority allocation during peak-risk windows.",
    )
    best = q("q_top_drivers", limit=50)
    if best.empty:
        empty_state()
    else:
        dataframe_card(
            best,
            "Top Drivers by Reliability Score",
            height=520,
            column_config={
                "reliability_score": st.column_config.ProgressColumn(
                    "Reliability", min_value=0, max_value=100, format="%.1f"
                )
            },
        )
        download_button(best, "top_drivers.csv")

        st.success(
            "**Allocation rule:** during high-traffic windows, route requests to drivers "
            "scoring above the fleet median first. Since cancellation rates are effectively "
            "identical across cities, allocation should key on live traffic rather than location.",
            icon=":material/route:",
        )
