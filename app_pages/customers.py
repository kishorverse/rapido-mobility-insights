"""Customer behaviour and cancellation-risk segmentation."""

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

st.title(":material/group: Customer Analysis")
st.caption("Demographics, ratings and the customers most likely to cancel.")

filters = st.session_state.get("filters", {})

tab_profile, tab_risk, tab_ratings = st.tabs(
    ["Demographics", "High-risk customers", "Ratings"]
)

with tab_profile:
    demographics = q("q_customer_demographics")
    if demographics.empty:
        empty_state()
    else:
        chart_card(
            charts.grouped_bar_comparison(
                demographics,
                x="age_band",
                y="customers",
                colour="gender",
                title="Customers by Age Band and Gender",
            )
        )
        chart_card(
            charts.grouped_bar_comparison(
                demographics,
                x="age_band",
                y="avg_cancel_rate",
                colour="gender",
                title="Average Cancellation Rate (%) by Segment",
            )
        )
        dataframe_card(demographics, "Customer Segments")
        st.caption(
            "Cancellation rate is broadly flat across demographic segments - consistent "
            "with the finding that trip conditions, not customer identity, drive outcomes."
        )

with tab_risk:
    section(
        "Customers with the highest observed cancellation rate",
        "Ranked on historical rate, restricted to customers with at least five bookings.",
    )

    limit = st.slider("Rows", 10, 200, 50, 10)
    risky = q("q_high_risk_customers", limit=limit)

    if risky.empty:
        empty_state()
    else:
        with st.container(horizontal=True):
            st.metric("Listed customers", f"{len(risky):,}", border=True)
            st.metric("Mean cancel rate", f"{risky['cancel_rate'].mean():.1f}%", border=True)
            st.metric("Mean rating", f"{risky['rating'].mean():.2f}", border=True)

        dataframe_card(
            risky,
            "High-Risk Customers",
            height=520,
            column_config={
                "cancel_rate": st.column_config.ProgressColumn(
                    "Cancel rate (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
                "rating": st.column_config.NumberColumn("Rating", format="%.1f"),
            },
        )
        download_button(risky, "high_risk_customers.csv")

        st.warning(
            "This table reflects **observed history**, not a prediction. For a forward-looking "
            "score on a specific booking, use the Live Prediction page - the risk model reaches "
            "ROC-AUC 0.851 using only information available before the trip starts.",
            icon=":material/warning:",
        )

with tab_ratings:
    ratings = q("q_customer_vs_driver_ratings")
    if ratings.empty:
        empty_state()
    else:
        chart_card(
            charts.grouped_bar_comparison(
                ratings,
                x="rating",
                y="people",
                colour="party",
                title="Rating Distribution: Customers vs Drivers",
            )
        )
        dataframe_card(ratings.pivot_table(
            index="rating", columns="party", values="people", fill_value=0
        ).reset_index(), "Rating Counts")
