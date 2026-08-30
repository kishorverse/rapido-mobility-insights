"""Paginated booking explorer.

Demonstrates the project guideline of never loading the full dataset into the
browser: pagination happens in SQL through LIMIT/OFFSET.
"""

from __future__ import annotations

import streamlit as st

from app_pages._helpers import download_button, empty_state, q, section
from rapido import queries

st.title(":material/table: Data Explorer")
st.caption("Browse individual bookings. Records are paged server-side, 50 rows per request.")

filters = st.session_state.get("filters", {})

kpis = q("q_kpi_summary", filters)
total = int(kpis["total_bookings"].iloc[0]) if not kpis.empty else 0

if not total:
    empty_state()
    st.stop()

page_size = 50
total_pages = max(1, -(-total // page_size))

section(f"{total:,} bookings match the current filters")

left, right = st.columns([3, 1])
with right:
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )
with left:
    st.caption(f"Page {page} of {total_pages:,}")

records = queries.q_bookings_page(filters, page=int(page), page_size=page_size)

if records.empty:
    empty_state()
else:
    st.dataframe(
        records,
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "booking_value": st.column_config.NumberColumn("Fare", format="₹%.2f"),
            "ride_distance_km": st.column_config.NumberColumn("Distance", format="%.2f km"),
            "surge_multiplier": st.column_config.NumberColumn("Surge", format="%.1fx"),
            "booking_ts": st.column_config.DatetimeColumn("Booked at"),
        },
    )
    download_button(records, f"bookings_page_{int(page)}.csv", "Download this page")

st.caption(
    "Only the visible page is fetched from MySQL. The `idx_bookings_ts` index serves "
    "the ORDER BY, keeping paging fast across all 100,000 rows."
)
