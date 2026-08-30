"""Rapido Intelligent Mobility Insights - Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from rapido import db  # noqa: E402


def configure_page() -> None:
    """Apply global page configuration."""
    st.set_page_config(
        page_title="Rapido Mobility Insights",
        page_icon=":material/local_taxi:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data(ttl="1h", show_spinner=False)
def _filter_options() -> dict:
    """Load sidebar filter values from the database."""
    from rapido import queries

    return queries.q_filter_options()


@st.cache_data(ttl="5m", show_spinner=False)
def _connection_ok() -> tuple[bool, str]:
    """Check database connectivity once per five minutes."""
    try:
        status = db.healthcheck()
        if not status.get("connected"):
            return False, status.get("error", "Unknown connection error.")
        if not status["tables"].get("bookings"):
            return False, "The bookings table is empty. Run scripts/run_etl.py."
        return True, f"MySQL {status.get('server_version', '')}"
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        return False, str(exc)


def sidebar_filters() -> dict:
    """Render the shared sidebar filters and return the active selection."""
    options = _filter_options()
    cities = options.get("cities") or config.CITIES
    vehicles = options.get("vehicle_types") or config.VEHICLE_TYPES

    with st.sidebar:
        st.markdown("### :material/tune: Filters")

        selected_cities = st.multiselect("City", cities, default=[], placeholder="All cities")
        selected_vehicles = st.multiselect(
            "Vehicle type", vehicles, default=[], placeholder="All vehicles"
        )
        selected_traffic = st.multiselect(
            "Traffic level", config.TRAFFIC_LEVELS, default=[], placeholder="All levels"
        )
        selected_weather = st.multiselect(
            "Weather", config.WEATHER_CONDITIONS, default=[], placeholder="All conditions"
        )

        date_min = options.get("date_min")
        date_max = options.get("date_max")
        default_start = date_min.date() if hasattr(date_min, "date") else date(2025, 1, 1)
        default_end = date_max.date() if hasattr(date_max, "date") else date(2025, 12, 31)

        date_range = st.date_input(
            "Booking date range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
        )

        hour_from, hour_to = st.slider("Hour of day", 0, 23, (0, 23))

        if st.button("Reset filters", icon=":material/restart_alt:", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        connected, detail = _connection_ok()
        if connected:
            st.caption(f":material/database: Connected - {detail}")
        else:
            st.error(f"Database unavailable: {detail}", icon=":material/error:")

    date_from, date_to = (
        date_range if isinstance(date_range, tuple) and len(date_range) == 2
        else (default_start, default_end)
    )

    return {
        "cities": selected_cities,
        "vehicle_types": selected_vehicles,
        "traffic_levels": selected_traffic,
        "weather_conditions": selected_weather,
        "date_from": date_from,
        "date_to": date_to,
        "hour_from": hour_from,
        "hour_to": hour_to,
    }


def main() -> None:
    """Compose navigation and run the selected page."""
    configure_page()

    navigation = st.navigation(
        {
            "Analytics": [
                st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:"),
                st.Page("app_pages/demand.py", title="Demand & Volume", icon=":material/trending_up:"),
                st.Page(
                    "app_pages/cancellations.py",
                    title="Cancellations",
                    icon=":material/cancel:",
                ),
                st.Page("app_pages/fares.py", title="Fares & Revenue", icon=":material/payments:"),
                st.Page("app_pages/customers.py", title="Customers", icon=":material/group:"),
                st.Page("app_pages/drivers.py", title="Drivers", icon=":material/two_wheeler:"),
                st.Page("app_pages/explorer.py", title="Data Explorer", icon=":material/table:"),
            ],
            "Machine Learning": [
                st.Page("app_pages/model_lab.py", title="Model Lab", icon=":material/science:"),
                st.Page("app_pages/predict.py", title="Live Prediction", icon=":material/bolt:"),
            ],
        },
        position="sidebar",
    )

    st.session_state["filters"] = sidebar_filters()
    navigation.run()


main()
