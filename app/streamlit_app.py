"""Rapido Intelligent Mobility Insights - Streamlit dashboard.

The whole presentation layer in one file: theme and Plotly figure builders,
caching and formatting helpers, the shared sidebar filters, and the nine pages
themselves. It holds no business logic - every number comes from ``src/``.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import data_preprocessing as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import predict as pr  # noqa: E402
import train_models as tm  # noqa: E402


st.set_page_config(
    page_title="Rapido Mobility Insights",
    page_icon=":material/local_taxi:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================== #
# Theme and figure builders
# =========================================================================== #

BRAND = "#F5C518"
SEQUENTIAL = "YlOrRd"
DIVERGING = "RdYlGn_r"

STATUS_COLOURS = {
    "Completed": "#2E9E5B",
    "Cancelled": "#D64545",
    "Incomplete": "#E8A33D",
}

CATEGORICAL = ["#3B7DD8", "#2E9E5B", "#E8A33D", "#D64545", "#8B5CF6", "#0EA5A5"]

#: Colour per accountable party, kept distinct from the status palette.
PARTY_COLOURS = {
    "Customer": "#3B7DD8",
    "Driver": "#D64545",
    "Platform": "#8B5CF6",
    "Unknown": "#94A3B8",
}


def apply_theme(figure: go.Figure, height: int = 400) -> go.Figure:
    """Apply the shared layout to any figure."""
    figure.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=20, t=55, b=45),
        title_font_size=16,
        font=dict(size=12),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def empty_figure(message: str = "No data for the selected filters") -> go.Figure:
    """Return a placeholder figure for empty result sets."""
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, font=dict(size=14, color="#888")
    )
    figure.update_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
    )
    return figure


def _guard(frame: pd.DataFrame) -> bool:
    """Return True when the frame has nothing to plot."""
    return frame is None or frame.empty


# --------------------------------------------------------------------------- #
# Volume and demand
# --------------------------------------------------------------------------- #


def line_rides_by_hour(frame: pd.DataFrame) -> go.Figure:
    """Ride volume across the 24-hour clock."""
    if _guard(frame):
        return empty_figure()
    figure = px.line(
        frame,
        x="hour_of_day",
        y="rides",
        markers=True,
        title="Ride Volume by Hour of Day",
        labels={"hour_of_day": "Hour", "rides": "Bookings"},
    )
    figure.update_traces(line_color="#3B7DD8", line_width=2.5)
    figure.update_xaxes(dtick=2)
    return apply_theme(figure)


def bar_rides_by_weekday(frame: pd.DataFrame) -> go.Figure:
    """Ride volume by day of week, in calendar order."""
    if _guard(frame):
        return empty_figure()
    ordered = frame.copy()
    ordered["day_of_week"] = pd.Categorical(
        ordered["day_of_week"], categories=dp.WEEKDAY_ORDER, ordered=True
    )
    ordered = ordered.sort_values("day_of_week")
    figure = px.bar(
        ordered,
        x="day_of_week",
        y="rides",
        title="Ride Volume by Day of Week",
        labels={"day_of_week": "Day", "rides": "Bookings"},
    )
    figure.update_traces(marker_color="#3B7DD8")
    return apply_theme(figure)


def bar_rides_by_city(frame: pd.DataFrame) -> go.Figure:
    """Ride volume by city."""
    if _guard(frame):
        return empty_figure()
    figure = px.bar(
        frame.sort_values("rides", ascending=True),
        x="rides",
        y="city",
        orientation="h",
        title="Ride Volume by City",
        labels={"rides": "Bookings", "city": ""},
    )
    figure.update_traces(marker_color="#3B7DD8")
    return apply_theme(figure)


def line_monthly_trend(frame: pd.DataFrame) -> go.Figure:
    """Monthly booking volume and revenue on twin axes."""
    if _guard(frame):
        return empty_figure()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["month_label"],
            y=frame["rides"],
            name="Bookings",
            line=dict(color="#3B7DD8", width=2.5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["month_label"],
            y=frame["revenue"],
            name="Revenue",
            yaxis="y2",
            line=dict(color="#2E9E5B", width=2.5, dash="dot"),
        )
    )
    figure.update_layout(
        title="Monthly Bookings and Revenue",
        yaxis=dict(title="Bookings"),
        yaxis2=dict(title="Revenue", overlaying="y", side="right", showgrid=False),
    )
    return apply_theme(figure)


def heatmap_demand(frame: pd.DataFrame) -> go.Figure:
    """Hour-by-weekday demand heatmap."""
    if _guard(frame):
        return empty_figure()
    pivot = frame.pivot_table(
        index="day_of_week", columns="hour_of_day", values="rides", aggfunc="sum"
    ).reindex(dp.WEEKDAY_ORDER)
    figure = px.imshow(
        pivot,
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        title="Demand Heatmap: Day vs Hour",
        labels=dict(x="Hour", y="", color="Bookings"),
    )
    return apply_theme(figure, height=380)


def line_wait_time_by_hour(frame: pd.DataFrame) -> go.Figure:
    """Average zone wait time across the day."""
    if _guard(frame):
        return empty_figure()
    figure = px.line(
        frame,
        x="hour_of_day",
        y="avg_wait_min",
        markers=True,
        title="Average Zone Wait Time by Hour",
        labels={"hour_of_day": "Hour", "avg_wait_min": "Wait (min)"},
    )
    figure.update_traces(line_color="#E8A33D", line_width=2.5)
    return apply_theme(figure)


# --------------------------------------------------------------------------- #
# Cancellations
# --------------------------------------------------------------------------- #


def heatmap_cancellation(frame: pd.DataFrame) -> go.Figure:
    """City-by-hour cancellation-rate heatmap."""
    if _guard(frame):
        return empty_figure()
    pivot = frame.pivot_table(
        index="city", columns="hour_of_day", values="cancel_rate", aggfunc="mean"
    )
    figure = px.imshow(
        pivot,
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        title="Cancellation Rate (%) by City and Hour",
        labels=dict(x="Hour", y="", color="Cancel %"),
    )
    return apply_theme(figure, height=340)


def stacked_status_by_category(frame: pd.DataFrame, category: str) -> go.Figure:
    """Outcome mix as a 100% stacked bar across a categorical driver."""
    if _guard(frame):
        return empty_figure()
    figure = px.bar(
        frame,
        x=category,
        y="share_pct",
        color="booking_status",
        title=f"Booking Outcome Mix by {category.replace('_', ' ').title()}",
        labels={"share_pct": "Share (%)", category: ""},
        color_discrete_map=STATUS_COLOURS,
    )
    figure.update_layout(barmode="stack")
    return apply_theme(figure)


def bar_cancellation_rate(frame: pd.DataFrame, category: str) -> go.Figure:
    """Cancellation rate by a categorical driver, sorted worst first."""
    if _guard(frame):
        return empty_figure()
    ordered = frame.sort_values("cancel_rate", ascending=True)
    figure = px.bar(
        ordered,
        x="cancel_rate",
        y=ordered[category].astype(str),
        orientation="h",
        title=f"Cancellation Rate by {category.replace('_', ' ').title()}",
        labels={"cancel_rate": "Cancellation Rate (%)", "y": ""},
        color="cancel_rate",
        color_continuous_scale=SEQUENTIAL,
    )
    figure.update_layout(coloraxis_showscale=False, yaxis_title="")
    return apply_theme(figure)


def pie_cancellation_reasons(frame: pd.DataFrame) -> go.Figure:
    """Breakdown of stated incomplete-ride reasons."""
    if _guard(frame):
        return empty_figure()
    figure = px.pie(
        frame,
        names="incomplete_ride_reason",
        values="rides",
        title="Incomplete Ride Reasons",
        hole=0.45,
        color_discrete_sequence=CATEGORICAL,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(figure)


def bar_reasons_by_party(frame: pd.DataFrame) -> go.Figure:
    """Stated reasons coloured by the party accountable for them.

    Separating customer-caused from driver-caused failures matters because the
    two need different interventions: no-show penalties versus driver coaching.
    """
    if _guard(frame):
        return empty_figure()
    figure = px.bar(
        frame.sort_values("rides"),
        x="rides",
        y="incomplete_ride_reason",
        color="responsible_party",
        orientation="h",
        title="Incomplete Ride Reasons by Accountable Party",
        labels={
            "rides": "Rides",
            "incomplete_ride_reason": "",
            "responsible_party": "Party",
        },
        color_discrete_map=PARTY_COLOURS,
        text="share_pct",
    )
    figure.update_traces(texttemplate="%{text}%", textposition="outside")
    return apply_theme(figure, height=380)


def pie_party_share(frame: pd.DataFrame) -> go.Figure:
    """Share of incomplete rides attributable to each party."""
    if _guard(frame):
        return empty_figure()
    grouped = frame.groupby("responsible_party", as_index=False)["rides"].sum()
    figure = px.pie(
        grouped,
        names="responsible_party",
        values="rides",
        title="Accountability Split",
        hole=0.45,
        color="responsible_party",
        color_discrete_map=PARTY_COLOURS,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(figure, height=380)


def bar_peak_cancellation_windows(frame: pd.DataFrame) -> go.Figure:
    """The worst city-hour windows by cancellation rate."""
    if _guard(frame):
        return empty_figure()
    labelled = frame.copy()
    labelled["window"] = (
        labelled["city"].astype(str)
        + " @ "
        + labelled["hour_of_day"].astype(str)
        + ":00"
    )
    labelled = labelled.sort_values("cancel_rate", ascending=True)
    figure = px.bar(
        labelled,
        x="cancel_rate",
        y="window",
        orientation="h",
        title="Peak Cancellation Windows",
        labels={"cancel_rate": "Cancellation Rate (%)", "window": ""},
        color="cancel_rate",
        color_continuous_scale=SEQUENTIAL,
    )
    figure.update_layout(coloraxis_showscale=False)
    return apply_theme(figure, height=460)


# --------------------------------------------------------------------------- #
# Fares
# --------------------------------------------------------------------------- #


def scatter_distance_fare(frame: pd.DataFrame) -> go.Figure:
    """Distance against fare, coloured by vehicle type."""
    if _guard(frame):
        return empty_figure()
    figure = px.scatter(
        frame,
        x="ride_distance_km",
        y="booking_value",
        color="vehicle_type",
        opacity=0.45,
        title="Ride Distance vs Booking Value",
        labels={
            "ride_distance_km": "Distance (km)",
            "booking_value": "Fare",
            "vehicle_type": "Vehicle",
        },
        color_discrete_sequence=CATEGORICAL,
        trendline="ols" if len(frame) <= 5000 else None,
    )
    figure.update_traces(marker=dict(size=5))
    return apply_theme(figure, height=440)


def box_fare_by_vehicle(frame: pd.DataFrame) -> go.Figure:
    """Fare distribution per vehicle type."""
    if _guard(frame):
        return empty_figure()
    figure = px.box(
        frame,
        x="vehicle_type",
        y="booking_value",
        color="vehicle_type",
        title="Fare Distribution by Vehicle Type",
        labels={"booking_value": "Fare", "vehicle_type": ""},
        color_discrete_sequence=CATEGORICAL,
    )
    figure.update_layout(showlegend=False)
    return apply_theme(figure)


def line_surge_by_hour(frame: pd.DataFrame) -> go.Figure:
    """Average surge multiplier across the day."""
    if _guard(frame):
        return empty_figure()
    figure = px.line(
        frame,
        x="hour_of_day",
        y="avg_surge",
        markers=True,
        title="Average Surge Multiplier by Hour",
        labels={"hour_of_day": "Hour", "avg_surge": "Surge"},
    )
    figure.update_traces(line_color="#D64545", line_width=2.5)
    figure.update_xaxes(dtick=2)
    return apply_theme(figure)


def treemap_revenue(frame: pd.DataFrame) -> go.Figure:
    """Revenue split by city and vehicle type."""
    if _guard(frame):
        return empty_figure()
    figure = px.treemap(
        frame,
        path=["city", "vehicle_type"],
        values="revenue",
        title="Revenue by City and Vehicle Type",
        color="revenue",
        color_continuous_scale="Blues",
    )
    return apply_theme(figure, height=440)


def bar_fare_per_km(frame: pd.DataFrame) -> go.Figure:
    """Average fare per kilometre by vehicle type and city."""
    if _guard(frame):
        return empty_figure()
    figure = px.bar(
        frame,
        x="city",
        y="avg_fare_per_km",
        color="vehicle_type",
        barmode="group",
        title="Average Fare per Kilometre",
        labels={"avg_fare_per_km": "Fare / km", "city": ""},
        color_discrete_sequence=CATEGORICAL,
    )
    return apply_theme(figure)


# --------------------------------------------------------------------------- #
# People
# --------------------------------------------------------------------------- #


def bar_top_locations(frame: pd.DataFrame) -> go.Figure:
    """Busiest pickup zones."""
    if _guard(frame):
        return empty_figure()
    ordered = frame.sort_values("rides", ascending=True)
    figure = px.bar(
        ordered,
        x="rides",
        y="zone",
        orientation="h",
        title="Busiest Pickup Zones",
        labels={"rides": "Bookings", "zone": ""},
    )
    figure.update_traces(marker_color="#3B7DD8")
    return apply_theme(figure, height=520)


def scatter_reliability_vs_delay(frame: pd.DataFrame) -> go.Figure:
    """Driver reliability score against average pickup delay."""
    if _guard(frame):
        return empty_figure()
    figure = px.scatter(
        frame,
        x="driver_reliability_score",
        y="avg_pickup_delay_min",
        color="avg_driver_rating",
        opacity=0.6,
        title="Driver Reliability vs Pickup Delay",
        labels={
            "driver_reliability_score": "Reliability Score",
            "avg_pickup_delay_min": "Avg Pickup Delay (min)",
            "avg_driver_rating": "Rating",
        },
        color_continuous_scale="Viridis",
    )
    return apply_theme(figure, height=440)


def grouped_bar_comparison(
    frame: pd.DataFrame, x: str, y: str, colour: str, title: str
) -> go.Figure:
    """Generic grouped bar for side-by-side comparisons."""
    if _guard(frame):
        return empty_figure()
    figure = px.bar(
        frame,
        x=x,
        y=y,
        color=colour,
        barmode="group",
        title=title,
        color_discrete_sequence=CATEGORICAL,
    )
    return apply_theme(figure)


# --------------------------------------------------------------------------- #
# Model diagnostics
# --------------------------------------------------------------------------- #


def confusion_matrix_fig(matrix: np.ndarray, labels: list[str]) -> go.Figure:
    """Annotated confusion matrix."""
    figure = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    return apply_theme(figure, height=420)


def roc_curve_fig(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """ROC curve with the random-classifier reference line."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            name=f"AUC = {auc:.3f}",
            line=dict(color="#3B7DD8", width=2.5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="Random", line=dict(color="#999", dash="dash")
        )
    )
    figure.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return apply_theme(figure)


def pr_curve_fig(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: float,
    baseline: float,
) -> go.Figure:
    """Precision-recall curve with the prevalence baseline."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            name=f"AP = {average_precision:.3f}",
            line=dict(color="#2E9E5B", width=2.5),
        )
    )
    figure.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="#999",
        annotation_text=f"Baseline = {baseline:.3f}",
    )
    figure.update_layout(
        title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision"
    )
    return apply_theme(figure)


def feature_importance_fig(frame: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar of the strongest features."""
    if _guard(frame):
        return empty_figure()
    top = frame.head(top_n).sort_values("importance", ascending=True)
    figure = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(top))} Features",
        labels={"importance": "Importance", "feature": ""},
    )
    figure.update_traces(marker_color="#3B7DD8")
    return apply_theme(figure, height=max(380, 22 * len(top)))


def residual_plot_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Predicted values against residuals."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    figure = px.scatter(
        x=y_pred,
        y=residuals,
        opacity=0.35,
        title="Residuals vs Predicted",
        labels={"x": "Predicted Fare", "y": "Residual"},
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#D64545")
    figure.update_traces(marker=dict(size=4, color="#3B7DD8"))
    return apply_theme(figure, height=400)


def actual_vs_predicted_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Actual against predicted values with the identity line."""
    figure = px.scatter(
        x=y_true,
        y=y_pred,
        opacity=0.35,
        title="Actual vs Predicted Fare",
        labels={"x": "Actual", "y": "Predicted"},
    )
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect",
            line=dict(color="#D64545", dash="dash"),
        )
    )
    figure.update_traces(marker=dict(size=4))
    return apply_theme(figure, height=400)


def gauge_risk(probability: float, title: str) -> go.Figure:
    """Risk gauge for a single prediction."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * probability,
            number={"suffix": "%"},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#333", "thickness": 0.25},
                "steps": [
                    {"range": [0, 30], "color": "#2E9E5B"},
                    {"range": [30, 60], "color": "#E8A33D"},
                    {"range": [60, 100], "color": "#D64545"},
                ],
            },
        )
    )
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return figure


# =========================================================================== #
# Data access and formatting helpers
# =========================================================================== #

CACHE_TTL = "10m"


def _cache_key(filters: dict | None) -> tuple:
    """Turn a filter dict into a hashable cache key."""
    if not filters:
        return ()
    return tuple(
        sorted(
            (key, tuple(value) if isinstance(value, list) else str(value))
            for key, value in filters.items()
            if value not in (None, [], "")
        )
    )


@st.cache_data(ttl=CACHE_TTL, max_entries=200, show_spinner=False)
def run_query(name: str, key: tuple, _filters: dict | None = None, **kwargs):
    """Run a named query from :mod:`feature_engineering` with caching.

    Args:
        name: Function name in ``feature_engineering``.
        key: Hashable filter signature; participates in the cache key.
        _filters: The real filter dict, excluded from hashing by the underscore.
        **kwargs: Extra query arguments.
    """
    function = getattr(fe, name, None)
    if function is None:
        raise ValueError(f"Unknown query {name!r}.")
    try:
        return (
            function(_filters, **kwargs) if _filters is not None else function(**kwargs)
        )
    except TypeError:
        return function(**kwargs)


def q(name: str, filters: dict | None = None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: run a cached query for the active filters."""
    return run_query(name, _cache_key(filters), _filters=filters, **kwargs)


@st.cache_resource(show_spinner=False)
def cached_model(name: str):
    """Load a trained model once per session."""
    return tm.load_model(name)


@st.cache_data(ttl="1h", show_spinner=False)
def cached_metrics() -> dict:
    """Load the stored model metrics."""
    return tm.load_metrics()


@st.cache_data(ttl="1h", show_spinner=False)
def filter_options() -> dict:
    """Fetch distinct filter values from the database."""
    return fe.q_filter_options()


@st.cache_data(ttl="5m", show_spinner=False)
def connection_ok() -> tuple[bool, str]:
    """Check database connectivity once per five minutes."""
    try:
        status = dp.healthcheck()
        if not status.get("connected"):
            return False, status.get("error", "Unknown connection error.")
        if not status["tables"].get("bookings"):
            return False, (
                "The bookings table is empty. "
                "Run: python src/data_preprocessing.py etl"
            )
        return True, f"MySQL {status.get('server_version', '')}"
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        return False, str(exc)


def format_currency(value: float | None) -> str:
    """Format a number as Indian rupees with a thousands separator."""
    if value is None or pd.isna(value):
        return "-"
    return f"₹{value:,.0f}"


def format_compact(value: float | None) -> str:
    """Format a large number compactly (K / L / Cr)."""
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1e7:
        return f"{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"{value / 1e5:.2f} L"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} K"
    return f"{value:,.0f}"


def format_pct(value: float | None, decimals: int = 1) -> str:
    """Format a number already expressed as a percentage."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}%"


def section(title: str, description: str | None = None) -> None:
    """Render a section heading with optional caption."""
    st.subheader(title)
    if description:
        st.caption(description)


def empty_state(message: str = "No data matches the current filters.") -> None:
    """Render a consistent empty-state notice."""
    st.info(message, icon=":material/filter_alt_off:")


def chart_card(figure, title: str | None = None) -> None:
    """Render a Plotly figure inside a bordered card."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        st.plotly_chart(figure, width="stretch")


def dataframe_card(
    frame: pd.DataFrame, title: str | None = None, height: int | None = None, **kwargs
) -> None:
    """Render a DataFrame inside a bordered card."""
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        if frame is None or frame.empty:
            empty_state()
            return
        # height must be omitted entirely when unset; None is not a valid value.
        if height is not None:
            kwargs["height"] = height
        st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def download_button(
    frame: pd.DataFrame, filename: str, label: str = "Download CSV"
) -> None:
    """Offer a DataFrame as a CSV download."""
    if frame is None or frame.empty:
        return
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
    )


def model_missing_notice(name: str) -> None:
    """Explain how to produce a missing model artefact."""
    st.warning(
        f"The **{name}** model has not been trained yet. "
        "Run `python src/train_models.py` from the project root.",
        icon=":material/model_training:",
    )


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of bordered metric cards."""
    with st.container(horizontal=True):
        for label, value, delta in metrics:
            st.metric(label, value, delta, border=True)


def active_filters() -> dict:
    """Return the filter selection made in the sidebar."""
    return st.session_state.get("filters", {})


# =========================================================================== #
# Sidebar
# =========================================================================== #


def sidebar_filters() -> dict:
    """Render the shared sidebar filters and return the active selection."""
    options = filter_options()
    cities = options.get("cities") or dp.CITIES
    vehicles = options.get("vehicle_types") or dp.VEHICLE_TYPES

    with st.sidebar:
        st.markdown("### :material/tune: Filters")

        selected_cities = st.multiselect(
            "City", cities, default=[], placeholder="All cities"
        )
        selected_vehicles = st.multiselect(
            "Vehicle type", vehicles, default=[], placeholder="All vehicles"
        )
        selected_traffic = st.multiselect(
            "Traffic level", dp.TRAFFIC_LEVELS, default=[], placeholder="All levels"
        )
        selected_weather = st.multiselect(
            "Weather",
            dp.WEATHER_CONDITIONS,
            default=[],
            placeholder="All conditions",
        )

        date_min = options.get("date_min")
        date_max = options.get("date_max")
        default_start = (
            date_min.date() if hasattr(date_min, "date") else date(2025, 1, 1)
        )
        default_end = date_max.date() if hasattr(date_max, "date") else date(2025, 12, 31)

        date_range = st.date_input(
            "Booking date range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
        )

        hour_from, hour_to = st.slider("Hour of day", 0, 23, (0, 23))

        if st.button(
            "Reset filters", icon=":material/restart_alt:", width="stretch"
        ):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        connected, detail = connection_ok()
        if connected:
            st.caption(f":material/database: Connected - {detail}")
        else:
            st.error(f"Database unavailable: {detail}", icon=":material/error:")

    date_from, date_to = (
        date_range
        if isinstance(date_range, tuple) and len(date_range) == 2
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


# =========================================================================== #
# Page: Overview
# =========================================================================== #


def page_overview() -> None:
    """Headline KPIs and the project's key findings."""
    st.title(":material/dashboard: Rapido Mobility Insights")
    st.caption(
        "Ride patterns, cancellation drivers and fare forecasting across five "
        "cities, 100,000 bookings, calendar year 2025."
    )

    filters = active_filters()
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
        chart_card(line_rides_by_hour(q("q_rides_by_hour", filters)))
    with right:
        chart_card(bar_rides_by_city(q("q_rides_by_city", filters)))

    chart_card(line_monthly_trend(q("q_monthly_trend", filters)))

    st.divider()
    section(
        "What the data says",
        "Findings verified in the analysis, each reproducible from the code in "
        "this repository.",
    )

    findings = [
        (
            ":material/traffic: Traffic and weather drive cancellations - not geography",
            "High traffic lifts cancellations to **33.5%** against ~18% otherwise, and "
            "heavy rain to **33.7%** against 10.0% in clear conditions. City and vehicle "
            "type are **not statistically significant** (chi-square p = 0.40 and p = 0.70). "
            "Cancellation rate varies only between 22.95% and 23.78% across the five cities.",
        ),
        (
            ":material/bolt: Surge pricing is the strongest single lever",
            "Cancellations climb from **5.3%** at no surge to **35.3%** above 2.0x - a "
            "near sevenfold increase. Surge is set by the platform, which makes this the "
            "most actionable finding here.",
        ),
        (
            ":material/rainy: Weather affects cancellations but not incompletions",
            "Incomplete-ride rates stay flat at 8.3-8.4% across all weather conditions, "
            "while traffic moves them from 5.1% to **14.8%**. Riders abandon bookings in "
            "bad weather; drivers fail to complete them in bad traffic. These are two "
            "different operational problems.",
        ),
        (
            ":material/functions: Fare follows a fixed tariff",
            "`booking_value = base_fare x surge x (1 +/- 5%)`, and base fare is exactly "
            "**flagfall + rate x distance** (Bike ₹20 + ₹8/km, Auto ₹40 + ₹12/km, "
            "Cab ₹80 + ₹18/km, R² = 1.000000). The fare model reaches 2.76% MAPE against "
            "a **2.50% theoretical floor** set by the noise term - it is effectively optimal.",
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
            "The heavy-rain and high-traffic combination is where surge and cancellation "
            "risk compound. Capping the multiplier in those windows attacks the single "
            "largest driver directly.",
        ),
        (
            "Pre-position drivers on traffic, not on city",
            "Since cancellation rates are effectively equal across cities, allocation "
            "should be driven by live traffic and demand-pressure signals per zone rather "
            "than by city-level targets.",
        ),
        (
            "Flag high-risk bookings at request time",
            "The cancellation model reaches ROC-AUC 0.851. Use the Live Prediction page to "
            "score a booking before dispatch and hold the driver assignment for high-risk "
            "requests.",
        ),
        (
            "Separate the two failure modes",
            "Weather-driven cancellations need rider-side intervention (fare guarantees, "
            "wait-time transparency). Traffic-driven incompletions need driver-side "
            "routing support.",
        ),
    ]

    for title, body in actions:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)


# =========================================================================== #
# Page: Demand and volume
# =========================================================================== #


def page_demand() -> None:
    """When and where rides are requested."""
    st.title(":material/trending_up: Demand & Volume")
    st.caption("Ride volume across time, city and pickup zone.")

    filters = active_filters()

    hourly = q("q_rides_by_hour", filters)
    if hourly.empty:
        empty_state()
        st.stop()

    peak = hourly.loc[hourly["rides"].idxmax()]
    trough = hourly.loc[hourly["rides"].idxmin()]

    with st.container(horizontal=True):
        st.metric(
            "Busiest hour",
            f"{int(peak['hour_of_day']):02d}:00",
            f"{int(peak['rides']):,} rides",
            border=True,
        )
        st.metric(
            "Quietest hour",
            f"{int(trough['hour_of_day']):02d}:00",
            f"{int(trough['rides']):,} rides",
            border=True,
        )
        st.metric("Total bookings", f"{int(hourly['rides'].sum()):,}", border=True)

    tab_time, tab_places, tab_zones = st.tabs(
        ["By time", "By city & route", "Zone demand"]
    )

    with tab_time:
        left, right = st.columns(2)
        with left:
            chart_card(line_rides_by_hour(hourly))
        with right:
            chart_card(bar_rides_by_weekday(q("q_rides_by_weekday", filters)))

        chart_card(heatmap_demand(q("q_demand_by_day_hour", filters)))
        chart_card(line_monthly_trend(q("q_monthly_trend", filters)))

    with tab_places:
        chart_card(bar_rides_by_city(q("q_rides_by_city", filters)))

        left, right = st.columns(2)
        with left:
            top_zones = q("q_top_pickup_locations", filters, limit=15)
            chart_card(bar_top_locations(top_zones))
        with right:
            routes = q("q_busiest_routes", filters, limit=15)
            dataframe_card(routes, "Busiest Routes", height=520)
            download_button(routes, "busiest_routes.csv")

    with tab_zones:
        section(
            "Zone-level demand",
            "Aggregated from location_demand.csv: requests, wait time and surge by "
            "zone, hour and vehicle type.",
        )

        wait = q("q_wait_time_by_hour")
        if wait.empty:
            empty_state()
        else:
            left, right = st.columns(2)
            with left:
                chart_card(line_wait_time_by_hour(wait))
            with right:
                chart_card(line_surge_by_hour(q("q_surge_by_hour", filters)))

            dataframe_card(
                q("q_demand_level_distribution"), "Demand Level Distribution"
            )
            st.caption(
                "The source data contains only Low and Medium demand levels - no High "
                "level is present, so this dimension is effectively binary."
            )


# =========================================================================== #
# Page: Cancellations
# =========================================================================== #


def page_cancellations() -> None:
    """Where, when and why bookings fail."""
    st.title(":material/cancel: Cancellation Analysis")
    st.caption("Which conditions cause bookings to fail, and by how much.")

    filters = active_filters()

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
                bar_cancellation_rate(
                    q("q_cancellation_by_category", filters, category="traffic_level"),
                    "traffic_level",
                )
            )
        with right:
            chart_card(
                bar_cancellation_rate(
                    q(
                        "q_cancellation_by_category",
                        filters,
                        category="weather_condition",
                    ),
                    "weather_condition",
                )
            )

        left, right = st.columns(2)
        with left:
            chart_card(
                stacked_status_by_category(
                    q("q_status_split_by_category", filters, category="traffic_level"),
                    "traffic_level",
                )
            )
        with right:
            chart_card(
                stacked_status_by_category(
                    q(
                        "q_status_split_by_category",
                        filters,
                        category="weather_condition",
                    ),
                    "weather_condition",
                )
            )

        surge = q("q_cancellation_by_surge", filters)
        chart_card(bar_cancellation_rate(surge, "surge_band"))
        st.info(
            "Cancellations rise from about 5% with no surge to about 35% above 2.0x. "
            "Surge is platform-controlled, which makes it the most actionable lever "
            "available.",
            icon=":material/lightbulb:",
        )

        dataframe_card(
            q("q_fare_by_conditions", filters), "Traffic x Weather Combinations"
        )

    with tab_windows:
        section(
            "Peak cancellation windows",
            "City-hour combinations with at least 50 bookings, ranked by cancellation "
            "rate.",
        )
        chart_card(
            heatmap_cancellation(q("q_cancellation_rate_by_city_hour", filters))
        )

        windows = q("q_peak_cancellation_windows", filters, limit=15)
        left, right = st.columns([3, 2])
        with left:
            chart_card(bar_peak_cancellation_windows(windows))
        with right:
            dataframe_card(windows, "Worst Windows", height=460)
            download_button(windows, "peak_cancellation_windows.csv")

    with tab_reasons:
        section(
            "Stated incomplete-ride reasons",
            "Recorded only for the 8,370 rides that ended Incomplete. Cancelled "
            "bookings carry no reason code in the source data.",
        )
        reasons = q("q_cancellation_reasons", filters)
        left, right = st.columns(2)
        with left:
            chart_card(pie_cancellation_reasons(reasons))
        with right:
            dataframe_card(reasons, "Reason Counts")
            st.caption(
                "Driver Delay dominates at roughly 4,700 of 8,370 incomplete rides, "
                "which is why the driver-risk model targets incompletion specifically."
            )

        section(
            "Customer vs driver accountability",
            "The source records what went wrong, not who is answerable for it. Reasons "
            "are attributed here so the two failure types can be costed separately: "
            "no-shows are a demand-side problem, delays and vehicle issues are a "
            "supply-side one.",
        )
        by_party = q("q_cancellation_reasons_by_party", filters)
        left, right = st.columns([3, 2])
        with left:
            chart_card(bar_reasons_by_party(by_party))
        with right:
            chart_card(pie_party_share(by_party))
        download_button(by_party, "reasons_by_party.csv")


# =========================================================================== #
# Page: Fares and revenue
# =========================================================================== #


def page_fares() -> None:
    """How fare is formed, how it varies, and where revenue comes from."""
    st.title(":material/payments: Fares & Revenue")
    st.caption("How fare is formed, how it varies, and where revenue comes from.")

    filters = active_filters()

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
            "Recovered directly from the data by linear fit, R² = 1.000000 for every "
            "vehicle type.",
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

        base = pr.estimate_base_fare(vehicle, distance)
        with st.container(horizontal=True):
            st.metric("Base fare", format_currency(base), border=True)
            st.metric("Surge applied", f"{surge:.1f}x", border=True)
            st.metric("Expected value", format_currency(base * surge), border=True)

    with tab_variation:
        scatter = q("q_distance_vs_fare", filters, sample=4000)
        chart_card(scatter_distance_fare(scatter))
        st.caption(
            "The three clean bands are the three vehicle tariffs; the spread within "
            "each band is surge."
        )

        left, right = st.columns(2)
        with left:
            chart_card(box_fare_by_vehicle(scatter))
        with right:
            chart_card(line_surge_by_hour(q("q_surge_by_hour", filters)))

        fare_table = q("q_fare_by_vehicle_city", filters)
        chart_card(bar_fare_per_km(fare_table))
        dataframe_card(fare_table, "Fare by City and Vehicle Type")
        download_button(fare_table, "fare_by_city_vehicle.csv")

    with tab_revenue:
        revenue = q("q_revenue_by_city_vehicle", filters)
        chart_card(treemap_revenue(revenue))
        dataframe_card(revenue, "Completed-Ride Revenue")
        download_button(revenue, "revenue_by_city_vehicle.csv")

        st.info(
            "Revenue here counts completed rides only. Cancelled bookings carry a "
            "quoted value in the source data but never convert.",
            icon=":material/info:",
        )


# =========================================================================== #
# Page: Customers
# =========================================================================== #


def page_customers() -> None:
    """Demographics, ratings and the customers most likely to cancel."""
    st.title(":material/group: Customer Analysis")
    st.caption("Demographics, ratings and the customers most likely to cancel.")

    tab_profile, tab_risk, tab_ratings = st.tabs(
        ["Demographics", "High-risk customers", "Ratings"]
    )

    with tab_profile:
        demographics = q("q_customer_demographics")
        if demographics.empty:
            empty_state()
        else:
            chart_card(
                grouped_bar_comparison(
                    demographics,
                    x="age_band",
                    y="customers",
                    colour="gender",
                    title="Customers by Age Band and Gender",
                )
            )
            chart_card(
                grouped_bar_comparison(
                    demographics,
                    x="age_band",
                    y="avg_cancel_rate",
                    colour="gender",
                    title="Average Cancellation Rate (%) by Segment",
                )
            )
            dataframe_card(demographics, "Customer Segments")
            st.caption(
                "Cancellation rate is broadly flat across demographic segments - "
                "consistent with the finding that trip conditions, not customer "
                "identity, drive outcomes."
            )

    with tab_risk:
        section(
            "Customers with the highest observed cancellation rate",
            "Ranked on historical rate, restricted to customers with at least five "
            "bookings.",
        )

        limit = st.slider("Rows", 10, 200, 50, 10)
        risky = q("q_high_risk_customers", limit=limit)

        if risky.empty:
            empty_state()
        else:
            with st.container(horizontal=True):
                st.metric("Listed customers", f"{len(risky):,}", border=True)
                st.metric(
                    "Mean cancel rate",
                    f"{risky['cancel_rate'].mean():.1f}%",
                    border=True,
                )
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
                "This table reflects **observed history**, not a prediction. For a "
                "forward-looking score on a specific booking, use the Live Prediction "
                "page - the risk model reaches ROC-AUC 0.851 using only information "
                "available before the trip starts.",
                icon=":material/warning:",
            )

    with tab_ratings:
        ratings = q("q_customer_vs_driver_ratings")
        if ratings.empty:
            empty_state()
        else:
            chart_card(
                grouped_bar_comparison(
                    ratings,
                    x="rating",
                    y="people",
                    colour="party",
                    title="Rating Distribution: Customers vs Drivers",
                )
            )
            dataframe_card(
                ratings.pivot_table(
                    index="rating", columns="party", values="people", fill_value=0
                ).reset_index(),
                "Rating Counts",
            )


# =========================================================================== #
# Page: Drivers
# =========================================================================== #


def page_drivers() -> None:
    """Reliability scoring, delay behaviour and allocation guidance."""
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
                st.metric(
                    "Mean rating",
                    f"{scatter['avg_driver_rating'].mean():.2f}",
                    border=True,
                )

            chart_card(scatter_reliability_vs_delay(scatter))
            st.caption(
                "Pickup delay is only loosely related to the composite score, because "
                "acceptance rate and rating carry most of the weight."
            )

    with tab_risk:
        section(
            "Drivers with the highest observed delay rate",
            "Ranked on historical delay rate, restricted to drivers with at least five "
            "assigned rides.",
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
                "Driver Delay accounts for roughly 4,700 of the 8,370 incomplete rides "
                "- the largest single reason. Traffic exposure raises the incompletion "
                "rate from 5.1% to 14.8%, so routing support matters more here than "
                "driver replacement.",
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
                "**Allocation rule:** during high-traffic windows, route requests to "
                "drivers scoring above the fleet median first. Since cancellation rates "
                "are effectively identical across cities, allocation should key on live "
                "traffic rather than location.",
                icon=":material/route:",
            )


# =========================================================================== #
# Page: Data explorer
# =========================================================================== #


def page_explorer() -> None:
    """Paginated booking explorer.

    Demonstrates the project guideline of never loading the full dataset into
    the browser: pagination happens in SQL through LIMIT/OFFSET.
    """
    st.title(":material/table: Data Explorer")
    st.caption(
        "Browse individual bookings. Records are paged server-side, 50 rows per "
        "request."
    )

    filters = active_filters()

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

    records = fe.q_bookings_page(filters, page=int(page), page_size=page_size)

    if records.empty:
        empty_state()
    else:
        st.dataframe(
            records,
            hide_index=True,
            width="stretch",
            height=560,
            column_config={
                "booking_value": st.column_config.NumberColumn(
                    "Fare", format="₹%.2f"
                ),
                "ride_distance_km": st.column_config.NumberColumn(
                    "Distance", format="%.2f km"
                ),
                "surge_multiplier": st.column_config.NumberColumn(
                    "Surge", format="%.1fx"
                ),
                "booking_ts": st.column_config.DatetimeColumn("Booked at"),
            },
        )
        download_button(
            records, f"bookings_page_{int(page)}.csv", "Download this page"
        )

    st.caption(
        "Only the visible page is fetched from MySQL. The `idx_bookings_ts` index "
        "serves the ORDER BY, keeping paging fast across all 100,000 rows."
    )


# =========================================================================== #
# Page: Model Lab
# =========================================================================== #

MODEL_TABS = {
    "Ride Outcome": ("outcome", "classification"),
    "Fare Prediction": ("fare", "regression"),
    "Cancellation Risk": ("customer_risk", "classification"),
    "Driver Delay Risk": ("driver_risk", "classification"),
}


@st.cache_data(ttl="1h", show_spinner="Scoring held-out test set...")
def _test_evaluation(model_key: str) -> dict:
    """Rebuild the held-out split and score the persisted model on it."""
    frame = dp.load_model_data()
    features, target = tm.build_dataset(frame, model_key)
    stratify = model_key != "fare"
    _, x_test, _, y_test = tm.split_train_test(features, target, stratify=stratify)

    pipeline, _ = cached_model(dp.MODEL_NAMES[model_key])
    predictions = pipeline.predict(x_test)

    result = {"y_test": np.asarray(y_test), "predictions": np.asarray(predictions)}
    if hasattr(pipeline, "predict_proba"):
        result["probabilities"] = pipeline.predict_proba(x_test)
        result["classes"] = list(pipeline.named_steps["model"].classes_)
    return result


def page_model_lab() -> None:
    """Leaderboards, evaluation metrics and feature importance."""
    st.title(":material/science: Model Lab")
    st.caption("How the four models were built, what they score, and what drives them.")

    metrics_store = cached_metrics()
    available = tm.list_models()

    if available.empty:
        model_missing_notice("prediction")
        st.stop()

    st.subheader("Model portfolio")
    overview_rows = []
    for label, (key, task) in MODEL_TABS.items():
        stored = metrics_store.get(dp.MODEL_NAMES[key], {})
        model_metrics = stored.get("metrics", {})
        overview_rows.append(
            {
                "Model": label,
                "Algorithm": stored.get("algorithm", "-"),
                "Task": task,
                "Headline metric": (
                    f"F1-macro {model_metrics.get('f1_macro')}"
                    if task == "classification"
                    else f"R² {model_metrics.get('r2')}"
                ),
                "Secondary": (
                    "ROC-AUC "
                    f"{model_metrics.get('roc_auc', model_metrics.get('roc_auc_ovr', '-'))}"
                    if task == "classification"
                    else f"MAPE {model_metrics.get('mape_pct')}%"
                ),
            }
        )
    dataframe_card(pd.DataFrame(overview_rows), "Trained Models")

    st.divider()

    tabs = st.tabs(list(MODEL_TABS))

    for tab, (label, (model_key, task)) in zip(tabs, MODEL_TABS.items()):
        with tab:
            artefact_name = dp.MODEL_NAMES[model_key]
            if not tm.model_exists(artefact_name):
                model_missing_notice(label)
                continue

            _, metadata = cached_model(artefact_name)
            model_metrics = metadata.get("metrics", {})

            st.markdown(f"**{metadata.get('description', label)}**")
            st.caption(
                f"Algorithm: `{metadata.get('algorithm', '-')}` · "
                f"train {metadata.get('n_train', 0):,} / "
                f"test {metadata.get('n_test', 0):,} rows · "
                f"trained {metadata.get('saved_at', '-')}"
            )

            if task == "classification":
                with st.container(horizontal=True):
                    st.metric(
                        "Accuracy", model_metrics.get("accuracy", "-"), border=True
                    )
                    st.metric(
                        "F1 (macro)", model_metrics.get("f1_macro", "-"), border=True
                    )
                    st.metric(
                        "Balanced accuracy",
                        model_metrics.get("balanced_accuracy", "-"),
                        border=True,
                    )
                    st.metric(
                        "ROC-AUC",
                        model_metrics.get(
                            "roc_auc", model_metrics.get("roc_auc_ovr", "-")
                        ),
                        border=True,
                    )
            else:
                with st.container(horizontal=True):
                    st.metric("R²", model_metrics.get("r2", "-"), border=True)
                    st.metric("RMSE", model_metrics.get("rmse", "-"), border=True)
                    st.metric("MAE", model_metrics.get("mae", "-"), border=True)
                    st.metric(
                        "MAPE", f"{model_metrics.get('mape_pct', '-')}%", border=True
                    )

            leaderboard = metadata.get("leaderboard")
            if leaderboard is not None:
                frame = (
                    pd.DataFrame(leaderboard)
                    if not isinstance(leaderboard, pd.DataFrame)
                    else leaderboard
                )
                dataframe_card(frame, "Candidate Leaderboard (held-out test set)")
                st.caption(
                    "`dummy` is the baseline - majority class for classification, mean "
                    "for regression. Every model must beat it to justify its complexity."
                )

            cv = metadata.get("cross_validation", {})
            if cv:
                with st.container(border=True):
                    st.markdown("**5-fold cross-validation on the training split**")
                    st.write(
                        f"{cv.get('metric')}: **{cv.get('mean')}** ± {cv.get('std')} "
                        f"across folds {cv.get('folds')}"
                    )
                    st.caption(
                        "A small standard deviation here means the held-out score is "
                        "stable, not a lucky split."
                    )

            evaluation = _test_evaluation(model_key)

            if task == "classification":
                labels = sorted(pd.Series(evaluation["y_test"]).unique())
                matrix = tm.confusion_matrix_df(
                    evaluation["y_test"], evaluation["predictions"], labels
                )
                left, right = st.columns(2)
                with left:
                    chart_card(
                        confusion_matrix_fig(
                            matrix.to_numpy(), [str(x) for x in labels]
                        )
                    )
                with right:
                    per_class = tm.per_class_metrics(
                        evaluation["y_test"], evaluation["predictions"], labels
                    )
                    dataframe_card(per_class, "Per-Class Metrics")

                if len(labels) == 2 and "probabilities" in evaluation:
                    fpr, tpr, auc = tm.roc_data(
                        evaluation["y_test"], evaluation["probabilities"]
                    )
                    precision, recall, ap, baseline = tm.pr_data(
                        evaluation["y_test"], evaluation["probabilities"]
                    )
                    left, right = st.columns(2)
                    with left:
                        chart_card(roc_curve_fig(fpr, tpr, auc))
                    with right:
                        chart_card(pr_curve_fig(precision, recall, ap, baseline))

                    thresholds = tm.threshold_table(
                        evaluation["y_test"], evaluation["probabilities"]
                    )
                    dataframe_card(thresholds, "Decision Threshold Trade-off")
                    st.caption(
                        "Operations picks the threshold: a lower cut-off catches more "
                        "failures but flags more bookings for intervention."
                    )
            else:
                left, right = st.columns(2)
                with left:
                    chart_card(
                        actual_vs_predicted_fig(
                            evaluation["y_test"][:4000],
                            evaluation["predictions"][:4000],
                        )
                    )
                with right:
                    chart_card(
                        residual_plot_fig(
                            evaluation["y_test"][:4000],
                            evaluation["predictions"][:4000],
                        )
                    )

                with st.container(border=True):
                    st.markdown("**Against the project benchmark**")
                    st.write(
                        "Predictions within ±10% of actual fare: "
                        f"**{100 * model_metrics.get('within_10_pct', 0):.2f}%**"
                    )
                    st.caption(
                        "The brief targets RMSE within ±10% of actual fare. Because "
                        "fare is a deterministic tariff plus ±5% uniform noise, the "
                        "theoretical minimum MAPE is 2.50%; this model reaches "
                        f"{model_metrics.get('mape_pct')}%, effectively the noise floor."
                    )

            importance = metadata.get("top_features")
            if importance is not None:
                frame = (
                    pd.DataFrame(importance)
                    if not isinstance(importance, pd.DataFrame)
                    else importance
                )
                if not frame.empty:
                    chart_card(feature_importance_fig(frame, top_n=15))
                    st.caption(
                        "Permutation importance measured on the held-out set, computed "
                        "over the original columns rather than one-hot fragments."
                    )

    st.divider()
    section("Leakage control", "Why these scores are trustworthy.")

    with st.container(border=True):
        st.markdown(
            """
Three columns in this dataset would produce spectacular, meaningless scores:

1. **`actual_ride_time_min`** is null for *every* non-Completed ride. Its null indicator alone
   reproduces the target perfectly. Blocked from all four models.
2. **`base_fare`** reproduces `booking_value` to within 5% once multiplied by surge. Blocked from
   the fare model; the deployed model predicts from distance, vehicle, conditions and surge only.
3. **`cancellation_rate` and `delay_rate`** in the dimension tables are whole-period aggregates
   that already include the booking being predicted, and the shipped `*_flag` columns are simply
   those rates thresholded. Replaced with expanding **prior-history** features computed over
   strictly earlier bookings.

`src/train_models.py` enforces this with `assert_no_leakage()`, which raises rather than warns.
            """
        )

    ablation = metrics_store.get("fare_leakage_ablation", {})
    if ablation:
        ablation_metrics = ablation.get("metrics", {})
        honest = (
            metrics_store.get(dp.MODEL_NAMES["fare"], {})
            .get("metrics", {})
            .get("r2")
        )
        with st.container(border=True):
            st.markdown("**Fare leakage ablation**")
            st.write(
                "Refitting the fare model *with* `base_fare` gives R² "
                f"**{ablation_metrics.get('r2')}** versus **{honest}** without it."
            )
            st.caption(
                "The gap is negligible, which confirms the tariff is already "
                "recoverable from distance and vehicle type - `base_fare` carries no "
                "independent information."
            )


# =========================================================================== #
# Page: Live prediction
# =========================================================================== #


def page_predict() -> None:
    """Score a hypothetical booking against all four models."""
    st.title(":material/bolt: Live Prediction")
    st.caption(
        "Describe a booking as it would look at request time. Every model below sees "
        "only information available before the trip starts."
    )

    availability = pr.available_models()
    if not any(availability.values()):
        model_missing_notice("prediction")
        st.stop()

    with st.form("booking_form"):
        st.markdown("**Trip details**")
        row_one = st.columns(3)
        with row_one[0]:
            city = st.selectbox("City", dp.CITIES, index=4)
        with row_one[1]:
            vehicle_type = st.selectbox("Vehicle type", dp.VEHICLE_TYPES, index=2)
        with row_one[2]:
            hour_of_day = st.slider("Hour of day", 0, 23, 18)

        row_two = st.columns(3)
        with row_two[0]:
            ride_distance_km = st.slider("Distance (km)", 1.0, 25.0, 10.0, 0.5)
        with row_two[1]:
            estimated_ride_time_min = st.slider(
                "Estimated time (min)", 3.0, 165.0, 35.0, 1.0
            )
        with row_two[2]:
            surge_multiplier = st.slider("Surge multiplier", 1.0, 2.3, 1.8, 0.1)

        st.markdown("**Conditions**")
        row_three = st.columns(3)
        with row_three[0]:
            traffic_level = st.segmented_control(
                "Traffic", dp.TRAFFIC_LEVELS, default="High", key="traffic"
            )
        with row_three[1]:
            weather_condition = st.segmented_control(
                "Weather", dp.WEATHER_CONDITIONS, default="Heavy Rain", key="weather"
            )
        with row_three[2]:
            is_weekend = st.toggle("Weekend booking", value=False)

        with st.expander("Customer and driver history (optional)"):
            history = st.columns(4)
            with history[0]:
                cust_prior_rides = st.number_input("Customer prior rides", 0, 50, 5)
            with history[1]:
                cust_prior_cancel_rate = st.slider(
                    "Customer prior cancel rate", 0.0, 1.0, 0.2, 0.05
                )
            with history[2]:
                drv_prior_rides = st.number_input("Driver prior rides", 0, 50, 10)
            with history[3]:
                drv_prior_incomplete_rate = st.slider(
                    "Driver prior incomplete rate", 0.0, 1.0, 0.08, 0.02
                )

            profile = st.columns(3)
            with profile[0]:
                avg_customer_rating = st.slider("Customer rating", 1.0, 5.0, 4.0, 0.1)
            with profile[1]:
                avg_driver_rating = st.slider("Driver rating", 1.0, 5.0, 4.2, 0.1)
            with profile[2]:
                acceptance_rate = st.slider(
                    "Driver acceptance rate", 0.0, 1.0, 0.8, 0.05
                )

        submitted = st.form_submit_button(
            "Run all models",
            icon=":material/play_arrow:",
            width="stretch",
            type="primary",
        )

    if not submitted:
        st.info(
            "Set the trip details above and run the models. The defaults describe a "
            "long evening Cab ride in heavy rain and high traffic at 1.8x surge - a "
            "high-risk booking.",
            icon=":material/info:",
        )
        st.stop()

    inputs = {
        "city": city,
        "vehicle_type": vehicle_type,
        "hour_of_day": hour_of_day,
        "ride_distance_km": ride_distance_km,
        "estimated_ride_time_min": estimated_ride_time_min,
        "surge_multiplier": surge_multiplier,
        "traffic_level": traffic_level or "Medium",
        "weather_condition": weather_condition or "Clear",
        "is_weekend": int(is_weekend),
        "cust_prior_rides": cust_prior_rides,
        "cust_prior_cancel_rate": cust_prior_cancel_rate,
        "cust_prior_completion_rate": 1 - cust_prior_cancel_rate,
        "drv_prior_rides": drv_prior_rides,
        "drv_prior_incomplete_rate": drv_prior_incomplete_rate,
        "avg_customer_rating": avg_customer_rating,
        "avg_driver_rating": avg_driver_rating,
        "acceptance_rate": acceptance_rate,
    }

    st.divider()

    # ----- 1. Ride outcome -------------------------------------------------- #

    section(
        "1. Ride outcome",
        "Multi-class prediction: Completed, Cancelled or Incomplete.",
    )

    if not availability["outcome"]:
        model_missing_notice("ride outcome")
    else:
        result = pr.predict_outcome(inputs)
        probabilities = result["probabilities"]
        icon = {
            "Completed": ":material/check_circle:",
            "Cancelled": ":material/cancel:",
            "Incomplete": ":material/error:",
        }.get(result["prediction"], ":material/help:")

        with st.container(border=True):
            st.markdown(f"### {icon} {result['prediction']}")
            st.caption(f"Confidence {100 * result['confidence']:.1f}%")

            bars = st.columns(len(probabilities))
            for column, (label, value) in zip(bars, probabilities.items()):
                with column:
                    st.metric(label, f"{100 * value:.1f}%", border=True)
                    st.progress(float(value))

    # ----- 2. Fare ---------------------------------------------------------- #

    section("2. Fare estimate", "Predicted before confirmation, without using base_fare.")

    if not availability["fare"]:
        model_missing_notice("fare prediction")
    else:
        fare = pr.predict_fare(inputs)
        with st.container(horizontal=True):
            st.metric(
                "Predicted fare", format_currency(fare["predicted_fare"]), border=True
            )
            st.metric(
                "Expected range",
                f"{format_currency(fare['lower_bound'])} - "
                f"{format_currency(fare['upper_bound'])}",
                border=True,
            )
            st.metric(
                "Tariff formula value",
                format_currency(fare["formula_fare"]),
                border=True,
            )

        st.caption(
            f"The tariff figure applies the recovered pricing rule for a {vehicle_type} "
            f"({pr.TARIFF[vehicle_type]['flagfall']:.0f} + "
            f"{pr.TARIFF[vehicle_type]['per_km']:.0f} per km) multiplied by surge. "
            "The model reaches this independently, from trip context alone."
        )

    # ----- 3. Risk ---------------------------------------------------------- #

    section("3. Risk scores", "Binary probabilities with an operational risk band.")

    customer = driver = None
    risk_left, risk_right = st.columns(2)

    with risk_left:
        if not availability["customer_risk"]:
            model_missing_notice("cancellation risk")
        else:
            customer = pr.predict_customer_risk(inputs)
            with st.container(border=True):
                st.plotly_chart(
                    gauge_risk(customer["probability"], "Cancellation risk"),
                    width="stretch",
                )
                st.metric("Risk level", customer["risk_level"], border=True)

    with risk_right:
        if not availability["driver_risk"]:
            model_missing_notice("driver delay risk")
        else:
            driver = pr.predict_driver_risk(inputs)
            with st.container(border=True):
                st.plotly_chart(
                    gauge_risk(driver["probability"], "Delay / incomplete risk"),
                    width="stretch",
                )
                st.metric("Risk level", driver["risk_level"], border=True)

    # ----- 4. Recommendation ------------------------------------------------ #

    if customer is not None and driver is not None:
        st.divider()
        section("Recommended action")

        cancel_probability = customer["probability"]
        delay_probability = driver["probability"]

        if cancel_probability >= 0.6:
            st.error(
                "**High cancellation risk.** Hold driver assignment until the rider "
                "confirms, and consider capping surge for this request - surge is the "
                "strongest single driver of cancellation in this dataset.",
                icon=":material/priority_high:",
            )
        elif cancel_probability >= 0.3:
            st.warning(
                "**Moderate cancellation risk.** Show an accurate ETA up front and "
                "avoid raising surge further on this request.",
                icon=":material/warning:",
            )
        else:
            st.success(
                "**Low cancellation risk.** Dispatch normally.",
                icon=":material/check_circle:",
            )

        if delay_probability >= 0.3:
            st.warning(
                "**Elevated delay risk.** Assign a driver scoring above the fleet "
                "median on reliability, and provide routing support - traffic exposure "
                "raises the incompletion rate from 5.1% to 14.8%.",
                icon=":material/route:",
            )

        with st.expander("What the models were given"):
            st.caption(
                "Fields you did not set are filled with dataset medians and modes. "
                "Engineered flags such as rush hour, distance band and adverse "
                "conditions are recomputed from your inputs so the row stays "
                "internally consistent."
            )
            st.dataframe(
                pd.DataFrame(
                    [{"field": key, "value": value} for key, value in inputs.items()]
                ),
                hide_index=True,
                width="stretch",
            )


# =========================================================================== #
# Navigation
# =========================================================================== #


def main() -> None:
    """Compose navigation and run the selected page."""
    # url_path is set explicitly on every page: without it Streamlit derives the
    # slug from the function name, giving URLs like /page_model_lab.
    navigation = st.navigation(
        {
            "Analytics": [
                st.Page(
                    page_overview,
                    title="Overview",
                    icon=":material/dashboard:",
                    default=True,
                ),
                st.Page(
                    page_demand,
                    title="Demand & Volume",
                    icon=":material/trending_up:",
                    url_path="demand",
                ),
                st.Page(
                    page_cancellations,
                    title="Cancellations",
                    icon=":material/cancel:",
                    url_path="cancellations",
                ),
                st.Page(
                    page_fares,
                    title="Fares & Revenue",
                    icon=":material/payments:",
                    url_path="fares",
                ),
                st.Page(
                    page_customers,
                    title="Customers",
                    icon=":material/group:",
                    url_path="customers",
                ),
                st.Page(
                    page_drivers,
                    title="Drivers",
                    icon=":material/two_wheeler:",
                    url_path="drivers",
                ),
                st.Page(
                    page_explorer,
                    title="Data Explorer",
                    icon=":material/table:",
                    url_path="explorer",
                ),
            ],
            "Machine Learning": [
                st.Page(
                    page_model_lab,
                    title="Model Lab",
                    icon=":material/science:",
                    url_path="model-lab",
                ),
                st.Page(
                    page_predict,
                    title="Live Prediction",
                    icon=":material/bolt:",
                    url_path="predict",
                ),
            ],
        },
        position="sidebar",
    )

    st.session_state["filters"] = sidebar_filters()
    navigation.run()


main()
