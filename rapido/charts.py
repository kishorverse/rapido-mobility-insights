"""Plotly figure builders.

Every figure in the dashboard is produced here, so colours, fonts and layout
conventions are defined once. Pages call these functions and render; they never
build figures inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

BRAND = "#F5C518"
SEQUENTIAL = "YlOrRd"
DIVERGING = "RdYlGn_r"

STATUS_COLOURS = {
    "Completed": "#2E9E5B",
    "Cancelled": "#D64545",
    "Incomplete": "#E8A33D",
}

CATEGORICAL = ["#3B7DD8", "#2E9E5B", "#E8A33D", "#D64545", "#8B5CF6", "#0EA5A5"]


def apply_theme(figure: go.Figure, height: int = 400) -> go.Figure:
    """Apply the shared layout to any figure."""
    figure.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=50, r=20, t=55, b=45),
        title_font_size=16,
        font=dict(size=12),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
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
        ordered["day_of_week"], categories=config.WEEKDAY_ORDER, ordered=True
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
    ).reindex(config.WEEKDAY_ORDER)
    figure = px.imshow(
        pivot,
        color_continuous_scale=SEQUENTIAL,
        aspect="auto",
        title="Demand Heatmap: Day vs Hour",
        labels=dict(x="Hour", y="", color="Bookings"),
    )
    return apply_theme(figure, height=380)


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


#: Colour per accountable party, kept distinct from the status palette.
PARTY_COLOURS = {
    "Customer": "#3B7DD8",
    "Driver": "#D64545",
    "Platform": "#8B5CF6",
    "Unknown": "#94A3B8",
}


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
        labelled["city"].astype(str) + " @ " + labelled["hour_of_day"].astype(str) + ":00"
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


def histogram_ratings(frame: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Distribution of a rating column."""
    if _guard(frame):
        return empty_figure()
    figure = px.histogram(
        frame, x=column, nbins=30, title=title, labels={column: "Rating"}
    )
    figure.update_traces(marker_color="#3B7DD8")
    return apply_theme(figure)


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
        go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc:.3f}", line=dict(color="#3B7DD8", width=2.5))
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            line=dict(color="#999", dash="dash"),
        )
    )
    figure.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return apply_theme(figure)


def pr_curve_fig(
    precision: np.ndarray, recall: np.ndarray, average_precision: float, baseline: float
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
