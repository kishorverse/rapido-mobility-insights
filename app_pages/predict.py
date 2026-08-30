"""Live prediction page: score a hypothetical booking against all four models."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from app_pages._helpers import format_currency, model_missing_notice, section
from rapido import charts
from rapido.models import serve

st.title(":material/bolt: Live Prediction")
st.caption(
    "Describe a booking as it would look at request time. Every model below sees only "
    "information available before the trip starts."
)

availability = serve.available_models()
if not any(availability.values()):
    model_missing_notice("prediction")
    st.stop()

with st.form("booking_form"):
    st.markdown("**Trip details**")
    row_one = st.columns(3)
    with row_one[0]:
        city = st.selectbox("City", config.CITIES, index=4)
    with row_one[1]:
        vehicle_type = st.selectbox("Vehicle type", config.VEHICLE_TYPES, index=2)
    with row_one[2]:
        hour_of_day = st.slider("Hour of day", 0, 23, 18)

    row_two = st.columns(3)
    with row_two[0]:
        ride_distance_km = st.slider("Distance (km)", 1.0, 25.0, 10.0, 0.5)
    with row_two[1]:
        estimated_ride_time_min = st.slider("Estimated time (min)", 3.0, 165.0, 35.0, 1.0)
    with row_two[2]:
        surge_multiplier = st.slider("Surge multiplier", 1.0, 2.3, 1.8, 0.1)

    st.markdown("**Conditions**")
    row_three = st.columns(3)
    with row_three[0]:
        traffic_level = st.segmented_control(
            "Traffic", config.TRAFFIC_LEVELS, default="High", key="traffic"
        )
    with row_three[1]:
        weather_condition = st.segmented_control(
            "Weather", config.WEATHER_CONDITIONS, default="Heavy Rain", key="weather"
        )
    with row_three[2]:
        is_weekend = st.toggle("Weekend booking", value=False)

    with st.expander("Customer and driver history (optional)"):
        history = st.columns(4)
        with history[0]:
            cust_prior_rides = st.number_input("Customer prior rides", 0, 50, 5)
        with history[1]:
            cust_prior_cancel_rate = st.slider("Customer prior cancel rate", 0.0, 1.0, 0.2, 0.05)
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
            acceptance_rate = st.slider("Driver acceptance rate", 0.0, 1.0, 0.8, 0.05)

    submitted = st.form_submit_button(
        "Run all models", icon=":material/play_arrow:", width="stretch", type="primary"
    )

if not submitted:
    st.info(
        "Set the trip details above and run the models. The defaults describe a long "
        "evening Cab ride in heavy rain and high traffic at 1.8x surge - a high-risk booking.",
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

# --------------------------------------------------------------------------- #
# Ride outcome
# --------------------------------------------------------------------------- #

section("1. Ride outcome", "Multi-class prediction: Completed, Cancelled or Incomplete.")

if not availability["outcome"]:
    model_missing_notice("ride outcome")
else:
    result = serve.predict_outcome(inputs)
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

# --------------------------------------------------------------------------- #
# Fare
# --------------------------------------------------------------------------- #

section("2. Fare estimate", "Predicted before confirmation, without using base_fare.")

if not availability["fare"]:
    model_missing_notice("fare prediction")
else:
    fare = serve.predict_fare(inputs)
    with st.container(horizontal=True):
        st.metric("Predicted fare", format_currency(fare["predicted_fare"]), border=True)
        st.metric(
            "Expected range",
            f"{format_currency(fare['lower_bound'])} - {format_currency(fare['upper_bound'])}",
            border=True,
        )
        st.metric("Tariff formula value", format_currency(fare["formula_fare"]), border=True)

    st.caption(
        f"The tariff figure applies the recovered pricing rule for a {vehicle_type} "
        f"({serve.TARIFF[vehicle_type]['flagfall']:.0f} + "
        f"{serve.TARIFF[vehicle_type]['per_km']:.0f} per km) multiplied by surge. "
        "The model reaches this independently, from trip context alone."
    )

# --------------------------------------------------------------------------- #
# Risk models
# --------------------------------------------------------------------------- #

section("3. Risk scores", "Binary probabilities with an operational risk band.")

risk_left, risk_right = st.columns(2)

with risk_left:
    if not availability["customer_risk"]:
        model_missing_notice("cancellation risk")
    else:
        customer = serve.predict_customer_risk(inputs)
        with st.container(border=True):
            st.plotly_chart(
                charts.gauge_risk(customer["probability"], "Cancellation risk"),
                width="stretch",
            )
            st.metric("Risk level", customer["risk_level"], border=True)

with risk_right:
    if not availability["driver_risk"]:
        model_missing_notice("driver delay risk")
    else:
        driver = serve.predict_driver_risk(inputs)
        with st.container(border=True):
            st.plotly_chart(
                charts.gauge_risk(driver["probability"], "Delay / incomplete risk"),
                width="stretch",
            )
            st.metric("Risk level", driver["risk_level"], border=True)

# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

if availability["customer_risk"] and availability["driver_risk"]:
    st.divider()
    section("Recommended action")

    cancel_probability = customer["probability"]
    delay_probability = driver["probability"]

    if cancel_probability >= 0.6:
        st.error(
            "**High cancellation risk.** Hold driver assignment until the rider confirms, and "
            "consider capping surge for this request - surge is the strongest single driver of "
            "cancellation in this dataset.",
            icon=":material/priority_high:",
        )
    elif cancel_probability >= 0.3:
        st.warning(
            "**Moderate cancellation risk.** Show an accurate ETA up front and avoid raising "
            "surge further on this request.",
            icon=":material/warning:",
        )
    else:
        st.success("**Low cancellation risk.** Dispatch normally.", icon=":material/check_circle:")

    if delay_probability >= 0.3:
        st.warning(
            "**Elevated delay risk.** Assign a driver scoring above the fleet median on "
            "reliability, and provide routing support - traffic exposure raises the "
            "incompletion rate from 5.1% to 14.8%.",
            icon=":material/route:",
        )

    with st.expander("What the models were given"):
        st.caption(
            "Fields you did not set are filled with dataset medians and modes. "
            "Engineered flags such as rush hour, distance band and adverse conditions are "
            "recomputed from your inputs so the row stays internally consistent."
        )
        st.dataframe(
            pd.DataFrame(
                [{"field": key, "value": value} for key, value in inputs.items()]
            ),
            hide_index=True,
            width="stretch",
        )
