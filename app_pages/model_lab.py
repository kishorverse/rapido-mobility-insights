"""Model Lab: leaderboards, evaluation metrics and feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_pages._helpers import (
    cached_metrics,
    cached_model,
    chart_card,
    dataframe_card,
    model_missing_notice,
    section,
)
import config
from rapido import charts
from rapido.models import dataset, evaluate, registry

st.title(":material/science: Model Lab")
st.caption("How the four models were built, what they score, and what drives them.")

metrics_store = cached_metrics()
available = registry.list_models()

if available.empty:
    model_missing_notice("prediction")
    st.stop()

MODEL_TABS = {
    "Ride Outcome": ("outcome", "classification"),
    "Fare Prediction": ("fare", "regression"),
    "Cancellation Risk": ("customer_risk", "classification"),
    "Driver Delay Risk": ("driver_risk", "classification"),
}


@st.cache_data(ttl="1h", show_spinner="Scoring held-out test set...")
def _test_evaluation(model_key: str) -> dict:
    """Rebuild the held-out split and score the persisted model on it."""
    from rapido import io

    frame = io.load_processed("features")
    features, target = dataset.build_dataset(frame, model_key)
    stratify = model_key != "fare"
    _, x_test, _, y_test = dataset.split_train_test(features, target, stratify=stratify)

    pipeline, _ = cached_model(config.MODEL_NAMES[model_key])
    predictions = pipeline.predict(x_test)

    result = {"y_test": np.asarray(y_test), "predictions": np.asarray(predictions)}
    if hasattr(pipeline, "predict_proba"):
        result["probabilities"] = pipeline.predict_proba(x_test)
        result["classes"] = list(pipeline.named_steps["model"].classes_)
    return result


st.subheader("Model portfolio")
overview_rows = []
for label, (key, task) in MODEL_TABS.items():
    stored = metrics_store.get(config.MODEL_NAMES[key], {})
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
                f"ROC-AUC {model_metrics.get('roc_auc', model_metrics.get('roc_auc_ovr', '-'))}"
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
        artefact_name = config.MODEL_NAMES[model_key]
        if not registry.model_exists(artefact_name):
            model_missing_notice(label)
            continue

        _, metadata = cached_model(artefact_name)
        model_metrics = metadata.get("metrics", {})

        st.markdown(f"**{metadata.get('description', label)}**")
        st.caption(
            f"Algorithm: `{metadata.get('algorithm', '-')}` · "
            f"train {metadata.get('n_train', 0):,} / test {metadata.get('n_test', 0):,} rows · "
            f"trained {metadata.get('saved_at', '-')}"
        )

        if task == "classification":
            with st.container(horizontal=True):
                st.metric("Accuracy", model_metrics.get("accuracy", "-"), border=True)
                st.metric("F1 (macro)", model_metrics.get("f1_macro", "-"), border=True)
                st.metric(
                    "Balanced accuracy",
                    model_metrics.get("balanced_accuracy", "-"),
                    border=True,
                )
                st.metric(
                    "ROC-AUC",
                    model_metrics.get("roc_auc", model_metrics.get("roc_auc_ovr", "-")),
                    border=True,
                )
        else:
            with st.container(horizontal=True):
                st.metric("R²", model_metrics.get("r2", "-"), border=True)
                st.metric("RMSE", model_metrics.get("rmse", "-"), border=True)
                st.metric("MAE", model_metrics.get("mae", "-"), border=True)
                st.metric("MAPE", f"{model_metrics.get('mape_pct', '-')}%", border=True)

        leaderboard = metadata.get("leaderboard")
        if leaderboard is not None:
            frame = (
                pd.DataFrame(leaderboard)
                if not isinstance(leaderboard, pd.DataFrame)
                else leaderboard
            )
            dataframe_card(frame, "Candidate Leaderboard (held-out test set)")
            st.caption(
                "`dummy` is the baseline - majority class for classification, mean for "
                "regression. Every model must beat it to justify its complexity."
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
                    "A small standard deviation here means the held-out score is stable, "
                    "not a lucky split."
                )

        evaluation = _test_evaluation(model_key)

        if task == "classification":
            labels = sorted(pd.Series(evaluation["y_test"]).unique())
            matrix = evaluate.confusion_matrix_df(
                evaluation["y_test"], evaluation["predictions"], labels
            )
            left, right = st.columns(2)
            with left:
                chart_card(
                    charts.confusion_matrix_fig(matrix.to_numpy(), [str(x) for x in labels])
                )
            with right:
                per_class = evaluate.per_class_metrics(
                    evaluation["y_test"], evaluation["predictions"], labels
                )
                dataframe_card(per_class, "Per-Class Metrics")

            if len(labels) == 2 and "probabilities" in evaluation:
                fpr, tpr, auc = evaluate.roc_data(
                    evaluation["y_test"], evaluation["probabilities"]
                )
                precision, recall, ap, baseline = evaluate.pr_data(
                    evaluation["y_test"], evaluation["probabilities"]
                )
                left, right = st.columns(2)
                with left:
                    chart_card(charts.roc_curve_fig(fpr, tpr, auc))
                with right:
                    chart_card(charts.pr_curve_fig(precision, recall, ap, baseline))

                thresholds = evaluate.threshold_table(
                    evaluation["y_test"], evaluation["probabilities"]
                )
                dataframe_card(thresholds, "Decision Threshold Trade-off")
                st.caption(
                    "Operations picks the threshold: a lower cut-off catches more failures "
                    "but flags more bookings for intervention."
                )
        else:
            left, right = st.columns(2)
            with left:
                chart_card(
                    charts.actual_vs_predicted_fig(
                        evaluation["y_test"][:4000], evaluation["predictions"][:4000]
                    )
                )
            with right:
                chart_card(
                    charts.residual_plot_fig(
                        evaluation["y_test"][:4000], evaluation["predictions"][:4000]
                    )
                )

            with st.container(border=True):
                st.markdown("**Against the project benchmark**")
                st.write(
                    f"Predictions within ±10% of actual fare: "
                    f"**{100 * model_metrics.get('within_10_pct', 0):.2f}%**"
                )
                st.caption(
                    "The brief targets RMSE within ±10% of actual fare. Because fare is a "
                    "deterministic tariff plus ±5% uniform noise, the theoretical minimum "
                    "MAPE is 2.50%; this model reaches "
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
                chart_card(charts.feature_importance_fig(frame, top_n=15))
                st.caption(
                    "Permutation importance measured on the held-out set, computed over the "
                    "original columns rather than one-hot fragments."
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

`rapido/models/dataset.py` enforces this with `assert_no_leakage()`, which raises rather than
warns, and the test suite fails if any blocked column reappears.
        """
    )

ablation = metrics_store.get("fare_leakage_ablation", {})
if ablation:
    ablation_metrics = ablation.get("metrics", {})
    with st.container(border=True):
        st.markdown("**Fare leakage ablation**")
        st.write(
            f"Refitting the fare model *with* `base_fare` gives R² "
            f"**{ablation_metrics.get('r2')}** versus **"
            f"{metrics_store.get(config.MODEL_NAMES['fare'], {}).get('metrics', {}).get('r2')}** "
            "without it."
        )
        st.caption(
            "The gap is negligible, which confirms the tariff is already recoverable from "
            "distance and vehicle type - `base_fare` carries no independent information."
        )
