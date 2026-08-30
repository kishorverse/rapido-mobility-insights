"""Shared helpers for the Streamlit pages.

Pages hold no business logic. Everything here is presentation: caching,
formatting and small layout utilities.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rapido import queries
from rapido.models import registry

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
    """Run a named query from :mod:`rapido.queries` with caching.

    Args:
        name: Function name in ``rapido.queries``.
        key: Hashable filter signature; participates in the cache key.
        _filters: The real filter dict, excluded from hashing by the underscore.
        **kwargs: Extra query arguments.
    """
    function = getattr(queries, name, None)
    if function is None:
        raise ValueError(f"Unknown query {name!r}.")
    try:
        return function(_filters, **kwargs) if _filters is not None else function(**kwargs)
    except TypeError:
        return function(**kwargs)


def q(name: str, filters: dict | None = None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: run a cached query for the active filters."""
    return run_query(name, _cache_key(filters), _filters=filters, **kwargs)


@st.cache_resource(show_spinner=False)
def cached_model(name: str):
    """Load a trained model once per session."""
    return registry.load_model(name)


@st.cache_data(ttl="1h", show_spinner=False)
def cached_metrics() -> dict:
    """Load the stored model metrics."""
    return registry.load_metrics()


@st.cache_data(ttl="1h", show_spinner=False)
def filter_options() -> dict:
    """Fetch distinct filter values from the database."""
    return queries.q_filter_options()


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def format_currency(value: float | None) -> str:
    """Format a number as Indian rupees with a thousands separator."""
    if value is None or pd.isna(value):
        return "-"
    return f"₹{value:,.0f}"


def format_compact(value: float | None) -> str:
    """Format a large number compactly (K / M / Cr)."""
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


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


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


def paginate(frame: pd.DataFrame, page_size: int = 25, key: str = "page") -> pd.DataFrame:
    """Render pagination controls and return the visible slice.

    The project guidelines call for avoiding full-data loads; the SQL layer
    pages server-side, and this handles in-memory frames the same way.
    """
    if frame is None or frame.empty:
        return frame

    total_pages = max(1, -(-len(frame) // page_size))
    left, right = st.columns([3, 1])
    with right:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=key,
        )
    with left:
        st.caption(f"{len(frame):,} rows across {total_pages} page(s)")

    start = (page - 1) * page_size
    return frame.iloc[start : start + page_size]


def download_button(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
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
        "Run `python scripts/train_all.py` from the project root.",
        icon=":material/model_training:",
    )


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of bordered metric cards."""
    with st.container(horizontal=True):
        for label, value, delta in metrics:
            st.metric(label, value, delta, border=True)
