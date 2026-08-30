"""Shared pytest fixtures and path setup for the Rapido test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rapido import io  # noqa: E402


@pytest.fixture(scope="session")
def bookings():
    """The raw booking fact table, loaded once per test session."""
    return io.load_bookings()


@pytest.fixture(scope="session")
def customers():
    """The raw customer dimension."""
    return io.load_customers()


@pytest.fixture(scope="session")
def drivers():
    """The raw driver dimension."""
    return io.load_drivers()


@pytest.fixture(scope="session")
def raw_frames():
    """All five raw source frames keyed by dataset name."""
    return io.load_all_raw()
