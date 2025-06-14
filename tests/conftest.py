"""
Shared pytest fixtures for ICARUS tests.
This module provides common fixtures that can be used across all test files.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Generator

import pytest

from ICARUS.database import Database

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


@pytest.fixture(scope="session")
def database_instance() -> Generator[Database, None, None]:
    """
    Session-scoped fixture that provides a properly initialized Database instance.

    This fixture:
    - Initializes the Database singleton once per test session
    - Loads all data from the database
    - Ensures Database.get_instance() works in all tests
    - Cleans up after all tests are done

    Returns:
        Database: The initialized database instance
    """
    # Database folder path - adjust if needed
    database_folder = os.path.join(os.path.dirname(__file__), "TestData")
    if not os.path.exists(database_folder):
        # database_folder = ".\\Data"  # Fallback to relative path
        os.makedirs(database_folder, exist_ok=True)

    # Clear any existing instance to ensure clean state
    Database._instance = None

    # Initialize the database singleton
    db = Database(database_folder)

    try:
        # Load all data
        db.load_all_data()
        print(f"Database initialized from: {database_folder}")
        yield db
    finally:
        # Cleanup after all tests
        Database._instance = None


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("database_instance")
def benchmark_airplane() -> Airplane:
    """Fixture that provides a benchmark airplane and state."""
    from .benchmark_plane_test import get_benchmark_plane

    return get_benchmark_plane("bmark")


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("database_instance")
def benchmark_state(benchmark_airplane: Airplane) -> State:
    """Fixture that provides a benchmark state for the airplane."""
    from .benchmark_plane_test import get_benchmark_state

    return get_benchmark_state(benchmark_airplane)
