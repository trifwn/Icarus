"""
Shared pytest fixtures for ICARUS tests.
This module provides common fixtures that can be used across all test files.
"""

from __future__ import annotations

import os
from typing import Generator

import numpy as np
import pytest

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.environment.definition import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import SymmetryAxes
from ICARUS.vehicle import WingSegment


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
    """Fixture that provides a benchmark airplane configuration.

    Args:
        name: Name for the airplane

    Returns:
        Tuple of (airplane, state)
    """
    origin: FloatArray = np.array([0.0, 0.0, 0.0], dtype=float)
    wing_position: FloatArray = np.array(
        [0, 0.0, 0.0],
        dtype=float,
    )
    wing_orientation: FloatArray = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    Simplewing = WingSegment(
        name="benchmark",
        root_airfoil=NACA4(M=0.04, P=0.4, XX=0.15),  # "NACA4415",
        origin=origin + wing_position,
        orientation=wing_orientation,
        symmetries=SymmetryAxes.Y,
        span=2 * 5,
        sweep_offset=0.0,
        root_chord=1.0,
        tip_chord=1.0,
        N=15,
        M=15,
        mass=1,
    )
    airplane = Airplane(Simplewing.name, main_wing=Simplewing)
    return airplane


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("database_instance")
def benchmark_state(benchmark_airplane: Airplane) -> State:
    """Fixture that provides a benchmark state for the airplane."""

    return State(
        name="Unstick",
        airplane=benchmark_airplane,
        u_freestream=100,  # Example freestream velocity
        environment=EARTH_ISA,
    )
