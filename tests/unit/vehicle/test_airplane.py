import numpy as np
import pytest

from ICARUS.airfoils import NACA4
from ICARUS.core.types import FloatArray
from ICARUS.vehicle.airplane import Airplane


@pytest.mark.unit
def test_benchmark_plane(benchmark_airplane: Airplane) -> None:
    """Test that benchmark plane can be created successfully."""
    # Test airplane properties
    assert benchmark_airplane is not None, "Airplane should be created"
    assert benchmark_airplane.name == "benchmark", "Airplane should have correct name"
    assert hasattr(benchmark_airplane, "main_wing"), "Airplane should have main wing"


@pytest.mark.unit
def test_wing_segment_properties(benchmark_airplane: Airplane) -> None:
    """Test wing segment properties."""
    wing = benchmark_airplane.main_wing

    # Test wing properties that we know exist
    assert wing.span == 10.0, "Wing should have correct span"
    assert wing.num_grid_points == 225, "Wing should have correct number of grid points"
    assert wing.num_panels == 196, "Wing should have correct number of panels"
    assert wing.S == 10.0, "Wing should have correct area"
    assert wing.mean_aerodynamic_chord == 1.0, "Wing should have correct MAC"
    assert hasattr(wing, "root_airfoil"), "Wing should have root airfoil"


@pytest.mark.unit
def test_wing_geometry(benchmark_airplane: Airplane) -> None:
    """Test wing geometry calculations including area, MAC, CG, and inertia."""
    print("Testing Geometry...")

    S = benchmark_airplane.S
    MAC = benchmark_airplane.mean_aerodynamic_chord
    CG = benchmark_airplane.CG
    INERTIA = benchmark_airplane.inertia

    # Expected values
    S_expected: tuple[float] = (10.0,)
    MAC_expected: tuple[float] = (1.0,)
    CG_expected: FloatArray = np.array([0.451, 0.0, 0.0])
    I_expected: FloatArray = np.array([2.077, 0.026, 2.103, 0.0, 0.0, 0.0])

    # Assertions with tolerances
    np.testing.assert_almost_equal(S, S_expected, decimal=4)
    np.testing.assert_almost_equal(MAC, MAC_expected, decimal=4)
    np.testing.assert_almost_equal(CG, CG_expected, decimal=3)
    np.testing.assert_almost_equal(INERTIA, I_expected, decimal=3)


def test_wing_properties(benchmark_airplane: Airplane) -> None:
    """Test airfoil properties of the wing."""
    wing = benchmark_airplane.main_wing
    airfoil = wing.root_airfoil

    # Test NACA4 airfoil properties
    assert isinstance(airfoil, NACA4), "Should be NACA4 airfoil"
    assert airfoil.m == 0.04, "Should have correct max camber"
    assert airfoil.p == 0.4, "Should have correct camber position"
    assert airfoil.xx == 0.15, "Should have correct thickness"


@pytest.mark.unit
def test_wing_geometry_properties(benchmark_airplane: Airplane) -> None:
    """Test individual wing properties."""

    # Test that properties exist and are reasonable
    assert hasattr(benchmark_airplane, "S"), "Airplane should have wing area property"
    assert hasattr(
        benchmark_airplane,
        "mean_aerodynamic_chord",
    ), "Airplane should have MAC property"
    assert hasattr(benchmark_airplane, "CG"), "Airplane should have CG property"
    assert hasattr(
        benchmark_airplane,
        "inertia",
    ), "Airplane should have inertia property"

    # Test that values are positive where expected
    assert benchmark_airplane.S > 0, "Wing area should be positive"
    assert benchmark_airplane.mean_aerodynamic_chord > 0, "MAC should be positive"
