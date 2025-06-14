import numpy as np
import pytest

from ICARUS.core.types import FloatArray
from ICARUS.vehicle.airplane import Airplane


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


def test_wing_geometry_properties(benchmark_airplane: Airplane) -> None:
    """Test individual wing properties."""

    # Test that properties exist and are reasonable
    assert hasattr(benchmark_airplane, "S"), "Airplane should have wing area property"
    assert hasattr(benchmark_airplane, "mean_aerodynamic_chord"), "Airplane should have MAC property"
    assert hasattr(benchmark_airplane, "CG"), "Airplane should have CG property"
    assert hasattr(benchmark_airplane, "inertia"), "Airplane should have inertia property"

    # Test that values are positive where expected
    assert benchmark_airplane.S > 0, "Wing area should be positive"
    assert benchmark_airplane.mean_aerodynamic_chord > 0, "MAC should be positive"
