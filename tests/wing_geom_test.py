import numpy as np
import pytest

from ICARUS.core.types import FloatArray


@pytest.mark.unit
def test_wing_geometry():
    """Test wing geometry calculations including area, MAC, CG, and inertia."""
    print("Testing Geometry...")

    from .benchmark_plane_test import get_benchmark_plane

    bmark = get_benchmark_plane("bmark")

    S = bmark.S
    MAC = bmark.mean_aerodynamic_chord
    CG = bmark.CG
    INERTIA = bmark.inertia

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


def test_wing_geometry_properties():
    """Test individual wing properties."""
    from .benchmark_plane_test import get_benchmark_plane

    bmark = get_benchmark_plane("test_props")

    # Test that properties exist and are reasonable
    assert hasattr(bmark, "S"), "Airplane should have wing area property"
    assert hasattr(bmark, "mean_aerodynamic_chord"), "Airplane should have MAC property"
    assert hasattr(bmark, "CG"), "Airplane should have CG property"
    assert hasattr(bmark, "inertia"), "Airplane should have inertia property"

    # Test that values are positive where expected
    assert bmark.S > 0, "Wing area should be positive"
    assert bmark.mean_aerodynamic_chord > 0, "MAC should be positive"


# Backward compatibility function
def geom() -> None:
    """Legacy function for backward compatibility."""
    test_wing_geometry()
