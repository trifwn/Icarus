"""
Summary of comprehensive JAX airfoil test suite implementation.

This file documents the comprehensive test suite created for task 15.
The test suite covers all requirements: 1.2, 2.1, 2.2, 8.2

Test Files Created:
1. test_comprehensive_jax_suite.py - Main comprehensive test suite
2. test_jax_gradient_verification.py - Detailed gradient verification
3. test_jax_jit_compilation_comprehensive.py - JIT compilation testing
4. test_jax_property_based_comprehensive.py - Property-based edge case testing
5. test_comprehensive_runner.py - Test runner utility

Test Coverage:
- Gradient tests using jax.grad for all operations ✓
- JIT compilation tests for all core methods ✓
- Numerical accuracy tests comparing with NumPy version ✓
- Property-based testing for edge cases ✓
- Integration tests with existing ICARUS workflows ✓
"""

import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


def test_comprehensive_suite_exists():
    """Verify that the comprehensive test suite files exist and are importable."""

    # Test that main comprehensive suite can be imported
    try:
        from . import test_comprehensive_jax_suite

        assert hasattr(test_comprehensive_jax_suite, "TestComprehensiveJaxGradients")
        assert hasattr(test_comprehensive_jax_suite, "TestComprehensiveJITCompilation")
        assert hasattr(test_comprehensive_jax_suite, "TestNumericalAccuracy")
        assert hasattr(test_comprehensive_jax_suite, "TestPropertyBasedEdgeCases")
        assert hasattr(test_comprehensive_jax_suite, "TestIcarusIntegration")
        print("✓ Main comprehensive test suite imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import main comprehensive test suite: {e}")

    # Test that gradient verification suite can be imported
    try:
        from . import test_jax_gradient_verification

        assert hasattr(test_jax_gradient_verification, "TestGradientVerification")
        print("✓ Gradient verification test suite imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import gradient verification suite: {e}")

    # Test that JIT compilation suite can be imported
    try:
        from . import test_jax_jit_compilation_comprehensive

        assert hasattr(
            test_jax_jit_compilation_comprehensive,
            "TestJITCompilationComprehensive",
        )
        print("✓ JIT compilation test suite imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import JIT compilation suite: {e}")

    # Test that property-based suite can be imported
    try:
        from . import test_jax_property_based_comprehensive

        assert hasattr(
            test_jax_property_based_comprehensive,
            "TestPropertyBasedComprehensive",
        )
        print("✓ Property-based test suite imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import property-based suite: {e}")


def test_basic_jax_airfoil_functionality():
    """Basic smoke test to verify JAX airfoil works."""

    # Create simple test airfoil
    upper = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.06, 0.08, 0.05, 0.0]])
    lower = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.04, -0.06, -0.03, 0.0]])

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="SmokeTest")

    # Basic assertions
    assert airfoil.n_points > 0
    assert airfoil.buffer_size >= airfoil.n_points
    assert jnp.isfinite(airfoil.max_thickness)
    assert airfoil.max_thickness > 0

    # Test basic operations
    query_x = jnp.array([0.5])
    thickness = airfoil.thickness(query_x)
    camber = airfoil.camber_line(query_x)
    y_upper = airfoil.y_upper(query_x)
    y_lower = airfoil.y_lower(query_x)

    assert jnp.isfinite(thickness[0])
    assert thickness[0] > 0
    assert jnp.isfinite(camber[0])
    assert jnp.isfinite(y_upper[0])
    assert jnp.isfinite(y_lower[0])
    assert y_upper[0] > y_lower[0]  # Upper should be above lower


def test_gradient_computation_basic():
    """Basic test of gradient computation."""
    import jax

    # Create test airfoil
    upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
    lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.05, 0.0]])

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="GradTest")

    # Define objective function
    def thickness_objective(airfoil):
        return airfoil.thickness(jnp.array([0.5]))[0]

    # Compute gradient
    grad_fn = jax.grad(thickness_objective)
    gradients = grad_fn(airfoil)

    # Basic checks
    assert isinstance(gradients, JaxAirfoil)
    assert gradients.n_points == airfoil.n_points

    # Check that gradients are finite
    grad_coords = gradients._coordinates[:, gradients._validity_mask]
    assert jnp.all(jnp.isfinite(grad_coords))


def test_jit_compilation_basic():
    """Basic test of JIT compilation."""
    import jax

    # Create test airfoil
    coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
    airfoil = JaxAirfoil(coords, name="JITTest")

    @jax.jit
    def jit_thickness(airfoil):
        return airfoil.thickness(jnp.array([0.5]))

    @jax.jit
    def jit_properties(airfoil):
        return airfoil.max_thickness, airfoil.max_camber

    # Test JIT compilation
    thickness = jit_thickness(airfoil)
    max_t, max_c = jit_properties(airfoil)

    # Basic checks
    assert jnp.isfinite(thickness[0])
    assert thickness[0] > 0
    assert jnp.isfinite(max_t)
    assert jnp.isfinite(max_c)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
