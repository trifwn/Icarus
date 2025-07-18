"""
Test runner for comprehensive JAX airfoil test suite.

This module provides a simple way to run and verify the comprehensive test suite.
"""

import sys
import traceback

import jax.numpy as jnp

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


def create_test_airfoil():
    """Create a simple test airfoil for testing."""
    upper = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.06, 0.08, 0.05, 0.0]])
    lower = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.04, -0.06, -0.03, 0.0]])
    return JaxAirfoil.from_upper_lower(upper, lower, name="TestAirfoil")


def test_basic_functionality():
    """Test basic functionality of the comprehensive test suite."""
    print("Testing basic JAX airfoil functionality...")

    try:
        # Create test airfoil
        airfoil = create_test_airfoil()
        print(f"✓ Created test airfoil with {airfoil.n_points} points")

        # Test basic operations
        query_x = jnp.array([0.5])
        thickness = airfoil.thickness(query_x)
        camber = airfoil.camber_line(query_x)
        y_upper = airfoil.y_upper(query_x)
        y_lower = airfoil.y_lower(query_x)

        print(f"✓ Thickness at x=0.5: {thickness[0]:.6f}")
        print(f"✓ Camber at x=0.5: {camber[0]:.6f}")
        print(f"✓ Upper surface at x=0.5: {y_upper[0]:.6f}")
        print(f"✓ Lower surface at x=0.5: {y_lower[0]:.6f}")

        # Test geometric properties
        max_thickness = airfoil.max_thickness
        max_camber = airfoil.max_camber
        chord_length = airfoil.chord_length

        print(f"✓ Max thickness: {max_thickness:.6f}")
        print(f"✓ Max camber: {max_camber:.6f}")
        print(f"✓ Chord length: {chord_length:.6f}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_gradient_functionality():
    """Test gradient functionality."""
    print("\nTesting gradient functionality...")

    try:
        import jax

        airfoil = create_test_airfoil()

        # Test gradient of thickness
        def thickness_objective(airfoil):
            query_x = jnp.array([0.5])
            return jnp.sum(airfoil.thickness(query_x))

        grad_fn = jax.grad(thickness_objective)
        gradients = grad_fn(airfoil)

        print("✓ Computed gradients successfully")
        print(f"✓ Gradient airfoil has {gradients.n_points} points")

        # Check that gradients are finite
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        if jnp.all(jnp.isfinite(grad_coords)):
            print("✓ All gradients are finite")
        else:
            print("✗ Some gradients are not finite")
            return False

        # Check that some gradients are non-zero
        if jnp.any(jnp.abs(grad_coords) > 1e-8):
            print("✓ Some gradients are non-zero")
        else:
            print("✗ All gradients are zero")
            return False

        return True

    except Exception as e:
        print(f"✗ Gradient functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_jit_functionality():
    """Test JIT compilation functionality."""
    print("\nTesting JIT compilation functionality...")

    try:
        import jax

        airfoil = create_test_airfoil()

        @jax.jit
        def jit_thickness_computation(airfoil):
            query_x = jnp.array([0.25, 0.5, 0.75])
            return airfoil.thickness(query_x)

        @jax.jit
        def jit_surface_queries(airfoil):
            query_x = jnp.array([0.5])
            y_upper = airfoil.y_upper(query_x)
            y_lower = airfoil.y_lower(query_x)
            return y_upper, y_lower

        # Test JIT compilation
        thickness = jit_thickness_computation(airfoil)
        y_upper, y_lower = jit_surface_queries(airfoil)

        print(f"✓ JIT thickness computation: {thickness}")
        print(f"✓ JIT surface queries: upper={y_upper[0]:.6f}, lower={y_lower[0]:.6f}")

        # Verify results are finite
        if (
            jnp.all(jnp.isfinite(thickness))
            and jnp.all(jnp.isfinite(y_upper))
            and jnp.all(jnp.isfinite(y_lower))
        ):
            print("✓ All JIT results are finite")
        else:
            print("✗ Some JIT results are not finite")
            return False

        return True

    except Exception as e:
        print(f"✗ JIT functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_numerical_accuracy():
    """Test numerical accuracy against reference implementation."""
    print("\nTesting numerical accuracy...")

    try:
        from ICARUS.airfoils import NACA4

        # Create reference NACA airfoil
        ref_naca = NACA4.from_digits("2412", n_points=30)

        # Create equivalent JAX airfoil
        x_coords, y_coords = ref_naca.get_coordinates()
        coords = jnp.array([x_coords, y_coords])
        jax_naca = JaxAirfoil(coords, name="NACA2412_JAX")

        # Compare thickness at several points
        query_x = jnp.array([0.25, 0.5, 0.75])
        ref_thickness = ref_naca.thickness(query_x)
        jax_thickness = jax_naca.thickness(query_x)

        max_error = jnp.max(jnp.abs(ref_thickness - jax_thickness))
        relative_error = max_error / jnp.max(ref_thickness)

        print(f"✓ Reference thickness: {ref_thickness}")
        print(f"✓ JAX thickness: {jax_thickness}")
        print(f"✓ Max absolute error: {max_error:.8f}")
        print(f"✓ Relative error: {relative_error:.6f}")

        if relative_error < 0.01:  # Less than 1% error
            print("✓ Numerical accuracy test passed")
            return True
        else:
            print("✗ Numerical accuracy test failed - error too large")
            return False

    except Exception as e:
        print(f"✗ Numerical accuracy test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("COMPREHENSIVE JAX AIRFOIL TEST SUITE")
    print("=" * 60)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Gradient Functionality", test_gradient_functionality),
        ("JIT Compilation", test_jit_functionality),
        ("Numerical Accuracy", test_numerical_accuracy),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name} Tests")
        print(f"{'-' * 40}")

        success = test_func()
        results.append((test_name, success))

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All comprehensive tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
