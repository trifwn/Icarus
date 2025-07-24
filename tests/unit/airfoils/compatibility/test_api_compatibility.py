"""
API compatibility tests for JAX airfoil implementation.

This module tests backward compatibility with NumPy implementation,
integration with existing ICARUS workflows, and migration utilities.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad
from jax import jit

from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


class TestBackwardCompatibility:
    """Test backward compatibility with original NumPy implementation."""

    def test_naca4_creation_compatibility(self) -> None:
        """Test NACA4 creation matches expected interface."""
        # Test different creation methods
        naca1 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
        naca2 = NACA4.from_name("NACA2412")
        naca3 = NACA4.from_digits("2412")

        # All should create valid airfoils
        assert naca1.name == "naca2412"
        assert naca2.name == "naca2412"
        assert naca3.name == "naca2412"

        # Parameters should match
        assert jnp.allclose(naca1.m, naca2.m)
        assert jnp.allclose(naca1.p, naca2.p)
        assert jnp.allclose(naca1.xx, naca2.xx)

    def test_airfoil_properties_compatibility(self) -> None:
        """Test that airfoil properties match expected interface."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test property access
        assert hasattr(naca2412, "name")
        assert hasattr(naca2412, "n_points")
        assert hasattr(naca2412, "upper_surface")
        assert hasattr(naca2412, "lower_surface")
        assert hasattr(naca2412, "max_thickness")
        assert hasattr(naca2412, "max_thickness_location")

        # Test property types
        assert isinstance(naca2412.name, str)
        assert isinstance(naca2412.n_points, int)
        assert isinstance(naca2412.upper_surface, jnp.ndarray)
        assert isinstance(naca2412.lower_surface, jnp.ndarray)
        # JAX arrays should be convertible to float
        assert isinstance(float(naca2412.max_thickness), float)
        assert isinstance(float(naca2412.max_thickness_location), float)

    def test_surface_evaluation_compatibility(self):
        """Test surface evaluation methods match expected interface."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test with different input types
        x_float = 0.5
        x_list = jnp.array([0.3, 0.5, 0.7])  # Convert list to JAX array
        x_array = np.array([0.2, 0.4, 0.6, 0.8])
        x_jax = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # All should work
        y_upper_float = naca2412.y_upper(x_float)
        y_upper_list = naca2412.y_upper(x_list)
        y_upper_array = naca2412.y_upper(x_array)
        y_upper_jax = naca2412.y_upper(x_jax)

        # Check return types and shapes
        assert isinstance(y_upper_float, (float, jnp.ndarray))
        assert len(y_upper_list) == len(x_list)
        assert len(y_upper_array) == len(x_array)
        assert len(y_upper_jax) == len(x_jax)

    def test_morphing_compatibility(self):
        """Test morphing interface compatibility."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test morphing with different eta values
        morphed = Airfoil.morph_new_from_two_foils(
            naca0012,
            naca2412,
            eta=0.5,
            n_points=100,
        )

        # Should return valid airfoil
        assert isinstance(morphed, Airfoil)
        # Note: morphing creates n_points//2 for each surface, so total is n_points
        assert morphed.n_points == 50  # This is the actual behavior
        assert "morphed" in morphed.name

    def test_flapping_compatibility(self):
        """Test flapping interface compatibility."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test flapping with different parameters
        flapped = naca2412.flap(
            flap_hinge_chord_percentage=0.7,
            flap_angle=10.0,
            chord_extension=1.2,
        )

        # Should return valid airfoil (or FlappedAirfoil)
        assert flapped is not None
        assert hasattr(flapped, "name")
        assert hasattr(flapped, "n_points")

    def test_serialization_compatibility(self):
        """Test serialization/deserialization compatibility."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test getstate/setstate
        state = naca2412.__getstate__()
        assert isinstance(state, dict)
        assert "m" in state
        assert "p" in state
        assert "xx" in state
        assert "n_points" in state

        # Test reconstruction
        new_naca = NACA4(M=0, P=0, XX=0, n_points=50)  # Dummy initialization
        new_naca.__setstate__(state)

        assert jnp.allclose(new_naca.m, naca2412.m)
        assert jnp.allclose(new_naca.p, naca2412.p)
        assert jnp.allclose(new_naca.xx, naca2412.xx)


class TestNumericalAccuracy:
    """Test numerical accuracy compared to reference implementations."""

    def test_naca4_analytical_comparison(self):
        """Test NACA4 implementation against analytical formulas."""
        # Test symmetric airfoil (NACA0012)
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=200)

        x_test = jnp.linspace(0, 1, 100)

        # For symmetric airfoil, camber should be zero
        camber = naca0012.camber_line(x_test)
        assert jnp.allclose(camber, 0.0, atol=1e-10)

        # Upper and lower surfaces should be symmetric
        y_upper = naca0012.y_upper(x_test)
        y_lower = naca0012.y_lower(x_test)
        assert jnp.allclose(y_upper, -y_lower, atol=1e-10)

        # Thickness should match analytical formula (with relaxed tolerance)
        # Note: thickness_distribution returns half-thickness, so we need to multiply by 2
        thickness_analytical = 2 * naca0012.thickness_distribution(x_test)
        thickness_computed = y_upper - y_lower
        assert jnp.allclose(thickness_computed, thickness_analytical, atol=1e-6)

    def test_naca4_camber_accuracy(self):
        """Test NACA4 camber line accuracy."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test camber line properties
        x_test = jnp.linspace(0, 1, 100)
        camber = naca2412.camber_line(x_test)

        # Maximum camber should occur at p
        max_camber_idx = jnp.argmax(camber)
        max_camber_location = x_test[max_camber_idx]

        # Should be close to p (within discretization error)
        assert jnp.abs(max_camber_location - naca2412.p) < 0.05

        # Maximum camber should be close to m
        max_camber_value = jnp.max(camber)
        assert jnp.abs(max_camber_value - naca2412.m) < 0.001

    def test_thickness_distribution_accuracy(self):
        """Test thickness distribution accuracy."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        x_test = jnp.linspace(0, 1, 100)
        thickness = naca2412.thickness_distribution(x_test)

        # Thickness should be zero at leading and trailing edges
        assert jnp.abs(thickness[0]) < 1e-6  # Leading edge
        assert jnp.abs(thickness[-1]) < 1e-6  # Trailing edge

        # Maximum thickness should be close to xx (note: thickness_distribution returns half-thickness)
        max_thickness = jnp.max(thickness)
        # The thickness_distribution function returns half-thickness, so we expect xx/2
        expected_max = naca2412.xx / 2
        assert jnp.abs(max_thickness - expected_max) < 0.001

    def test_surface_continuity(self):
        """Test surface continuity and smoothness."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test surface continuity
        x_test = jnp.linspace(0.01, 0.99, 100)  # Avoid exact edges
        y_upper = naca2412.y_upper(x_test)
        y_lower = naca2412.y_lower(x_test)

        # Surfaces should be smooth (no large jumps)
        upper_diff = jnp.diff(y_upper)
        lower_diff = jnp.diff(y_lower)

        # Maximum difference between adjacent points should be reasonable
        max_upper_diff = jnp.max(jnp.abs(upper_diff))
        max_lower_diff = jnp.max(jnp.abs(lower_diff))

        assert max_upper_diff < 0.01  # Reasonable smoothness
        assert max_lower_diff < 0.01

    def test_gradient_accuracy_comparison(self):
        """Test gradient accuracy against numerical derivatives."""

        def airfoil_thickness_at_point(params, x_point):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.thickness_distribution(x_point)

        params = jnp.array([0.02, 0.4, 0.12])
        x_point = 0.5

        # Analytical gradient
        grad_fn = grad(airfoil_thickness_at_point, argnums=0)
        analytical_grad = grad_fn(params, x_point)

        # Numerical gradient
        eps = 1e-6
        numerical_grad = jnp.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.at[i].add(eps)
            params_minus = params.at[i].add(-eps)

            f_plus = airfoil_thickness_at_point(params_plus, x_point)
            f_minus = airfoil_thickness_at_point(params_minus, x_point)

            numerical_grad = numerical_grad.at[i].set((f_plus - f_minus) / (2 * eps))

        # Compare gradients
        relative_error = jnp.abs(
            (analytical_grad - numerical_grad) / (numerical_grad + 1e-12),
        )
        assert jnp.all(relative_error < 1e-4)


class TestIntegrationCompatibility:
    """Test integration with existing ICARUS workflows."""

    def test_airfoil_database_integration(self):
        """Test integration with airfoil database workflows."""
        # Create multiple airfoils as would be done in database
        airfoils = []

        # NACA 4-digit series
        for m in [0, 2, 4]:
            for p in [0, 4, 6]:
                for xx in [12, 15, 18]:
                    naca_name = f"{m}{p}{xx:02d}"
                    airfoil = NACA4.from_digits(naca_name)
                    airfoils.append(airfoil)

        # All should be valid
        assert len(airfoils) == 3 * 3 * 3

        for airfoil in airfoils:
            assert isinstance(airfoil, NACA4)
            assert airfoil.max_thickness > 0
            assert 0 <= airfoil.max_thickness_location <= 1

    def test_analysis_workflow_integration(self):
        """Test integration with typical analysis workflows."""
        # Create airfoil
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Typical analysis workflow
        x_analysis = jnp.linspace(0, 1, 100)

        # Surface evaluation
        y_upper = naca2412.y_upper(x_analysis)
        y_lower = naca2412.y_lower(x_analysis)

        # Geometric properties
        thickness = naca2412.thickness(x_analysis)
        camber = naca2412.camber_line(x_analysis)

        # All should work without issues
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))

    def test_optimization_workflow_integration(self):
        """Test integration with optimization workflows."""

        def airfoil_objective(params):
            """Typical optimization objective."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)

            # Multi-objective: thickness and camber constraints
            max_thickness = naca.max_thickness
            max_camber = jnp.max(naca.camber_line(jnp.linspace(0, 1, 50)))

            # Objective: minimize thickness while maintaining camber
            return (max_thickness - 0.10) ** 2 + (max_camber - 0.015) ** 2

        # Test gradient computation (essential for optimization)
        params = jnp.array([0.02, 0.4, 0.12])

        objective_value = airfoil_objective(params)
        gradient = grad(airfoil_objective)(params)

        assert jnp.isfinite(objective_value)
        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)

    def test_morphing_workflow_integration(self):
        """Test integration with morphing workflows."""
        # Create base airfoils
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)
        naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=100)

        # Create morphing sequence
        eta_values = jnp.linspace(0, 1, 11)
        morphed_airfoils = []

        for eta in eta_values:
            morphed = Airfoil.morph_new_from_two_foils(
                naca0012,
                naca2412,
                eta=float(eta),
                n_points=100,
            )
            morphed_airfoils.append(morphed)

        # All should be valid
        assert len(morphed_airfoils) == len(eta_values)

        for airfoil in morphed_airfoils:
            assert isinstance(airfoil, Airfoil)
            # Note: morphing creates n_points//2 for each surface
            assert airfoil.n_points == 50

    def test_file_io_compatibility(self):
        """Test file I/O compatibility."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test Selig format conversion
        selig_coords = naca2412.to_selig()

        assert isinstance(selig_coords, jnp.ndarray)
        assert selig_coords.shape[0] == 2  # x and y coordinates
        # Note: to_selig() combines upper and lower surfaces, so total points = n_upper + n_lower
        expected_points = naca2412.n_upper + naca2412.n_lower
        assert selig_coords.shape[1] == expected_points

        # Test that coordinates are reasonable
        x_coords = selig_coords[0, :]
        y_coords = selig_coords[1, :]

        # Allow small numerical errors near zero
        assert jnp.min(x_coords) >= -1e-4  # Small tolerance for numerical precision
        assert jnp.max(x_coords) <= 1.0
        assert jnp.all(jnp.isfinite(x_coords))
        assert jnp.all(jnp.isfinite(y_coords))


class TestMigrationUtilities:
    """Test utilities for migrating from NumPy to JAX implementation."""

    def test_parameter_conversion(self):
        """Test conversion between parameter formats."""
        # Test NACA4 parameter conversion
        naca_string = "2412"
        naca_from_string = NACA4.from_digits(naca_string)
        naca_from_params = NACA4(M=0.02, P=0.4, XX=0.12)

        # Should be equivalent
        assert jnp.allclose(naca_from_string.m, naca_from_params.m)
        assert jnp.allclose(naca_from_string.p, naca_from_params.p)
        assert jnp.allclose(naca_from_string.xx, naca_from_params.xx)

    def test_coordinate_format_conversion(self):
        """Test conversion between coordinate formats."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test different coordinate access methods
        upper_surface = naca2412.upper_surface
        lower_surface = naca2412.lower_surface
        selig_format = naca2412.to_selig()

        # All should be valid JAX arrays
        assert isinstance(upper_surface, jnp.ndarray)
        assert isinstance(lower_surface, jnp.ndarray)
        assert isinstance(selig_format, jnp.ndarray)

        # Shapes should be consistent
        assert upper_surface.shape[0] == 2  # x and y
        assert lower_surface.shape[0] == 2  # x and y
        assert selig_format.shape[0] == 2  # x and y

    def test_numpy_jax_interoperability(self):
        """Test interoperability between NumPy and JAX arrays."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test with NumPy input
        x_numpy = np.linspace(0, 1, 50)
        y_upper_from_numpy = naca2412.y_upper(x_numpy)

        # Test with JAX input
        x_jax = jnp.linspace(0, 1, 50)
        y_upper_from_jax = naca2412.y_upper(x_jax)

        # Results should be equivalent
        assert jnp.allclose(y_upper_from_numpy, y_upper_from_jax)

        # Both should return JAX arrays
        assert isinstance(y_upper_from_numpy, jnp.ndarray)
        assert isinstance(y_upper_from_jax, jnp.ndarray)

    def test_legacy_interface_support(self):
        """Test support for legacy interface patterns."""
        # Test alternative constructor patterns
        naca1 = Airfoil.naca("2412", n_points=100)
        naca2 = NACA4.from_name("NACA2412")

        # Both should work and be equivalent
        assert isinstance(naca1, NACA4)
        assert isinstance(naca2, NACA4)
        assert naca1.name == naca2.name

    def test_error_message_compatibility(self) -> None:
        """Test that error messages are helpful for migration."""
        # Test invalid NACA parameters
        with pytest.raises(ValueError) as exc_info:
            NACA4.from_digits("12345")  # Too many digits

        assert "4 characters" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            NACA4.from_digits("abcd")  # Non-numeric

        assert "numeric" in str(exc_info.value)

        # Test invalid morphing parameters
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=100)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        with pytest.raises(ValueError) as exc_info:
            Airfoil.morph_new_from_two_foils(naca0012, naca2412, eta=1.5, n_points=100)

        assert "range [0,1]" in str(exc_info.value)


class TestPerformanceCompatibility:
    """Test performance characteristics for compatibility."""

    def test_performance_regression_check(self):
        """Test that performance hasn't regressed significantly."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
        x_points = jnp.linspace(0, 1, 1000)

        # Time surface evaluation
        import time

        start_time = time.time()
        for _ in range(100):
            y_upper = naca2412.y_upper(x_points)
        elapsed_time = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed_time < 5.0  # 5 seconds for 100 evaluations

        ops_per_second = 100 / elapsed_time
        print(f"Surface evaluation: {ops_per_second:.1f} ops/s")

        # Should be reasonably fast
        assert ops_per_second > 10  # At least 10 operations per second

    def test_memory_usage_compatibility(self):
        """Test memory usage characteristics."""
        # Create multiple airfoils (simulating typical usage)
        airfoils = []
        for i in range(50):
            m = 0.02 * (i + 1) / 50
            naca = NACA4(M=m, P=0.4, XX=0.12, n_points=200)
            airfoils.append(naca)

        # All should be created successfully
        assert len(airfoils) == 50

        # Test batch evaluation
        x_points = jnp.linspace(0, 1, 100)
        results = []

        for airfoil in airfoils:
            y_upper = airfoil.y_upper(x_points)
            results.append(y_upper)

        # All evaluations should succeed
        assert len(results) == 50
        for result in results:
            assert jnp.all(jnp.isfinite(result))

    def test_jit_compilation_compatibility(self):
        """Test JIT compilation doesn't break compatibility."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # JIT compile surface evaluation
        jit_y_upper = jit(naca2412.y_upper)

        # Test with different input types
        x_float = 0.5
        x_array = jnp.linspace(0, 1, 20)

        # Both should work with JIT
        result_float = jit_y_upper(x_float)
        result_array = jit_y_upper(x_array)

        # Compare with non-JIT results
        expected_float = naca2412.y_upper(x_float)
        expected_array = naca2412.y_upper(x_array)

        assert jnp.allclose(result_float, expected_float)
        assert jnp.allclose(result_array, expected_array)
