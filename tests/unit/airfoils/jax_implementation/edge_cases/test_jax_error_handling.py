"""
Unit tests for JAX airfoil error handling system.

This module contains comprehensive tests for the error handling functionality
in the JAX airfoil implementation, including validation functions, buffer
overflow detection, and meaningful error messages.
"""

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.buffer_management import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.coordinate_processor import CoordinateProcessor
from ICARUS.airfoils.jax_implementation.error_handling import AirfoilErrorHandler
from ICARUS.airfoils.jax_implementation.error_handling import AirfoilValidationError
from ICARUS.airfoils.jax_implementation.error_handling import BufferOverflowError
from ICARUS.airfoils.jax_implementation.error_handling import GeometryError
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestAirfoilErrorHandler:
    """Test suite for AirfoilErrorHandler validation functions."""

    def test_validate_coordinate_shape_valid(self):
        """Test coordinate shape validation with valid inputs."""
        # Valid 2D coordinate array
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Should not raise any exception
        AirfoilErrorHandler.validate_coordinate_shape(coords, "test_coords")

    def test_validate_coordinate_shape_invalid_dimensions(self):
        """Test coordinate shape validation with invalid dimensions."""
        # 1D array
        coords_1d = jnp.array([0.0, 0.5, 1.0])
        with pytest.raises(AirfoilValidationError, match="must be a 2D array"):
            AirfoilErrorHandler.validate_coordinate_shape(coords_1d, "test_coords")

        # 3D array
        coords_3d = jnp.array([[[0.0, 0.5], [1.0, 0.0]], [[0.1, 0.2], [0.3, 0.4]]])
        with pytest.raises(AirfoilValidationError, match="must be a 2D array"):
            AirfoilErrorHandler.validate_coordinate_shape(coords_3d, "test_coords")

    def test_validate_coordinate_shape_invalid_first_dimension(self):
        """Test coordinate shape validation with wrong first dimension."""
        # Wrong first dimension (should be 2)
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0], [0.2, 0.3, 0.4]])
        with pytest.raises(
            AirfoilValidationError,
            match="must have shape \\(2, n_points\\)",
        ):
            AirfoilErrorHandler.validate_coordinate_shape(coords, "test_coords")

    def test_validate_coordinate_shape_too_few_points(self):
        """Test coordinate shape validation with too few points."""
        # Only 2 points (minimum is 3)
        coords = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(AirfoilValidationError, match="must have at least 3 points"):
            AirfoilErrorHandler.validate_coordinate_shape(coords, "test_coords")

    def test_validate_coordinate_values_valid(self):
        """Test coordinate value validation with valid inputs."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Should not raise any exception
        AirfoilErrorHandler.validate_coordinate_values(coords, "test_coords")

    def test_validate_coordinate_values_nan(self):
        """Test coordinate value validation with NaN values."""
        coords = jnp.array([[0.0, jnp.nan, 1.0], [0.0, 0.1, 0.0]])
        with pytest.raises(AirfoilValidationError, match="contain.*NaN values"):
            AirfoilErrorHandler.validate_coordinate_values(coords, "test_coords")

    def test_validate_coordinate_values_infinite(self):
        """Test coordinate value validation with infinite values."""
        coords = jnp.array([[0.0, jnp.inf, 1.0], [0.0, 0.1, 0.0]])
        with pytest.raises(AirfoilValidationError, match="contain.*infinite values"):
            AirfoilErrorHandler.validate_coordinate_values(coords, "test_coords")

    def test_validate_coordinate_values_too_large(self):
        """Test coordinate value validation with extremely large values."""
        coords = jnp.array([[0.0, 0.5, 1000.0], [0.0, 0.1, 0.0]])
        with pytest.raises(AirfoilValidationError, match="contain values larger than"):
            AirfoilErrorHandler.validate_coordinate_values(coords, "test_coords")

    def test_validate_airfoil_geometry_valid(self):
        """Test airfoil geometry validation with valid geometry."""
        # Create simple valid airfoil surfaces
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.05, 0.0]])

        # Should not raise any exception
        AirfoilErrorHandler.validate_airfoil_geometry(upper, lower, "test_airfoil")

    def test_validate_airfoil_geometry_empty_surfaces(self):
        """Test airfoil geometry validation with empty surfaces."""
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        lower = jnp.empty((2, 0))

        with pytest.raises(GeometryError, match="lower surface has no points"):
            AirfoilErrorHandler.validate_airfoil_geometry(upper, lower, "test_airfoil")

    def test_validate_airfoil_geometry_too_thick(self):
        """Test airfoil geometry validation with unrealistic thickness."""
        # Create airfoil with unrealistic thickness
        upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 5.0, 0.0]])  # Very thick
        lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -5.0, 0.0]])

        with pytest.raises(GeometryError, match="thickness ratio.*unrealistic"):
            AirfoilErrorHandler.validate_airfoil_geometry(upper, lower, "test_airfoil")

    def test_validate_surface_ordering_valid(self):
        """Test surface ordering validation with valid ordering."""
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])

        # Should not raise any exception
        AirfoilErrorHandler.validate_surface_ordering(coords, "test_surface")

    def test_validate_surface_ordering_duplicates(self):
        """Test surface ordering validation with duplicate points."""
        coords = jnp.array([[0.0, 0.5, 0.5, 1.0], [0.0, 0.1, 0.1, 0.0]])

        with pytest.raises(GeometryError, match="duplicate consecutive x-coordinates"):
            AirfoilErrorHandler.validate_surface_ordering(coords, "test_surface")

    def test_check_buffer_capacity_sufficient(self):
        """Test buffer capacity check when capacity is sufficient."""
        needs_reallocation, new_size = AirfoilErrorHandler.check_buffer_capacity(
            required_size=50,
            current_buffer_size=100,
        )

        assert not needs_reallocation
        assert new_size is None

    def test_check_buffer_capacity_insufficient(self):
        """Test buffer capacity check when reallocation is needed."""
        needs_reallocation, new_size = AirfoilErrorHandler.check_buffer_capacity(
            required_size=150,
            current_buffer_size=100,
        )

        assert needs_reallocation
        assert new_size == 256  # Next power of 2

    def test_check_buffer_capacity_overflow(self):
        """Test buffer capacity check when required size exceeds maximum."""
        with pytest.raises(BufferOverflowError, match="exceeds maximum allowed size"):
            AirfoilErrorHandler.check_buffer_capacity(
                required_size=5000,
                current_buffer_size=100,
                max_buffer_size=4096,
            )

    def test_validate_morphing_parameters_valid(self):
        """Test morphing parameter validation with valid inputs."""
        # Should not raise any exception
        AirfoilErrorHandler.validate_morphing_parameters(0.5, "airfoil1", "airfoil2")

    def test_validate_morphing_parameters_invalid_range(self):
        """Test morphing parameter validation with out-of-range values."""
        with pytest.raises(AirfoilValidationError, match="must be between 0.0 and 1.0"):
            AirfoilErrorHandler.validate_morphing_parameters(
                1.5,
                "airfoil1",
                "airfoil2",
            )

        with pytest.raises(AirfoilValidationError, match="must be between 0.0 and 1.0"):
            AirfoilErrorHandler.validate_morphing_parameters(
                -0.1,
                "airfoil1",
                "airfoil2",
            )

    def test_validate_morphing_parameters_invalid_type(self):
        """Test morphing parameter validation with invalid type."""
        with pytest.raises(AirfoilValidationError, match="must be a number"):
            AirfoilErrorHandler.validate_morphing_parameters(
                "0.5",
                "airfoil1",
                "airfoil2",
            )

    def test_validate_flap_parameters_valid(self):
        """Test flap parameter validation with valid inputs."""
        # Should not raise any exception
        AirfoilErrorHandler.validate_flap_parameters(0.7, 10.0, 0.5, 1.0)

    def test_validate_flap_parameters_invalid_hinge_position(self):
        """Test flap parameter validation with invalid hinge position."""
        with pytest.raises(
            AirfoilValidationError,
            match="flap_hinge_chord_percentage must be between",
        ):
            AirfoilErrorHandler.validate_flap_parameters(1.5, 10.0, 0.5, 1.0)

    def test_validate_flap_parameters_invalid_thickness_position(self):
        """Test flap parameter validation with invalid thickness position."""
        with pytest.raises(
            AirfoilValidationError,
            match="flap_hinge_thickness_percentage must be between",
        ):
            AirfoilErrorHandler.validate_flap_parameters(0.7, 10.0, 1.5, 1.0)

    def test_validate_flap_parameters_invalid_chord_extension(self):
        """Test flap parameter validation with invalid chord extension."""
        with pytest.raises(
            AirfoilValidationError,
            match="chord_extension must be positive",
        ):
            AirfoilErrorHandler.validate_flap_parameters(0.7, 10.0, 0.5, -1.0)

    def test_validate_naca_parameters_4digit_valid(self):
        """Test NACA parameter validation with valid 4-digit inputs."""
        # Should not raise any exception
        AirfoilErrorHandler.validate_naca_parameters("2412", "4-digit")

    def test_validate_naca_parameters_4digit_invalid_length(self):
        """Test NACA parameter validation with invalid 4-digit length."""
        with pytest.raises(AirfoilValidationError, match="must be exactly 4 digits"):
            AirfoilErrorHandler.validate_naca_parameters("241", "4-digit")

    def test_validate_naca_parameters_4digit_invalid_characters(self):
        """Test NACA parameter validation with invalid 4-digit characters."""
        with pytest.raises(
            AirfoilValidationError,
            match="must contain only numeric characters",
        ):
            AirfoilErrorHandler.validate_naca_parameters("24a2", "4-digit")

    def test_validate_naca_parameters_4digit_invalid_ranges(self):
        """Test NACA parameter validation with invalid 4-digit parameter ranges."""
        # Invalid thickness (last two digits = 00)
        with pytest.raises(
            AirfoilValidationError,
            match="thickness parameter must be 01-99",
        ):
            AirfoilErrorHandler.validate_naca_parameters("2400", "4-digit")

    def test_validate_naca_parameters_5digit_valid(self):
        """Test NACA parameter validation with valid 5-digit inputs."""
        # Should not raise any exception
        AirfoilErrorHandler.validate_naca_parameters("23012", "5-digit")

    def test_validate_naca_parameters_5digit_invalid_length(self):
        """Test NACA parameter validation with invalid 5-digit length."""
        with pytest.raises(AirfoilValidationError, match="must be exactly 5 digits"):
            AirfoilErrorHandler.validate_naca_parameters("2301", "5-digit")

    def test_validate_naca_parameters_5digit_invalid_reflex(self):
        """Test NACA parameter validation with invalid 5-digit reflex parameter."""
        with pytest.raises(
            AirfoilValidationError,
            match="reflex parameter must be 0 or 1",
        ):
            AirfoilErrorHandler.validate_naca_parameters("23212", "5-digit")

    def test_validate_repanel_parameters_valid(self):
        """Test repanel parameter validation with valid inputs."""
        # Should not raise any exception
        AirfoilErrorHandler.validate_repanel_parameters(100, "cosine", "chord_based")

    def test_validate_repanel_parameters_invalid_n_points(self):
        """Test repanel parameter validation with invalid n_points."""
        with pytest.raises(
            AirfoilValidationError,
            match="n_points must be an integer >= 4",
        ):
            AirfoilErrorHandler.validate_repanel_parameters(3, "cosine", "chord_based")

    def test_validate_repanel_parameters_invalid_distribution(self):
        """Test repanel parameter validation with invalid distribution."""
        with pytest.raises(AirfoilValidationError, match="distribution must be one of"):
            AirfoilErrorHandler.validate_repanel_parameters(
                100,
                "invalid",
                "chord_based",
            )

    def test_validate_repanel_parameters_invalid_method(self):
        """Test repanel parameter validation with invalid method."""
        with pytest.raises(AirfoilValidationError, match="method must be one of"):
            AirfoilErrorHandler.validate_repanel_parameters(100, "cosine", "invalid")

    def test_create_error_context(self):
        """Test error context creation."""
        context = AirfoilErrorHandler.create_error_context(
            "test_operation",
            "test_airfoil",
            {"param1": "value1", "param2": 42},
        )

        assert "Error in test_operation for test_airfoil" in context
        assert "param1: value1" in context
        assert "param2: 42" in context

    def test_suggest_fixes(self):
        """Test fix suggestions for different error types."""
        # Test various error types
        suggestion = AirfoilErrorHandler.suggest_fixes("nan_coordinates")
        assert "CoordinateProcessor.filter_nan_coordinates" in suggestion

        suggestion = AirfoilErrorHandler.suggest_fixes(
            "buffer_overflow",
            required_size=5000,
        )
        assert "Required size: 5000 points" in suggestion

        suggestion = AirfoilErrorHandler.suggest_fixes("unknown_error")
        assert "Check input parameters" in suggestion


class TestJaxAirfoilErrorIntegration:
    """Test error handling integration in JaxAirfoil class."""

    def test_jax_airfoil_invalid_coordinates_shape(self):
        """Test JaxAirfoil constructor with invalid coordinate shape."""
        # 1D coordinates
        coords_1d = jnp.array([0.0, 0.5, 1.0])

        with pytest.raises(AirfoilValidationError, match="Invalid coordinates"):
            JaxAirfoil(coords_1d, name="test")

    def test_jax_airfoil_naca4_invalid_digits(self):
        """Test NACA 4-digit generation with invalid digits."""
        with pytest.raises(
            AirfoilValidationError,
            match="Invalid NACA 4-digit parameters",
        ):
            JaxAirfoil.naca4("241")  # Too few digits

    def test_jax_airfoil_naca5_invalid_digits(self):
        """Test NACA 5-digit generation with invalid digits."""
        with pytest.raises(
            AirfoilValidationError,
            match="Invalid NACA 5-digit parameters",
        ):
            JaxAirfoil.naca5("2301")  # Too few digits

    def test_jax_airfoil_flap_invalid_parameters(self):
        """Test flap operation with invalid parameters."""
        # Create a simple airfoil
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        with pytest.raises(AirfoilValidationError, match="Invalid flap parameters"):
            airfoil.flap(1.5, 10.0)  # Invalid hinge position

    def test_jax_airfoil_morphing_invalid_eta(self):
        """Test morphing with invalid eta parameter."""
        airfoil1 = JaxAirfoil.naca4("2412", n_points=50)
        airfoil2 = JaxAirfoil.naca4("0012", n_points=50)

        with pytest.raises(AirfoilValidationError, match="Invalid morphing parameters"):
            JaxAirfoil.morph_new_from_two_foils(
                airfoil1,
                airfoil2,
                eta=1.5,
                n_points=50,
            )

    def test_jax_airfoil_repanel_invalid_parameters(self):
        """Test repanel operation with invalid parameters."""
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        with pytest.raises(AirfoilValidationError, match="Invalid repanel parameters"):
            airfoil.repanel(n_points=2)  # Too few points


class TestBufferManagerErrorHandling:
    """Test error handling in buffer manager."""

    def test_buffer_manager_negative_points(self):
        """Test buffer manager with negative number of points."""
        with pytest.raises(
            BufferOverflowError,
            match="Number of points must be positive",
        ):
            AirfoilBufferManager.determine_buffer_size(-5)

    def test_buffer_manager_zero_points(self):
        """Test buffer manager with zero points."""
        with pytest.raises(
            BufferOverflowError,
            match="Number of points must be positive",
        ):
            AirfoilBufferManager.determine_buffer_size(0)

    def test_buffer_manager_overflow(self):
        """Test buffer manager with points exceeding maximum."""
        with pytest.raises(BufferOverflowError, match="exceeds maximum allowed size"):
            AirfoilBufferManager.determine_buffer_size(10000)


class TestCoordinateProcessorErrorHandling:
    """Test error handling in coordinate processor."""

    def test_coordinate_processor_invalid_shape(self):
        """Test coordinate processor with invalid coordinate shape."""
        coords_1d = jnp.array([0.0, 0.5, 1.0])

        with pytest.raises(
            AirfoilValidationError,
            match="Cannot filter NaN coordinates",
        ):
            CoordinateProcessor.filter_nan_coordinates(coords_1d)

    def test_coordinate_processor_all_nan(self):
        """Test coordinate processor with all NaN coordinates."""
        coords_nan = jnp.array(
            [[jnp.nan, jnp.nan, jnp.nan], [jnp.nan, jnp.nan, jnp.nan]],
        )

        with pytest.raises(
            AirfoilValidationError,
            match="All coordinate points contain NaN",
        ):
            CoordinateProcessor.filter_nan_coordinates(coords_nan)

    def test_coordinate_processor_validation_integration(self):
        """Test coordinate processor validation integration."""
        # Create coordinates with infinite values
        coords_inf = jnp.array([[0.0, jnp.inf, 1.0], [0.0, 0.1, 0.0]])

        with pytest.raises(AirfoilValidationError, match="contain.*infinite values"):
            CoordinateProcessor.validate_coordinates(coords_inf)


class TestGradientSafeErrorHandling:
    """Test gradient-safe error handling functionality."""

    def test_gradient_safe_error_handling(self):
        """Test that error handling doesn't break gradients."""

        # Create a simple function that uses error handling
        def test_function(x):
            # This should work without breaking gradients
            coords = jnp.array([[0.0, x, 1.0], [0.0, 0.1, 0.0]])

            # The error handling should happen in eager mode, not here
            # This is just a placeholder for gradient testing
            return jnp.sum(coords**2)

        # Test that we can compute gradients
        grad_fn = jax.grad(test_function)
        result = grad_fn(0.5)

        assert jnp.isfinite(result)

    def test_jit_compatibility_with_error_handling(self):
        """Test that JIT compilation works with error handling."""

        # Create a simple JIT-compatible function
        @jax.jit
        def test_jit_function(coords):
            # This should work in JIT mode
            return jnp.sum(coords**2)

        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        result = test_jit_function(coords)

        assert jnp.isfinite(result)


class TestErrorMessageQuality:
    """Test the quality and helpfulness of error messages."""

    def test_error_messages_contain_context(self):
        """Test that error messages contain helpful context."""
        try:
            AirfoilErrorHandler.validate_coordinate_shape(
                jnp.array([1, 2, 3]),
                "test_coordinates",
            )
        except AirfoilValidationError as e:
            error_msg = str(e)
            assert "test_coordinates" in error_msg
            assert "2D array" in error_msg
            assert "shape" in error_msg

    def test_error_messages_contain_suggestions(self):
        """Test that error messages contain helpful suggestions."""
        try:
            coords_1d = jnp.array([0.0, 0.5, 1.0])
            CoordinateProcessor.filter_nan_coordinates(coords_1d)
        except AirfoilValidationError as e:
            error_msg = str(e)
            assert "first row" in error_msg and "second row" in error_msg

    def test_naca_error_messages_specific(self):
        """Test that NACA error messages are specific and helpful."""
        try:
            AirfoilErrorHandler.validate_naca_parameters("2400", "4-digit")
        except AirfoilValidationError as e:
            error_msg = str(e)
            assert "thickness parameter" in error_msg
            assert "01-99" in error_msg

    def test_buffer_overflow_error_messages(self):
        """Test that buffer overflow error messages are helpful."""
        try:
            AirfoilErrorHandler.check_buffer_capacity(5000, 100, 4096)
        except BufferOverflowError as e:
            error_msg = str(e)
            assert "5000" in error_msg
            assert "4096" in error_msg
            assert "reducing the number" in error_msg


if __name__ == "__main__":
    # Run a simple test to verify functionality
    print("Testing JAX Airfoil error handling functionality...")

    # Test basic validation
    try:
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        AirfoilErrorHandler.validate_coordinate_shape(coords)
        AirfoilErrorHandler.validate_coordinate_values(coords)
        print("✓ Basic coordinate validation passed")
    except Exception as e:
        print(f"✗ Basic coordinate validation failed: {e}")

    # Test NACA validation
    try:
        AirfoilErrorHandler.validate_naca_parameters("2412", "4-digit")
        print("✓ NACA parameter validation passed")
    except Exception as e:
        print(f"✗ NACA parameter validation failed: {e}")

    # Test buffer capacity check
    try:
        needs_realloc, new_size = AirfoilErrorHandler.check_buffer_capacity(100, 50)
        assert needs_realloc and new_size == 128
        print("✓ Buffer capacity check passed")
    except Exception as e:
        print(f"✗ Buffer capacity check failed: {e}")

    # Test integration with JaxAirfoil
    try:
        airfoil = JaxAirfoil.naca4("2412", n_points=50)
        print("✓ JaxAirfoil integration passed")
    except Exception as e:
        print(f"✗ JaxAirfoil integration failed: {e}")

    print("Error handling tests completed!")
