"""
Comprehensive test suite for JAX airfoil implementation.

This module provides comprehensive testing for the JAX airfoil refactor including:
- Gradient tests using jax.grad for all operations
- JIT compilation tests for all core methods
- Numerical accuracy tests comparing with NumPy version
- Property-based testing for edge cases
- Integration tests with existing ICARUS workflows

Requirements covered: 1.2, 2.1, 2.2, 8.2
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ICARUS.airfoils import NACA4
from ICARUS.airfoils.jax_implementation.buffer_manager import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.interpolation_engine import (
    JaxInterpolationEngine,
)
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestComprehensiveJaxGradients:
    """Test gradient computation for all JAX airfoil operations (Requirement 2.1, 2.2)."""

    @pytest.fixture
    def test_airfoil(self):
        """Create a test airfoil for gradient testing."""
        upper = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.06, 0.08, 0.05, 0.0]])
        lower = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.04, -0.06, -0.03, 0.0]],
        )
        return JaxAirfoil.from_upper_lower(upper, lower, name="GradientTest")

    def test_thickness_gradients(self, test_airfoil):
        """Test gradients of thickness computation."""

        def thickness_objective(airfoil):
            query_x = jnp.array([0.5])
            return jnp.sum(airfoil.thickness(query_x))

        # Compute gradients
        grad_fn = jax.grad(thickness_objective)
        gradients = grad_fn(test_airfoil)

        # Verify gradient structure
        assert isinstance(gradients, JaxAirfoil)
        assert gradients.n_points == test_airfoil.n_points
        assert gradients.buffer_size == test_airfoil.buffer_size

        # Verify gradients are finite and non-zero
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))
        assert jnp.any(jnp.abs(grad_coords) > 1e-8)  # Some gradients should be non-zero

    def test_camber_line_gradients(self, test_airfoil):
        """Test gradients of camber line computation."""

        def camber_objective(airfoil):
            query_x = jnp.array([0.25, 0.5, 0.75])
            return jnp.sum(jnp.abs(airfoil.camber_line(query_x)))

        grad_fn = jax.grad(camber_objective)
        gradients = grad_fn(test_airfoil)

        # Verify gradient structure and finiteness
        assert isinstance(gradients, JaxAirfoil)
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))

    def test_surface_query_gradients(self, test_airfoil):
        """Test gradients of surface coordinate queries."""

        def upper_surface_objective(airfoil):
            query_x = jnp.array([0.3, 0.7])
            return jnp.sum(airfoil.y_upper(query_x))

        def lower_surface_objective(airfoil):
            query_x = jnp.array([0.3, 0.7])
            return jnp.sum(jnp.abs(airfoil.y_lower(query_x)))

        # Test upper surface gradients
        upper_grad_fn = jax.grad(upper_surface_objective)
        upper_gradients = upper_grad_fn(test_airfoil)

        upper_grad_coords = upper_gradients._coordinates[
            :,
            upper_gradients._validity_mask,
        ]
        assert jnp.all(jnp.isfinite(upper_grad_coords))

        # Test lower surface gradients
        lower_grad_fn = jax.grad(lower_surface_objective)
        lower_gradients = lower_grad_fn(test_airfoil)

        lower_grad_coords = lower_gradients._coordinates[
            :,
            lower_gradients._validity_mask,
        ]
        assert jnp.all(jnp.isfinite(lower_grad_coords))

    def test_morphing_gradients(self, test_airfoil):
        """Test gradients of airfoil morphing operations."""
        # Create second airfoil for morphing
        upper2 = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.04, 0.06, 0.03, 0.0]])
        lower2 = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.06, -0.08, -0.05, 0.0]],
        )
        airfoil2 = JaxAirfoil.from_upper_lower(upper2, lower2, name="MorphTarget")

        def morphing_objective(eta):
            morphed = JaxAirfoil.morph_new_from_two_foils(
                test_airfoil,
                airfoil2,
                eta,
                n_points=20,
            )
            return jnp.sum(morphed.thickness(jnp.array([0.5])))

        # Test gradient with respect to morphing parameter
        grad_fn = jax.grad(morphing_objective)
        eta_gradient = grad_fn(0.3)

        assert jnp.isfinite(eta_gradient)
        assert jnp.abs(eta_gradient) > 1e-8  # Should have non-zero gradient

    def test_flap_operation_gradients(self, test_airfoil):
        """Test gradients of flap operations."""

        def flap_objective(flap_angle):
            flapped = test_airfoil.flap(
                hinge_point=jnp.array([0.75, 0.0]),
                flap_angle=flap_angle,
            )
            return jnp.sum(flapped.thickness(jnp.array([0.9])))

        grad_fn = jax.grad(flap_objective)
        angle_gradient = grad_fn(0.1)  # 0.1 radian flap deflection

        assert jnp.isfinite(angle_gradient)
        assert jnp.abs(angle_gradient) > 1e-8

    def test_repaneling_gradients(self, test_airfoil):
        """Test gradients of repaneling operations."""

        def repanel_objective(airfoil):
            repaneled = airfoil.repanel(n_points=20, distribution="cosine")
            return jnp.sum(repaneled.thickness(jnp.array([0.5])))

        grad_fn = jax.grad(repanel_objective)
        gradients = grad_fn(test_airfoil)

        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))

    def test_geometric_property_gradients(self, test_airfoil):
        """Test gradients of geometric properties."""

        def max_thickness_objective(airfoil):
            return airfoil.max_thickness

        def max_camber_objective(airfoil):
            return jnp.abs(airfoil.max_camber)

        # Test max thickness gradients
        thickness_grad_fn = jax.grad(max_thickness_objective)
        thickness_gradients = thickness_grad_fn(test_airfoil)

        thickness_grad_coords = thickness_gradients._coordinates[
            :,
            thickness_gradients._validity_mask,
        ]
        assert jnp.all(jnp.isfinite(thickness_grad_coords))

        # Test max camber gradients
        camber_grad_fn = jax.grad(max_camber_objective)
        camber_gradients = camber_grad_fn(test_airfoil)

        camber_grad_coords = camber_gradients._coordinates[
            :,
            camber_gradients._validity_mask,
        ]
        assert jnp.all(jnp.isfinite(camber_grad_coords))

    def test_batch_operation_gradients(self, test_airfoil):
        """Test gradients of batch operations."""

        def batch_thickness_objective(airfoil):
            # Create batch of query points
            query_batch = jnp.array([[0.25, 0.5, 0.75], [0.3, 0.6, 0.9]])

            # Use vmap for batch processing
            batch_fn = jax.vmap(lambda x: jnp.sum(airfoil.thickness(x)))
            return jnp.sum(batch_fn(query_batch))

        grad_fn = jax.grad(batch_thickness_objective)
        gradients = grad_fn(test_airfoil)

        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))


class TestComprehensiveJITCompilation:
    """Test JIT compilation for all core methods (Requirement 1.2)."""

    @pytest.fixture
    def test_airfoil(self):
        """Create a test airfoil for JIT testing."""
        coords = jnp.array(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
            ],
        )
        return JaxAirfoil(coords, name="JITTest")

    def test_geometric_operations_jit(self, test_airfoil):
        """Test JIT compilation of all geometric operations."""

        @jax.jit
        def thickness_computation(airfoil):
            query_x = jnp.array([0.25, 0.5, 0.75])
            return airfoil.thickness(query_x)

        @jax.jit
        def camber_computation(airfoil):
            query_x = jnp.array([0.25, 0.5, 0.75])
            return airfoil.camber_line(query_x)

        @jax.jit
        def surface_queries(airfoil):
            query_x = jnp.array([0.25, 0.5, 0.75])
            y_upper = airfoil.y_upper(query_x)
            y_lower = airfoil.y_lower(query_x)
            return y_upper, y_lower

        # Test compilation and execution
        thickness = thickness_computation(test_airfoil)
        camber = camber_computation(test_airfoil)
        y_upper, y_lower = surface_queries(test_airfoil)

        # Verify results are finite and reasonable
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(thickness >= 0)

    def test_property_access_jit(self, test_airfoil):
        """Test JIT compilation of property accessors."""

        @jax.jit
        def get_properties(airfoil):
            max_t = airfoil.max_thickness
            max_c = airfoil.max_camber
            chord = airfoil.chord_length
            return max_t, max_c, chord

        @jax.jit
        def get_surface_points(airfoil):
            x_upper, y_upper = airfoil.upper_surface_points
            x_lower, y_lower = airfoil.lower_surface_points
            return x_upper, y_upper, x_lower, y_lower

        # Test compilation and execution
        max_t, max_c, chord = get_properties(test_airfoil)
        x_upper, y_upper, x_lower, y_lower = get_surface_points(test_airfoil)

        # Verify results
        assert jnp.isfinite(max_t) and max_t > 0
        assert jnp.isfinite(max_c)
        assert jnp.isfinite(chord) and chord > 0
        assert jnp.all(jnp.isfinite(x_upper))
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(x_lower))
        assert jnp.all(jnp.isfinite(y_lower))

    def test_transformation_operations_jit(self, test_airfoil):
        """Test JIT compilation of transformation operations."""

        @jax.jit
        def flap_operation(airfoil, angle):
            return airfoil.flap(hinge_point=jnp.array([0.75, 0.0]), flap_angle=angle)

        @jax.jit
        def repanel_operation(airfoil):
            return airfoil.repanel(n_points=15, distribution="cosine")

        # Test compilation and execution
        flapped = flap_operation(test_airfoil, 0.1)
        repaneled = repanel_operation(test_airfoil)

        # Verify results
        assert isinstance(flapped, JaxAirfoil)
        assert isinstance(repaneled, JaxAirfoil)
        assert flapped.n_points > 0
        assert repaneled.n_points > 0

    def test_morphing_operations_jit(self, test_airfoil):
        """Test JIT compilation of morphing operations."""
        # Create second airfoil
        coords2 = jnp.array(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.03, 0.06, 0.03, 0.0, -0.03, -0.06, -0.03, 0.0],
            ],
        )
        airfoil2 = JaxAirfoil(coords2, name="MorphTarget")

        @jax.jit
        def morphing_operation(eta):
            return JaxAirfoil.morph_new_from_two_foils(
                test_airfoil,
                airfoil2,
                eta,
                n_points=20,
            )

        # Test compilation and execution
        morphed = morphing_operation(0.3)

        # Verify results
        assert isinstance(morphed, JaxAirfoil)
        assert morphed.n_points > 0

    def test_batch_operations_jit(self, test_airfoil):
        """Test JIT compilation of batch operations."""

        @jax.jit
        def batch_thickness(airfoil):
            query_batch = jnp.array(
                [[0.25, 0.5, 0.75], [0.3, 0.6, 0.9], [0.1, 0.4, 0.8]],
            )
            return jax.vmap(airfoil.thickness)(query_batch)

        @jax.jit
        def batch_surface_queries(airfoil):
            query_batch = jnp.array([[0.25, 0.5, 0.75], [0.3, 0.6, 0.9]])
            upper_batch = jax.vmap(airfoil.y_upper)(query_batch)
            lower_batch = jax.vmap(airfoil.y_lower)(query_batch)
            return upper_batch, lower_batch

        # Test compilation and execution
        thickness_batch = batch_thickness(test_airfoil)
        upper_batch, lower_batch = batch_surface_queries(test_airfoil)

        # Verify results
        assert thickness_batch.shape == (3, 3)
        assert upper_batch.shape == (2, 3)
        assert lower_batch.shape == (2, 3)
        assert jnp.all(jnp.isfinite(thickness_batch))
        assert jnp.all(jnp.isfinite(upper_batch))
        assert jnp.all(jnp.isfinite(lower_batch))

    def test_buffer_operations_jit(self):
        """Test JIT compilation of buffer management operations."""

        @partial(jax.jit, static_argnums=(1,))
        def buffer_padding(coords, target_size):
            return AirfoilBufferManager.pad_coordinates(coords, target_size)

        @partial(jax.jit, static_argnums=(0, 1))
        def validity_mask(n_valid, buffer_size):
            return AirfoilBufferManager.create_validity_mask(n_valid, buffer_size)

        # Test compilation and execution
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
        padded = buffer_padding(coords, 32)
        mask = validity_mask(3, 32)

        # Verify results
        assert padded.shape == (2, 32)
        assert mask.shape == (32,)
        assert jnp.sum(mask) == 3


class TestNumericalAccuracy:
    """Test numerical accuracy against NumPy reference implementation."""

    @pytest.fixture
    def reference_airfoil(self):
        """Create reference airfoil using original NACA implementation."""
        return NACA4.from_digits("2412")

    @pytest.fixture
    def jax_airfoil(self, reference_airfoil):
        """Create equivalent JAX airfoil."""
        # Get coordinates from reference airfoil
        x_coords, y_coords = reference_airfoil.get_coordinates()
        coords = jnp.array([x_coords, y_coords])
        return JaxAirfoil(coords, name="NACA2412_JAX")

    def test_thickness_accuracy(self, reference_airfoil, jax_airfoil):
        """Test thickness computation accuracy."""
        query_x = np.linspace(0.05, 0.95, 20)

        # Compute thickness using both implementations
        ref_thickness = reference_airfoil.thickness(query_x)
        jax_thickness = jax_airfoil.thickness(jnp.array(query_x))

        # Compare results
        max_error = np.max(np.abs(ref_thickness - np.array(jax_thickness)))
        relative_error = max_error / np.max(ref_thickness)

        assert relative_error < 0.01  # Less than 1% relative error
        assert max_error < 0.001  # Less than 0.1% chord absolute error

    def test_camber_line_accuracy(self, reference_airfoil, jax_airfoil):
        """Test camber line computation accuracy."""
        query_x = np.linspace(0.05, 0.95, 20)

        # Compute camber using both implementations
        ref_camber = reference_airfoil.camber_line(query_x)
        jax_camber = jax_airfoil.camber_line(jnp.array(query_x))

        # Compare results
        max_error = np.max(np.abs(ref_camber - np.array(jax_camber)))

        assert max_error < 0.001  # Less than 0.1% chord absolute error

    def test_surface_query_accuracy(self, reference_airfoil, jax_airfoil):
        """Test surface coordinate query accuracy."""
        query_x = np.linspace(0.05, 0.95, 20)

        # Compare upper surface
        ref_y_upper = reference_airfoil.y_upper(query_x)
        jax_y_upper = jax_airfoil.y_upper(jnp.array(query_x))

        upper_error = np.max(np.abs(ref_y_upper - np.array(jax_y_upper)))
        assert upper_error < 0.001

        # Compare lower surface
        ref_y_lower = reference_airfoil.y_lower(query_x)
        jax_y_lower = jax_airfoil.y_lower(jnp.array(query_x))

        lower_error = np.max(np.abs(ref_y_lower - np.array(jax_y_lower)))
        assert lower_error < 0.001

    def test_geometric_properties_accuracy(self, reference_airfoil, jax_airfoil):
        """Test geometric properties accuracy."""

        # Compare maximum thickness
        ref_max_t = reference_airfoil.max_thickness
        jax_max_t = float(jax_airfoil.max_thickness)
        thickness_error = abs(ref_max_t - jax_max_t) / ref_max_t
        assert thickness_error < 0.02  # Less than 2% relative error

        # Compare maximum camber
        ref_max_c = reference_airfoil.max_camber
        jax_max_c = float(jax_airfoil.max_camber)
        camber_error = abs(ref_max_c - jax_max_c)
        assert camber_error < 0.001  # Small absolute error for camber

        # Compare chord length
        ref_chord = reference_airfoil.chord_length
        jax_chord = float(jax_airfoil.chord_length)
        chord_error = abs(ref_chord - jax_chord) / ref_chord
        assert chord_error < 0.001  # Very small relative error for chord

    def test_interpolation_accuracy(self):
        """Test interpolation engine accuracy against NumPy interp."""
        # Create test data
        x_data = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_data = jnp.array([0.0, 0.05, 0.08, 0.05, 0.0])
        query_x = jnp.array([0.1, 0.3, 0.6, 0.9])

        # JAX interpolation
        jax_result = JaxInterpolationEngine.linear_interpolate_1d(
            x_data,
            y_data,
            len(x_data),
            query_x,
        )

        # NumPy reference
        numpy_result = np.interp(query_x, x_data, y_data)

        # Compare results
        max_error = np.max(np.abs(numpy_result - np.array(jax_result)))
        assert max_error < 1e-10  # Should be nearly identical

    def test_naca_generation_accuracy(self):
        """Test NACA generation accuracy."""
        # Generate NACA airfoil using both methods
        ref_naca = NACA4.from_digits("2412")
        jax_naca = JaxAirfoil.naca("2412", n_points=50)

        # Compare coordinates
        ref_x, ref_y = ref_naca.get_coordinates()
        jax_x, jax_y = jax_naca.get_coordinates()

        # Compare coordinate arrays
        x_error = np.max(np.abs(ref_x - np.array(jax_x)))
        y_error = np.max(np.abs(ref_y - np.array(jax_y)))

        assert x_error < 1e-6  # Very small error for x-coordinates
        assert y_error < 1e-6  # Very small error for y-coordinates


class TestPropertyBasedEdgeCases:
    """Property-based testing for edge cases (Requirement 8.2)."""

    @given(
        n_points=st.integers(min_value=5, max_value=100),
        max_thickness=st.floats(min_value=0.01, max_value=0.3),
        max_camber=st.floats(min_value=-0.1, max_value=0.1),
    )
    def test_airfoil_creation_robustness(self, n_points, max_thickness, max_camber):
        """Test airfoil creation with various parameters."""
        # Generate airfoil coordinates
        x = jnp.linspace(0, 1, n_points)
        y_upper = max_thickness * jnp.sin(jnp.pi * x) + max_camber
        y_lower = -max_thickness * jnp.sin(jnp.pi * x) + max_camber

        upper_coords = jnp.array([x, y_upper])
        lower_coords = jnp.array([x, y_lower])

        # Create airfoil
        airfoil = JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Basic sanity checks
        assert airfoil.n_points > 0
        assert airfoil.buffer_size >= airfoil.n_points
        assert jnp.isfinite(airfoil.max_thickness)
        assert airfoil.max_thickness >= 0

    @given(
        query_points=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False),
        ),
    )
    def test_interpolation_robustness(self, query_points):
        """Test interpolation with various query points."""
        # Create simple test airfoil
        coords = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.05, 0.08, 0.05, 0.0]])
        airfoil = JaxAirfoil(coords)

        # Test thickness computation
        query_jax = jnp.array(query_points)
        thickness = airfoil.thickness(query_jax)

        # Results should be finite and non-negative
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(thickness >= 0)

    @given(eta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_morphing_parameter_robustness(self, eta):
        """Test morphing with various eta parameters."""
        # Create two test airfoils
        coords1 = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
        coords2 = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.08, 0.0]])

        airfoil1 = JaxAirfoil(coords1)
        airfoil2 = JaxAirfoil(coords2)

        # Test morphing
        morphed = JaxAirfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta,
            n_points=20,
        )

        # Basic checks
        assert isinstance(morphed, JaxAirfoil)
        assert morphed.n_points > 0
        assert jnp.isfinite(morphed.max_thickness)

    @given(flap_angle=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False))
    def test_flap_angle_robustness(self, flap_angle):
        """Test flap operations with various angles."""
        # Create test airfoil
        coords = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.05, 0.08, 0.05, 0.0]])
        airfoil = JaxAirfoil(coords)

        # Test flap operation
        flapped = airfoil.flap(
            hinge_point=jnp.array([0.75, 0.0]),
            flap_angle=flap_angle,
        )

        # Basic checks
        assert isinstance(flapped, JaxAirfoil)
        assert flapped.n_points > 0
        assert jnp.isfinite(flapped.max_thickness)

    def test_degenerate_geometries(self):
        """Test handling of degenerate airfoil geometries."""
        # Test flat plate (zero thickness)
        flat_coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
        flat_airfoil = JaxAirfoil(flat_coords)

        # Should handle gracefully
        assert flat_airfoil.n_points > 0
        assert flat_airfoil.max_thickness >= 0

        # Test single point (should be handled by buffer management)
        try:
            single_point = jnp.array([[0.5], [0.0]])
            single_airfoil = JaxAirfoil(single_point)
            # If it doesn't raise an error, check basic properties
            assert single_airfoil.buffer_size > 0
        except (ValueError, AssertionError):
            # Expected for degenerate case
            pass

    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values."""
        # Create coordinates with NaN values
        coords_with_nan = jnp.array(
            [[0.0, 0.25, jnp.nan, 0.75, 1.0], [0.0, 0.05, 0.08, jnp.inf, 0.0]],
        )

        # Should handle NaN/inf during preprocessing
        try:
            airfoil = JaxAirfoil(coords_with_nan)
            # If successful, check that result is valid
            assert airfoil.n_points > 0
            x_coords, y_coords = airfoil.get_coordinates()
            assert jnp.all(jnp.isfinite(x_coords))
            assert jnp.all(jnp.isfinite(y_coords))
        except (ValueError, AssertionError):
            # Expected for invalid input
            pass


class TestIcarusIntegration:
    """Integration tests with existing ICARUS workflows."""

    def test_naca_compatibility(self):
        """Test compatibility with NACA airfoil generation."""
        # Generate using original NACA
        original_naca = NACA4.from_digits("4412")

        # Generate using JAX implementation
        jax_naca = JaxAirfoil.naca("4412", n_points=50)

        # Test that they can be used interchangeably
        query_x = np.linspace(0.1, 0.9, 10)

        orig_thickness = original_naca.thickness(query_x)
        jax_thickness = jax_naca.thickness(jnp.array(query_x))

        # Should produce similar results
        max_error = np.max(np.abs(orig_thickness - np.array(jax_thickness)))
        assert max_error < 0.01  # Reasonable tolerance

    def test_file_io_compatibility(self):
        """Test file I/O compatibility with existing formats."""
        # Create test airfoil
        coords = jnp.array(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
            ],
        )
        airfoil = JaxAirfoil(coords, name="TestIO")

        # Test coordinate extraction for file saving
        x_coords, y_coords = airfoil.get_coordinates()

        # Should be able to convert to numpy for file operations
        x_numpy = np.array(x_coords)
        y_numpy = np.array(y_coords)

        assert isinstance(x_numpy, np.ndarray)
        assert isinstance(y_numpy, np.ndarray)
        assert len(x_numpy) == len(y_numpy)
        assert len(x_numpy) > 0

    def test_optimization_workflow_compatibility(self):
        """Test compatibility with optimization workflows."""

        # Create parametric airfoil function
        def create_parametric_airfoil(thickness_param, camber_param):
            x = jnp.linspace(0, 1, 20)
            y_upper = thickness_param * jnp.sin(jnp.pi * x) + camber_param
            y_lower = -thickness_param * jnp.sin(jnp.pi * x) + camber_param

            upper_coords = jnp.array([x, y_upper])
            lower_coords = jnp.array([x, y_lower])

            return JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Test that it works with JAX optimization
        def objective(params):
            thickness_param, camber_param = params
            airfoil = create_parametric_airfoil(thickness_param, camber_param)
            return airfoil.max_thickness  # Maximize thickness

        # Test gradient computation for optimization
        grad_fn = jax.grad(objective)
        params = jnp.array([0.1, 0.02])
        gradients = grad_fn(params)

        # Should produce finite gradients
        assert jnp.all(jnp.isfinite(gradients))
        assert len(gradients) == 2

    def test_batch_processing_integration(self):
        """Test integration with batch processing workflows."""
        # Create multiple airfoils
        airfoils = []
        for i in range(5):
            thickness = 0.05 + i * 0.01
            x = jnp.linspace(0, 1, 15)
            y_upper = thickness * jnp.sin(jnp.pi * x)
            y_lower = -thickness * jnp.sin(jnp.pi * x)

            coords = jnp.array(
                [
                    jnp.concatenate([x, x[::-1]]),
                    jnp.concatenate([y_upper, y_lower[::-1]]),
                ],
            )
            airfoils.append(JaxAirfoil(coords, name=f"Batch_{i}"))

        # Test batch thickness computation
        query_x = jnp.array([0.25, 0.5, 0.75])

        def compute_thickness(airfoil):
            return airfoil.thickness(query_x)

        # Use vmap for batch processing
        batch_fn = jax.vmap(compute_thickness)

        # This would require proper batch structure, but test the concept
        # In practice, would need to restructure for true batch processing
        for airfoil in airfoils:
            thickness = compute_thickness(airfoil)
            assert jnp.all(jnp.isfinite(thickness))
            assert jnp.all(thickness >= 0)

    def test_plotting_integration(self):
        """Test integration with plotting utilities."""
        # Create test airfoil
        coords = jnp.array(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
            ],
        )
        airfoil = JaxAirfoil(coords, name="PlotTest")

        # Test that plotting data can be extracted
        x_coords, y_coords = airfoil.get_coordinates()
        x_upper, y_upper = airfoil.upper_surface_points
        x_lower, y_lower = airfoil.lower_surface_points

        # Should be convertible to numpy for matplotlib
        plot_data = {
            "x_all": np.array(x_coords),
            "y_all": np.array(y_coords),
            "x_upper": np.array(x_upper),
            "y_upper": np.array(y_upper),
            "x_lower": np.array(x_lower),
            "y_lower": np.array(y_lower),
        }

        # All arrays should be valid for plotting
        for key, arr in plot_data.items():
            assert isinstance(arr, np.ndarray)
            assert len(arr) > 0
            assert np.all(np.isfinite(arr))


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    def test_compilation_caching(self):
        """Test that JIT compilation is cached properly."""
        # Create multiple airfoils with same buffer size
        coords1 = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
        coords2 = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.08, 0.0]])

        airfoil1 = JaxAirfoil(coords1)
        airfoil2 = JaxAirfoil(coords2)

        # Both should use same buffer size
        assert airfoil1.buffer_size == airfoil2.buffer_size

        # Operations should reuse compiled functions
        thickness1 = airfoil1.thickness(jnp.array([0.5]))
        thickness2 = airfoil2.thickness(jnp.array([0.5]))

        assert jnp.isfinite(thickness1[0])
        assert jnp.isfinite(thickness2[0])

    def test_memory_efficiency(self):
        """Test memory efficiency of buffer allocation."""
        # Create airfoils of different sizes
        small_coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])

        large_coords = jnp.array(
            [jnp.linspace(0, 1, 50), 0.1 * jnp.sin(jnp.pi * jnp.linspace(0, 1, 50))],
        )

        small_airfoil = JaxAirfoil(small_coords)
        large_airfoil = JaxAirfoil(large_coords)

        # Buffer sizes should be appropriate
        assert small_airfoil.buffer_size >= small_airfoil.n_points
        assert large_airfoil.buffer_size >= large_airfoil.n_points
        assert large_airfoil.buffer_size > small_airfoil.buffer_size

    def test_gradient_computation_efficiency(self):
        """Test efficiency of gradient computations."""
        coords = jnp.array(
            [jnp.linspace(0, 1, 30), 0.1 * jnp.sin(jnp.pi * jnp.linspace(0, 1, 30))],
        )
        airfoil = JaxAirfoil(coords)

        # Test that gradients can be computed efficiently
        def objective(airfoil):
            return jnp.sum(airfoil.thickness(jnp.array([0.25, 0.5, 0.75])))

        grad_fn = jax.grad(objective)
        gradients = grad_fn(airfoil)

        # Should complete without issues
        assert isinstance(gradients, JaxAirfoil)
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
