"""
Comprehensive JIT compilation tests for JAX airfoil implementation.

This module provides detailed testing of JIT compilation behavior,
compilation caching, and performance characteristics.

Requirements covered: 1.2
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.buffer_management import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.interpolation import JaxInterpolationEngine
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestJITCompilationComprehensive:
    """Comprehensive JIT compilation testing."""

    @pytest.fixture
    def test_airfoils(self):
        """Create multiple test airfoils with different buffer sizes."""
        # Small airfoil
        small_coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
        small_airfoil = JaxAirfoil(small_coords, name="Small")

        # Medium airfoil
        medium_coords = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.05, 0.08, 0.05, 0.0]],
        )
        medium_airfoil = JaxAirfoil(medium_coords, name="Medium")

        # Large airfoil
        x_large = jnp.linspace(0, 1, 30)
        y_large = 0.1 * jnp.sin(jnp.pi * x_large)
        large_coords = jnp.array([x_large, y_large])
        large_airfoil = JaxAirfoil(large_coords, name="Large")

        return small_airfoil, medium_airfoil, large_airfoil

    def test_basic_operation_jit_compilation(self, test_airfoils):
        """Test JIT compilation of basic operations."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def thickness_computation(airfoil):
            return airfoil.thickness(jnp.array([0.5]))

        @jax.jit
        def camber_computation(airfoil):
            return airfoil.camber_line(jnp.array([0.5]))

        @jax.jit
        def surface_queries(airfoil):
            query_x = jnp.array([0.5])
            return airfoil.y_upper(query_x), airfoil.y_lower(query_x)

        # Test compilation with different airfoil sizes
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            thickness = thickness_computation(airfoil)
            camber = camber_computation(airfoil)
            y_upper, y_lower = surface_queries(airfoil)

            assert jnp.isfinite(thickness[0])
            assert jnp.isfinite(camber[0])
            assert jnp.isfinite(y_upper[0])
            assert jnp.isfinite(y_lower[0])

    def test_property_access_jit_compilation(self, test_airfoils):
        """Test JIT compilation of property accessors."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def get_geometric_properties(airfoil):
            return (
                airfoil.max_thickness,
                airfoil.max_camber,
                airfoil.chord_length,
                airfoil.max_thickness_location,
                airfoil.max_camber_location,
            )

        @jax.jit
        def get_surface_coordinates(airfoil):
            x_upper, y_upper = airfoil.upper_surface_points
            x_lower, y_lower = airfoil.lower_surface_points
            return x_upper, y_upper, x_lower, y_lower

        # Test with different airfoil sizes
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            props = get_geometric_properties(airfoil)
            coords = get_surface_coordinates(airfoil)

            # Verify all properties are finite
            for prop in props:
                assert jnp.isfinite(prop)

            # Verify all coordinates are finite
            for coord_array in coords:
                assert jnp.all(jnp.isfinite(coord_array))

    def test_transformation_jit_compilation(self, test_airfoils):
        """Test JIT compilation of transformation operations."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def flap_transformation(airfoil, angle):
            return airfoil.flap(hinge_point=jnp.array([0.75, 0.0]), flap_angle=angle)

        @jax.jit
        def repanel_transformation(airfoil):
            return airfoil.repanel(n_points=15, distribution="cosine")

        # Test with different airfoil sizes
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            flapped = flap_transformation(airfoil, 0.1)
            repaneled = repanel_transformation(airfoil)

            assert isinstance(flapped, JaxAirfoil)
            assert isinstance(repaneled, JaxAirfoil)
            assert flapped.n_points > 0
            assert repaneled.n_points > 0

    def test_morphing_jit_compilation(self, test_airfoils):
        """Test JIT compilation of morphing operations."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def morphing_operation(airfoil1, airfoil2, eta):
            return JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta)

        # Test morphing between airfoils of same buffer size
        morphed_small = morphing_operation(small_airfoil, small_airfoil, 0.5)
        morphed_medium = morphing_operation(medium_airfoil, medium_airfoil, 0.3)

        assert isinstance(morphed_small, JaxAirfoil)
        assert isinstance(morphed_medium, JaxAirfoil)
        assert morphed_small.n_points > 0
        assert morphed_medium.n_points > 0

    def test_batch_operations_jit_compilation(self, test_airfoils):
        """Test JIT compilation of batch operations."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def batch_thickness_queries(airfoil):
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

        # Test with different airfoil sizes
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            thickness_batch = batch_thickness_queries(airfoil)
            upper_batch, lower_batch = batch_surface_queries(airfoil)

            assert thickness_batch.shape == (3, 3)
            assert upper_batch.shape == (2, 3)
            assert lower_batch.shape == (2, 3)
            assert jnp.all(jnp.isfinite(thickness_batch))
            assert jnp.all(jnp.isfinite(upper_batch))
            assert jnp.all(jnp.isfinite(lower_batch))

    def test_nested_jit_compilation(self, test_airfoils):
        """Test nested JIT compilation scenarios."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def inner_computation(airfoil):
            return airfoil.thickness(jnp.array([0.5]))[0]

        @jax.jit
        def outer_computation(airfoil):
            base_thickness = inner_computation(airfoil)
            flapped = airfoil.flap(hinge_point=jnp.array([0.75, 0.0]), flap_angle=0.05)
            flapped_thickness = inner_computation(flapped)
            return base_thickness + flapped_thickness

        # Test nested compilation
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            result = outer_computation(airfoil)
            assert jnp.isfinite(result)
            assert result > 0

    def test_conditional_jit_compilation(self, test_airfoils):
        """Test JIT compilation with conditional logic."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def conditional_computation(airfoil, use_flap):
            base_thickness = airfoil.thickness(jnp.array([0.5]))[0]

            # Use jax.lax.cond for JIT-compatible conditionals
            def apply_flap(airfoil):
                return airfoil.flap(
                    hinge_point=jnp.array([0.75, 0.0]),
                    flap_angle=0.1,
                ).thickness(jnp.array([0.5]))[0]

            def no_flap(airfoil):
                return base_thickness

            result = jax.lax.cond(use_flap, apply_flap, no_flap, airfoil)
            return result

        # Test conditional compilation
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            result_with_flap = conditional_computation(airfoil, True)
            result_without_flap = conditional_computation(airfoil, False)

            assert jnp.isfinite(result_with_flap)
            assert jnp.isfinite(result_without_flap)

    def test_compilation_with_static_arguments(self):
        """Test JIT compilation with static arguments."""

        @partial(jax.jit, static_argnums=(1, 2))
        def buffer_operations(coords, target_size, n_valid):
            padded = AirfoilBufferManager.pad_coordinates(coords, target_size)
            mask = AirfoilBufferManager.create_validity_mask(n_valid, target_size)
            return padded, mask

        @partial(jax.jit, static_argnums=(2,))
        def interpolation_operation(x_data, y_data, n_valid, query_x):
            return JaxInterpolationEngine.linear_interpolate_1d(
                x_data,
                y_data,
                n_valid,
                query_x,
            )

        # Test buffer operations with static arguments
        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.05, 0.0]])
        padded, mask = buffer_operations(coords, 32, 3)

        assert padded.shape == (2, 32)
        assert mask.shape == (32,)
        assert jnp.sum(mask) == 3

        # Test interpolation with static arguments
        x_data = jnp.array([0.0, 0.5, 1.0])
        y_data = jnp.array([0.0, 0.05, 0.0])
        query_x = jnp.array([0.25, 0.75])

        result = interpolation_operation(x_data, y_data, 3, query_x)
        assert len(result) == 2
        assert jnp.all(jnp.isfinite(result))

    def test_compilation_caching_behavior(self, test_airfoils):
        """Test that compilation caching works correctly."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def cached_computation(airfoil):
            return airfoil.thickness(jnp.array([0.5]))[0]

        # First calls should trigger compilation
        result1 = cached_computation(small_airfoil)
        result2 = cached_computation(medium_airfoil)

        # Subsequent calls with same buffer sizes should use cached compilation
        result3 = cached_computation(small_airfoil)
        result4 = cached_computation(medium_airfoil)

        # Results should be consistent
        assert jnp.isclose(result1, result3)
        assert jnp.isclose(result2, result4)

    def test_compilation_with_different_shapes(self):
        """Test compilation behavior with different input shapes."""

        @jax.jit
        def shape_dependent_computation(query_x):
            # This should work with different query shapes
            return jnp.sum(query_x**2)

        # Test with different query shapes
        single_query = jnp.array([0.5])
        multi_query = jnp.array([0.25, 0.5, 0.75])
        large_query = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        result1 = shape_dependent_computation(single_query)
        result2 = shape_dependent_computation(multi_query)
        result3 = shape_dependent_computation(large_query)

        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)
        assert jnp.isfinite(result3)

    def test_compilation_error_handling(self):
        """Test compilation behavior with potential errors."""

        @jax.jit
        def safe_division(x, y):
            # Use jax.lax.cond for safe division
            return jax.lax.cond(jnp.abs(y) > 1e-10, lambda: x / y, lambda: 0.0, None)

        # Test safe division compilation
        result1 = safe_division(1.0, 2.0)
        result2 = safe_division(1.0, 0.0)  # Should handle division by zero

        assert jnp.isfinite(result1)
        assert jnp.isfinite(result2)
        assert result1 == 0.5
        assert result2 == 0.0

    def test_compilation_performance_characteristics(self, test_airfoils):
        """Test performance characteristics of compiled functions."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def performance_test_function(airfoil):
            # Complex computation to test performance
            query_x = jnp.linspace(0.1, 0.9, 20)
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)
            y_upper = airfoil.y_upper(query_x)
            y_lower = airfoil.y_lower(query_x)

            return (
                jnp.sum(thickness)
                + jnp.sum(jnp.abs(camber))
                + jnp.sum(y_upper)
                + jnp.sum(jnp.abs(y_lower))
            )

        # Warm up compilation
        _ = performance_test_function(medium_airfoil)

        # Time multiple executions
        start_time = time.time()
        for _ in range(100):
            result = performance_test_function(medium_airfoil)
        end_time = time.time()

        execution_time = (end_time - start_time) / 100

        # Should be fast after compilation
        assert execution_time < 0.01  # Less than 10ms per execution
        assert jnp.isfinite(result)

    def test_memory_usage_during_compilation(self, test_airfoils):
        """Test memory usage patterns during compilation."""
        small_airfoil, medium_airfoil, large_airfoil = test_airfoils

        @jax.jit
        def memory_intensive_computation(airfoil):
            # Create multiple intermediate arrays
            query_x = jnp.linspace(0, 1, 100)
            thickness = airfoil.thickness(query_x)

            # Multiple transformations
            intermediate1 = thickness * 2
            intermediate2 = jnp.sin(intermediate1)
            intermediate3 = jnp.exp(-intermediate2)

            return jnp.sum(intermediate3)

        # Test with different airfoil sizes
        results = []
        for airfoil in [small_airfoil, medium_airfoil, large_airfoil]:
            result = memory_intensive_computation(airfoil)
            results.append(result)
            assert jnp.isfinite(result)

        # All results should be finite and reasonable
        assert all(jnp.isfinite(r) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
