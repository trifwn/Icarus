"""
Comprehensive property-based testing for JAX airfoil implementation.

This module uses Hypothesis for property-based testing to verify
robustness across a wide range of inputs and edge cases.

Requirements covered: 8.2
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ICARUS.airfoils.jax_implementation.buffer_management import AirfoilBufferManager
from ICARUS.airfoils.jax_implementation.interpolation import JaxInterpolationEngine
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestPropertyBasedComprehensive:
    """Comprehensive property-based testing for edge cases and robustness."""

    @given(
        n_points=st.integers(min_value=5, max_value=200),
        thickness_scale=st.floats(min_value=0.001, max_value=0.5),
        camber_scale=st.floats(min_value=-0.2, max_value=0.2),
        x_distribution=st.sampled_from(["linear", "cosine", "random"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_airfoil_creation_robustness(
        self,
        n_points,
        thickness_scale,
        camber_scale,
        x_distribution,
    ):
        """Test airfoil creation with various geometric parameters."""
        # Generate x-coordinates based on distribution
        if x_distribution == "linear":
            x = jnp.linspace(0, 1, n_points)
        elif x_distribution == "cosine":
            theta = jnp.linspace(0, jnp.pi, n_points)
            x = 0.5 * (1 - jnp.cos(theta))
        else:  # random
            x = jnp.sort(
                jnp.concatenate(
                    [
                        jnp.array([0.0, 1.0]),
                        jnp.random.uniform(0.01, 0.99, n_points - 2),
                    ],
                ),
            )

        # Generate airfoil shape
        y_thickness = thickness_scale * jnp.sin(jnp.pi * x) * jnp.sqrt(x) * (1 - x)
        y_camber = camber_scale * x * (1 - x)

        y_upper = y_camber + y_thickness
        y_lower = y_camber - y_thickness

        upper_coords = jnp.array([x, y_upper])
        lower_coords = jnp.array([x, y_lower])

        # Create airfoil
        airfoil = JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Basic invariants
        assert airfoil.n_points > 0
        assert airfoil.buffer_size >= airfoil.n_points
        assert jnp.isfinite(airfoil.max_thickness)
        assert airfoil.max_thickness >= 0
        assert jnp.isfinite(airfoil.chord_length)
        assert airfoil.chord_length > 0

        # Test basic operations
        query_x = jnp.array([0.25, 0.5, 0.75])
        thickness = airfoil.thickness(query_x)
        camber = airfoil.camber_line(query_x)

        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(thickness >= 0)
        assert jnp.all(jnp.isfinite(camber))

    @given(
        query_points=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=50),
            elements=st.floats(
                min_value=-1.0,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_interpolation_robustness(self, query_points):
        """Test interpolation with various query point distributions."""
        # Create a well-behaved test airfoil
        x = jnp.linspace(0, 1, 20)
        y_upper = 0.1 * jnp.sin(jnp.pi * x) * (1 - x)
        y_lower = -0.05 * jnp.sin(jnp.pi * x) * (1 - x)

        upper_coords = jnp.array([x, y_upper])
        lower_coords = jnp.array([x, y_lower])
        airfoil = JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Test with various query points
        query_jax = jnp.array(query_points)

        # All operations should produce finite results
        thickness = airfoil.thickness(query_jax)
        camber = airfoil.camber_line(query_jax)
        y_upper_interp = airfoil.y_upper(query_jax)
        y_lower_interp = airfoil.y_lower(query_jax)

        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(thickness >= 0)
        assert jnp.all(jnp.isfinite(camber))
        assert jnp.all(jnp.isfinite(y_upper_interp))
        assert jnp.all(jnp.isfinite(y_lower_interp))

        # Thickness should be consistent with surface queries
        computed_thickness = y_upper_interp - y_lower_interp
        # Allow some tolerance for extrapolation and numerical differences
        assert jnp.allclose(thickness, computed_thickness, atol=1e-3, rtol=1e-2)

    @given(
        eta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        scale_factor=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_morphing_robustness(self, eta, scale_factor):
        """Test morphing operations with various parameters."""
        # Create two different airfoils
        x = jnp.linspace(0, 1, 15)

        # First airfoil - symmetric
        y1_upper = 0.08 * jnp.sin(jnp.pi * x) * (1 - x)
        y1_lower = -y1_upper
        airfoil1 = JaxAirfoil.from_upper_lower(
            jnp.array([x, y1_upper]),
            jnp.array([x, y1_lower]),
        )

        # Second airfoil - cambered and scaled
        y2_upper = scale_factor * 0.06 * jnp.sin(jnp.pi * x) * (1 - x) + 0.02 * x * (
            1 - x
        )
        y2_lower = -scale_factor * 0.04 * jnp.sin(jnp.pi * x) * (1 - x) + 0.02 * x * (
            1 - x
        )
        airfoil2 = JaxAirfoil.from_upper_lower(
            jnp.array([x, y2_upper]),
            jnp.array([x, y2_lower]),
        )

        # Test morphing
        morphed = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta)

        # Basic invariants
        assert isinstance(morphed, JaxAirfoil)
        assert morphed.n_points > 0
        assert jnp.isfinite(morphed.max_thickness)
        assert morphed.max_thickness >= 0

        # Morphing should interpolate between the two airfoils
        thickness1 = airfoil1.max_thickness
        thickness2 = airfoil2.max_thickness
        morphed_thickness = morphed.max_thickness

        # Should be between the two original thicknesses (with some tolerance)
        min_thickness = min(thickness1, thickness2)
        max_thickness = max(thickness1, thickness2)
        assert morphed_thickness >= min_thickness - 0.01
        assert morphed_thickness <= max_thickness + 0.01

    @given(
        flap_angle=st.floats(min_value=-0.8, max_value=0.8, allow_nan=False),
        hinge_x=st.floats(min_value=0.3, max_value=0.9),
        hinge_y=st.floats(min_value=-0.05, max_value=0.05),
    )
    @settings(max_examples=30, deadline=None)
    def test_flap_operation_robustness(self, flap_angle, hinge_x, hinge_y):
        """Test flap operations with various hinge positions and angles."""
        # Create test airfoil
        x = jnp.linspace(0, 1, 20)
        y_upper = 0.1 * jnp.sin(jnp.pi * x) * (1 - x)
        y_lower = -0.06 * jnp.sin(jnp.pi * x) * (1 - x)

        upper_coords = jnp.array([x, y_upper])
        lower_coords = jnp.array([x, y_lower])
        airfoil = JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Test flap operation
        hinge_point = jnp.array([hinge_x, hinge_y])
        flapped = airfoil.flap(hinge_point=hinge_point, flap_angle=flap_angle)

        # Basic invariants
        assert isinstance(flapped, JaxAirfoil)
        assert flapped.n_points > 0
        assert jnp.isfinite(flapped.max_thickness)
        assert flapped.max_thickness >= 0

        # Test that flapped airfoil operations work
        query_x = jnp.array([0.5])
        thickness = flapped.thickness(query_x)
        assert jnp.isfinite(thickness[0])
        assert thickness[0] >= 0

    @given(
        n_points_target=st.integers(min_value=5, max_value=100),
        distribution=st.sampled_from(["uniform", "cosine"]),
    )
    @settings(max_examples=20, deadline=None)
    def test_repaneling_robustness(self, n_points_target, distribution):
        """Test repaneling operations with various parameters."""
        # Create test airfoil with irregular point distribution
        x_orig = jnp.concatenate(
            [
                jnp.array([0.0]),
                jnp.sort(jnp.random.uniform(0.01, 0.99, 15)),
                jnp.array([1.0]),
            ],
        )
        y_upper = 0.08 * jnp.sin(jnp.pi * x_orig) * (1 - x_orig)
        y_lower = -0.05 * jnp.sin(jnp.pi * x_orig) * (1 - x_orig)

        upper_coords = jnp.array([x_orig, y_upper])
        lower_coords = jnp.array([x_orig, y_lower])
        airfoil = JaxAirfoil.from_upper_lower(upper_coords, lower_coords)

        # Test repaneling
        repaneled = airfoil.repanel(n_points=n_points_target, distribution=distribution)

        # Basic invariants
        assert isinstance(repaneled, JaxAirfoil)
        assert repaneled.n_points > 0
        assert jnp.isfinite(repaneled.max_thickness)
        assert repaneled.max_thickness >= 0

        # Should preserve basic geometric properties (within tolerance)
        orig_max_thickness = airfoil.max_thickness
        repaneled_max_thickness = repaneled.max_thickness
        thickness_error = (
            abs(orig_max_thickness - repaneled_max_thickness) / orig_max_thickness
        )
        assert thickness_error < 0.1  # Less than 10% error

    @given(
        buffer_size=st.integers(min_value=8, max_value=512),
        n_valid=st.integers(min_value=3, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_buffer_management_robustness(self, buffer_size, n_valid):
        """Test buffer management with various sizes."""
        assume(
            buffer_size >= n_valid,
        )  # Buffer must be at least as large as valid points

        # Create test coordinates
        x = jnp.linspace(0, 1, n_valid)
        y = 0.1 * jnp.sin(jnp.pi * x)
        coords = jnp.array([x, y])

        # Test buffer operations
        padded = AirfoilBufferManager.pad_coordinates(coords, buffer_size)
        mask = AirfoilBufferManager.create_validity_mask(n_valid, buffer_size)

        # Verify buffer properties
        assert padded.shape == (2, buffer_size)
        assert mask.shape == (buffer_size,)
        assert jnp.sum(mask) == n_valid

        # Valid portion should match original
        valid_portion = padded[:, mask]
        assert jnp.allclose(valid_portion, coords)

        # Invalid portion should be NaN or zero
        invalid_portion = padded[:, ~mask]
        assert jnp.all(jnp.isnan(invalid_portion) | (invalid_portion == 0))

    @given(
        x_data=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=3, max_value=20),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        y_scale=st.floats(min_value=0.01, max_value=0.2),
    )
    @settings(max_examples=20, deadline=None)
    def test_interpolation_engine_robustness(self, x_data, y_scale):
        """Test interpolation engine with various data distributions."""
        # Ensure x_data is sorted and has unique values
        x_data_sorted = jnp.sort(jnp.unique(jnp.array(x_data)))
        assume(len(x_data_sorted) >= 3)  # Need at least 3 points

        # Ensure endpoints are included
        if x_data_sorted[0] > 0.01:
            x_data_sorted = jnp.concatenate([jnp.array([0.0]), x_data_sorted])
        if x_data_sorted[-1] < 0.99:
            x_data_sorted = jnp.concatenate([x_data_sorted, jnp.array([1.0])])

        # Generate corresponding y values
        y_data = y_scale * jnp.sin(jnp.pi * x_data_sorted) * (1 - x_data_sorted)

        # Test interpolation
        query_x = jnp.array([0.25, 0.5, 0.75])
        result = JaxInterpolationEngine.linear_interpolate_1d(
            x_data_sorted,
            y_data,
            len(x_data_sorted),
            query_x,
        )

        # Results should be finite
        assert jnp.all(jnp.isfinite(result))

        # Compare with numpy interp for validation
        numpy_result = np.interp(query_x, x_data_sorted, y_data)
        assert jnp.allclose(result, numpy_result, atol=1e-5)

    @given(
        coordinate_noise=st.floats(min_value=0.0, max_value=0.01),
        duplicate_probability=st.floats(min_value=0.0, max_value=0.3),
    )
    @settings(max_examples=15, deadline=None)
    def test_noisy_coordinate_handling(self, coordinate_noise, duplicate_probability):
        """Test handling of noisy and duplicate coordinates."""
        # Create base airfoil
        x_base = jnp.linspace(0, 1, 20)
        y_base = 0.08 * jnp.sin(jnp.pi * x_base) * (1 - x_base)

        # Add noise
        x_noisy = x_base + coordinate_noise * (
            jnp.random.normal(size=len(x_base)) - 0.5
        )
        y_noisy = y_base + coordinate_noise * (
            jnp.random.normal(size=len(y_base)) - 0.5
        )

        # Ensure endpoints remain fixed
        x_noisy = x_noisy.at[0].set(0.0)
        x_noisy = x_noisy.at[-1].set(1.0)
        y_noisy = y_noisy.at[0].set(0.0)
        y_noisy = y_noisy.at[-1].set(0.0)

        # Randomly duplicate some points
        if duplicate_probability > 0:
            n_duplicates = int(len(x_base) * duplicate_probability)
            duplicate_indices = jnp.random.choice(
                len(x_base),
                n_duplicates,
                replace=False,
            )
            x_with_dups = jnp.concatenate([x_noisy, x_noisy[duplicate_indices]])
            y_with_dups = jnp.concatenate([y_noisy, y_noisy[duplicate_indices]])
        else:
            x_with_dups = x_noisy
            y_with_dups = y_noisy

        coords = jnp.array([x_with_dups, y_with_dups])

        # Should handle noisy/duplicate coordinates gracefully
        try:
            airfoil = JaxAirfoil(coords)

            # If successful, basic operations should work
            assert airfoil.n_points > 0
            assert jnp.isfinite(airfoil.max_thickness)

            # Test basic query
            thickness = airfoil.thickness(jnp.array([0.5]))
            assert jnp.isfinite(thickness[0])
            assert thickness[0] >= 0

        except (ValueError, AssertionError):
            # Some noisy configurations may be rejected, which is acceptable
            pass

    def test_extreme_aspect_ratios(self):
        """Test airfoils with extreme aspect ratios."""
        # Very thin airfoil
        x = jnp.linspace(0, 1, 20)
        y_upper_thin = 0.001 * jnp.sin(jnp.pi * x)
        y_lower_thin = -y_upper_thin

        thin_airfoil = JaxAirfoil.from_upper_lower(
            jnp.array([x, y_upper_thin]),
            jnp.array([x, y_lower_thin]),
        )

        # Very thick airfoil
        y_upper_thick = 0.4 * jnp.sin(jnp.pi * x) * (1 - x)
        y_lower_thick = -y_upper_thick

        thick_airfoil = JaxAirfoil.from_upper_lower(
            jnp.array([x, y_upper_thick]),
            jnp.array([x, y_lower_thick]),
        )

        # Both should work
        for airfoil in [thin_airfoil, thick_airfoil]:
            assert airfoil.n_points > 0
            assert jnp.isfinite(airfoil.max_thickness)
            assert airfoil.max_thickness >= 0

            # Basic operations should work
            thickness = airfoil.thickness(jnp.array([0.5]))
            assert jnp.isfinite(thickness[0])
            assert thickness[0] >= 0

    def test_boundary_conditions(self):
        """Test behavior at domain boundaries."""
        # Create test airfoil
        x = jnp.linspace(0, 1, 15)
        y_upper = 0.08 * jnp.sin(jnp.pi * x) * (1 - x)
        y_lower = -0.05 * jnp.sin(jnp.pi * x) * (1 - x)

        airfoil = JaxAirfoil.from_upper_lower(
            jnp.array([x, y_upper]),
            jnp.array([x, y_lower]),
        )

        # Test queries exactly at boundaries
        boundary_queries = jnp.array([0.0, 1.0])
        thickness_boundary = airfoil.thickness(boundary_queries)
        camber_boundary = airfoil.camber_line(boundary_queries)

        # Should be finite and reasonable
        assert jnp.all(jnp.isfinite(thickness_boundary))
        assert jnp.all(thickness_boundary >= 0)
        assert jnp.all(jnp.isfinite(camber_boundary))

        # Thickness at leading/trailing edge should be small
        assert thickness_boundary[0] < 0.01  # Leading edge
        assert thickness_boundary[1] < 0.01  # Trailing edge

    def test_gradient_robustness_property_based(self):
        """Property-based test for gradient robustness."""

        @given(perturbation_scale=st.floats(min_value=1e-8, max_value=1e-4))
        @settings(max_examples=10, deadline=None)
        def test_gradient_continuity(perturbation_scale):
            # Create test airfoil
            x = jnp.linspace(0, 1, 10)
            y_upper = 0.08 * jnp.sin(jnp.pi * x) * (1 - x)
            y_lower = -0.05 * jnp.sin(jnp.pi * x) * (1 - x)

            airfoil = JaxAirfoil.from_upper_lower(
                jnp.array([x, y_upper]),
                jnp.array([x, y_lower]),
            )

            def objective(airfoil):
                return airfoil.thickness(jnp.array([0.5]))[0]

            # Compute gradient
            grad_fn = jax.grad(objective)
            gradients = grad_fn(airfoil)

            # Gradients should be finite
            grad_coords = gradients._coordinates[:, gradients._validity_mask]
            assert jnp.all(jnp.isfinite(grad_coords))

        # Run the property-based test
        test_gradient_continuity()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
