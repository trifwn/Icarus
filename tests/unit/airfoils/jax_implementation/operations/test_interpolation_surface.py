"""
Interpolation and surface query tests for JAX airfoil implementation.

This module tests the interpolation and surface query functionality including:
- Surface coordinate queries
- Interpolation accuracy
- Edge case handling for extrapolation
- Gradient preservation through interpolation operations

Requirements covered: 2.1, 2.2, 6.1, 6.2, 7.3
"""

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestSurfaceQueries:
    """Test surface coordinate queries and interpolation."""

    @pytest.fixture
    def test_airfoil(self):
        """Create a test airfoil for surface queries."""
        return JaxAirfoil.naca4("2412", n_points=100)

    def test_thickness_queries(self, test_airfoil):
        """Test thickness queries at various x-coordinates."""
        # Test single point query
        x_query = jnp.array([0.5])
        thickness = test_airfoil.thickness(x_query)

        assert thickness.shape == (1,)
        assert jnp.isfinite(thickness[0])
        assert thickness[0] > 0

        # Test multiple point queries
        x_queries = jnp.linspace(0.0, 1.0, 21)
        thicknesses = test_airfoil.thickness(x_queries)

        assert thicknesses.shape == (21,)
        assert jnp.all(jnp.isfinite(thicknesses))
        assert jnp.all(thicknesses >= 0)

        # Maximum thickness should be around 12% for NACA 2412
        max_thickness = jnp.max(thicknesses)
        assert 0.10 < max_thickness < 0.14

    def test_camber_line_queries(self, test_airfoil):
        """Test camber line queries."""
        x_queries = jnp.linspace(0.0, 1.0, 21)
        camber = test_airfoil.camber_line(x_queries)

        assert camber.shape == (21,)
        assert jnp.all(jnp.isfinite(camber))

        # For NACA 2412, camber should be positive in forward section
        assert jnp.max(camber) > 0

        # Camber should be zero at leading and trailing edges
        assert jnp.abs(camber[0]) < 1e-6  # Leading edge
        assert jnp.abs(camber[-1]) < 1e-6  # Trailing edge

    def test_upper_lower_surface_queries(self, test_airfoil):
        """Test upper and lower surface coordinate queries."""
        x_queries = jnp.linspace(0.0, 1.0, 21)

        y_upper = test_airfoil.y_upper(x_queries)
        y_lower = test_airfoil.y_lower(x_queries)

        assert y_upper.shape == (21,)
        assert y_lower.shape == (21,)
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))

        # Upper surface should be above lower surface
        assert jnp.all(y_upper >= y_lower)

        # At leading and trailing edges, upper and lower should meet
        assert jnp.abs(y_upper[0] - y_lower[0]) < 1e-6  # Leading edge
        assert jnp.abs(y_upper[-1] - y_lower[-1]) < 1e-6  # Trailing edge

    def test_interpolation_accuracy(self, test_airfoil):
        """Test interpolation accuracy against analytical solutions."""
        # For NACA airfoils, we can compute analytical thickness distribution
        x_test = jnp.array([0.25, 0.5, 0.75])

        # Get interpolated thickness
        thickness_interp = test_airfoil.thickness(x_test)

        # Compute analytical NACA 2412 thickness (12% max thickness)
        t = 0.12  # 12% thickness
        thickness_analytical = (
            t
            * 5
            * (
                0.2969 * jnp.sqrt(x_test)
                - 0.1260 * x_test
                - 0.3516 * x_test**2
                + 0.2843 * x_test**3
                - 0.1015 * x_test**4
            )
        )

        # Check relative error is small
        relative_error = (
            jnp.abs(thickness_interp - thickness_analytical) / thickness_analytical
        )
        assert jnp.all(relative_error < 0.05)  # Less than 5% error

    def test_extrapolation_handling(self, test_airfoil):
        """Test handling of extrapolation beyond airfoil bounds."""
        # Query points outside [0, 1] range
        x_extrap = jnp.array([-0.1, 1.1])

        # Should handle extrapolation gracefully
        thickness = test_airfoil.thickness(x_extrap)
        camber = test_airfoil.camber_line(x_extrap)
        y_upper = test_airfoil.y_upper(x_extrap)
        y_lower = test_airfoil.y_lower(x_extrap)

        # Results should be finite (not NaN)
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))

    def test_boundary_conditions(self, test_airfoil):
        """Test interpolation at boundary conditions."""
        # Test at exact leading and trailing edges
        x_boundaries = jnp.array([0.0, 1.0])

        thickness = test_airfoil.thickness(x_boundaries)
        y_upper = test_airfoil.y_upper(x_boundaries)
        y_lower = test_airfoil.y_lower(x_boundaries)

        # At boundaries, thickness should be very small
        assert jnp.all(thickness < 1e-3)

        # Upper and lower surfaces should meet at boundaries
        assert jnp.abs(y_upper[0] - y_lower[0]) < 1e-6
        assert jnp.abs(y_upper[1] - y_lower[1]) < 1e-6

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_jit_compatibility(self, test_airfoil, jit_compile):
        """Test JIT compatibility of interpolation operations."""
        x_queries = jnp.linspace(0.1, 0.9, 10)

        def query_all_surfaces(airfoil, x):
            thickness = airfoil.thickness(x)
            camber = airfoil.camber_line(x)
            y_upper = airfoil.y_upper(x)
            y_lower = airfoil.y_lower(x)
            return thickness, camber, y_upper, y_lower

        if jit_compile:
            query_all_surfaces = jax.jit(query_all_surfaces)

        thickness, camber, y_upper, y_lower = query_all_surfaces(
            test_airfoil,
            x_queries,
        )

        # Check results are valid
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))


class TestGradientPreservation:
    """Test gradient preservation through interpolation operations."""

    def test_thickness_gradients(self):
        """Test that thickness queries preserve gradients."""
        airfoil = JaxAirfoil.naca4("0012", n_points=50)

        def thickness_objective(af):
            x_query = jnp.array([0.3, 0.7])
            return jnp.sum(af.thickness(x_query))

        # Compute gradient
        grad_fn = jax.grad(thickness_objective)
        gradients = grad_fn(airfoil)

        # Check gradient structure
        assert isinstance(gradients, JaxAirfoil)
        assert gradients.n_points == airfoil.n_points

        # Check gradients are finite
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))

    def test_surface_query_gradients(self):
        """Test gradients for surface coordinate queries."""
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        def surface_objective(af):
            x_query = jnp.array([0.5])
            y_upper = af.y_upper(x_query)
            y_lower = af.y_lower(x_query)
            return y_upper[0] - y_lower[0]  # Thickness at x=0.5

        # Compute gradient
        grad_fn = jax.grad(surface_objective)
        gradients = grad_fn(airfoil)

        # Check gradient is valid
        assert isinstance(gradients, JaxAirfoil)
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))

    def test_camber_gradients(self):
        """Test gradients for camber line queries."""
        airfoil = JaxAirfoil.naca4("4415", n_points=50)

        def camber_objective(af):
            x_query = jnp.linspace(0.1, 0.9, 5)
            camber = af.camber_line(x_query)
            return jnp.sum(camber**2)  # Sum of squared camber

        # Compute gradient
        grad_fn = jax.grad(camber_objective)
        gradients = grad_fn(airfoil)

        # Check gradient structure
        assert isinstance(gradients, JaxAirfoil)
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))


class TestInterpolationMethods:
    """Test different interpolation methods and their properties."""

    def test_interpolation_smoothness(self):
        """Test that interpolation produces smooth results."""
        airfoil = JaxAirfoil.naca4("0012", n_points=50)

        # Query at many closely spaced points
        x_dense = jnp.linspace(0.01, 0.99, 100)
        thickness = airfoil.thickness(x_dense)

        # Check that thickness varies smoothly (no large jumps)
        thickness_diff = jnp.diff(thickness)
        max_jump = jnp.max(jnp.abs(thickness_diff))

        # Maximum jump should be reasonable for smooth airfoil
        assert max_jump < 0.01

    def test_interpolation_monotonicity(self):
        """Test interpolation behavior in monotonic regions."""
        airfoil = JaxAirfoil.naca4("0012", n_points=50)

        # For symmetric airfoil, thickness should decrease from max to trailing edge
        x_aft = jnp.linspace(0.3, 1.0, 20)  # Aft section where thickness decreases
        thickness = airfoil.thickness(x_aft)

        # Check general decreasing trend (allowing for small numerical variations)
        thickness_trend = jnp.diff(thickness)
        decreasing_count = jnp.sum(thickness_trend <= 0.001)  # Allow small increases

        # Most differences should be decreasing or nearly constant
        assert decreasing_count >= len(thickness_trend) * 0.8

    def test_interpolation_conservation(self):
        """Test that interpolation conserves airfoil properties."""
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        # Query at original coordinate locations
        x_coords, _ = airfoil.get_coordinates()
        x_unique = jnp.unique(x_coords)

        # Interpolated values at original points should match closely
        thickness_interp = airfoil.thickness(x_unique)

        # Compute thickness directly from coordinates
        y_upper_direct = airfoil.y_upper(x_unique)
        y_lower_direct = airfoil.y_lower(x_unique)
        thickness_direct = y_upper_direct - y_lower_direct

        # Should match closely (within interpolation tolerance)
        max_error = jnp.max(jnp.abs(thickness_interp - thickness_direct))
        assert max_error < 1e-6
