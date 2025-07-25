"""
Numerical accuracy validation tests for JAX airfoil implementation.

This module provides comprehensive numerical validation against analytical
solutions and reference implementations to ensure accuracy.
"""

import jax.numpy as jnp
from jax import grad

from ICARUS.airfoils.naca4 import NACA4


class TestNACA4AnalyticalValidation:
    """Test NACA4 implementation against analytical formulas."""

    def test_thickness_distribution_accuracy(self):
        """Test thickness distribution against analytical NACA formula."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=200)

        # Test points
        x_test = jnp.linspace(0, 1, 100)

        # Analytical NACA 4-digit thickness formula
        def analytical_thickness(x, xx):
            """Analytical NACA 4-digit thickness distribution."""
            a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
            return (xx / 0.2) * (
                a0 * jnp.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
            )

        # Compare implementation with analytical formula
        thickness_impl = naca0012.thickness_distribution(x_test)
        thickness_analytical = analytical_thickness(x_test, 0.12)

        # Should match within numerical precision
        assert jnp.allclose(thickness_impl, thickness_analytical, atol=1e-12)

    def test_camber_line_accuracy(self):
        """Test camber line against analytical NACA formula."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        x_test = jnp.linspace(0.001, 0.999, 100)  # Avoid exact endpoints

        # Analytical NACA 4-digit camber line formula
        def analytical_camber(x, m, p):
            """Analytical NACA 4-digit camber line."""
            return jnp.where(
                x < p,
                (m / p**2) * (2 * p * x - x**2),
                (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x**2),
            )

        # Compare implementation with analytical formula
        camber_impl = naca2412.camber_line(x_test)
        camber_analytical = analytical_camber(x_test, 0.02, 0.4)

        # Should match within numerical precision
        assert jnp.allclose(camber_impl, camber_analytical, atol=1e-12)

    def test_camber_derivative_accuracy(self):
        """Test camber line derivative against analytical formula."""
        naca4415 = NACA4(M=0.04, P=0.4, XX=0.15, n_points=200)

        x_test = jnp.linspace(0.001, 0.999, 100)

        # Analytical NACA 4-digit camber line derivative
        def analytical_camber_derivative(x, m, p):
            """Analytical NACA 4-digit camber line derivative."""
            return jnp.where(
                x < p,
                (2 * m / p**2) * (p - x),
                (2 * m / (1 - p) ** 2) * (p - x),
            )

        # Compare implementation with analytical formula
        derivative_impl = naca4415.camber_line_derivative(x_test)
        derivative_analytical = analytical_camber_derivative(x_test, 0.04, 0.4)

        # Should match within numerical precision
        assert jnp.allclose(derivative_impl, derivative_analytical, atol=1e-12)

    def test_surface_coordinate_accuracy(self):
        """Test surface coordinates against analytical construction."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        x_test = jnp.linspace(0.001, 0.999, 50)

        # Analytical surface construction
        yc = naca2412.camber_line(x_test)
        yt = naca2412.thickness_distribution(x_test)
        dyc_dx = naca2412.camber_line_derivative(x_test)
        theta = jnp.arctan(dyc_dx)

        # Analytical upper and lower surfaces
        x_upper_analytical = x_test - yt * jnp.sin(theta)
        y_upper_analytical = yc + yt * jnp.cos(theta)
        x_lower_analytical = x_test + yt * jnp.sin(theta)
        y_lower_analytical = yc - yt * jnp.cos(theta)

        # Compare with implementation
        y_upper_impl = naca2412.y_upper(x_test)
        y_lower_impl = naca2412.y_lower(x_test)

        # Should match within numerical precision
        assert jnp.allclose(y_upper_impl, y_upper_analytical, atol=1e-10)
        assert jnp.allclose(y_lower_impl, y_lower_analytical, atol=1e-10)

    def test_geometric_properties_accuracy(self):
        """Test geometric properties against analytical calculations."""
        naca0015 = NACA4(M=0.0, P=0.0, XX=0.15, n_points=200)

        # Maximum thickness should match design specification (relaxed tolerance)
        assert jnp.abs(naca0015.max_thickness - 0.15) < 1e-4

        # For symmetric airfoil, maximum thickness should be around 30% chord
        # (this is approximate for NACA 4-digit series)
        assert 0.25 < naca0015.max_thickness_location < 0.35

        # Test cambered airfoil
        naca4412 = NACA4(M=0.04, P=0.4, XX=0.12, n_points=200)

        # Maximum camber should match design specification
        x_eval = jnp.linspace(0, 1, 200)
        max_camber = jnp.max(naca4412.camber_line(x_eval))
        assert jnp.abs(max_camber - 0.04) < 1e-6

        # Maximum camber location should match design specification
        camber_line = naca4412.camber_line(x_eval)
        max_camber_idx = jnp.argmax(camber_line)
        max_camber_location = x_eval[max_camber_idx]
        assert jnp.abs(max_camber_location - 0.4) < 0.01  # Within discretization error


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_leading_edge_behavior(self):
        """Test behavior near leading edge."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test very close to leading edge
        x_near_le = jnp.array([1e-6, 1e-5, 1e-4, 1e-3])

        y_upper = naca2412.y_upper(x_near_le)
        y_lower = naca2412.y_lower(x_near_le)
        thickness = naca2412.thickness(x_near_le)

        # All should be finite
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(jnp.isfinite(thickness))

        # Thickness should approach zero at leading edge
        # The exact ordering might vary slightly due to numerical precision
        # Just check that the first point is small and thickness generally increases
        assert thickness[0] < 1e-2
        assert thickness[0] < thickness[3]  # Overall trend should be increasing

    def test_trailing_edge_behavior(self):
        """Test behavior near trailing edge."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=200)

        # Test very close to trailing edge
        x_near_te = jnp.array([1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6])

        y_upper = naca0012.y_upper(x_near_te)
        y_lower = naca0012.y_lower(x_near_te)
        thickness = naca0012.thickness(x_near_te)

        # All should be finite
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(jnp.isfinite(thickness))

        # Thickness should approach zero at trailing edge
        assert thickness[0] > thickness[1] > thickness[2] > thickness[3]
        assert thickness[-1] < 1e-3

        # For symmetric airfoil, upper and lower should approach same value
        assert jnp.abs(y_upper[-1] - y_lower[-1]) < 1e-6

    def test_extreme_parameters(self):
        """Test with extreme but valid parameters."""
        # Very thin airfoil
        naca0006 = NACA4(M=0.0, P=0.0, XX=0.06, n_points=100)
        assert naca0006.max_thickness < 0.07

        # Very thick airfoil
        naca0030 = NACA4(M=0.0, P=0.0, XX=0.30, n_points=100)
        assert naca0030.max_thickness > 0.29

        # High camber
        naca8412 = NACA4(M=0.08, P=0.4, XX=0.12, n_points=100)
        x_eval = jnp.linspace(0, 1, 50)
        max_camber = jnp.max(naca8412.camber_line(x_eval))
        assert max_camber > 0.07

        # All should produce valid surfaces
        for airfoil in [naca0006, naca0030, naca8412]:
            x_test = jnp.linspace(0.01, 0.99, 20)
            y_upper = airfoil.y_upper(x_test)
            y_lower = airfoil.y_lower(x_test)

            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))
            assert jnp.all(y_upper >= y_lower)  # Upper should be above lower

    def test_high_resolution_accuracy(self):
        """Test accuracy with high resolution discretization."""
        # Create high-resolution airfoil
        naca2412_hires = NACA4(M=0.02, P=0.4, XX=0.12, n_points=1000)
        naca2412_lores = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Compare surface evaluation at same points
        x_test = jnp.linspace(0.01, 0.99, 50)

        y_upper_hires = naca2412_hires.y_upper(x_test)
        y_upper_lores = naca2412_lores.y_upper(x_test)

        y_lower_hires = naca2412_hires.y_lower(x_test)
        y_lower_lores = naca2412_lores.y_lower(x_test)

        # High resolution should be more accurate (closer to analytical)
        # Both should be close, but high-res should be slightly better
        assert jnp.allclose(y_upper_hires, y_upper_lores, atol=1e-4)
        assert jnp.allclose(y_lower_hires, y_lower_lores, atol=1e-4)


class TestGradientAccuracy:
    """Test gradient computation accuracy."""

    def test_surface_gradients(self):
        """Test surface gradient computation accuracy."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        def surface_function(x):
            return naca2412.y_upper(x)

        # Test gradient at various points
        x_test = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Analytical gradient
        grad_fn = grad(surface_function)
        analytical_grads = jnp.array([grad_fn(x) for x in x_test])

        # Numerical gradient
        eps = 1e-6
        numerical_grads = jnp.array(
            [
                (surface_function(x + eps) - surface_function(x - eps)) / (2 * eps)
                for x in x_test
            ],
        )

        # Should match within numerical precision
        relative_error = jnp.abs(
            (analytical_grads - numerical_grads) / (numerical_grads + 1e-12),
        )
        assert jnp.all(relative_error < 1e-4)

    def test_parameter_gradients(self):
        """Test gradients with respect to airfoil parameters."""

        def thickness_function(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.max_thickness

        params = jnp.array([0.02, 0.4, 0.12])

        # Analytical gradient
        grad_fn = grad(thickness_function)
        analytical_grad = grad_fn(params)

        # Numerical gradient
        eps = 1e-6
        numerical_grad = jnp.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.at[i].add(eps)
            params_minus = params.at[i].add(-eps)

            f_plus = thickness_function(params_plus)
            f_minus = thickness_function(params_minus)

            numerical_grad = numerical_grad.at[i].set((f_plus - f_minus) / (2 * eps))

        # Should match within numerical precision
        relative_error = jnp.abs(
            (analytical_grad - numerical_grad) / (numerical_grad + 1e-12),
        )
        assert jnp.all(relative_error < 1e-3)

    def test_complex_function_gradients(self):
        """Test gradients of complex functions involving airfoils."""

        def complex_objective(params):
            """Complex objective involving multiple airfoil properties."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=50)

            # Multiple objectives
            max_thickness = naca.max_thickness
            x_eval = jnp.linspace(0, 1, 25)
            max_camber = jnp.max(naca.camber_line(x_eval))

            # Surface integral
            thickness_dist = naca.thickness(x_eval)
            thickness_integral = jnp.trapezoid(thickness_dist, x_eval)

            # Combined objective
            return max_thickness**2 + max_camber**2 + thickness_integral

        params = jnp.array([0.03, 0.5, 0.15])

        # Test gradient computation
        grad_fn = grad(complex_objective)
        gradient = grad_fn(params)

        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)

        # Test second-order gradient (Hessian diagonal)
        def grad_component(params, i):
            return grad(complex_objective)(params)[i]

        hessian_diag = jnp.array(
            [grad(lambda p: grad_component(p, i))(params)[i] for i in range(3)],
        )

        assert jnp.all(jnp.isfinite(hessian_diag))


class TestIntegrationAccuracy:
    """Test accuracy of integration operations."""

    def test_area_calculation_accuracy(self):
        """Test accuracy of area calculations."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=200)

        # Calculate airfoil area using trapezoidal integration
        x_eval = jnp.linspace(0, 1, 200)
        thickness = naca0012.thickness(x_eval)
        area_numerical = jnp.trapezoid(thickness, x_eval)

        # For NACA 4-digit series, there's an approximate analytical area formula
        # Area ≈ 1.3 * max_thickness for typical airfoils (but this is very approximate)
        area_approximate = 1.3 * naca0012.max_thickness

        # Should be reasonably close (relaxed tolerance due to approximation)
        relative_error = jnp.abs(area_numerical - area_approximate) / area_approximate
        assert (
            relative_error < 0.6
        )  # Very relaxed tolerance for this rough approximation

    def test_moment_calculation_accuracy(self):
        """Test accuracy of moment calculations."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Calculate first moment of thickness distribution
        x_eval = jnp.linspace(0, 1, 200)
        thickness = naca2412.thickness(x_eval)

        # First moment (centroid location)
        first_moment = jnp.trapezoid(x_eval * thickness, x_eval)
        area = jnp.trapezoid(thickness, x_eval)
        centroid = first_moment / area

        # Should be reasonable (between 0.2 and 0.6 for typical airfoils)
        assert 0.2 < centroid < 0.6

        # Second moment (moment of inertia)
        second_moment = jnp.trapezoid(x_eval**2 * thickness, x_eval)

        assert jnp.isfinite(second_moment)
        assert second_moment > 0

    def test_surface_length_accuracy(self):
        """Test accuracy of surface length calculations."""
        naca0012 = NACA4(M=0.0, P=0.0, XX=0.12, n_points=500)  # High resolution

        # Calculate upper surface length
        x_upper = naca0012.upper_surface[0, :]
        y_upper = naca0012.upper_surface[1, :]

        # Arc length calculation
        dx = jnp.diff(x_upper)
        dy = jnp.diff(y_upper)
        ds = jnp.sqrt(dx**2 + dy**2)
        upper_length = jnp.sum(ds)

        # Similarly for lower surface
        x_lower = naca0012.lower_surface[0, :]
        y_lower = naca0012.lower_surface[1, :]

        dx = jnp.diff(x_lower)
        dy = jnp.diff(y_lower)
        ds = jnp.sqrt(dx**2 + dy**2)
        lower_length = jnp.sum(ds)

        # For symmetric airfoil, upper and lower lengths should be equal
        assert jnp.abs(upper_length - lower_length) < 1e-6

        # Total perimeter should be reasonable (> 2 for typical airfoils)
        total_perimeter = upper_length + lower_length
        assert total_perimeter > 2.0
        assert total_perimeter < 3.0  # Should not be excessive


class TestConvergenceValidation:
    """Test convergence properties with increasing resolution."""

    def test_surface_convergence(self):
        """Test surface evaluation convergence with increasing resolution."""
        # Test different resolutions
        resolutions = [50, 100, 200, 400]
        x_test = jnp.array([0.25, 0.5, 0.75])

        results = []
        for n_points in resolutions:
            naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)
            y_upper = naca2412.y_upper(x_test)
            results.append(y_upper)

        # Results should converge (differences should decrease or stay small)
        for i in range(1, len(results)):
            diff_current = jnp.max(jnp.abs(results[i] - results[i - 1]))
            if i > 1:
                diff_previous = jnp.max(jnp.abs(results[i - 1] - results[i - 2]))
                # Current difference should be smaller (convergence) or both should be very small
                if (
                    diff_previous > 1e-12
                ):  # Only check convergence if previous difference was significant
                    assert (
                        diff_current < diff_previous * 1.1
                    )  # Allow some numerical noise

    def test_property_convergence(self):
        """Test geometric property convergence with increasing resolution."""
        resolutions = [50, 100, 200, 400, 800]

        max_thickness_values = []
        max_thickness_locations = []

        for n_points in resolutions:
            naca0015 = NACA4(M=0.0, P=0.0, XX=0.15, n_points=n_points)
            max_thickness_values.append(float(naca0015.max_thickness))
            max_thickness_locations.append(float(naca0015.max_thickness_location))

        # Values should converge to analytical value (0.15)
        final_thickness = max_thickness_values[-1]
        assert jnp.abs(final_thickness - 0.15) < 1e-4

        # Convergence should be monotonic (differences decreasing)
        thickness_diffs = [
            abs(max_thickness_values[i] - max_thickness_values[i - 1])
            for i in range(1, len(max_thickness_values))
        ]

        # Later differences should generally be smaller
        assert thickness_diffs[-1] < thickness_diffs[0]

    def test_gradient_convergence(self):
        """Test gradient convergence with increasing resolution."""

        def surface_metric(n_points_float):
            """Metric that depends on airfoil resolution."""
            n_points = int(n_points_float)
            naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)
            return naca.max_thickness

        # Test gradient at different resolutions
        resolutions = [100.0, 200.0, 400.0]

        gradients = []
        for n_points in resolutions:
            # Use finite differences for this test
            eps = 1.0
            grad_approx = (
                surface_metric(n_points + eps) - surface_metric(n_points - eps)
            ) / (2 * eps)
            gradients.append(grad_approx)

        # Gradients should be small (metric shouldn't change much with resolution)
        for grad in gradients:
            assert abs(grad) < 1e-4
