"""
Gradient verification tests for JAX airfoil implementation.

This module provides detailed gradient testing using finite differences
to verify the correctness of automatic differentiation.

Requirements covered: 2.1, 2.2
"""

import jax
import jax.numpy as jnp
import pytest

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestGradientVerification:
    """Verify gradients using finite difference approximations."""

    @pytest.fixture
    def test_airfoil(self):
        """Create a test airfoil for gradient verification."""
        upper = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.06, 0.08, 0.05, 0.0]])
        lower = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.04, -0.06, -0.03, 0.0]],
        )
        return JaxAirfoil.from_upper_lower(upper, lower, name="GradVerify")

    def finite_difference_gradient(self, func, airfoil, eps=1e-6):
        """Compute finite difference gradient of function with respect to airfoil coordinates."""
        base_value = func(airfoil)

        # Get valid coordinates
        valid_coords = airfoil._coordinates[:, airfoil._validity_mask]
        n_coords = valid_coords.shape[1]

        gradients = jnp.zeros_like(valid_coords)

        for i in range(2):  # x and y coordinates
            for j in range(n_coords):
                # Create perturbed airfoil
                perturbed_coords = airfoil._coordinates.at[
                    :,
                    airfoil._validity_mask,
                ].set(valid_coords.at[i, j].add(eps))

                # Create new airfoil with perturbed coordinates
                perturbed_airfoil = JaxAirfoil(
                    perturbed_coords[:, airfoil._validity_mask],
                    name=airfoil.name,
                )

                # Compute finite difference
                perturbed_value = func(perturbed_airfoil)
                gradients = gradients.at[i, j].set((perturbed_value - base_value) / eps)

        return gradients

    def test_thickness_gradient_verification(self, test_airfoil):
        """Verify thickness gradients using finite differences."""

        def thickness_func(airfoil):
            return jnp.sum(airfoil.thickness(jnp.array([0.5])))

        # Compute analytical gradient
        grad_fn = jax.grad(thickness_func)
        analytical_grad = grad_fn(test_airfoil)
        analytical_coords = analytical_grad._coordinates[
            :,
            analytical_grad._validity_mask,
        ]

        # Compute finite difference gradient
        fd_grad = self.finite_difference_gradient(thickness_func, test_airfoil)

        # Compare gradients
        max_error = jnp.max(jnp.abs(analytical_coords - fd_grad))
        relative_error = max_error / (jnp.max(jnp.abs(fd_grad)) + 1e-10)

        assert relative_error < 0.01  # Less than 1% relative error
        assert max_error < 1e-3  # Small absolute error

    def test_camber_gradient_verification(self, test_airfoil):
        """Verify camber line gradients using finite differences."""

        def camber_func(airfoil):
            return jnp.sum(jnp.abs(airfoil.camber_line(jnp.array([0.3, 0.7]))))

        # Compute analytical gradient
        grad_fn = jax.grad(camber_func)
        analytical_grad = grad_fn(test_airfoil)
        analytical_coords = analytical_grad._coordinates[
            :,
            analytical_grad._validity_mask,
        ]

        # Compute finite difference gradient
        fd_grad = self.finite_difference_gradient(camber_func, test_airfoil)

        # Compare gradients
        max_error = jnp.max(jnp.abs(analytical_coords - fd_grad))
        relative_error = max_error / (jnp.max(jnp.abs(fd_grad)) + 1e-10)

        assert relative_error < 0.01  # Less than 1% relative error

    def test_surface_query_gradient_verification(self, test_airfoil):
        """Verify surface query gradients using finite differences."""

        def upper_surface_func(airfoil):
            return jnp.sum(airfoil.y_upper(jnp.array([0.25, 0.75])))

        def lower_surface_func(airfoil):
            return jnp.sum(jnp.abs(airfoil.y_lower(jnp.array([0.25, 0.75]))))

        # Test upper surface gradients
        upper_grad_fn = jax.grad(upper_surface_func)
        upper_analytical = upper_grad_fn(test_airfoil)
        upper_analytical_coords = upper_analytical._coordinates[
            :,
            upper_analytical._validity_mask,
        ]

        upper_fd_grad = self.finite_difference_gradient(
            upper_surface_func,
            test_airfoil,
        )

        upper_max_error = jnp.max(jnp.abs(upper_analytical_coords - upper_fd_grad))
        upper_relative_error = upper_max_error / (
            jnp.max(jnp.abs(upper_fd_grad)) + 1e-10
        )

        assert upper_relative_error < 0.01

        # Test lower surface gradients
        lower_grad_fn = jax.grad(lower_surface_func)
        lower_analytical = lower_grad_fn(test_airfoil)
        lower_analytical_coords = lower_analytical._coordinates[
            :,
            lower_analytical._validity_mask,
        ]

        lower_fd_grad = self.finite_difference_gradient(
            lower_surface_func,
            test_airfoil,
        )

        lower_max_error = jnp.max(jnp.abs(lower_analytical_coords - lower_fd_grad))
        lower_relative_error = lower_max_error / (
            jnp.max(jnp.abs(lower_fd_grad)) + 1e-10
        )

        assert lower_relative_error < 0.01

    def test_geometric_property_gradient_verification(self, test_airfoil):
        """Verify geometric property gradients using finite differences."""

        def max_thickness_func(airfoil):
            return airfoil.max_thickness

        # Compute analytical gradient
        grad_fn = jax.grad(max_thickness_func)
        analytical_grad = grad_fn(test_airfoil)
        analytical_coords = analytical_grad._coordinates[
            :,
            analytical_grad._validity_mask,
        ]

        # Compute finite difference gradient
        fd_grad = self.finite_difference_gradient(max_thickness_func, test_airfoil)

        # Compare gradients (may be less accurate for max operations)
        max_error = jnp.max(jnp.abs(analytical_coords - fd_grad))
        relative_error = max_error / (jnp.max(jnp.abs(fd_grad)) + 1e-10)

        assert relative_error < 0.05  # Allow higher tolerance for max operations

    def test_morphing_parameter_gradient_verification(self, test_airfoil):
        """Verify morphing parameter gradients using finite differences."""
        # Create second airfoil for morphing
        upper2 = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.04, 0.06, 0.03, 0.0]])
        lower2 = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.06, -0.08, -0.05, 0.0]],
        )
        airfoil2 = JaxAirfoil.from_upper_lower(upper2, lower2, name="MorphTarget")

        def morphing_func(eta):
            morphed = JaxAirfoil.morph_new_from_two_foils(test_airfoil, airfoil2, eta)
            return jnp.sum(morphed.thickness(jnp.array([0.5])))

        # Compute analytical gradient
        grad_fn = jax.grad(morphing_func)
        analytical_grad = grad_fn(0.3)

        # Compute finite difference gradient
        eps = 1e-6
        fd_grad = (morphing_func(0.3 + eps) - morphing_func(0.3 - eps)) / (2 * eps)

        # Compare gradients
        error = abs(analytical_grad - fd_grad)
        relative_error = error / (abs(fd_grad) + 1e-10)

        assert relative_error < 0.01

    def test_flap_angle_gradient_verification(self, test_airfoil):
        """Verify flap angle gradients using finite differences."""

        def flap_func(angle):
            flapped = test_airfoil.flap(
                hinge_point=jnp.array([0.75, 0.0]),
                flap_angle=angle,
            )
            return jnp.sum(flapped.thickness(jnp.array([0.9])))

        # Compute analytical gradient
        grad_fn = jax.grad(flap_func)
        analytical_grad = grad_fn(0.1)

        # Compute finite difference gradient
        eps = 1e-6
        fd_grad = (flap_func(0.1 + eps) - flap_func(0.1 - eps)) / (2 * eps)

        # Compare gradients
        error = abs(analytical_grad - fd_grad)
        relative_error = error / (abs(fd_grad) + 1e-10)

        assert relative_error < 0.01

    def test_higher_order_gradients(self, test_airfoil):
        """Test second-order gradients (Hessian)."""

        def thickness_func(airfoil):
            return airfoil.thickness(jnp.array([0.5]))[0]

        # Compute second-order gradient
        hessian_fn = jax.hessian(thickness_func)
        hessian = hessian_fn(test_airfoil)

        # Hessian should be finite
        hessian_coords = hessian._coordinates[:, hessian._validity_mask]
        assert jnp.all(jnp.isfinite(hessian_coords))

    def test_gradient_flow_through_transformations(self, test_airfoil):
        """Test gradient flow through chained transformations."""

        def chained_transformation(airfoil):
            # Chain multiple operations
            flapped = airfoil.flap(hinge_point=jnp.array([0.75, 0.0]), flap_angle=0.05)
            repaneled = flapped.repanel(n_points=15, distribution="cosine")
            return jnp.sum(repaneled.thickness(jnp.array([0.5])))

        # Compute gradient through chained operations
        grad_fn = jax.grad(chained_transformation)
        gradients = grad_fn(test_airfoil)

        # Gradients should be finite
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        assert jnp.all(jnp.isfinite(grad_coords))
        assert jnp.any(jnp.abs(grad_coords) > 1e-8)  # Should have non-zero gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
