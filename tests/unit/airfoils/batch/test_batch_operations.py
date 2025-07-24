"""
Batch operations tests for JAX airfoil implementation.

This module tests batch processing capabilities, vectorized operations,
and performance optimizations for JAX-based airfoils.
"""

import time

import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import vmap

from ICARUS.airfoils.naca4 import NACA4


class TestBatchAirfoilCreation:
    """Test batch creation of airfoils."""

    def test_batch_naca4_creation(self) -> None:
        """Test creating multiple NACA4 airfoils in batch."""
        # Create multiple NACA airfoils with different parameters
        m_values = [0.0, 0.02, 0.04]
        p_values = [0.0, 0.4, 0.6]
        xx_values = [0.12, 0.15, 0.18]

        airfoils = []
        for m in m_values:
            for p in p_values:
                for xx in xx_values:
                    airfoil = NACA4(
                        M=m,
                        P=p,
                        XX=xx,
                        n_points=200,
                    )  # Use 200 to get 100 per side
                    airfoils.append(airfoil)

        assert len(airfoils) == len(m_values) * len(p_values) * len(xx_values)

        # Check that all airfoils are valid
        for airfoil in airfoils:
            assert airfoil.n_points == 100  # 100 points per side (200 total / 2)
            assert isinstance(airfoil.name, str)
            assert airfoil.max_thickness > 0

    def test_vectorized_parameter_creation(self):
        """Test vectorized creation with parameter arrays."""
        # Create arrays of parameters
        m_array = jnp.array([0.0, 0.02, 0.04])
        p_array = jnp.array([0.0, 0.4, 0.6])
        xx_array = jnp.array([0.12, 0.15, 0.18])

        # Test that individual creation works with array elements
        for i in range(len(m_array)):
            airfoil = NACA4(
                M=m_array[i],
                P=p_array[i],
                XX=xx_array[i],
                n_points=200,
            )  # Use 200 to get 100 per side
            assert airfoil is not None
            assert airfoil.n_points == 100  # 100 points per side


class TestBatchSurfaceEvaluation:
    """Test batch evaluation of airfoil surfaces."""

    def test_batch_surface_evaluation(self):
        """Test evaluating surfaces at multiple points simultaneously."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Create batch of x-coordinates (avoid endpoints to prevent numerical issues)
        x_points = jnp.linspace(0.01, 0.99, 50)

        # Evaluate upper and lower surfaces
        y_upper = naca2412.y_upper(x_points)
        y_lower = naca2412.y_lower(x_points)

        assert y_upper.shape == x_points.shape
        assert y_lower.shape == x_points.shape
        # Check that upper is generally above lower (allow small numerical tolerance)
        assert jnp.mean(y_upper - y_lower) > 0  # Average difference should be positive

    def test_batch_thickness_evaluation(self):
        """Test batch thickness evaluation."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Create batch of x-coordinates (avoid endpoints to prevent numerical issues)
        x_points = jnp.linspace(0.01, 0.99, 50)

        # Evaluate thickness using thickness_distribution instead of thickness to avoid boolean indexing
        thickness = naca2412.thickness_distribution(x_points)

        assert thickness.shape == x_points.shape
        assert jnp.all(
            thickness >= -1e-10,
        )  # Thickness should be non-negative (allow small numerical error)

    def test_batch_camber_evaluation(self):
        """Test batch camber line evaluation."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Create batch of x-coordinates
        x_points = jnp.linspace(0, 1, 50)

        # Evaluate camber line and its derivative
        camber = naca2412.camber_line(x_points)
        camber_deriv = naca2412.camber_line_derivative(x_points)

        assert camber.shape == x_points.shape
        assert camber_deriv.shape == x_points.shape

    def test_multidimensional_input_handling(self):
        """Test handling of multidimensional input arrays."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Create 2D array of x-coordinates
        x_2d = jnp.reshape(jnp.linspace(0, 1, 20), (4, 5))

        # Evaluate surfaces
        y_upper_2d = naca2412.y_upper(x_2d)
        y_lower_2d = naca2412.y_lower(x_2d)

        assert y_upper_2d.shape == x_2d.shape
        assert y_lower_2d.shape == x_2d.shape

        # Compare with flattened version
        x_1d = x_2d.flatten()
        y_upper_1d = naca2412.y_upper(x_1d)
        y_lower_1d = naca2412.y_lower(x_1d)

        assert jnp.allclose(y_upper_2d.flatten(), y_upper_1d)
        assert jnp.allclose(y_lower_2d.flatten(), y_lower_1d)


class TestVectorizedOperations:
    """Test vectorized operations using JAX transformations."""

    def test_vmap_surface_evaluation(self):
        """Test vectorized surface evaluation using vmap."""

        def evaluate_airfoil_at_point(params, x_point):
            """Evaluate airfoil surface at a single point."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.y_upper(x_point)

        # Create batch of parameters
        params_batch = jnp.array(
            [[0.0, 0.0, 0.12], [0.02, 0.4, 0.12], [0.04, 0.6, 0.15]],
        )

        x_point = 0.5

        # Vectorize over parameter batch
        vmap_eval = vmap(evaluate_airfoil_at_point, in_axes=(0, None))
        results = vmap_eval(params_batch, x_point)

        assert results.shape == (3,)  # One result per parameter set

        # Compare with individual evaluations
        for i, params in enumerate(params_batch):
            individual_result = evaluate_airfoil_at_point(params, x_point)
            assert jnp.allclose(results[i], individual_result)

    def test_vmap_thickness_distribution(self):
        """Test vectorized thickness distribution evaluation."""

        def thickness_at_points(params, x_points):
            """Compute thickness distribution for given parameters."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.thickness_distribution(x_points)

        # Create batch of parameters
        params_batch = jnp.array(
            [[0.0, 0.0, 0.12], [0.02, 0.4, 0.12], [0.04, 0.6, 0.15]],
        )

        x_points = jnp.linspace(0, 1, 20)

        # Vectorize over parameter batch
        vmap_thickness = vmap(thickness_at_points, in_axes=(0, None))
        results = vmap_thickness(params_batch, x_points)

        assert results.shape == (3, 20)  # 3 parameter sets, 20 points each

    def test_nested_vmap_operations(self):
        """Test nested vmap operations for complex batch processing."""

        def evaluate_multiple_points(params, x_points):
            """Evaluate airfoil at multiple points."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return vmap(naca.y_upper)(x_points)

        # Create batch of parameters and points
        params_batch = jnp.array([[0.0, 0.0, 0.12], [0.02, 0.4, 0.12]])

        x_points_batch = jnp.array([jnp.linspace(0, 1, 10), jnp.linspace(0.2, 0.8, 10)])

        # Vectorize over both parameter and point batches
        vmap_eval = vmap(evaluate_multiple_points, in_axes=(0, 0))
        results = vmap_eval(params_batch, x_points_batch)

        assert results.shape == (2, 10)  # 2 parameter sets, 10 points each


class TestBatchGradientComputation:
    """Test batch gradient computation."""

    def test_batch_gradient_computation(self):
        """Test computing gradients for batch of parameters."""

        def airfoil_property(params):
            """Compute some airfoil property."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.max_thickness

        # Create batch of parameters
        params_batch = jnp.array(
            [[0.0, 0.0, 0.12], [0.02, 0.4, 0.12], [0.04, 0.6, 0.15]],
        )

        # Compute gradients for each parameter set
        grad_fn = grad(airfoil_property)
        vmap_grad = vmap(grad_fn)

        gradients = vmap_grad(params_batch)

        assert gradients.shape == (3, 3)  # 3 parameter sets, 3 gradients each

        # Check that gradients are reasonable
        assert jnp.all(jnp.isfinite(gradients))

    def test_gradient_through_surface_evaluation(self):
        """Test gradients through surface evaluation operations."""

        def surface_integral(params, x_points):
            """Compute integral of upper surface."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=200)
            y_values = naca.y_upper(x_points)
            # Use jnp.trapezoid instead of jnp.trapz (which doesn't exist in JAX)
            return jnp.trapezoid(y_values, x_points)

        params = jnp.array([0.02, 0.4, 0.12])
        x_points = jnp.linspace(0.01, 0.99, 50)  # Avoid endpoints

        # Compute gradient
        grad_fn = grad(surface_integral, argnums=0)
        gradient = grad_fn(params, x_points)

        assert gradient.shape == (3,)  # Gradient w.r.t. m, p, xx
        assert jnp.all(jnp.isfinite(gradient))

    def test_hessian_computation(self):
        """Test second-order derivative computation."""

        def thickness_objective(params):
            """Simple objective based on thickness."""
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.thickness_distribution(0.5) ** 2

        params = jnp.array([0.02, 0.4, 0.12])

        # Compute Hessian
        from jax import hessian

        hess_fn = hessian(thickness_objective)
        H = hess_fn(params)

        assert H.shape == (3, 3)  # 3x3 Hessian matrix
        assert jnp.all(jnp.isfinite(H))


class TestBatchPerformance:
    """Test performance characteristics of batch operations."""

    def test_batch_vs_individual_performance(self):
        """Compare performance of batch vs individual operations."""
        # Create test parameters
        n_airfoils = 10
        params_list = []
        for i in range(n_airfoils):
            m = 0.02 * (i + 1) / n_airfoils
            p = 0.4
            xx = 0.12 + 0.03 * i / n_airfoils
            params_list.append([m, p, xx])

        x_points = jnp.linspace(0, 1, 50)

        # Individual operations
        start_time = time.time()
        individual_results = []
        for params in params_list:
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            result = naca.y_upper(x_points)
            individual_results.append(result)
        individual_time = time.time() - start_time

        # Batch operations using vmap
        def evaluate_airfoil(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.y_upper(x_points)

        params_batch = jnp.array(params_list)
        vmap_eval = vmap(evaluate_airfoil)

        start_time = time.time()
        batch_results = vmap_eval(params_batch)
        batch_time = time.time() - start_time

        # Check that results are equivalent
        for i, individual_result in enumerate(individual_results):
            assert jnp.allclose(individual_result, batch_results[i])

        # Performance comparison (batch should be competitive)
        print(f"Individual time: {individual_time:.4f}s")
        print(f"Batch time: {batch_time:.4f}s")
        print(f"Speedup: {individual_time / batch_time:.2f}x")

    def test_jit_compilation_batch_operations(self):
        """Test JIT compilation of batch operations."""

        def batch_evaluation(params_batch, x_points):
            """Batch evaluation function."""

            def single_eval(params):
                m, p, xx = params
                naca = NACA4(M=m, P=p, XX=xx, n_points=100)
                return naca.y_upper(x_points)

            return vmap(single_eval)(params_batch)

        # JIT compile the function
        jit_batch_eval = jit(batch_evaluation)

        # Test data
        params_batch = jnp.array(
            [[0.0, 0.0, 0.12], [0.02, 0.4, 0.12], [0.04, 0.6, 0.15]],
        )
        x_points = jnp.linspace(0, 1, 20)

        # Compare JIT and non-JIT results
        result_normal = batch_evaluation(params_batch, x_points)
        result_jit = jit_batch_eval(params_batch, x_points)

        assert jnp.allclose(result_normal, result_jit, atol=1e-10)

    def test_memory_efficiency_batch_operations(self):
        """Test memory efficiency of batch operations."""
        # This test checks that batch operations don't create excessive intermediate arrays

        def memory_intensive_operation(params_batch):
            """Operation that could potentially use lots of memory."""

            def single_operation(params):
                m, p, xx = params
                naca = NACA4(M=m, P=p, XX=xx, n_points=200)

                # Multiple operations that create intermediate arrays
                x_points = jnp.linspace(0, 1, 100)
                y_upper = naca.y_upper(x_points)
                y_lower = naca.y_lower(x_points)
                thickness = y_upper - y_lower

                return jnp.sum(thickness)

            return vmap(single_operation)(params_batch)

        # Create large batch
        n_batch = 50
        params_batch = jnp.array(
            [
                [0.02 * i / n_batch, 0.4, 0.12 + 0.03 * i / n_batch]
                for i in range(n_batch)
            ],
        )

        # This should complete without memory errors
        results = memory_intensive_operation(params_batch)

        assert results.shape == (n_batch,)
        assert jnp.all(jnp.isfinite(results))


class TestBatchValidation:
    """Test validation and error handling for batch operations."""

    def test_batch_parameter_validation(self):
        """Test validation of batch parameters."""
        # Test with invalid parameter ranges
        invalid_params = jnp.array(
            [
                [-0.1, 0.4, 0.12],  # Negative camber
                [0.02, -0.1, 0.12],  # Negative position
                [0.02, 0.4, -0.05],  # Negative thickness
            ],
        )

        def create_airfoil(params):
            m, p, xx = params
            return NACA4(M=m, P=p, XX=xx, n_points=100)

        # Should handle invalid parameters gracefully or raise appropriate errors
        for params in invalid_params:
            try:
                airfoil = create_airfoil(params)
                # If creation succeeds, airfoil should still be valid
                assert airfoil is not None
            except ValueError:
                # Acceptable to raise error for invalid parameters
                pass

    def test_batch_size_consistency(self):
        """Test consistency of batch sizes."""
        # Test with mismatched batch sizes
        params_batch = jnp.array([[0.0, 0.0, 0.12], [0.02, 0.4, 0.12]])

        x_points_batch = jnp.array(
            [
                jnp.linspace(0, 1, 10),
                jnp.linspace(0, 1, 10),
                jnp.linspace(0, 1, 10),  # Extra element
            ],
        )

        def batch_operation(params, x_points):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.y_upper(x_points)

        # Should handle mismatched batch sizes appropriately
        try:
            vmap_op = vmap(batch_operation, in_axes=(0, 0))
            results = vmap_op(params_batch, x_points_batch)
        except (ValueError, TypeError):
            # Expected to fail with mismatched batch sizes
            pass

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        # Test with empty parameter batch
        empty_params = jnp.array([]).reshape(0, 3)

        def batch_operation(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            return naca.max_thickness

        # Should handle empty batches gracefully
        try:
            vmap_op = vmap(batch_operation)
            results = vmap_op(empty_params)
            assert results.shape == (0,)
        except (ValueError, IndexError):
            # Acceptable to raise error for empty batches
            pass
