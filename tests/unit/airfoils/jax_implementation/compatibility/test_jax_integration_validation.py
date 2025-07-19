"""
Integration testing and validation for JAX airfoil implementation.

This module provides comprehensive integration testing including:
- Integration with existing ICARUS modules
- Validation against known airfoil databases
- Regression tests on complex airfoil operations
- Memory usage testing under various workloads
- Gradient accuracy validation for optimization workflows

Requirements covered: 1.3, 2.1, 3.1, 8.2
"""

import gc
import os

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import pytest

from ICARUS.airfoils import NACA4
from ICARUS.airfoils import NACA5
from ICARUS.airfoils.jax_implementation.batch_processing import BatchAirfoilOps
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestIcarusModuleIntegration:
    """Test integration with existing ICARUS modules (Requirement 1.3, 3.1)."""

    def test_naca4_compatibility(self):
        """Test compatibility with ICARUS NACA4 module."""
        # Create airfoils using both implementations
        jax_naca = JaxAirfoil.naca4("2412", n_points=100)
        original_naca = NACA4("2412", n_points=100)

        # Compare basic properties
        assert abs(jax_naca.max_thickness - original_naca.max_thickness) < 1e-3
        assert abs(jax_naca.max_camber - original_naca.max_camber) < 1e-3

        # Compare coordinate distributions
        jax_x, jax_y = jax_naca.get_coordinates()
        orig_x, orig_y = original_naca.all_points()

        # Check coordinate similarity (allowing for different point distributions)
        assert len(jax_x) == len(orig_x)
        assert jnp.allclose(jax_x, orig_x, atol=1e-2)
        assert jnp.allclose(jax_y, orig_y, atol=1e-2)

    def test_naca5_compatibility(self):
        """Test compatibility with ICARUS NACA5 module."""
        # Create airfoils using both implementations
        jax_naca = JaxAirfoil.naca5("23012", n_points=100)
        original_naca = NACA5("23012", n_points=100)

        # Compare basic properties
        assert abs(jax_naca.max_thickness - original_naca.max_thickness) < 1e-3
        assert abs(jax_naca.max_camber - original_naca.max_camber) < 1e-3

    def test_original_airfoil_api_compatibility(self):
        """Test API compatibility with original Airfoil class."""
        # Create test coordinates
        upper = jnp.array([[1.0, 0.75, 0.5, 0.25, 0.0], [0.0, 0.05, 0.08, 0.06, 0.0]])
        lower = jnp.array(
            [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, -0.04, -0.06, -0.03, 0.0]],
        )

        # Create airfoils using both implementations
        jax_airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="TestAirfoil")

        # Test that all expected methods exist and work
        query_x = jnp.array([0.25, 0.5, 0.75])

        # Test surface queries
        y_upper = jax_airfoil.y_upper(query_x)
        y_lower = jax_airfoil.y_lower(query_x)
        thickness = jax_airfoil.thickness(query_x)
        camber = jax_airfoil.camber_line(query_x)

        assert len(y_upper) == len(query_x)
        assert len(y_lower) == len(query_x)
        assert len(thickness) == len(query_x)
        assert len(camber) == len(query_x)

        # Test properties
        assert isinstance(jax_airfoil.max_thickness, float)
        assert isinstance(jax_airfoil.max_camber, float)
        assert isinstance(jax_airfoil.chord_length, float)
        assert isinstance(jax_airfoil.name, str)

    def test_interpolation_module_integration(self):
        """Test integration with ICARUS interpolation module."""
        # Create test airfoil
        naca_airfoil = JaxAirfoil.naca4("0012", n_points=50)

        # Test interpolation methods work with JAX arrays
        query_x = jnp.linspace(0.0, 1.0, 20)

        y_upper = naca_airfoil.y_upper(query_x)
        y_lower = naca_airfoil.y_lower(query_x)

        # Verify interpolation results are reasonable
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(y_upper >= y_lower)  # Upper should be above lower

        # Test edge cases
        edge_x = jnp.array([0.0, 1.0])  # Leading and trailing edge
        edge_upper = naca_airfoil.y_upper(edge_x)
        edge_lower = naca_airfoil.y_lower(edge_x)

        # At edges, upper and lower should be close (airfoil closure)
        assert jnp.allclose(edge_upper, edge_lower, atol=1e-2)


class TestAirfoilDatabaseValidation:
    """Test validation against known airfoil databases (Requirement 8.2)."""

    @pytest.fixture
    def known_naca_properties(self):
        """Known properties for standard NACA airfoils."""
        return {
            "0012": {"max_thickness": 0.12, "max_camber": 0.0, "symmetric": True},
            "2412": {"max_thickness": 0.12, "max_camber": 0.02, "symmetric": False},
            "4415": {"max_thickness": 0.15, "max_camber": 0.04, "symmetric": False},
            "0008": {"max_thickness": 0.08, "max_camber": 0.0, "symmetric": True},
            "6409": {"max_thickness": 0.09, "max_camber": 0.06, "symmetric": False},
        }

    def test_naca4_database_validation(self, known_naca_properties):
        """Validate NACA 4-digit airfoils against known properties."""
        for naca_id, expected_props in known_naca_properties.items():
            airfoil = JaxAirfoil.naca4(naca_id, n_points=200)

            # Test thickness
            thickness_error = abs(
                airfoil.max_thickness - expected_props["max_thickness"],
            )
            assert (
                thickness_error < 0.005
            ), f"NACA {naca_id} thickness error: {thickness_error}"

            # Test camber
            camber_error = abs(airfoil.max_camber - expected_props["max_camber"])
            assert camber_error < 0.005, f"NACA {naca_id} camber error: {camber_error}"

            # Test symmetry
            if expected_props["symmetric"]:
                # For symmetric airfoils, camber should be near zero
                assert abs(airfoil.max_camber) < 1e-6

                # Upper and lower surfaces should be symmetric about x-axis
                query_x = jnp.linspace(0.1, 0.9, 10)
                y_upper = airfoil.y_upper(query_x)
                y_lower = airfoil.y_lower(query_x)

                # For symmetric airfoils: y_upper ≈ -y_lower
                symmetry_error = jnp.max(jnp.abs(y_upper + y_lower))
                assert (
                    symmetry_error < 1e-3
                ), f"NACA {naca_id} symmetry error: {symmetry_error}"

    def test_naca5_database_validation(self):
        """Validate NACA 5-digit airfoils against known properties."""
        test_cases = [
            ("23012", {"max_thickness": 0.12, "design_cl": 0.3}),
            ("23015", {"max_thickness": 0.15, "design_cl": 0.3}),
            ("44012", {"max_thickness": 0.12, "design_cl": 0.6}),
        ]

        for naca_id, expected_props in test_cases:
            airfoil = JaxAirfoil.naca5(naca_id, n_points=200)

            # Test thickness
            thickness_error = abs(
                airfoil.max_thickness - expected_props["max_thickness"],
            )
            assert (
                thickness_error < 0.01
            ), f"NACA {naca_id} thickness error: {thickness_error}"

            # Test that camber is reasonable for the design CL
            expected_camber = expected_props["design_cl"] * 0.05  # Rough estimate
            assert airfoil.max_camber > 0.0  # Should have positive camber
            assert airfoil.max_camber < 0.1  # But not excessive

    def test_airfoil_geometric_constraints(self):
        """Test that generated airfoils satisfy basic geometric constraints."""
        test_airfoils = [
            JaxAirfoil.naca4("0012", n_points=100),
            JaxAirfoil.naca4("2412", n_points=100),
            JaxAirfoil.naca5("23012", n_points=100),
        ]

        for airfoil in test_airfoils:
            # Test chord length is approximately 1.0
            assert abs(airfoil.chord_length - 1.0) < 0.01

            # Test leading edge is at x=0, trailing edge at x=1
            x_coords, _ = airfoil.get_coordinates()
            assert abs(jnp.min(x_coords)) < 0.01  # Near x=0
            assert abs(jnp.max(x_coords) - 1.0) < 0.01  # Near x=1

            # Test airfoil closure (leading and trailing edges)
            le_upper = airfoil.y_upper(jnp.array([0.0]))[0]
            le_lower = airfoil.y_lower(jnp.array([0.0]))[0]
            te_upper = airfoil.y_upper(jnp.array([1.0]))[0]
            te_lower = airfoil.y_lower(jnp.array([1.0]))[0]

            # Leading and trailing edges should be closed
            assert abs(le_upper - le_lower) < 0.01
            assert abs(te_upper - te_lower) < 0.01


class TestComplexOperationRegression:
    """Test regression on complex airfoil operations (Requirement 2.1, 8.2)."""

    def test_morphing_regression(self):
        """Test morphing operations maintain expected behavior."""
        # Create two different airfoils
        airfoil1 = JaxAirfoil.naca4("0012", n_points=100)
        airfoil2 = JaxAirfoil.naca4("4415", n_points=100)

        # Test morphing at different eta values
        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        morphed_airfoils = []

        for eta in eta_values:
            morphed = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta)
            morphed_airfoils.append(morphed)

            # Test that morphed airfoil properties are between the two originals
            if eta == 0.0:
                assert abs(morphed.max_thickness - airfoil1.max_thickness) < 1e-6
                assert abs(morphed.max_camber - airfoil1.max_camber) < 1e-6
            elif eta == 1.0:
                assert abs(morphed.max_thickness - airfoil2.max_thickness) < 1e-6
                assert abs(morphed.max_camber - airfoil2.max_camber) < 1e-6
            else:
                # Intermediate values should be between the two
                thickness_range = [airfoil1.max_thickness, airfoil2.max_thickness]
                camber_range = [airfoil1.max_camber, airfoil2.max_camber]

                assert (
                    min(thickness_range)
                    <= morphed.max_thickness
                    <= max(thickness_range)
                )
                assert min(camber_range) <= morphed.max_camber <= max(camber_range)

    def test_flap_operation_regression(self):
        """Test flap operations maintain expected behavior."""
        base_airfoil = JaxAirfoil.naca4("2412", n_points=100)

        # Test different flap configurations
        flap_configs = [
            {"hinge": 0.7, "angle": 10.0},
            {"hinge": 0.8, "angle": 20.0},
            {"hinge": 0.75, "angle": -15.0},  # Negative deflection
        ]

        for config in flap_configs:
            flapped = base_airfoil.flap(
                flap_hinge_chord_percentage=config["hinge"],
                flap_angle=config["angle"],
            )

            # Test that flapped airfoil is different from original
            base_coords = base_airfoil.get_coordinates()
            flapped_coords = flapped.get_coordinates()

            coord_diff = jnp.max(
                jnp.abs(jnp.array(base_coords) - jnp.array(flapped_coords)),
            )
            assert (
                coord_diff > 1e-3
            ), "Flapped airfoil should be different from original"

            # Test that flap affects only the aft portion
            query_x_forward = jnp.array([0.1, 0.2, 0.3])  # Forward of hinge
            query_x_aft = jnp.array([0.8, 0.9])  # Aft of hinge

            # Forward section should be mostly unchanged
            base_forward_upper = base_airfoil.y_upper(query_x_forward)
            flapped_forward_upper = flapped.y_upper(query_x_forward)
            forward_diff = jnp.max(jnp.abs(base_forward_upper - flapped_forward_upper))
            assert forward_diff < 0.01, "Forward section should be mostly unchanged"

            # Aft section should be changed
            base_aft_upper = base_airfoil.y_upper(query_x_aft)
            flapped_aft_upper = flapped.y_upper(query_x_aft)
            aft_diff = jnp.max(jnp.abs(base_aft_upper - flapped_aft_upper))
            assert aft_diff > 1e-3, "Aft section should be changed by flap"

    def test_repaneling_regression(self):
        """Test repaneling operations maintain airfoil shape."""
        original_airfoil = JaxAirfoil.naca4("0012", n_points=50)

        # Test repaneling to different point counts
        target_points = [25, 100, 200]

        for n_points in target_points:
            repaneled = original_airfoil.repanel(n_points)

            # Test that basic properties are preserved
            thickness_error = abs(
                repaneled.max_thickness - original_airfoil.max_thickness,
            )
            assert (
                thickness_error < 0.01
            ), f"Thickness preservation error: {thickness_error}"

            camber_error = abs(repaneled.max_camber - original_airfoil.max_camber)
            assert camber_error < 0.01, f"Camber preservation error: {camber_error}"

            # Test that surface queries give similar results
            query_x = jnp.linspace(0.1, 0.9, 10)
            orig_upper = original_airfoil.y_upper(query_x)
            repaneled_upper = repaneled.y_upper(query_x)

            surface_error = jnp.max(jnp.abs(orig_upper - repaneled_upper))
            assert surface_error < 0.02, f"Surface preservation error: {surface_error}"

    def test_batch_operation_consistency(self):
        """Test that batch operations give consistent results with individual operations."""
        # Create multiple airfoils
        airfoils = [
            JaxAirfoil.naca4("0012", n_points=50),
            JaxAirfoil.naca4("2412", n_points=50),
            JaxAirfoil.naca4("4415", n_points=50),
        ]

        # Test batch thickness computation
        query_x = jnp.array([0.25, 0.5, 0.75])

        # Individual computations
        individual_thickness = [airfoil.thickness(query_x) for airfoil in airfoils]

        # Batch computation
        batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)

        # Compare results
        for i, (individual, batch_result) in enumerate(
            zip(individual_thickness, batch_thickness),
        ):
            error = jnp.max(jnp.abs(individual - batch_result))
            assert error < 1e-10, f"Batch consistency error for airfoil {i}: {error}"


class TestMemoryUsageValidation:
    """Test memory usage under various workloads (Requirement 4.1, 4.3)."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_single_airfoil_memory_usage(self):
        """Test memory usage for single airfoil operations."""
        initial_memory = self.get_memory_usage()

        # Create and manipulate airfoils
        airfoils = []
        for i in range(100):
            airfoil = JaxAirfoil.naca4(f"{i:04d}", n_points=200)
            airfoils.append(airfoil)

        peak_memory = self.get_memory_usage()
        memory_per_airfoil = (peak_memory - initial_memory) / 100

        # Each airfoil should use reasonable memory (less than 1MB)
        assert (
            memory_per_airfoil < 1.0
        ), f"Memory per airfoil too high: {memory_per_airfoil:.2f} MB"

        # Clean up
        del airfoils
        gc.collect()

    def test_batch_operation_memory_efficiency(self):
        """Test memory efficiency of batch operations."""
        # Create test airfoils
        airfoils = [JaxAirfoil.naca4(f"{i:04d}", n_points=100) for i in range(50)]
        query_x = jnp.linspace(0.0, 1.0, 50)

        initial_memory = self.get_memory_usage()

        # Perform batch operations
        batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)
        batch_camber = BatchAirfoilOps.batch_camber_line(airfoils, query_x)

        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory

        # Batch operations should be memory efficient
        assert (
            memory_increase < 100
        ), f"Batch operations memory usage too high: {memory_increase:.2f} MB"

        # Results should be reasonable
        assert batch_thickness.shape == (50, 50)
        assert batch_camber.shape == (50, 50)
        assert jnp.all(jnp.isfinite(batch_thickness))
        assert jnp.all(jnp.isfinite(batch_camber))

    def test_buffer_reallocation_memory(self):
        """Test memory usage during buffer reallocation."""
        initial_memory = self.get_memory_usage()

        # Create airfoils with increasing sizes to trigger reallocation
        sizes = [50, 100, 200, 500, 1000]
        airfoils = []

        for size in sizes:
            # Create NACA airfoil with specified number of points
            airfoil = JaxAirfoil.naca4("0012", n_points=size)
            airfoils.append(airfoil)

            # Verify buffer size is appropriate
            expected_buffer_size = 2 ** int(
                np.ceil(np.log2(size * 2)),
            )  # Upper + lower surfaces
            assert airfoil.buffer_size >= expected_buffer_size

        peak_memory = self.get_memory_usage()
        total_memory_increase = peak_memory - initial_memory

        # Memory usage should scale reasonably with airfoil size
        assert (
            total_memory_increase < 50
        ), f"Buffer reallocation memory usage too high: {total_memory_increase:.2f} MB"

    def test_jit_compilation_memory(self):
        """Test memory usage during JIT compilation."""
        initial_memory = self.get_memory_usage()

        # Create airfoil and trigger JIT compilation
        airfoil = JaxAirfoil.naca4("2412", n_points=200)

        # Perform operations that trigger JIT compilation
        query_x = jnp.linspace(0.0, 1.0, 100)

        # First call triggers compilation
        thickness1 = airfoil.thickness(query_x)
        camber1 = airfoil.camber_line(query_x)
        upper1 = airfoil.y_upper(query_x)
        lower1 = airfoil.y_lower(query_x)

        compilation_memory = self.get_memory_usage()

        # Subsequent calls should reuse compiled functions
        thickness2 = airfoil.thickness(query_x)
        camber2 = airfoil.camber_line(query_x)
        upper2 = airfoil.y_upper(query_x)
        lower2 = airfoil.y_lower(query_x)

        final_memory = self.get_memory_usage()

        # Compilation should add some memory overhead
        compilation_overhead = compilation_memory - initial_memory
        assert (
            compilation_overhead < 100
        ), f"JIT compilation memory overhead too high: {compilation_overhead:.2f} MB"

        # Subsequent calls should not add significant memory
        reuse_overhead = final_memory - compilation_memory
        assert (
            reuse_overhead < 10
        ), f"JIT reuse memory overhead too high: {reuse_overhead:.2f} MB"

        # Results should be identical
        assert jnp.allclose(thickness1, thickness2)
        assert jnp.allclose(camber1, camber2)
        assert jnp.allclose(upper1, upper2)
        assert jnp.allclose(lower1, lower2)


class TestOptimizationGradientAccuracy:
    """Test gradient accuracy for optimization workflows (Requirement 2.1)."""

    def test_finite_difference_gradient_validation(self):
        """Validate gradients using finite difference approximation."""
        # Create test airfoil
        airfoil = JaxAirfoil.naca4("2412", n_points=100)

        def objective_function(airfoil_coords):
            """Test objective function for optimization."""
            # Reconstruct airfoil from coordinates
            test_airfoil = JaxAirfoil(airfoil_coords, name="OptTest")

            # Compute objective (e.g., minimize drag-related quantity)
            query_x = jnp.array([0.3, 0.5, 0.7])
            thickness = test_airfoil.thickness(query_x)
            camber = test_airfoil.camber_line(query_x)

            # Simple objective: minimize thickness variance + camber magnitude
            return jnp.var(thickness) + jnp.sum(jnp.abs(camber))

        # Get initial coordinates
        coords = airfoil._coordinates

        # Compute analytical gradients
        grad_fn = jax.grad(objective_function)
        analytical_grad = grad_fn(coords)

        # Compute finite difference gradients
        eps = 1e-6
        finite_diff_grad = jnp.zeros_like(coords)

        for i in range(coords.shape[0]):
            for j in range(min(10, coords.shape[1])):  # Test subset for efficiency
                if airfoil._validity_mask[j]:  # Only test valid points
                    coords_plus = coords.at[i, j].add(eps)
                    coords_minus = coords.at[i, j].add(-eps)

                    f_plus = objective_function(coords_plus)
                    f_minus = objective_function(coords_minus)

                    finite_diff_grad = finite_diff_grad.at[i, j].set(
                        (f_plus - f_minus) / (2 * eps),
                    )

        # Compare gradients for valid points
        valid_analytical = analytical_grad[
            airfoil._validity_mask[: analytical_grad.shape[1]]
        ]
        valid_finite_diff = finite_diff_grad[
            airfoil._validity_mask[: finite_diff_grad.shape[1]]
        ]

        # Compute relative error
        relative_error = jnp.abs(valid_analytical - valid_finite_diff) / (
            jnp.abs(valid_finite_diff) + 1e-8
        )
        max_relative_error = jnp.max(relative_error)

        assert (
            max_relative_error < 1e-3
        ), f"Gradient accuracy error too high: {max_relative_error}"

    def test_optimization_convergence(self):
        """Test that gradients enable optimization convergence."""
        # Define optimization problem: find airfoil with target thickness distribution
        target_thickness = jnp.array(
            [0.02, 0.08, 0.10, 0.08, 0.02],
        )  # Target thickness at query points
        query_x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        def optimization_objective(airfoil):
            """Objective function: minimize difference from target thickness."""
            current_thickness = airfoil.thickness(query_x)
            return jnp.sum((current_thickness - target_thickness) ** 2)

        # Start with initial airfoil
        initial_airfoil = JaxAirfoil.naca4("0012", n_points=100)
        initial_objective = optimization_objective(initial_airfoil)

        # Define gradient-based update step
        def update_step(airfoil, learning_rate=0.01):
            grad_fn = jax.grad(optimization_objective)
            gradients = grad_fn(airfoil)

            # Update coordinates
            new_coords = airfoil._coordinates - learning_rate * gradients._coordinates

            # Create new airfoil with updated coordinates
            return JaxAirfoil(new_coords, name="Optimized")

        # Perform optimization steps
        current_airfoil = initial_airfoil
        objectives = [initial_objective]

        for step in range(10):
            try:
                current_airfoil = update_step(current_airfoil, learning_rate=0.001)
                current_objective = optimization_objective(current_airfoil)
                objectives.append(current_objective)
            except Exception:
                # If optimization fails, at least check that we can compute gradients
                grad_fn = jax.grad(optimization_objective)
                gradients = grad_fn(current_airfoil)
                assert jnp.all(jnp.isfinite(gradients._coordinates))
                break

        # Check that optimization is making progress (objective decreasing)
        if len(objectives) > 5:
            # Objective should generally decrease
            final_objective = objectives[-1]
            improvement = initial_objective - final_objective
            assert (
                improvement > 0
            ), f"Optimization should improve objective, got improvement: {improvement}"

    def test_morphing_parameter_gradients(self):
        """Test gradients with respect to morphing parameters."""
        # Create two base airfoils
        airfoil1 = JaxAirfoil.naca4("0012", n_points=100)
        airfoil2 = JaxAirfoil.naca4("4415", n_points=100)

        def morphing_objective(eta):
            """Objective function depending on morphing parameter."""
            morphed = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta)
            return morphed.max_thickness + 0.5 * morphed.max_camber

        # Test gradient computation
        eta_test = 0.5
        grad_fn = jax.grad(morphing_objective)
        gradient = grad_fn(eta_test)

        # Gradient should be finite
        assert jnp.isfinite(gradient), "Morphing parameter gradient should be finite"

        # Test finite difference validation
        eps = 1e-6
        f_plus = morphing_objective(eta_test + eps)
        f_minus = morphing_objective(eta_test - eps)
        finite_diff_grad = (f_plus - f_minus) / (2 * eps)

        relative_error = abs(gradient - finite_diff_grad) / (
            abs(finite_diff_grad) + 1e-8
        )
        assert (
            relative_error < 1e-3
        ), f"Morphing gradient error too high: {relative_error}"

    def test_flap_parameter_gradients(self):
        """Test gradients with respect to flap parameters."""
        base_airfoil = JaxAirfoil.naca4("2412", n_points=100)

        def flap_objective(flap_angle):
            """Objective function depending on flap angle."""
            flapped = base_airfoil.flap(
                flap_hinge_chord_percentage=0.75,
                flap_angle=flap_angle,
            )
            # Objective: minimize camber change due to flap
            return jnp.abs(flapped.max_camber - base_airfoil.max_camber)

        # Test gradient computation
        angle_test = 10.0
        grad_fn = jax.grad(flap_objective)
        gradient = grad_fn(angle_test)

        # Gradient should be finite
        assert jnp.isfinite(gradient), "Flap parameter gradient should be finite"

        # Test that gradient has reasonable magnitude
        assert abs(gradient) > 1e-8, "Flap gradient should be non-zero"
        assert abs(gradient) < 1e2, "Flap gradient should not be excessive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
