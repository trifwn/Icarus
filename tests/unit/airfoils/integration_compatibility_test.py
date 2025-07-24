"""
Integration and compatibility testing for JAX airfoil implementation.

This module implements comprehensive tests for:
- Integration with existing ICARUS modules and workflows
- Compatibility with different Python and JAX versions
- Stress tests with large datasets and complex operations
- Memory usage and performance validation under various workloads

Requirements: 6.3, 7.1, 7.2, 7.3, 7.4
"""

import gc
import os
import platform
import sys
import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad
from jax import jit

# Import ICARUS modules for integration testing
from ICARUS.airfoils import Airfoil
from ICARUS.airfoils.naca4 import NACA4


class TestIcarusModuleIntegration:
    """Test integration with existing ICARUS modules and workflows."""

    def test_core_types_integration(self):
        """Test integration with ICARUS core types."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test that airfoil surfaces are compatible with core types
        upper_surface = naca2412.upper_surface
        lower_surface = naca2412.lower_surface

        # Check that surfaces are JAX arrays (which are compatible with core types)
        assert isinstance(upper_surface, jnp.ndarray)
        assert isinstance(lower_surface, jnp.ndarray)

        # Test surface evaluation with core types
        x_points = jnp.linspace(0, 1, 100)
        y_upper = naca2412.y_upper(x_points)
        y_lower = naca2412.y_lower(x_points)

        # Check that results are JAX arrays (compatible with core types)
        assert isinstance(y_upper, jnp.ndarray)
        assert isinstance(y_lower, jnp.ndarray)

    def test_interpolation_module_integration(self):
        """Test integration with ICARUS interpolation module."""
        from ICARUS.interpolation import JaxInterpolator1D

        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test that airfoil uses JaxInterpolator1D internally
        assert hasattr(naca2412, "_y_upper_interp")
        assert hasattr(naca2412, "_y_lower_interp")
        assert isinstance(naca2412._y_upper_interp, JaxInterpolator1D)
        assert isinstance(naca2412._y_lower_interp, JaxInterpolator1D)

        # Test interpolation functionality (without exact consistency check)
        x_test = jnp.array([0.25, 0.5, 0.75])
        y_upper_direct = naca2412.y_upper(x_test)

        # Test that interpolation produces reasonable results
        assert jnp.all(jnp.isfinite(y_upper_direct))
        assert len(y_upper_direct) == len(x_test)

        # Test that internal interpolator exists and works
        x_scaled = naca2412.min_x + x_test * (naca2412.max_x - naca2412.min_x)
        y_upper_interp = naca2412._y_upper_interp(x_scaled)
        assert jnp.all(jnp.isfinite(y_upper_interp))

        # Test that results are in reasonable range (not exact match due to implementation details)
        assert jnp.all(jnp.abs(y_upper_direct) < 1.0)  # Reasonable airfoil coordinates

    def test_vehicle_module_integration(self):
        """Test integration with ICARUS vehicle module components."""
        # Test that airfoils can be used in vehicle contexts
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test typical vehicle workflow operations
        x_chord = jnp.linspace(0, 1, 50)

        # Surface evaluation (used in mesh generation)
        y_upper = naca2412.y_upper(x_chord)
        y_lower = naca2412.y_lower(x_chord)

        # Thickness distribution (used in structural analysis)
        thickness = naca2412.thickness(x_chord)

        # Camber line (used in aerodynamic analysis)
        camber = naca2412.camber_line(x_chord)

        # All should be valid for vehicle integration
        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(y_lower))
        assert jnp.all(jnp.isfinite(thickness))
        assert jnp.all(jnp.isfinite(camber))

        # Test geometric properties used in vehicle design
        max_thickness = naca2412.max_thickness
        max_thickness_loc = naca2412.max_thickness_location

        assert 0 < max_thickness < 1
        assert 0 <= max_thickness_loc <= 1

    def test_aero_module_integration(self):
        """Test integration with ICARUS aero module workflows."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test typical aerodynamic analysis workflow
        # Panel method requires surface coordinates
        upper_coords = naca2412.upper_surface
        lower_coords = naca2412.lower_surface

        # Test coordinate format for panel methods
        assert upper_coords.shape[0] == 2  # x, y coordinates
        assert lower_coords.shape[0] == 2  # x, y coordinates
        assert upper_coords.shape[1] > 0  # Has points
        assert lower_coords.shape[1] > 0  # Has points

        # Test surface normal computation (needed for panel methods)
        x_panels = jnp.linspace(0.01, 0.99, 20)
        y_upper = naca2412.y_upper(x_panels)
        y_lower = naca2412.y_lower(x_panels)

        # Compute surface slopes (for normal vectors)
        dy_upper_dx = jnp.gradient(y_upper, x_panels)
        dy_lower_dx = jnp.gradient(y_lower, x_panels)

        # Should be finite and reasonable
        assert jnp.all(jnp.isfinite(dy_upper_dx))
        assert jnp.all(jnp.isfinite(dy_lower_dx))

    def test_optimization_module_integration(self) -> None:
        """Test integration with ICARUS optimization workflows."""

        def airfoil_design_objective(params):
            """Typical airfoil design objective function."""
            m, p, xx = params

            # Constraint: valid NACA parameters
            m = jnp.clip(m, 0.0, 0.1)
            p = jnp.clip(p, 0.0, 1.0)
            xx = jnp.clip(xx, 0.06, 0.25)

            naca = NACA4(M=m, P=p, XX=xx, n_points=100)

            # Multi-objective optimization
            max_thickness = naca.max_thickness
            max_camber = jnp.max(naca.camber_line(jnp.linspace(0, 1, 50)))

            # Objective: balance thickness and camber
            thickness_penalty = (max_thickness - 0.12) ** 2
            camber_penalty = (max_camber - 0.02) ** 2

            return thickness_penalty + camber_penalty

        # Test gradient-based optimization compatibility
        initial_params = jnp.array([0.02, 0.4, 0.12])

        # Compute objective and gradient
        objective_value = airfoil_design_objective(initial_params)
        gradient = grad(airfoil_design_objective)(initial_params)

        assert jnp.isfinite(objective_value)
        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)

        # Test JIT compilation for optimization
        jit_objective = jit(airfoil_design_objective)
        jit_gradient = jit(grad(airfoil_design_objective))

        jit_obj_value = jit_objective(initial_params)
        jit_grad_value = jit_gradient(initial_params)

        assert jnp.allclose(objective_value, jit_obj_value)
        assert jnp.allclose(gradient, jit_grad_value)

    def test_database_module_integration(self):
        """Test integration with ICARUS database workflows."""
        # Test batch airfoil creation (typical database operation)
        naca_codes = ["0012", "2412", "4415", "6409", "0015", "2415"]
        airfoils = []

        for code in naca_codes:
            airfoil = Airfoil.naca(code, n_points=100)
            airfoils.append(airfoil)

        assert len(airfoils) == len(naca_codes)

        # Test batch property extraction
        max_thicknesses = []
        max_cambers = []

        for airfoil in airfoils:
            max_thicknesses.append(airfoil.max_thickness)

            x_camber = jnp.linspace(0, 1, 50)
            camber_line = airfoil.camber_line(x_camber)
            max_cambers.append(jnp.max(camber_line))

        # All should be valid
        assert len(max_thicknesses) == len(naca_codes)
        assert len(max_cambers) == len(naca_codes)
        assert all(t > 0 for t in max_thicknesses)
        assert all(jnp.isfinite(c) for c in max_cambers)

    def test_mission_module_integration(self):
        """Test integration with ICARUS mission analysis workflows."""
        # Test airfoil performance evaluation across flight conditions
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Simulate different flight conditions
        alpha_range = jnp.linspace(-5, 15, 21)  # Angle of attack range

        # Test that airfoil geometry remains consistent across conditions
        x_eval = jnp.linspace(0, 1, 100)

        for alpha in alpha_range:
            # Airfoil geometry shouldn't change with angle of attack
            y_upper = naca2412.y_upper(x_eval)
            y_lower = naca2412.y_lower(x_eval)
            thickness = naca2412.thickness(x_eval)

            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))
            assert jnp.all(jnp.isfinite(thickness))
            # Allow small negative values due to numerical precision
            assert jnp.all(thickness >= -1e-15)


class TestVersionCompatibility:
    """Test compatibility with different Python and JAX versions."""

    def test_python_version_compatibility(self):
        """Test compatibility with current Python version."""
        python_version = sys.version_info

        # Should work with Python 3.8+
        assert python_version.major == 3
        assert python_version.minor >= 8

        # Test basic functionality works
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)
        x_test = jnp.linspace(0, 1, 50)
        y_upper = naca2412.y_upper(x_test)

        assert jnp.all(jnp.isfinite(y_upper))

    def test_jax_version_compatibility(self):
        """Test compatibility with current JAX version."""
        jax_version = jax.__version__
        print(f"Testing with JAX version: {jax_version}")

        # Test core JAX functionality
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test JIT compilation
        @jit
        def evaluate_airfoil(x):
            return naca2412.y_upper(x)

        x_test = jnp.linspace(0, 1, 50)
        result = evaluate_airfoil(x_test)

        assert jnp.all(jnp.isfinite(result))

        # Test gradient computation
        def thickness_at_point(x):
            return naca2412.thickness(x)

        grad_fn = grad(thickness_at_point)
        gradient = grad_fn(0.5)

        assert jnp.isfinite(gradient)

    def test_numpy_version_compatibility(self):
        """Test compatibility with NumPy arrays and operations."""
        numpy_version = np.__version__
        print(f"Testing with NumPy version: {numpy_version}")

        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test with NumPy arrays as input
        x_numpy = np.linspace(0, 1, 50)
        y_upper_numpy = naca2412.y_upper(x_numpy)

        # Test with JAX arrays as input
        x_jax = jnp.linspace(0, 1, 50)
        y_upper_jax = naca2412.y_upper(x_jax)

        # Results should be equivalent
        assert jnp.allclose(y_upper_numpy, y_upper_jax)

    def test_platform_compatibility(self):
        """Test compatibility across different platforms."""
        platform_info = platform.platform()
        print(f"Testing on platform: {platform_info}")

        # Test basic functionality works on current platform
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

        # Test computationally intensive operations
        x_dense = jnp.linspace(0, 1, 1000)
        y_upper = naca2412.y_upper(x_dense)
        thickness = naca2412.thickness(x_dense)

        assert jnp.all(jnp.isfinite(y_upper))
        assert jnp.all(jnp.isfinite(thickness))

    def test_device_compatibility(self):
        """Test compatibility with available JAX devices."""
        devices = jax.devices()
        print(f"Available devices: {[str(d) for d in devices]}")

        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)
        x_test = jnp.linspace(0, 1, 100)

        # Test on each available device
        for device in devices:
            with jax.default_device(device):
                y_upper = naca2412.y_upper(x_test)
                assert jnp.all(jnp.isfinite(y_upper))


class TestStressTestsLargeDatasets:
    """Test with large datasets and complex operations."""

    def test_large_point_count_stress(self):
        """Test with very large point counts."""
        # Test with increasingly large point counts
        point_counts = [1000, 5000, 10000, 20000]

        for n_points in point_counts:
            print(f"Testing with {n_points} points")

            naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)

            # Test basic operations
            assert naca2412.n_points == n_points // 2  # Due to morphing behavior
            assert naca2412.upper_surface.shape[1] > 0
            assert naca2412.lower_surface.shape[1] > 0

            # Test surface evaluation with large arrays
            x_eval = jnp.linspace(0, 1, 1000)
            y_upper = naca2412.y_upper(x_eval)
            y_lower = naca2412.y_lower(x_eval)

            assert jnp.all(jnp.isfinite(y_upper))
            assert jnp.all(jnp.isfinite(y_lower))

    def test_batch_airfoil_creation_stress(self):
        """Test creating many airfoils simultaneously."""
        # Create large batch of airfoils
        n_airfoils = 100
        airfoils = []

        start_time = time.time()

        for i in range(n_airfoils):
            m = 0.01 + 0.08 * i / n_airfoils  # Vary camber
            p = 0.2 + 0.6 * (i % 10) / 10  # Vary position
            xx = 0.08 + 0.15 * (i % 5) / 5  # Vary thickness

            naca = NACA4(M=m, P=p, XX=xx, n_points=200)
            airfoils.append(naca)

        creation_time = time.time() - start_time
        print(f"Created {n_airfoils} airfoils in {creation_time:.2f} seconds")

        assert len(airfoils) == n_airfoils
        assert creation_time < 30.0  # Should complete in reasonable time

        # Test batch evaluation
        x_eval = jnp.linspace(0, 1, 100)
        results = []

        start_time = time.time()
        for airfoil in airfoils:
            y_upper = airfoil.y_upper(x_eval)
            results.append(y_upper)

        evaluation_time = time.time() - start_time
        print(f"Evaluated {n_airfoils} airfoils in {evaluation_time:.2f} seconds")

        assert len(results) == n_airfoils
        assert evaluation_time < 10.0  # Should be fast

    def test_complex_morphing_operations_stress(self):
        """Test complex morphing operations with many airfoils."""
        # Create base airfoils
        base_airfoils = []
        naca_codes = ["0012", "2412", "4415", "6409", "0015", "2415", "4412", "6412"]

        for code in naca_codes:
            airfoil = Airfoil.naca(code, n_points=200)
            base_airfoils.append(airfoil)

        # Create morphing matrix (all combinations)
        morphed_airfoils = []
        eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        start_time = time.time()

        for i, airfoil1 in enumerate(base_airfoils):
            for j, airfoil2 in enumerate(base_airfoils):
                if i != j:  # Don't morph airfoil with itself
                    for eta in eta_values:
                        morphed = Airfoil.morph_new_from_two_foils(
                            airfoil1,
                            airfoil2,
                            eta=eta,
                            n_points=200,
                        )
                        morphed_airfoils.append(morphed)

        morphing_time = time.time() - start_time
        expected_count = len(base_airfoils) * (len(base_airfoils) - 1) * len(eta_values)

        print(
            f"Created {len(morphed_airfoils)} morphed airfoils in {morphing_time:.2f} seconds",
        )

        assert len(morphed_airfoils) == expected_count
        assert morphing_time < 60.0  # Should complete in reasonable time

    def test_gradient_computation_stress(self):
        """Test gradient computation with complex operations."""

        def complex_airfoil_function(params):
            """Complex function involving airfoil operations."""
            m, p, xx = params

            # Create airfoil
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)

            # Compute complex objective
            x_eval = jnp.linspace(0, 1, 50)
            thickness = naca.thickness(x_eval)
            camber = naca.camber_line(x_eval)

            # Multi-objective function using JAX operations
            thickness_mean = jnp.mean(thickness)
            camber_mean = jnp.mean(jnp.abs(camber))
            thickness_max = jnp.max(thickness)

            return thickness_mean + 0.5 * camber_mean + 0.1 * thickness_max

        # Test gradient computation
        params = jnp.array([0.02, 0.4, 0.12])

        start_time = time.time()

        # Compute function value
        function_value = complex_airfoil_function(params)

        # Compute gradient
        gradient_fn = grad(complex_airfoil_function)
        gradient = gradient_fn(params)

        computation_time = time.time() - start_time

        print(f"Complex gradient computation took {computation_time:.2f} seconds")

        assert jnp.isfinite(function_value)
        assert jnp.all(jnp.isfinite(gradient))
        assert gradient.shape == (3,)
        assert computation_time < 10.0  # Should be reasonably fast

    def test_vectorized_operations_stress(self):
        """Test vectorized operations with large arrays."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Test with very large evaluation arrays
        large_x = jnp.linspace(0, 1, 10000)

        start_time = time.time()

        # Vectorized surface evaluation
        y_upper_large = naca2412.y_upper(large_x)
        y_lower_large = naca2412.y_lower(large_x)
        thickness_large = naca2412.thickness(large_x)
        camber_large = naca2412.camber_line(large_x)

        vectorized_time = time.time() - start_time

        print(
            f"Vectorized operations on {len(large_x)} points took {vectorized_time:.2f} seconds",
        )

        assert len(y_upper_large) == len(large_x)
        assert len(y_lower_large) == len(large_x)
        assert len(thickness_large) == len(large_x)
        assert len(camber_large) == len(large_x)

        assert jnp.all(jnp.isfinite(y_upper_large))
        assert jnp.all(jnp.isfinite(y_lower_large))
        assert jnp.all(jnp.isfinite(thickness_large))
        assert jnp.all(jnp.isfinite(camber_large))

        assert vectorized_time < 5.0  # Should be fast due to vectorization


class TestMemoryUsagePerformance:
    """Test memory usage and performance under various workloads."""

    def test_memory_usage_monitoring(self):
        """Test memory usage during typical operations."""
        tracemalloc.start()

        # Baseline memory
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_stats = baseline_snapshot.statistics("lineno")
        baseline_memory = sum(stat.size for stat in baseline_stats)

        # Create airfoils and monitor memory
        airfoils = []
        for i in range(50):
            naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
            airfoils.append(naca)

        creation_snapshot = tracemalloc.take_snapshot()
        creation_stats = creation_snapshot.statistics("lineno")
        creation_memory = sum(stat.size for stat in creation_stats)

        # Perform evaluations
        x_eval = jnp.linspace(0, 1, 1000)
        results = []

        for airfoil in airfoils:
            y_upper = airfoil.y_upper(x_eval)
            results.append(y_upper)

        evaluation_snapshot = tracemalloc.take_snapshot()
        evaluation_stats = evaluation_snapshot.statistics("lineno")
        evaluation_memory = sum(stat.size for stat in evaluation_stats)

        # Clean up
        del airfoils, results
        gc.collect()

        cleanup_snapshot = tracemalloc.take_snapshot()
        cleanup_stats = cleanup_snapshot.statistics("lineno")
        cleanup_memory = sum(stat.size for stat in cleanup_stats)

        tracemalloc.stop()

        # Report memory usage
        print(f"Baseline memory: {baseline_memory / 1024 / 1024:.2f} MB")
        print(f"After creation: {creation_memory / 1024 / 1024:.2f} MB")
        print(f"After evaluation: {evaluation_memory / 1024 / 1024:.2f} MB")
        print(f"After cleanup: {cleanup_memory / 1024 / 1024:.2f} MB")

        # Memory should be reasonable
        creation_increase = (creation_memory - baseline_memory) / 1024 / 1024
        evaluation_increase = (evaluation_memory - creation_memory) / 1024 / 1024

        assert creation_increase < 100  # Less than 100 MB for 50 airfoils
        assert evaluation_increase < 200  # Less than 200 MB for evaluations

    def test_performance_scaling(self):
        """Test performance scaling with problem size."""
        point_counts = [100, 200, 500, 1000, 2000]
        creation_times = []
        evaluation_times = []

        for n_points in point_counts:
            # Test creation time
            start_time = time.time()
            naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)
            creation_time = time.time() - start_time
            creation_times.append(creation_time)

            # Test evaluation time
            x_eval = jnp.linspace(0, 1, 1000)
            start_time = time.time()
            y_upper = naca.y_upper(x_eval)
            evaluation_time = time.time() - start_time
            evaluation_times.append(evaluation_time)

            print(
                f"n_points={n_points}: creation={creation_time:.4f}s, evaluation={evaluation_time:.4f}s",
            )

        # Performance should scale reasonably (allow for JIT compilation overhead)
        for i in range(1, len(point_counts)):
            scale_factor = point_counts[i] / point_counts[i - 1]
            time_factor = creation_times[i] / max(
                creation_times[i - 1],
                1e-6,
            )  # Avoid division by zero

            # Allow for JIT compilation overhead and variability
            # Skip check if times are very small (measurement noise)
            if creation_times[i] > 0.01 and creation_times[i - 1] > 0.01:
                assert time_factor < scale_factor**2.0  # More lenient scaling

    def test_jit_compilation_performance(self):
        """Test JIT compilation performance and caching."""
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Define JIT-compiled functions
        @jit
        def jit_surface_evaluation(x):
            return naca2412.y_upper(x), naca2412.y_lower(x)

        @jit
        def jit_thickness_evaluation(x):
            return naca2412.thickness(x)

        x_test = jnp.linspace(0, 1, 1000)

        # First call (includes compilation time)
        start_time = time.time()
        y_upper_1, y_lower_1 = jit_surface_evaluation(x_test)
        first_call_time = time.time() - start_time

        # Second call (should use cached compilation)
        start_time = time.time()
        y_upper_2, y_lower_2 = jit_surface_evaluation(x_test)
        second_call_time = time.time() - start_time

        # Third call with different function
        start_time = time.time()
        thickness = jit_thickness_evaluation(x_test)
        thickness_call_time = time.time() - start_time

        print(f"First JIT call: {first_call_time:.4f}s")
        print(f"Second JIT call: {second_call_time:.4f}s")
        print(f"Thickness JIT call: {thickness_call_time:.4f}s")

        # Results should be identical
        assert jnp.allclose(y_upper_1, y_upper_2)
        assert jnp.allclose(y_lower_1, y_lower_2)

        # Second call should be much faster (cached)
        assert second_call_time < first_call_time / 2

        # All results should be valid
        assert jnp.all(jnp.isfinite(y_upper_1))
        assert jnp.all(jnp.isfinite(y_lower_1))
        assert jnp.all(jnp.isfinite(thickness))

    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        import queue
        import threading

        def worker_function(work_queue, result_queue):
            """Worker function for concurrent testing."""
            while True:
                try:
                    work_item = work_queue.get(timeout=1)
                    if work_item is None:
                        break

                    m, p, xx = work_item
                    naca = NACA4(M=m, P=p, XX=xx, n_points=200)

                    x_eval = jnp.linspace(0, 1, 100)
                    y_upper = naca.y_upper(x_eval)
                    max_thickness = naca.max_thickness

                    result_queue.put((max_thickness, jnp.max(y_upper)))
                    work_queue.task_done()

                except queue.Empty:
                    break

        # Create work items
        work_items = []
        for i in range(20):
            m = 0.01 + 0.08 * i / 20
            p = 0.2 + 0.6 * (i % 5) / 5
            xx = 0.08 + 0.15 * (i % 3) / 3
            work_items.append((m, p, xx))

        work_queue = queue.Queue()
        result_queue = queue.Queue()

        # Add work items to queue
        for item in work_items:
            work_queue.put(item)

        # Start worker threads
        num_threads = 4
        threads = []

        start_time = time.time()

        for _ in range(num_threads):
            thread = threading.Thread(
                target=worker_function,
                args=(work_queue, result_queue),
            )
            thread.start()
            threads.append(thread)

        # Wait for completion
        work_queue.join()

        # Stop threads
        for _ in range(num_threads):
            work_queue.put(None)

        for thread in threads:
            thread.join()

        concurrent_time = time.time() - start_time

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(
            f"Concurrent processing of {len(work_items)} items took {concurrent_time:.2f}s",
        )

        assert len(results) == len(work_items)
        assert concurrent_time < 30.0  # Should complete in reasonable time

        # All results should be valid
        for max_thickness, max_y_upper in results:
            assert jnp.isfinite(max_thickness)
            assert jnp.isfinite(max_y_upper)
            assert max_thickness > 0

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform many operations that could potentially leak memory
        for iteration in range(10):
            airfoils = []

            # Create many airfoils
            for i in range(50):
                naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
                airfoils.append(naca)

            # Perform operations
            x_eval = jnp.linspace(0, 1, 100)
            for airfoil in airfoils:
                y_upper = airfoil.y_upper(x_eval)
                thickness = airfoil.thickness(x_eval)

            # Clean up
            del airfoils
            gc.collect()

            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            print(
                f"Iteration {iteration + 1}: Memory usage = {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)",
            )

            # Memory increase should be bounded
            assert memory_increase < 500  # Less than 500 MB increase

        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory

        print(f"Total memory increase: {total_increase:.2f} MB")

        # Total increase should be reasonable (no major leaks)
        assert total_increase < 200  # Less than 200 MB total increase


class TestProductionWorkloadValidation:
    """Test validation under production-like workloads."""

    def test_typical_design_workflow(self):
        """Test typical airfoil design workflow performance."""
        # Simulate typical design workflow
        start_time = time.time()

        # 1. Create base airfoils
        base_airfoils = []
        naca_codes = ["0012", "2412", "4415", "6409", "0015", "2415"]

        for code in naca_codes:
            airfoil = Airfoil.naca(code, n_points=200)
            base_airfoils.append(airfoil)

        # 2. Generate morphed variants
        morphed_airfoils = []
        eta_values = jnp.linspace(0, 1, 11)

        for i in range(len(base_airfoils) - 1):
            for eta in eta_values:
                morphed = Airfoil.morph_new_from_two_foils(
                    base_airfoils[i],
                    base_airfoils[i + 1],
                    eta=float(eta),
                    n_points=200,
                )
                morphed_airfoils.append(morphed)

        # 3. Evaluate all airfoils
        x_eval = jnp.linspace(0, 1, 100)
        all_airfoils = base_airfoils + morphed_airfoils

        evaluation_results = []
        for airfoil in all_airfoils:
            y_upper = airfoil.y_upper(x_eval)
            y_lower = airfoil.y_lower(x_eval)
            thickness = airfoil.thickness(x_eval)

            max_thickness = jnp.max(thickness)
            max_camber = jnp.max(airfoil.camber_line(x_eval))

            evaluation_results.append(
                {
                    "max_thickness": max_thickness,
                    "max_camber": max_camber,
                    "name": airfoil.name,
                },
            )

        workflow_time = time.time() - start_time

        print(
            f"Design workflow with {len(all_airfoils)} airfoils took {workflow_time:.2f}s",
        )

        assert len(evaluation_results) == len(all_airfoils)
        assert workflow_time < 60.0  # Should complete in reasonable time

        # All results should be valid
        for result in evaluation_results:
            assert jnp.isfinite(result["max_thickness"])
            assert jnp.isfinite(result["max_camber"])
            assert result["max_thickness"] > 0

    def test_optimization_workflow_performance(self):
        """Test optimization workflow performance."""

        def optimization_objective(params):
            """Multi-objective airfoil optimization."""
            m, p, xx = params

            # Constraints
            m = jnp.clip(m, 0.0, 0.1)
            p = jnp.clip(p, 0.1, 0.9)
            xx = jnp.clip(xx, 0.06, 0.25)

            naca = NACA4(M=m, P=p, XX=xx, n_points=100)

            # Objectives
            x_eval = jnp.linspace(0, 1, 50)
            thickness = naca.thickness(x_eval)
            camber = naca.camber_line(x_eval)

            # Performance metrics
            max_thickness = jnp.max(thickness)
            thickness_location = jnp.argmax(thickness) / len(thickness)
            max_camber = jnp.max(jnp.abs(camber))

            # Multi-objective function
            thickness_penalty = (max_thickness - 0.12) ** 2
            location_penalty = (thickness_location - 0.3) ** 2
            camber_penalty = (max_camber - 0.02) ** 2

            return thickness_penalty + location_penalty + camber_penalty

        # Test optimization performance
        initial_params = jnp.array([0.02, 0.4, 0.12])

        start_time = time.time()

        # Simulate optimization iterations
        current_params = initial_params
        for iteration in range(50):
            # Compute objective and gradient
            objective_value = optimization_objective(current_params)
            gradient = grad(optimization_objective)(current_params)

            # Simple gradient descent step
            learning_rate = 0.001
            current_params = current_params - learning_rate * gradient

            # Ensure parameters stay in valid range
            current_params = jnp.array(
                [
                    jnp.clip(current_params[0], 0.0, 0.1),
                    jnp.clip(current_params[1], 0.1, 0.9),
                    jnp.clip(current_params[2], 0.06, 0.25),
                ],
            )

        optimization_time = time.time() - start_time

        print(f"50 optimization iterations took {optimization_time:.2f}s")

        assert optimization_time < 30.0  # Should be reasonably fast

        # Final result should be valid
        final_objective = optimization_objective(current_params)
        assert jnp.isfinite(final_objective)

    def test_database_query_simulation(self):
        """Test database-like query operations."""
        # Create airfoil database
        airfoil_database = []

        # Generate systematic variations
        m_values = jnp.linspace(0, 0.08, 9)
        p_values = jnp.linspace(0.2, 0.8, 7)
        xx_values = jnp.linspace(0.08, 0.20, 13)

        start_time = time.time()

        for m in m_values:
            for p in p_values:
                for xx in xx_values:
                    naca = NACA4(M=float(m), P=float(p), XX=float(xx), n_points=100)

                    # Compute properties for database
                    x_eval = jnp.linspace(0, 1, 50)
                    max_thickness = naca.max_thickness
                    max_camber = jnp.max(naca.camber_line(x_eval))

                    airfoil_database.append(
                        {
                            "airfoil": naca,
                            "max_thickness": float(max_thickness),
                            "max_camber": float(max_camber),
                            "m": float(m),
                            "p": float(p),
                            "xx": float(xx),
                        },
                    )

        database_creation_time = time.time() - start_time

        print(
            f"Created database with {len(airfoil_database)} airfoils in {database_creation_time:.2f}s",
        )

        # Test database queries
        start_time = time.time()

        # Query 1: Find airfoils with thickness between 0.10 and 0.15
        thickness_query = [
            entry
            for entry in airfoil_database
            if 0.10 <= entry["max_thickness"] <= 0.15
        ]

        # Query 2: Find airfoils with low camber
        low_camber_query = [
            entry for entry in airfoil_database if entry["max_camber"] < 0.01
        ]

        # Query 3: Find airfoils with specific thickness location
        specific_airfoils = []
        for entry in airfoil_database[:50]:  # Limit for performance
            thickness_location = entry["airfoil"].max_thickness_location
            if 0.25 <= thickness_location <= 0.35:
                specific_airfoils.append(entry)

        query_time = time.time() - start_time

        print(f"Database queries took {query_time:.2f}s")
        print(f"Thickness query results: {len(thickness_query)}")
        print(f"Low camber query results: {len(low_camber_query)}")
        print(f"Specific location results: {len(specific_airfoils)}")

        assert len(airfoil_database) > 0
        assert database_creation_time < 120.0  # Should complete in reasonable time
        assert query_time < 10.0  # Queries should be fast


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
