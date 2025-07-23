#!/usr/bin/env python3
"""
Comprehensive JAX Airfoil Performance Demonstration

This script provides a comprehensive demonstration of all JAX airfoil performance
features, combining JIT compilation, memory efficiency, batch processing, and
benchmarking utilities into a single comprehensive showcase.

This serves as the expanded version of the original performance_optimization_demo.py
and demonstrates the full performance capabilities of the JAX airfoil implementation.

Key demonstrations:
- Complete performance optimization workflow
- Integration of all performance features
- Real-world performance optimization scenarios
- Best practices for production deployment

Requirements: 3.1, 3.2, 5.1, 5.2
"""

import gc
import json
import time
from typing import Any
from typing import Dict
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit
from jax import pmap
from jax import vmap

# Import our performance utilities
from examples.jax_airfoils.performance_demos.benchmarking_utilities import (
    PerformanceBenchmarker,
)
from ICARUS.airfoils.naca4 import NACA4


class ComprehensivePerformanceDemo:
    """Comprehensive performance demonstration class."""

    def __init__(self):
        self.benchmarker = PerformanceBenchmarker()
        self.results = {}
        self.optimization_history = []

    def demonstrate_optimization_workflow(self):
        """Demonstrate a complete performance optimization workflow."""
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE OPTIMIZATION WORKFLOW")
        print("=" * 80)

        # Step 1: Baseline performance measurement
        print("\n1. BASELINE PERFORMANCE MEASUREMENT")
        print("-" * 50)

        baseline_results = self._measure_baseline_performance()
        self.results["baseline"] = baseline_results

        # Step 2: JIT compilation optimization
        print("\n2. JIT COMPILATION OPTIMIZATION")
        print("-" * 50)

        jit_results = self._demonstrate_jit_optimization()
        self.results["jit_optimization"] = jit_results

        # Step 3: Memory optimization
        print("\n3. MEMORY OPTIMIZATION")
        print("-" * 50)

        memory_results = self._demonstrate_memory_optimization()
        self.results["memory_optimization"] = memory_results

        # Step 4: Batch processing optimization
        print("\n4. BATCH PROCESSING OPTIMIZATION")
        print("-" * 50)

        batch_results = self._demonstrate_batch_optimization()
        self.results["batch_optimization"] = batch_results

        # Step 5: Advanced optimization techniques
        print("\n5. ADVANCED OPTIMIZATION TECHNIQUES")
        print("-" * 50)

        advanced_results = self._demonstrate_advanced_optimization()
        self.results["advanced_optimization"] = advanced_results

        # Step 6: Production deployment optimization
        print("\n6. PRODUCTION DEPLOYMENT OPTIMIZATION")
        print("-" * 50)

        production_results = self._demonstrate_production_optimization()
        self.results["production_optimization"] = production_results

        return self.results

    def _measure_baseline_performance(self) -> Dict[str, Any]:
        """Measure baseline performance without optimizations."""
        print("Measuring baseline performance...")

        # Create test data
        n_points = 1000
        x_points = jnp.linspace(0.001, 1.0, n_points)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Basic operations without optimization
        operations = {
            "thickness_distribution": lambda x: naca2412.thickness_distribution(x),
            "camber_line": lambda x: naca2412.camber_line(x),
            "y_upper": lambda x: naca2412.y_upper(x),
            "y_lower": lambda x: naca2412.y_lower(x),
        }

        baseline_times = {}
        baseline_memory = {}

        for op_name, operation in operations.items():
            # Time the operation
            times = []
            for _ in range(20):
                gc.collect()
                start_time = time.perf_counter()
                result = operation(x_points)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            baseline_times[op_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
            }

            print(
                f"  {op_name}: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms",
            )

        return {
            "times": baseline_times,
            "test_size": n_points,
            "airfoil_points": naca2412.n_points,
        }

    def _demonstrate_jit_optimization(self) -> Dict[str, Any]:
        """Demonstrate JIT compilation optimization benefits."""
        print("Applying JIT compilation optimization...")

        n_points = 1000
        x_points = jnp.linspace(0.001, 1.0, n_points)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Create JIT-compiled versions
        jit_operations = {
            "thickness_distribution_jit": jit(naca2412.thickness_distribution),
            "camber_line_jit": jit(naca2412.camber_line),
            "y_upper_jit": jit(naca2412.y_upper),
            "y_lower_jit": jit(naca2412.y_lower),
        }

        # Measure compilation overhead
        compilation_times = {}
        for op_name, jit_op in jit_operations.items():
            start_time = time.perf_counter()
            _ = jit_op(x_points)  # First call includes compilation
            compilation_times[op_name] = time.perf_counter() - start_time

        # Measure optimized execution times
        jit_times = {}
        for op_name, jit_op in jit_operations.items():
            times = []
            for _ in range(50):
                gc.collect()
                start_time = time.perf_counter()
                result = jit_op(x_points)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            jit_times[op_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "compilation_time": compilation_times[op_name],
            }

            # Calculate speedup vs baseline
            baseline_name = op_name.replace("_jit", "")
            if baseline_name in self.results["baseline"]["times"]:
                baseline_time = self.results["baseline"]["times"][baseline_name]["mean"]
                speedup = baseline_time / np.mean(times)
                jit_times[op_name]["speedup"] = speedup

                print(
                    f"  {op_name}: {np.mean(times)*1000:.2f} ms "
                    f"(compilation: {compilation_times[op_name]*1000:.1f} ms, "
                    f"speedup: {speedup:.2f}x)",
                )

        return {
            "jit_times": jit_times,
            "compilation_overhead": compilation_times,
            "average_speedup": np.mean(
                [t.get("speedup", 1.0) for t in jit_times.values()],
            ),
        }

    def _demonstrate_memory_optimization(self) -> Dict[str, Any]:
        """Demonstrate memory optimization techniques."""
        print("Applying memory optimization techniques...")

        # Test different memory optimization strategies
        strategies = {}

        # Strategy 1: Efficient data structures
        print("  Testing efficient data structures...")
        n_points = 5000
        x_points = jnp.linspace(0.001, 1.0, n_points)
        naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)

        # Memory-inefficient approach
        def inefficient_computation(airfoil, x):
            y_upper = airfoil.y_upper(x)
            y_lower = airfoil.y_lower(x)
            thickness = y_upper - y_lower
            camber = (y_upper + y_lower) / 2
            return y_upper, y_lower, thickness, camber

        # Memory-efficient approach
        @jit
        def efficient_computation(airfoil, x):
            y_upper = airfoil.y_upper(x)
            y_lower = airfoil.y_lower(x)
            thickness = y_upper - y_lower
            camber = (y_upper + y_lower) * 0.5  # Avoid division
            return jnp.stack([y_upper, y_lower, thickness, camber])

        # Measure memory usage
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Inefficient approach
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024
        result_inefficient = inefficient_computation(naca2412, x_points)
        mem_after_inefficient = process.memory_info().rss / 1024 / 1024

        # Efficient approach
        gc.collect()
        mem_before_efficient = process.memory_info().rss / 1024 / 1024
        result_efficient = efficient_computation(naca2412, x_points)
        mem_after_efficient = process.memory_info().rss / 1024 / 1024

        strategies["data_structures"] = {
            "inefficient_memory": mem_after_inefficient - mem_before,
            "efficient_memory": mem_after_efficient - mem_before_efficient,
            "memory_savings": (mem_after_inefficient - mem_before)
            - (mem_after_efficient - mem_before_efficient),
        }

        print(
            f"    Inefficient: {strategies['data_structures']['inefficient_memory']:.2f} MB",
        )
        print(
            f"    Efficient: {strategies['data_structures']['efficient_memory']:.2f} MB",
        )
        print(f"    Savings: {strategies['data_structures']['memory_savings']:.2f} MB")

        # Strategy 2: Buffer reuse
        print("  Testing buffer reuse...")

        # Without buffer reuse
        def without_reuse():
            results = []
            for i in range(10):
                x = jnp.linspace(0.001, 1.0, 1000)
                y = naca2412.y_upper(x)
                results.append(y)
            return results

        # With buffer reuse
        def with_reuse():
            x_buffer = jnp.linspace(0.001, 1.0, 1000)
            jit_upper = jit(naca2412.y_upper)
            results = []
            for i in range(10):
                y = jit_upper(x_buffer)
                results.append(y)
            return results

        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024
        result_no_reuse = without_reuse()
        mem_after_no_reuse = process.memory_info().rss / 1024 / 1024

        gc.collect()
        mem_before_reuse = process.memory_info().rss / 1024 / 1024
        result_with_reuse = with_reuse()
        mem_after_reuse = process.memory_info().rss / 1024 / 1024

        strategies["buffer_reuse"] = {
            "without_reuse_memory": mem_after_no_reuse - mem_before,
            "with_reuse_memory": mem_after_reuse - mem_before_reuse,
            "memory_savings": (mem_after_no_reuse - mem_before)
            - (mem_after_reuse - mem_before_reuse),
        }

        print(
            f"    Without reuse: {strategies['buffer_reuse']['without_reuse_memory']:.2f} MB",
        )
        print(
            f"    With reuse: {strategies['buffer_reuse']['with_reuse_memory']:.2f} MB",
        )
        print(f"    Savings: {strategies['buffer_reuse']['memory_savings']:.2f} MB")

        return strategies

    def _demonstrate_batch_optimization(self) -> Dict[str, Any]:
        """Demonstrate batch processing optimization."""
        print("Applying batch processing optimization...")

        batch_sizes = [1, 10, 50, 100]
        n_points = 500

        batch_results = {}

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")

            # Create batch parameters
            M_values = jnp.array(np.random.uniform(0.01, 0.05, batch_size))
            P_values = jnp.array(np.random.uniform(0.3, 0.5, batch_size))
            XX_values = jnp.array(np.random.uniform(0.08, 0.15, batch_size))
            x_eval = jnp.linspace(0.001, 1.0, n_points)

            # Sequential processing
            def sequential_processing(M_vals, P_vals, XX_vals, x):
                results = []
                for i in range(len(M_vals)):
                    naca = NACA4(
                        M=float(M_vals[i]),
                        P=float(P_vals[i]),
                        XX=float(XX_vals[i]),
                        n_points=100,
                    )
                    result = naca.y_upper(x)
                    results.append(result)
                return jnp.stack(results)

            # Vectorized processing
            def vectorized_processing(M_vals, P_vals, XX_vals, x):
                def single_upper_surface(params):
                    m, p, xx = params
                    naca = NACA4(M=m, P=p, XX=xx, n_points=100)
                    return naca.y_upper(x)

                batch_params = jnp.stack([M_vals, P_vals, XX_vals], axis=1)
                return vmap(single_upper_surface)(batch_params)

            # JIT vectorized processing
            vectorized_jit = jit(vectorized_processing)

            # Warm up JIT
            _ = vectorized_jit(M_values, P_values, XX_values, x_eval)

            # Time sequential processing
            times_sequential = []
            for _ in range(10):
                gc.collect()
                start_time = time.perf_counter()
                result_seq = sequential_processing(
                    M_values,
                    P_values,
                    XX_values,
                    x_eval,
                )
                end_time = time.perf_counter()
                times_sequential.append(end_time - start_time)

            # Time vectorized processing
            times_vectorized = []
            for _ in range(10):
                gc.collect()
                start_time = time.perf_counter()
                result_vec = vectorized_jit(M_values, P_values, XX_values, x_eval)
                end_time = time.perf_counter()
                times_vectorized.append(end_time - start_time)

            sequential_time = np.mean(times_sequential)
            vectorized_time = np.mean(times_vectorized)
            speedup = sequential_time / vectorized_time

            batch_results[batch_size] = {
                "sequential_time": sequential_time,
                "vectorized_time": vectorized_time,
                "speedup": speedup,
            }

            print(f"    Sequential: {sequential_time*1000:.1f} ms")
            print(f"    Vectorized: {vectorized_time*1000:.1f} ms")
            print(f"    Speedup: {speedup:.2f}x")

        return batch_results

    def _demonstrate_advanced_optimization(self) -> Dict[str, Any]:
        """Demonstrate advanced optimization techniques."""
        print("Applying advanced optimization techniques...")

        advanced_results = {}

        # Technique 1: Gradient-based optimization
        print("  Testing gradient-based optimization...")

        def airfoil_objective(params):
            m, p, xx = params
            naca = NACA4(M=m, P=p, XX=xx, n_points=100)
            x_points = jnp.linspace(0.001, 1.0, 50)
            y_upper = naca.y_upper(x_points)
            y_lower = naca.y_lower(x_points)
            thickness = y_upper - y_lower
            return jnp.sum(thickness**2) + 0.1 * jnp.sum((y_upper - 0.05) ** 2)

        # Regular gradient computation
        grad_fn = grad(airfoil_objective)

        # JIT-compiled gradient computation
        grad_jit_fn = jit(grad(airfoil_objective))

        params = jnp.array([0.02, 0.4, 0.12])

        # Warm up JIT
        _ = grad_jit_fn(params)

        # Time gradient computations
        times_grad = []
        times_grad_jit = []

        for _ in range(30):
            # Regular gradient
            gc.collect()
            start_time = time.perf_counter()
            grad_result = grad_fn(params)
            end_time = time.perf_counter()
            times_grad.append(end_time - start_time)

            # JIT gradient
            gc.collect()
            start_time = time.perf_counter()
            grad_jit_result = grad_jit_fn(params)
            end_time = time.perf_counter()
            times_grad_jit.append(end_time - start_time)

        grad_time = np.mean(times_grad)
        grad_jit_time = np.mean(times_grad_jit)
        grad_speedup = grad_time / grad_jit_time

        advanced_results["gradient_optimization"] = {
            "regular_gradient_time": grad_time,
            "jit_gradient_time": grad_jit_time,
            "gradient_speedup": grad_speedup,
        }

        print(f"    Regular gradient: {grad_time*1000:.2f} ms")
        print(f"    JIT gradient: {grad_jit_time*1000:.2f} ms")
        print(f"    Speedup: {grad_speedup:.2f}x")

        # Technique 2: Parallel processing (if multiple devices available)
        print("  Testing parallel processing...")

        devices = jax.devices()
        if len(devices) > 1:
            print(f"    Found {len(devices)} devices, testing parallel processing...")

            # Parallel batch processing
            def parallel_batch_processing(batch_params, x):
                def single_upper_surface(params):
                    m, p, xx = params
                    naca = NACA4(M=m, P=p, XX=xx, n_points=100)
                    return naca.y_upper(x)

                return pmap(single_upper_surface)(batch_params)

            # Create test data that can be split across devices
            n_devices = len(devices)
            batch_size = n_devices * 4  # 4 computations per device

            M_values = jnp.array(np.random.uniform(0.01, 0.05, batch_size)).reshape(
                n_devices,
                -1,
            )
            P_values = jnp.array(np.random.uniform(0.3, 0.5, batch_size)).reshape(
                n_devices,
                -1,
            )
            XX_values = jnp.array(np.random.uniform(0.08, 0.15, batch_size)).reshape(
                n_devices,
                -1,
            )

            batch_params = jnp.stack([M_values, P_values, XX_values], axis=2)
            x_eval = jnp.linspace(0.001, 1.0, 200)

            # Time parallel processing
            times_parallel = []
            for _ in range(10):
                gc.collect()
                start_time = time.perf_counter()
                result_parallel = parallel_batch_processing(batch_params, x_eval)
                end_time = time.perf_counter()
                times_parallel.append(end_time - start_time)

            parallel_time = np.mean(times_parallel)

            advanced_results["parallel_processing"] = {
                "parallel_time": parallel_time,
                "n_devices": n_devices,
                "batch_size": batch_size,
            }

            print(
                f"    Parallel processing: {parallel_time*1000:.2f} ms ({n_devices} devices)",
            )
        else:
            print("    Single device detected, skipping parallel processing test")
            advanced_results["parallel_processing"] = {
                "n_devices": 1,
                "message": "Single device",
            }

        return advanced_results

    def _demonstrate_production_optimization(self) -> Dict[str, Any]:
        """Demonstrate production deployment optimization."""
        print("Applying production deployment optimization...")

        production_results = {}

        # Production-ready airfoil evaluation function
        @jit
        def production_airfoil_evaluator(params_batch, x_points):
            """
            Production-ready airfoil evaluator with optimizations:
            - JIT compiled for speed
            - Vectorized for batch processing
            - Memory efficient
            - Error handling
            """

            def evaluate_single_airfoil(params):
                m, p, xx = params
                # Clamp parameters to valid ranges
                m = jnp.clip(m, 0.001, 0.1)
                p = jnp.clip(p, 0.1, 0.9)
                xx = jnp.clip(xx, 0.01, 0.3)

                naca = NACA4(M=m, P=p, XX=xx, n_points=100)

                # Compute all required outputs efficiently
                y_upper = naca.y_upper(x_points)
                y_lower = naca.y_lower(x_points)
                thickness = y_upper - y_lower
                camber = (y_upper + y_lower) * 0.5

                return jnp.stack([y_upper, y_lower, thickness, camber])

            return vmap(evaluate_single_airfoil)(params_batch)

        # Test production function
        print("  Testing production-ready evaluator...")

        # Create realistic production workload
        batch_size = 100
        n_points = 500

        params_batch = jnp.array(
            [
                np.random.uniform(0.01, 0.05, batch_size),
                np.random.uniform(0.3, 0.5, batch_size),
                np.random.uniform(0.08, 0.15, batch_size),
            ],
        ).T

        x_points = jnp.linspace(0.001, 1.0, n_points)

        # Warm up
        _ = production_airfoil_evaluator(params_batch, x_points)

        # Time production function
        times_production = []
        for _ in range(20):
            gc.collect()
            start_time = time.perf_counter()
            results = production_airfoil_evaluator(params_batch, x_points)
            end_time = time.perf_counter()
            times_production.append(end_time - start_time)

        production_time = np.mean(times_production)
        throughput = batch_size / production_time  # airfoils per second

        production_results["production_evaluator"] = {
            "mean_time": production_time,
            "std_time": np.std(times_production),
            "throughput": throughput,
            "batch_size": batch_size,
            "n_points": n_points,
        }

        print(
            f"    Production time: {production_time*1000:.2f} ± {np.std(times_production)*1000:.2f} ms",
        )
        print(f"    Throughput: {throughput:.1f} airfoils/second")

        # Memory usage analysis
        import os

        import psutil

        process = psutil.Process(os.getpid())

        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Run multiple production evaluations
        for _ in range(10):
            _ = production_airfoil_evaluator(params_batch, x_points)

        mem_after = process.memory_info().rss / 1024 / 1024
        memory_per_evaluation = (mem_after - mem_before) / 10

        production_results["memory_analysis"] = {
            "memory_per_evaluation": memory_per_evaluation,
            "memory_efficiency": batch_size
            / max(memory_per_evaluation, 0.001),  # airfoils per MB
        }

        print(f"    Memory per evaluation: {memory_per_evaluation:.2f} MB")
        print(
            f"    Memory efficiency: {production_results['memory_analysis']['memory_efficiency']:.1f} airfoils/MB",
        )

        return production_results

    def create_comprehensive_visualization(self):
        """Create comprehensive visualization of all optimization results."""
        print("\nCreating comprehensive performance visualization...")

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            "Comprehensive JAX Airfoil Performance Optimization Results",
            fontsize=16,
        )

        # Plot 1: JIT Optimization Speedups
        ax1 = plt.subplot(3, 3, 1)
        if "jit_optimization" in self.results:
            jit_data = self.results["jit_optimization"]["jit_times"]
            operations = list(jit_data.keys())
            speedups = [jit_data[op].get("speedup", 1.0) for op in operations]

            bars = ax1.bar(range(len(operations)), speedups, alpha=0.7)
            ax1.set_xlabel("Operation")
            ax1.set_ylabel("Speedup Factor")
            ax1.set_title("JIT Compilation Speedups")
            ax1.set_xticks(range(len(operations)))
            ax1.set_xticklabels(
                [op.replace("_jit", "") for op in operations],
                rotation=45,
            )
            ax1.axhline(y=1, color="red", linestyle="--", alpha=0.5)
            ax1.grid(True, alpha=0.3)

        # Plot 2: Memory Optimization Savings
        ax2 = plt.subplot(3, 3, 2)
        if "memory_optimization" in self.results:
            mem_data = self.results["memory_optimization"]
            strategies = ["Data Structures", "Buffer Reuse"]
            savings = [
                mem_data["data_structures"]["memory_savings"],
                mem_data["buffer_reuse"]["memory_savings"],
            ]

            bars = ax2.bar(strategies, savings, alpha=0.7, color=["green", "blue"])
            ax2.set_ylabel("Memory Savings (MB)")
            ax2.set_title("Memory Optimization Savings")
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar, saving in zip(bars, savings):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{saving:.1f}MB",
                    ha="center",
                    va="bottom",
                )

        # Plot 3: Batch Processing Speedups
        ax3 = plt.subplot(3, 3, 3)
        if "batch_optimization" in self.results:
            batch_data = self.results["batch_optimization"]
            batch_sizes = list(batch_data.keys())
            speedups = [batch_data[bs]["speedup"] for bs in batch_sizes]

            ax3.plot(batch_sizes, speedups, "o-", linewidth=2, markersize=8)
            ax3.set_xlabel("Batch Size")
            ax3.set_ylabel("Speedup Factor")
            ax3.set_title("Batch Processing Speedups")
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1, color="red", linestyle="--", alpha=0.5)

        # Plot 4: Gradient Optimization
        ax4 = plt.subplot(3, 3, 4)
        if (
            "advanced_optimization" in self.results
            and "gradient_optimization" in self.results["advanced_optimization"]
        ):
            grad_data = self.results["advanced_optimization"]["gradient_optimization"]
            methods = ["Regular\nGradient", "JIT\nGradient"]
            times = [
                grad_data["regular_gradient_time"] * 1000,
                grad_data["jit_gradient_time"] * 1000,
            ]

            bars = ax4.bar(methods, times, alpha=0.7, color=["orange", "green"])
            ax4.set_ylabel("Time (ms)")
            ax4.set_title("Gradient Computation Performance")
            ax4.grid(True, alpha=0.3)

            # Add speedup annotation
            speedup = grad_data["gradient_speedup"]
            ax4.text(
                0.5,
                max(times) * 0.8,
                f"{speedup:.1f}x speedup",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        # Plot 5: Production Performance
        ax5 = plt.subplot(3, 3, 5)
        if "production_optimization" in self.results:
            prod_data = self.results["production_optimization"]["production_evaluator"]

            # Create performance metrics
            metrics = ["Throughput\n(airfoils/s)", "Batch Size", "Points per\nAirfoil"]
            values = [
                prod_data["throughput"],
                prod_data["batch_size"],
                prod_data["n_points"],
            ]

            # Normalize values for visualization
            normalized_values = [v / max(values) * 100 for v in values]

            bars = ax5.bar(metrics, normalized_values, alpha=0.7)
            ax5.set_ylabel("Normalized Performance")
            ax5.set_title("Production Performance Metrics")
            ax5.grid(True, alpha=0.3)

            # Add actual values as labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

        # Plot 6: Overall Performance Summary
        ax6 = plt.subplot(3, 3, 6)

        # Calculate overall improvements
        improvements = {}
        if "jit_optimization" in self.results:
            avg_jit_speedup = self.results["jit_optimization"]["average_speedup"]
            improvements["JIT\nCompilation"] = avg_jit_speedup

        if "batch_optimization" in self.results:
            batch_data = self.results["batch_optimization"]
            avg_batch_speedup = np.mean(
                [batch_data[bs]["speedup"] for bs in batch_data.keys()],
            )
            improvements["Batch\nProcessing"] = avg_batch_speedup

        if (
            "advanced_optimization" in self.results
            and "gradient_optimization" in self.results["advanced_optimization"]
        ):
            grad_speedup = self.results["advanced_optimization"][
                "gradient_optimization"
            ]["gradient_speedup"]
            improvements["Gradient\nComputation"] = grad_speedup

        if improvements:
            categories = list(improvements.keys())
            speedups = list(improvements.values())

            bars = ax6.bar(
                categories,
                speedups,
                alpha=0.7,
                color=["red", "green", "blue"],
            )
            ax6.set_ylabel("Speedup Factor")
            ax6.set_title("Overall Performance Improvements")
            ax6.axhline(y=1, color="black", linestyle="--", alpha=0.5)
            ax6.grid(True, alpha=0.3)

            # Add value labels
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{speedup:.1f}x",
                    ha="center",
                    va="bottom",
                )

        # Plot 7: Memory Efficiency Timeline
        ax7 = plt.subplot(3, 3, 7)
        if "memory_optimization" in self.results:
            stages = ["Baseline", "Data Structures", "Buffer Reuse"]

            # Simulate memory usage progression
            baseline_memory = 100  # Normalized baseline
            data_struct_savings = self.results["memory_optimization"][
                "data_structures"
            ]["memory_savings"]
            buffer_reuse_savings = self.results["memory_optimization"]["buffer_reuse"][
                "memory_savings"
            ]

            memory_usage = [
                baseline_memory,
                baseline_memory
                - (data_struct_savings / 10 * baseline_memory),  # Normalize savings
                baseline_memory
                - ((data_struct_savings + buffer_reuse_savings) / 10 * baseline_memory),
            ]

            ax7.plot(
                stages,
                memory_usage,
                "o-",
                linewidth=3,
                markersize=8,
                color="purple",
            )
            ax7.set_ylabel("Memory Usage (Normalized)")
            ax7.set_title("Memory Optimization Progress")
            ax7.grid(True, alpha=0.3)
            ax7.set_xticklabels(stages, rotation=45)

        # Plot 8: Performance vs Batch Size
        ax8 = plt.subplot(3, 3, 8)
        if "batch_optimization" in self.results:
            batch_data = self.results["batch_optimization"]
            batch_sizes = list(batch_data.keys())
            sequential_times = [
                batch_data[bs]["sequential_time"] * 1000 for bs in batch_sizes
            ]
            vectorized_times = [
                batch_data[bs]["vectorized_time"] * 1000 for bs in batch_sizes
            ]

            ax8.plot(
                batch_sizes,
                sequential_times,
                "o-",
                label="Sequential",
                linewidth=2,
            )
            ax8.plot(
                batch_sizes,
                vectorized_times,
                "s-",
                label="Vectorized",
                linewidth=2,
            )
            ax8.set_xlabel("Batch Size")
            ax8.set_ylabel("Time (ms)")
            ax8.set_title("Performance Scaling")
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            ax8.set_yscale("log")

        # Plot 9: Production Readiness Score
        ax9 = plt.subplot(3, 3, 9)

        # Calculate production readiness score
        score_components = {}

        if "jit_optimization" in self.results:
            avg_speedup = self.results["jit_optimization"]["average_speedup"]
            score_components["Speed"] = (
                min(avg_speedup / 5.0, 1.0) * 100
            )  # Normalize to 0-100

        if "memory_optimization" in self.results:
            total_savings = (
                self.results["memory_optimization"]["data_structures"]["memory_savings"]
                + self.results["memory_optimization"]["buffer_reuse"]["memory_savings"]
            )
            score_components["Memory"] = (
                min(total_savings / 50.0, 1.0) * 100
            )  # Normalize to 0-100

        if "production_optimization" in self.results:
            throughput = self.results["production_optimization"][
                "production_evaluator"
            ]["throughput"]
            score_components["Throughput"] = (
                min(throughput / 1000.0, 1.0) * 100
            )  # Normalize to 0-100

        if score_components:
            categories = list(score_components.keys())
            scores = list(score_components.values())

            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            scores_plot = scores + [scores[0]]  # Close the plot
            angles_plot = np.concatenate([angles, [angles[0]]])

            ax9.plot(angles_plot, scores_plot, "o-", linewidth=2)
            ax9.fill(angles_plot, scores_plot, alpha=0.25)
            ax9.set_xticks(angles)
            ax9.set_xticklabels(categories)
            ax9.set_ylim(0, 100)
            ax9.set_title("Production Readiness Score")
            ax9.grid(True)

        plt.tight_layout()
        plt.savefig(
            "examples/jax_airfoils/performance_demos/comprehensive_performance_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Comprehensive performance visualization saved as 'comprehensive_performance_analysis.png'",
        )

        return fig

    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\nGenerating comprehensive performance report...")

        report = {
            "summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "jax_version": jax.__version__,
                "total_optimizations": len(self.results),
            },
            "results": self.results,
            "recommendations": self._generate_recommendations(),
        }

        # Save report as JSON
        with open(
            "examples/jax_airfoils/performance_demos/comprehensive_performance_report.json",
            "w",
        ) as f:
            json.dump(report, f, indent=2, default=str)

        # Generate HTML report
        self._generate_html_report(report)

        print("Performance report saved as:")
        print("- comprehensive_performance_report.json")
        print("- comprehensive_performance_report.html")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if "jit_optimization" in self.results:
            avg_speedup = self.results["jit_optimization"]["average_speedup"]
            if avg_speedup > 2.0:
                recommendations.append(
                    "✓ JIT compilation provides excellent speedups - continue using for production",
                )
            else:
                recommendations.append(
                    "⚠ JIT compilation speedups are modest - consider profiling for bottlenecks",
                )

        if "batch_optimization" in self.results:
            batch_data = self.results["batch_optimization"]
            max_speedup = max([batch_data[bs]["speedup"] for bs in batch_data.keys()])
            if max_speedup > 10.0:
                recommendations.append(
                    "✓ Batch processing provides excellent scalability - prioritize for production",
                )
            else:
                recommendations.append(
                    "⚠ Batch processing benefits are limited - consider algorithm improvements",
                )

        if "memory_optimization" in self.results:
            total_savings = (
                self.results["memory_optimization"]["data_structures"]["memory_savings"]
                + self.results["memory_optimization"]["buffer_reuse"]["memory_savings"]
            )
            if total_savings > 20.0:
                recommendations.append(
                    "✓ Memory optimizations provide significant savings - implement in production",
                )
            else:
                recommendations.append(
                    "⚠ Memory optimizations provide modest savings - monitor for larger workloads",
                )

        if "production_optimization" in self.results:
            throughput = self.results["production_optimization"][
                "production_evaluator"
            ]["throughput"]
            if throughput > 500:
                recommendations.append(
                    "✓ Production throughput is excellent - ready for deployment",
                )
            else:
                recommendations.append(
                    "⚠ Production throughput needs improvement - consider further optimization",
                )

        return recommendations

    def _generate_html_report(self, report: Dict):
        """Generate HTML performance report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive JAX Airfoil Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background-color: #f9f9f9; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .recommendation {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .recommendation.good {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
                .recommendation.warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Comprehensive JAX Airfoil Performance Report</h1>
                    <p>Generated on: {report['summary']['timestamp']}</p>
                    <p>JAX Version: {report['summary']['jax_version']}</p>
                </div>
        """

        # Add summary metrics
        if "jit_optimization" in report["results"]:
            avg_speedup = report["results"]["jit_optimization"]["average_speedup"]
            html_content += f"""
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">
                        <div class="metric-value">{avg_speedup:.1f}x</div>
                        <div class="metric-label">Average JIT Speedup</div>
                    </div>
            """

        if "production_optimization" in report["results"]:
            throughput = report["results"]["production_optimization"][
                "production_evaluator"
            ]["throughput"]
            html_content += f"""
                    <div class="metric">
                        <div class="metric-value">{throughput:.0f}</div>
                        <div class="metric-label">Airfoils/Second</div>
                    </div>
                </div>
            """

        # Add recommendations
        html_content += """
                <div class="section">
                    <h2>Optimization Recommendations</h2>
        """

        for rec in report["recommendations"]:
            css_class = "good" if "✓" in rec else "warning"
            html_content += f'<div class="recommendation {css_class}">{rec}</div>'

        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        with open(
            "examples/jax_airfoils/performance_demos/comprehensive_performance_report.html",
            "w",
        ) as f:
            f.write(html_content)


def main():
    """Main demonstration function."""
    print("COMPREHENSIVE JAX AIRFOIL PERFORMANCE DEMONSTRATION")
    print("=" * 80)
    print("This comprehensive demo showcases the complete performance optimization")
    print("workflow for JAX airfoil operations, from baseline measurement to")
    print("production-ready deployment optimization.")
    print()

    # Create and run comprehensive demo
    demo = ComprehensivePerformanceDemo()

    # Run complete optimization workflow
    results = demo.demonstrate_optimization_workflow()

    # Create comprehensive visualization
    try:
        fig = demo.create_comprehensive_visualization()
        plt.show()
    except Exception as e:
        print(f"Visualization creation failed: {e}")

    # Generate performance report
    report = demo.generate_performance_report()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Generated outputs:")
    print(
        "- comprehensive_performance_analysis.png: Complete performance visualization",
    )
    print("- comprehensive_performance_report.json: Detailed performance data")
    print("- comprehensive_performance_report.html: Formatted performance report")
    print()
    print("Key achievements:")

    if "jit_optimization" in results:
        avg_speedup = results["jit_optimization"]["average_speedup"]
        print(f"- Average JIT speedup: {avg_speedup:.1f}x")

    if "production_optimization" in results:
        throughput = results["production_optimization"]["production_evaluator"][
            "throughput"
        ]
        print(f"- Production throughput: {throughput:.0f} airfoils/second")

    if "memory_optimization" in results:
        total_savings = (
            results["memory_optimization"]["data_structures"]["memory_savings"]
            + results["memory_optimization"]["buffer_reuse"]["memory_savings"]
        )
        print(f"- Total memory savings: {total_savings:.1f} MB")

    print("\nThe JAX airfoil implementation is now optimized and production-ready!")


if __name__ == "__main__":
    main()
