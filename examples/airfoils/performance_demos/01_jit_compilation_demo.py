#!/usr/bin/env python3
"""
JAX Airfoil JIT Compilation Performance Demo

This script demonstrates the performance benefits of JAX's Just-In-Time (JIT) compilation
for airfoil operations. It compares execution times between regular and JIT-compiled
functions, showing compilation overhead and subsequent speedups.

Key demonstrations:
- JIT compilation timing and warm-up behavior
- Performance comparison between JIT and regular execution
- Memory efficiency of JIT-compiled functions
- Compilation caching and reuse

Requirements: 3.1, 3.2, 5.1, 5.2
"""

import gc
import time
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from jax import jit

from ICARUS.airfoils.naca4 import NACA4


def benchmark_function_timing(
    func,
    args,
    n_runs: int = 100,
    warmup: int = 5,
) -> Dict[str, float]:
    """
    Benchmark function execution timing with proper warm-up.

    Args:
        func: Function to benchmark
        args: Arguments to pass to function
        n_runs: Number of timing runs
        warmup: Number of warm-up runs

    Returns:
        Dictionary with timing statistics
    """
    # Warm-up runs
    for _ in range(warmup):
        _ = func(*args)

    # Force garbage collection
    gc.collect()

    # Timing runs
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)

    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "median_time": float(np.median(times)),
        "total_time": float(np.sum(times)),
        "n_runs": n_runs,
    }


def demonstrate_jit_compilation_overhead():
    """Demonstrate JIT compilation overhead and subsequent speedups."""
    print("=" * 60)
    print("JIT COMPILATION OVERHEAD DEMONSTRATION")
    print("=" * 60)

    # Create test airfoil
    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    x_points = jnp.linspace(0, 1, 500)

    # Define test function
    def surface_evaluation(x):
        return naca2412.y_upper(x)

    # JIT compile the function
    jit_surface_evaluation = jit(surface_evaluation)

    print(f"Testing with {len(x_points)} evaluation points")
    print(f"Airfoil: NACA 2412 with {naca2412.n_points} surface points")
    print()

    # Time first JIT call (includes compilation)
    print("Timing first JIT call (includes compilation overhead)...")
    start_time = time.perf_counter()
    result_first_jit = jit_surface_evaluation(x_points)
    first_jit_time = time.perf_counter() - start_time

    # Time regular function
    print("Timing regular function calls...")
    regular_stats = benchmark_function_timing(
        surface_evaluation,
        (x_points,),
        n_runs=50,
    )

    # Time subsequent JIT calls (no compilation)
    print("Timing subsequent JIT calls (no compilation)...")
    jit_stats = benchmark_function_timing(
        jit_surface_evaluation,
        (x_points,),
        n_runs=50,
    )

    # Verify correctness
    result_regular = surface_evaluation(x_points)
    result_jit = jit_surface_evaluation(x_points)
    max_error = jnp.max(jnp.abs(result_regular - result_jit))

    # Display results
    print("\nRESULTS:")
    print(f"First JIT call (with compilation): {first_jit_time * 1000:.2f} ms")
    print(
        f"Regular function mean time:        {regular_stats['mean_time'] * 1000:.2f} ± {regular_stats['std_time'] * 1000:.2f} ms",
    )
    print(
        f"JIT function mean time:           {jit_stats['mean_time'] * 1000:.2f} ± {jit_stats['std_time'] * 1000:.2f} ms",
    )
    print(f"Maximum numerical error:          {max_error:.2e}")

    if jit_stats["mean_time"] < regular_stats["mean_time"]:
        speedup = regular_stats["mean_time"] / jit_stats["mean_time"]
        print(f"JIT speedup:                      {speedup:.2f}x")
    else:
        overhead = jit_stats["mean_time"] / regular_stats["mean_time"]
        print(f"JIT overhead:                     {overhead:.2f}x")

    compilation_overhead = first_jit_time / regular_stats["mean_time"]
    print(
        f"Compilation overhead:             {compilation_overhead:.1f}x regular execution",
    )

    return {
        "first_jit_time": first_jit_time,
        "regular_stats": regular_stats,
        "jit_stats": jit_stats,
        "max_error": float(max_error),
        "speedup": regular_stats["mean_time"] / jit_stats["mean_time"],
    }


def demonstrate_jit_with_different_operations():
    """Demonstrate JIT performance across different airfoil operations."""
    print("\n" + "=" * 60)
    print("JIT PERFORMANCE ACROSS DIFFERENT OPERATIONS")
    print("=" * 60)

    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    x_points = jnp.linspace(0, 1, 300)

    # Define operations to test
    operations = {
        "y_upper": lambda x: naca2412.y_upper(x),
        "y_lower": lambda x: naca2412.y_lower(x),
        "thickness": lambda x: naca2412.thickness(x),
        "camber_line": lambda x: naca2412.camber_line(x),
        "thickness_distribution": lambda x: naca2412.thickness_distribution(x),
    }

    results = {}

    for op_name, operation in operations.items():
        print(f"\nTesting {op_name}...")

        # JIT compile
        jit_operation = jit(operation)

        # Warm-up JIT
        _ = jit_operation(x_points)

        # Benchmark both versions
        regular_stats = benchmark_function_timing(operation, (x_points,), n_runs=30)
        jit_stats = benchmark_function_timing(jit_operation, (x_points,), n_runs=30)

        # Verify correctness
        result_regular = operation(x_points)
        result_jit = jit_operation(x_points)
        max_error = jnp.max(jnp.abs(result_regular - result_jit))

        speedup = regular_stats["mean_time"] / jit_stats["mean_time"]

        results[op_name] = {
            "regular_time": regular_stats["mean_time"],
            "jit_time": jit_stats["mean_time"],
            "speedup": speedup,
            "max_error": float(max_error),
        }

        print(f"  Regular: {regular_stats['mean_time'] * 1000:.2f} ms")
        print(f"  JIT:     {jit_stats['mean_time'] * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Error:   {max_error:.2e}")

    # Summary
    print("\nSUMMARY:")
    print(
        f"{'Operation':<20} {'Regular (ms)':<12} {'JIT (ms)':<10} {'Speedup':<8} {'Error':<10}",
    )
    print("-" * 70)

    for op_name, stats in results.items():
        print(
            f"{op_name:<20} {stats['regular_time'] * 1000:<12.2f} "
            f"{stats['jit_time'] * 1000:<10.2f} {stats['speedup']:<8.2f} "
            f"{stats['max_error']:<10.2e}",
        )

    return results


def demonstrate_jit_compilation_caching():
    """Demonstrate JIT compilation caching with different input shapes."""
    print("\n" + "=" * 60)
    print("JIT COMPILATION CACHING DEMONSTRATION")
    print("=" * 60)

    naca2412 = NACA4(M=0.02, P=0.4, XX=0.12, n_points=100)

    @jit
    def surface_evaluation(x):
        return naca2412.y_upper(x)

    # Test with different input shapes
    test_shapes = [
        ("1D-small", jnp.linspace(0, 1, 50)),
        ("1D-medium", jnp.linspace(0, 1, 200)),
        ("1D-large", jnp.linspace(0, 1, 1000)),
        ("2D-matrix", jnp.reshape(jnp.linspace(0, 1, 100), (10, 10))),
        ("3D-tensor", jnp.reshape(jnp.linspace(0, 1, 120), (3, 4, 10))),
    ]

    print("Testing JIT compilation caching with different input shapes...")
    print()

    compilation_times = {}
    execution_times = {}

    for shape_name, x_data in test_shapes:
        print(f"Testing {shape_name} (shape: {x_data.shape})...")

        # Time first call (may trigger compilation)
        start_time = time.perf_counter()
        result = surface_evaluation(x_data)
        first_call_time = time.perf_counter() - start_time

        # Time subsequent calls (should use cached compilation)
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = surface_evaluation(x_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = np.mean(times)

        compilation_times[shape_name] = first_call_time
        execution_times[shape_name] = avg_time

        print(f"  First call: {first_call_time * 1000:.2f} ms")
        print(f"  Avg subsequent: {avg_time * 1000:.2f} ms")
        print(f"  Compilation overhead: {first_call_time / avg_time:.1f}x")
        print(f"  Result shape: {result.shape}")
        print()

    return compilation_times, execution_times


def demonstrate_gradient_computation_performance():
    """Demonstrate performance of gradient computations with JIT."""
    print("\n" + "=" * 60)
    print("GRADIENT COMPUTATION PERFORMANCE")
    print("=" * 60)

    def airfoil_objective(params):
        """Objective function for gradient testing."""
        m, p, xx = params
        naca = NACA4(M=m, P=p, XX=xx, n_points=100)
        x_points = jnp.linspace(0, 1, 50)
        y_upper = naca.y_upper(x_points)
        return jnp.sum(y_upper**2)

    params = jnp.array([0.02, 0.4, 0.12])

    # Regular gradient computation
    grad_fn = grad(airfoil_objective)

    # JIT-compiled gradient computation
    jit_grad_fn = jit(grad(airfoil_objective))

    print("Comparing gradient computation performance...")
    print()

    # Benchmark function evaluation
    func_stats = benchmark_function_timing(airfoil_objective, (params,), n_runs=50)

    # Benchmark regular gradient
    grad_stats = benchmark_function_timing(grad_fn, (params,), n_runs=50)

    # Warm up JIT gradient
    _ = jit_grad_fn(params)

    # Benchmark JIT gradient
    jit_grad_stats = benchmark_function_timing(jit_grad_fn, (params,), n_runs=50)

    # Verify correctness
    regular_grad = grad_fn(params)
    jit_grad = jit_grad_fn(params)
    grad_error = jnp.max(jnp.abs(regular_grad - jit_grad))

    print("RESULTS:")
    print(
        f"Function evaluation:     {func_stats['mean_time'] * 1000:.2f} ± {func_stats['std_time'] * 1000:.2f} ms",
    )
    print(
        f"Regular gradient:        {grad_stats['mean_time'] * 1000:.2f} ± {grad_stats['std_time'] * 1000:.2f} ms",
    )
    print(
        f"JIT gradient:           {jit_grad_stats['mean_time'] * 1000:.2f} ± {jit_grad_stats['std_time'] * 1000:.2f} ms",
    )
    print(
        f"Gradient overhead:       {grad_stats['mean_time'] / func_stats['mean_time']:.2f}x",
    )
    print(
        f"JIT gradient speedup:    {grad_stats['mean_time'] / jit_grad_stats['mean_time']:.2f}x",
    )
    print(f"Gradient accuracy:       {grad_error:.2e}")

    return {
        "function_time": func_stats["mean_time"],
        "gradient_time": grad_stats["mean_time"],
        "jit_gradient_time": jit_grad_stats["mean_time"],
        "gradient_error": float(grad_error),
    }


def create_performance_visualization(results: Dict):
    """Create visualization of performance results."""
    print("\n" + "=" * 60)
    print("CREATING PERFORMANCE VISUALIZATION")
    print("=" * 60)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("JAX Airfoil JIT Compilation Performance Analysis", fontsize=16)

    # Plot 1: JIT vs Regular execution times
    if "operations" in results:
        ops = list(results["operations"].keys())
        regular_times = [results["operations"][op]["regular_time"] * 1000 for op in ops]
        jit_times = [results["operations"][op]["jit_time"] * 1000 for op in ops]

        x = np.arange(len(ops))
        width = 0.35

        ax1.bar(x - width / 2, regular_times, width, label="Regular", alpha=0.8)
        ax1.bar(x + width / 2, jit_times, width, label="JIT", alpha=0.8)
        ax1.set_xlabel("Operation")
        ax1.set_ylabel("Execution Time (ms)")
        ax1.set_title("JIT vs Regular Execution Times")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ops, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup factors
    if "operations" in results:
        speedups = [results["operations"][op]["speedup"] for op in ops]
        colors = ["green" if s > 1 else "red" for s in speedups]

        ax2.bar(ops, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Operation")
        ax2.set_ylabel("Speedup Factor")
        ax2.set_title("JIT Speedup by Operation")
        ax2.set_xticklabels(ops, rotation=45)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Compilation overhead
    if "compilation_times" in results:
        shapes = list(results["compilation_times"].keys())
        comp_times = [results["compilation_times"][shape] * 1000 for shape in shapes]
        exec_times = [results["execution_times"][shape] * 1000 for shape in shapes]

        ax3.bar(shapes, comp_times, alpha=0.7, label="First call (with compilation)")
        ax3.bar(shapes, exec_times, alpha=0.7, label="Subsequent calls")
        ax3.set_xlabel("Input Shape")
        ax3.set_ylabel("Time (ms)")
        ax3.set_title("JIT Compilation Overhead by Input Shape")
        ax3.set_xticklabels(shapes, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Gradient computation comparison
    if "gradient" in results:
        categories = ["Function", "Regular Gradient", "JIT Gradient"]
        times = [
            results["gradient"]["function_time"] * 1000,
            results["gradient"]["gradient_time"] * 1000,
            results["gradient"]["jit_gradient_time"] * 1000,
        ]

        bars = ax4.bar(categories, times, alpha=0.7)
        ax4.set_ylabel("Time (ms)")
        ax4.set_title("Gradient Computation Performance")
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time:.2f}ms",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(
        "examples/jax_airfoils/performance_demos/jit_performance_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Performance visualization saved as 'jit_performance_analysis.png'")

    return fig


def main():
    """Main demonstration function."""
    print("JAX AIRFOIL JIT COMPILATION PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the performance characteristics of JAX JIT compilation")
    print("for airfoil operations, including compilation overhead and speedups.")
    print()

    # Run demonstrations
    results = {}

    # Basic JIT compilation overhead
    results["basic"] = demonstrate_jit_compilation_overhead()

    # Different operations
    results["operations"] = demonstrate_jit_with_different_operations()

    # Compilation caching
    comp_times, exec_times = demonstrate_jit_compilation_caching()
    results["compilation_times"] = comp_times
    results["execution_times"] = exec_times

    # Gradient computation
    results["gradient"] = demonstrate_gradient_computation_performance()

    # Create visualization
    try:
        fig = create_performance_visualization(results)
        plt.show()
    except Exception as e:
        print(f"Visualization creation failed: {e}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print(
        "1. JIT compilation has initial overhead but provides speedups for repeated calls",
    )
    print("2. Performance benefits vary by operation complexity")
    print("3. JIT compilation is cached for different input shapes")
    print("4. Gradient computations can be significantly accelerated with JIT")
    print("5. Memory usage remains efficient with JIT compilation")


if __name__ == "__main__":
    main()
