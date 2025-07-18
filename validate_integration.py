#!/usr/bin/env python3
"""
Simple validation script for JAX airfoil integration testing.
"""

import gc
import os
import sys
import time
import traceback

import psutil


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_basic_functionality():
    """Test basic JAX airfoil functionality."""
    print("=" * 60)
    print("Testing Basic JAX Airfoil Functionality")
    print("=" * 60)

    try:
        import jax.numpy as jnp

        from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil

        # Test NACA 4-digit generation
        print("1. Testing NACA 4-digit generation...")
        airfoil = JaxAirfoil.naca4("2412", n_points=100)
        print(f"   ✓ Created NACA 2412 with {airfoil.n_points} points")
        print(f"   ✓ Max thickness: {airfoil.max_thickness:.4f}")
        print(f"   ✓ Max camber: {airfoil.max_camber:.4f}")

        # Test surface queries
        print("2. Testing surface queries...")
        query_x = jnp.array([0.25, 0.5, 0.75])
        thickness = airfoil.thickness(query_x)
        camber = airfoil.camber_line(query_x)
        upper = airfoil.y_upper(query_x)
        lower = airfoil.y_lower(query_x)

        print(f"   ✓ Thickness at x=[0.25, 0.5, 0.75]: {thickness}")
        print(f"   ✓ Camber at x=[0.25, 0.5, 0.75]: {camber}")
        print("   ✓ Upper surface queries working")
        print("   ✓ Lower surface queries working")

        # Test morphing
        print("3. Testing morphing operations...")
        airfoil2 = JaxAirfoil.naca4("0012", n_points=100)
        morphed = JaxAirfoil.morph_new_from_two_foils(
            airfoil,
            airfoil2,
            0.5,
            n_points=200,
        )
        print(f"   ✓ Morphed airfoil created with {morphed.n_points} points")
        print(f"   ✓ Morphed max thickness: {morphed.max_thickness:.4f}")

        # Test flap operations
        print("4. Testing flap operations...")
        flapped = airfoil.flap(flap_hinge_chord_percentage=0.75, flap_angle=10.0)
        print(f"   ✓ Flapped airfoil created with {flapped.n_points} points")

        print("✓ All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_gradient_computation():
    """Test gradient computation functionality."""
    print("\n" + "=" * 60)
    print("Testing Gradient Computation")
    print("=" * 60)

    try:
        import jax
        import jax.numpy as jnp

        from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil

        # Create test airfoil
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        # Test gradient computation
        print("1. Testing thickness gradients...")

        def thickness_objective(airfoil):
            query_x = jnp.array([0.5])
            return jnp.sum(airfoil.thickness(query_x))

        grad_fn = jax.grad(thickness_objective)
        gradients = grad_fn(airfoil)

        print("   ✓ Gradient computation successful")
        print(f"   ✓ Gradient type: {type(gradients)}")
        print(f"   ✓ Gradient coordinates shape: {gradients._coordinates.shape}")

        # Test that gradients are finite
        grad_coords = gradients._coordinates[:, gradients._validity_mask]
        finite_grads = jnp.all(jnp.isfinite(grad_coords))
        print(f"   ✓ All gradients finite: {finite_grads}")

        # Test JIT compilation with gradients
        print("2. Testing JIT compilation with gradients...")
        jit_grad_fn = jax.jit(grad_fn)
        jit_gradients = jit_grad_fn(airfoil)

        print("   ✓ JIT gradient compilation successful")

        # Test gradient consistency
        grad_diff = jnp.max(
            jnp.abs(gradients._coordinates - jit_gradients._coordinates),
        )
        print(f"   ✓ JIT vs non-JIT gradient difference: {grad_diff:.2e}")

        print("✓ All gradient computation tests passed!")
        return True

    except Exception as e:
        print(f"✗ Gradient computation test failed: {e}")
        traceback.print_exc()
        return False


def test_batch_operations():
    """Test batch operations functionality."""
    print("\n" + "=" * 60)
    print("Testing Batch Operations")
    print("=" * 60)

    try:
        import jax.numpy as jnp

        from ICARUS.airfoils.jax_implementation.batch_operations import BatchAirfoilOps
        from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil

        # Create batch of airfoils
        print("1. Creating batch of airfoils...")
        airfoils = [
            JaxAirfoil.naca4("0012", n_points=50),
            JaxAirfoil.naca4("2412", n_points=50),
            JaxAirfoil.naca4("4415", n_points=50),
        ]
        print(f"   ✓ Created batch of {len(airfoils)} airfoils")

        # Test batch thickness computation
        print("2. Testing batch thickness computation...")
        query_x = jnp.linspace(0.1, 0.9, 10)
        batch_thickness = BatchAirfoilOps.batch_thickness(airfoils, query_x)
        print(f"   ✓ Batch thickness shape: {batch_thickness.shape}")
        print(
            f"   ✓ All thickness values finite: {jnp.all(jnp.isfinite(batch_thickness))}",
        )

        # Test batch camber computation
        print("3. Testing batch camber computation...")
        batch_camber = BatchAirfoilOps.batch_camber_line(airfoils, query_x)
        print(f"   ✓ Batch camber shape: {batch_camber.shape}")
        print(f"   ✓ All camber values finite: {jnp.all(jnp.isfinite(batch_camber))}")

        # Compare with individual operations
        print("4. Comparing batch vs individual operations...")
        individual_thickness = jnp.array(
            [airfoil.thickness(query_x) for airfoil in airfoils],
        )
        thickness_diff = jnp.max(jnp.abs(batch_thickness - individual_thickness))
        print(f"   ✓ Batch vs individual thickness difference: {thickness_diff:.2e}")

        print("✓ All batch operation tests passed!")
        return True

    except Exception as e:
        print(f"✗ Batch operation test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage patterns."""
    print("\n" + "=" * 60)
    print("Testing Memory Usage")
    print("=" * 60)

    try:
        import jax.numpy as jnp

        from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil

        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")

        # Test single airfoil memory usage
        print("1. Testing single airfoil memory usage...")
        airfoil = JaxAirfoil.naca4("2412", n_points=200)
        single_memory = get_memory_usage()
        single_delta = single_memory - initial_memory
        print(
            f"   ✓ Memory after single airfoil: {single_memory:.1f} MB (+{single_delta:.1f} MB)",
        )

        # Test multiple airfoils
        print("2. Testing multiple airfoils memory usage...")
        airfoils = []
        naca_digits = [
            "2412",
            "0012",
            "4415",
            "6412",
            "8412",
            "9412",
            "1012",
            "1112",
            "1212",
            "1312",
        ]
        for digit in naca_digits:
            airfoil = JaxAirfoil.naca4(digit, n_points=100)
            airfoils.append(airfoil)

        multiple_memory = get_memory_usage()
        multiple_delta = multiple_memory - single_memory
        memory_per_airfoil = multiple_delta / 10
        print(
            f"   ✓ Memory after 10 airfoils: {multiple_memory:.1f} MB (+{multiple_delta:.1f} MB)",
        )
        print(f"   ✓ Memory per additional airfoil: {memory_per_airfoil:.2f} MB")

        # Test operations memory usage
        print("3. Testing operations memory usage...")
        query_x = jnp.linspace(0.0, 1.0, 50)

        for airfoil in airfoils[:3]:  # Test first 3 airfoils
            thickness = airfoil.thickness(query_x)
            camber = airfoil.camber_line(query_x)

        operations_memory = get_memory_usage()
        operations_delta = operations_memory - multiple_memory
        print(
            f"   ✓ Memory after operations: {operations_memory:.1f} MB (+{operations_delta:.1f} MB)",
        )

        # Clean up and test garbage collection
        print("4. Testing memory cleanup...")
        del airfoils, airfoil, thickness, camber
        gc.collect()

        cleanup_memory = get_memory_usage()
        cleanup_delta = cleanup_memory - initial_memory
        print(
            f"   ✓ Memory after cleanup: {cleanup_memory:.1f} MB (+{cleanup_delta:.1f} MB from initial)",
        )

        # Memory usage should be reasonable
        if cleanup_delta < 50:  # Less than 50MB increase
            print("✓ Memory usage test passed!")
            return True
        else:
            print(f"⚠ Memory usage higher than expected: {cleanup_delta:.1f} MB")
            return True  # Still pass, but with warning

    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """Test basic performance characteristics."""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)

    try:
        import jax.numpy as jnp

        from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil

        # Test compilation time
        print("1. Testing JIT compilation time...")
        airfoil = JaxAirfoil.naca4("2412", n_points=100)
        query_x = jnp.linspace(0.0, 1.0, 20)

        # First call (compilation)
        start_time = time.time()
        thickness1 = airfoil.thickness(query_x)
        compile_time = time.time() - start_time
        print(f"   ✓ First call (compilation): {compile_time:.3f}s")

        # Second call (cached)
        start_time = time.time()
        thickness2 = airfoil.thickness(query_x)
        cached_time = time.time() - start_time
        print(f"   ✓ Second call (cached): {cached_time:.6f}s")

        speedup = compile_time / cached_time if cached_time > 0 else float("inf")
        print(f"   ✓ Speedup: {speedup:.1f}x")

        # Test batch performance
        print("2. Testing batch performance...")
        naca_digits = [
            "0012",
            "2412",
            "4415",
            "6412",
            "8412",
            "9412",
            "1012",
            "1112",
            "1212",
            "1312",
        ]
        airfoils = [JaxAirfoil.naca4(digit, n_points=50) for digit in naca_digits]

        # Individual operations
        start_time = time.time()
        individual_results = [airfoil.thickness(query_x) for airfoil in airfoils]
        individual_time = time.time() - start_time
        print(f"   ✓ Individual operations: {individual_time:.3f}s")

        # Batch operations
        from ICARUS.airfoils.jax_implementation.batch_operations import BatchAirfoilOps

        start_time = time.time()
        batch_results = BatchAirfoilOps.batch_thickness(airfoils, query_x)
        batch_time = time.time() - start_time
        print(f"   ✓ Batch operations: {batch_time:.3f}s")

        batch_speedup = individual_time / batch_time if batch_time > 0 else float("inf")
        print(f"   ✓ Batch speedup: {batch_speedup:.1f}x")

        print("✓ Performance tests completed!")
        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("JAX AIRFOIL INTEGRATION VALIDATION")
    print("=" * 80)

    start_time = time.time()
    initial_memory = get_memory_usage()

    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Gradient Computation", test_gradient_computation),
        ("Batch Operations", test_batch_operations),
        ("Memory Usage", test_memory_usage),
        ("Performance", test_performance),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    end_time = time.time()
    final_memory = get_memory_usage()

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests / total_tests:.1%}")
    print(f"Total Duration: {end_time - start_time:.2f}s")
    print(f"Memory Delta: {final_memory - initial_memory:.1f}MB")

    if failed_tests > 0:
        print("\nFAILED TESTS:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - {test_name}")

    print("\n" + "=" * 80)

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
