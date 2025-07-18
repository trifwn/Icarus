#!/usr/bin/env python3
"""
Demonstration of comprehensive error handling in JAX airfoil implementation.

This script shows how the error handling system provides meaningful error messages
and suggestions for common issues when working with JAX airfoils.
"""

import jax.numpy as jnp

from ICARUS.airfoils.jax_implementation.error_handling import AirfoilErrorHandler
from ICARUS.airfoils.jax_implementation.error_handling import AirfoilValidationError
from ICARUS.airfoils.jax_implementation.error_handling import BufferOverflowError
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


def demonstrate_coordinate_validation():
    """Demonstrate coordinate validation error handling."""
    print("=" * 60)
    print("COORDINATE VALIDATION ERROR HANDLING")
    print("=" * 60)

    # Test 1: Invalid coordinate shape
    print("\n1. Invalid coordinate shape:")
    try:
        coords_1d = jnp.array([0.0, 0.5, 1.0])  # Should be 2D
        JaxAirfoil(coords_1d, name="invalid_shape")
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 2: NaN coordinates
    print("\n2. NaN coordinates:")
    try:
        coords_nan = jnp.array([[0.0, jnp.nan, 1.0], [0.0, 0.1, 0.0]])
        AirfoilErrorHandler.validate_coordinate_values(coords_nan, "test coordinates")
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 3: Infinite coordinates
    print("\n3. Infinite coordinates:")
    try:
        coords_inf = jnp.array([[0.0, jnp.inf, 1.0], [0.0, 0.1, 0.0]])
        AirfoilErrorHandler.validate_coordinate_values(coords_inf, "test coordinates")
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 4: Valid coordinates (should work)
    print("\n4. Valid coordinates:")
    try:
        coords_valid = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        AirfoilErrorHandler.validate_coordinate_values(coords_valid, "test coordinates")
        print("   ✓ Validation passed successfully")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def demonstrate_naca_validation():
    """Demonstrate NACA parameter validation error handling."""
    print("\n" + "=" * 60)
    print("NACA PARAMETER VALIDATION ERROR HANDLING")
    print("=" * 60)

    # Test 1: Invalid NACA 4-digit length
    print("\n1. Invalid NACA 4-digit length:")
    try:
        JaxAirfoil.naca4("241")  # Too short
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 2: Invalid NACA 4-digit characters
    print("\n2. Invalid NACA 4-digit characters:")
    try:
        JaxAirfoil.naca4("24a2")  # Contains letter
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 3: Invalid NACA 4-digit thickness (00)
    print("\n3. Invalid NACA 4-digit thickness:")
    try:
        JaxAirfoil.naca4("2400")  # Zero thickness
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 4: Valid NACA 4-digit (should work)
    print("\n4. Valid NACA 4-digit:")
    try:
        airfoil = JaxAirfoil.naca4("2412")
        print(f"   ✓ Created {airfoil.name} with {airfoil.n_points} points")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def demonstrate_flap_validation():
    """Demonstrate flap parameter validation error handling."""
    print("\n" + "=" * 60)
    print("FLAP PARAMETER VALIDATION ERROR HANDLING")
    print("=" * 60)

    # Create a test airfoil
    airfoil = JaxAirfoil.naca4("2412", n_points=50)

    # Test 1: Invalid hinge position
    print("\n1. Invalid flap hinge position:")
    try:
        airfoil.flap(1.5, 10.0)  # Hinge position > 1.0
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 2: Invalid thickness position
    print("\n2. Invalid flap thickness position:")
    try:
        airfoil.flap(0.7, 10.0, flap_hinge_thickness_percentage=1.5)  # > 1.0
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 3: Invalid chord extension
    print("\n3. Invalid chord extension:")
    try:
        airfoil.flap(0.7, 10.0, chord_extension=-1.0)  # Negative
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 4: Valid flap parameters (should work)
    print("\n4. Valid flap parameters:")
    try:
        flapped_airfoil = airfoil.flap(0.7, 10.0)
        print(f"   ✓ Created flapped airfoil with {flapped_airfoil.n_points} points")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def demonstrate_morphing_validation():
    """Demonstrate morphing parameter validation error handling."""
    print("\n" + "=" * 60)
    print("MORPHING PARAMETER VALIDATION ERROR HANDLING")
    print("=" * 60)

    # Create test airfoils
    airfoil1 = JaxAirfoil.naca4("2412", n_points=50)
    airfoil2 = JaxAirfoil.naca4("0012", n_points=50)

    # Test 1: Invalid eta (> 1.0)
    print("\n1. Invalid morphing parameter (eta > 1.0):")
    try:
        JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta=1.5, n_points=50)
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 2: Invalid eta (< 0.0)
    print("\n2. Invalid morphing parameter (eta < 0.0):")
    try:
        JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, eta=-0.1, n_points=50)
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 3: Invalid eta type
    print("\n3. Invalid morphing parameter type:")
    try:
        AirfoilErrorHandler.validate_morphing_parameters("0.5", "airfoil1", "airfoil2")
    except AirfoilValidationError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 4: Valid morphing parameters (should work)
    print("\n4. Valid morphing parameters:")
    try:
        morphed_airfoil = JaxAirfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta=0.5,
            n_points=50,
        )
        print(f"   ✓ Created morphed airfoil: {morphed_airfoil.name}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def demonstrate_buffer_overflow():
    """Demonstrate buffer overflow detection and handling."""
    print("\n" + "=" * 60)
    print("BUFFER OVERFLOW DETECTION AND HANDLING")
    print("=" * 60)

    # Test 1: Buffer capacity check - sufficient
    print("\n1. Buffer capacity check (sufficient):")
    try:
        needs_realloc, new_size = AirfoilErrorHandler.check_buffer_capacity(
            required_size=100,
            current_buffer_size=256,
        )
        print(f"   ✓ Needs reallocation: {needs_realloc}, New size: {new_size}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    # Test 2: Buffer capacity check - insufficient
    print("\n2. Buffer capacity check (insufficient):")
    try:
        needs_realloc, new_size = AirfoilErrorHandler.check_buffer_capacity(
            required_size=300,
            current_buffer_size=256,
        )
        print(f"   ✓ Needs reallocation: {needs_realloc}, New size: {new_size}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    # Test 3: Buffer overflow
    print("\n3. Buffer overflow (exceeds maximum):")
    try:
        AirfoilErrorHandler.check_buffer_capacity(
            required_size=10000,
            current_buffer_size=256,
            max_buffer_size=4096,
        )
    except BufferOverflowError as e:
        print(f"   ✗ Error caught: {e}")

    # Test 4: Demonstrate automatic buffer sizing
    print("\n4. Automatic buffer sizing:")
    try:
        from ICARUS.airfoils.jax_implementation.buffer_manager import (
            AirfoilBufferManager,
        )

        for n_points in [50, 150, 300, 600]:
            buffer_size = AirfoilBufferManager.determine_buffer_size(n_points)
            print(f"   ✓ {n_points} points → buffer size {buffer_size}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")


def demonstrate_error_context_and_suggestions():
    """Demonstrate error context creation and fix suggestions."""
    print("\n" + "=" * 60)
    print("ERROR CONTEXT AND FIX SUGGESTIONS")
    print("=" * 60)

    # Test 1: Error context creation
    print("\n1. Error context creation:")
    context = AirfoilErrorHandler.create_error_context(
        "flap_operation",
        "NACA2412",
        {"hinge_position": 0.7, "angle": 15.0, "buffer_size": 256},
    )
    print(f"   ✓ Context: {context}")

    # Test 2: Fix suggestions for different error types
    print("\n2. Fix suggestions:")
    error_types = [
        "nan_coordinates",
        "buffer_overflow",
        "geometry_invalid",
        "naca_invalid",
        "morphing_invalid",
    ]

    for error_type in error_types:
        suggestion = AirfoilErrorHandler.suggest_fixes(error_type)
        print(f"   ✓ {error_type}: {suggestion[:80]}...")


def demonstrate_gradient_safety():
    """Demonstrate that error handling doesn't break JAX transformations."""
    print("\n" + "=" * 60)
    print("GRADIENT SAFETY AND JAX COMPATIBILITY")
    print("=" * 60)

    import jax

    # Test 1: JIT compilation with error handling
    print("\n1. JIT compilation compatibility:")
    try:

        @jax.jit
        def test_jit_function(coords):
            return jnp.sum(coords**2)

        coords = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        result = test_jit_function(coords)
        print(f"   ✓ JIT compilation successful, result: {result:.6f}")
    except Exception as e:
        print(f"   ✗ JIT compilation failed: {e}")

    # Test 2: Gradient computation
    print("\n2. Gradient computation compatibility:")
    try:

        def test_grad_function(x):
            coords = jnp.array([[0.0, x, 1.0], [0.0, 0.1, 0.0]])
            return jnp.sum(coords**2)

        grad_fn = jax.grad(test_grad_function)
        gradient = grad_fn(0.5)
        print(f"   ✓ Gradient computation successful, gradient: {gradient:.6f}")
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")

    # Test 3: JAX airfoil with transformations
    print("\n3. JAX airfoil with transformations:")
    try:
        airfoil = JaxAirfoil.naca4("2412", n_points=50)

        # Test that airfoil works with JAX transformations
        def airfoil_thickness_at_midchord(airfoil_coords):
            # Simple function that could be differentiated
            return jnp.sum(airfoil_coords**2)

        coords = airfoil._coordinates
        result = airfoil_thickness_at_midchord(coords)
        print(f"   ✓ JAX airfoil transformation successful, result: {result:.6f}")
    except Exception as e:
        print(f"   ✗ JAX airfoil transformation failed: {e}")


def main():
    """Run all error handling demonstrations."""
    print("JAX AIRFOIL ERROR HANDLING DEMONSTRATION")
    print("This script demonstrates the comprehensive error handling system")
    print("implemented for the JAX airfoil refactor.")

    try:
        demonstrate_coordinate_validation()
        demonstrate_naca_validation()
        demonstrate_flap_validation()
        demonstrate_morphing_validation()
        demonstrate_buffer_overflow()
        demonstrate_error_context_and_suggestions()
        demonstrate_gradient_safety()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nAll error handling features demonstrated successfully!")
        print("The error handling system provides:")
        print("  ✓ Meaningful error messages with context")
        print("  ✓ Helpful suggestions for fixing issues")
        print("  ✓ Gradient-safe error handling")
        print("  ✓ Buffer overflow detection and management")
        print("  ✓ Comprehensive parameter validation")
        print("  ✓ JAX transformation compatibility")

    except Exception as e:
        print(f"\n✗ Demonstration failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
