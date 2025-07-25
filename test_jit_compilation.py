#!/usr/bin/env python3
"""
Test JIT compilation of all geometric operations.

This test verifies that all geometric operations can be JIT compiled successfully.
"""

import jax
import jax.numpy as jnp
import numpy as np
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.airfoils.airfoil import Airfoil


def test_jit_compilation_all_methods():
    """Test that all geometric operations can be JIT compiled."""
    print("=== Testing JIT Compilation of All Methods ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    geometry = airfoil.geometry

    # Test JIT compilation of all methods
    test_x = jnp.array([0.5])

    # Test y_upper JIT compilation
    @jax.jit
    def test_y_upper_jit(geom, x):
        return geom.y_upper(x)

    result = test_y_upper_jit(geometry, test_x)
    print(f"✓ y_upper JIT compilation works: {result}")

    # Test y_lower JIT compilation
    @jax.jit
    def test_y_lower_jit(geom, x):
        return geom.y_lower(x)

    result = test_y_lower_jit(geometry, test_x)
    print(f"✓ y_lower JIT compilation works: {result}")

    # Test thickness JIT compilation
    @jax.jit
    def test_thickness_jit(geom, x):
        return geom.thickness(x)

    result = test_thickness_jit(geometry, test_x)
    print(f"✓ thickness JIT compilation works: {result}")

    # Test camber_line JIT compilation
    @jax.jit
    def test_camber_line_jit(geom, x):
        return geom.camber_line(x)

    result = test_camber_line_jit(geometry, test_x)
    print(f"✓ camber_line JIT compilation works: {result}")

    # Test property access in JIT context
    @jax.jit
    def test_properties_jit(geom):
        return geom.max_thickness, geom.max_thickness_location

    max_thick, max_thick_loc = test_properties_jit(geometry)
    print(
        f"✓ Properties JIT compilation works: max_thickness={max_thick}, max_thickness_location={max_thick_loc}"
    )

    print("✓ All JIT compilation tests passed!")


def test_numpy_jax_conversion():
    """Test seamless conversion between NumPy and JAX arrays."""
    print("\n=== Testing NumPy/JAX Array Conversion ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test different input types and verify output types
    test_inputs = [
        ("Python float", 0.5),
        ("Python list", [0.25, 0.5, 0.75]),
        ("NumPy array", np.array([0.25, 0.5, 0.75])),
        ("JAX array", jnp.array([0.25, 0.5, 0.75])),
    ]

    for input_name, test_input in test_inputs:
        print(f"\nTesting {input_name}:")

        # Test y_upper
        result = airfoil.y_upper(test_input)
        print(
            f"  y_upper result type: {type(result)}, shape: {getattr(result, 'shape', 'scalar')}"
        )

        # Test y_lower
        result = airfoil.y_lower(test_input)
        print(
            f"  y_lower result type: {type(result)}, shape: {getattr(result, 'shape', 'scalar')}"
        )

        # Test thickness
        result = airfoil.thickness(test_input)
        print(
            f"  thickness result type: {type(result)}, shape: {getattr(result, 'shape', 'scalar')}"
        )

        # Test camber_line
        result = airfoil.camber_line(test_input)
        print(
            f"  camber_line result type: {type(result)}, shape: {getattr(result, 'shape', 'scalar')}"
        )

    print("\n✓ All NumPy/JAX conversion tests passed!")


def test_direct_jax_geometry_access():
    """Test direct access to JAX geometry for advanced operations."""
    print("\n=== Testing Direct JAX Geometry Access ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test both geometry properties
    geometry1 = airfoil.geometry
    geometry2 = airfoil.jax_geometry

    # Verify they are the same object
    assert geometry1 is geometry2, "geometry and jax_geometry should be the same object"
    print("✓ geometry and jax_geometry properties return same object")

    # Test that it's JAX-compatible
    assert hasattr(geometry1, "tree_flatten"), (
        "Geometry should be JAX pytree compatible"
    )
    assert hasattr(geometry1, "tree_unflatten"), (
        "Geometry should be JAX pytree compatible"
    )
    print("✓ Geometry is JAX pytree compatible")

    # Test advanced JAX operations
    @jax.jit
    def compute_multiple_properties(geom):
        x = jnp.array([0.25, 0.5, 0.75])
        return {
            "y_upper": geom.y_upper(x),
            "y_lower": geom.y_lower(x),
            "thickness": geom.thickness(x),
            "camber": geom.camber_line(x),
            "max_thickness": geom.max_thickness,
        }

    results = compute_multiple_properties(geometry1)
    print(f"✓ Advanced JIT operations work: {list(results.keys())}")

    # Test gradient computation
    def thickness_sum(x):
        return geometry1.thickness(x).sum()

    grad_fn = jax.grad(thickness_sum)
    gradient = grad_fn(jnp.array([0.5]))
    print(f"✓ Gradient computation works: {gradient}")

    # Test vmap
    def eval_at_point(x):
        return {
            "thickness": geometry1.thickness(x),
            "camber": geometry1.camber_line(x),
        }

    x_batch = jnp.array([[0.25], [0.5], [0.75]])
    batch_results = jax.vmap(eval_at_point)(x_batch)
    print(
        f"✓ Batch processing (vmap) works: thickness shape {batch_results['thickness'].shape}"
    )

    print("✓ All direct JAX geometry access tests passed!")


if __name__ == "__main__":
    test_jit_compilation_all_methods()
    test_numpy_jax_conversion()
    test_direct_jax_geometry_access()
    print("\n🎉 All JIT compilation and conversion tests passed!")
