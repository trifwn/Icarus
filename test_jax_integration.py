#!/usr/bin/env python3
"""
Integration test for JAX airfoil backend delegation.

This test verifies that the main Airfoil class correctly delegates to the JAX backend
while preserving API compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.airfoils.airfoil import Airfoil


def test_jax_backend_delegation():
    """Test that Airfoil methods delegate to JAX backend correctly."""
    print("=== Testing JAX Backend Delegation ===")

    # Create NACA4 airfoil and extract coordinates
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface

    # Create Airfoil from coordinates
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test that geometry property exists and is JAX-compatible
    geometry = airfoil.geometry
    assert hasattr(airfoil, "geometry"), "Airfoil should have geometry property"
    assert hasattr(geometry, "tree_flatten"), "Geometry should be JAX pytree compatible"

    print(f"✓ Geometry property exists: {type(geometry)}")

    # Test basic method delegation
    x_test = 0.5
    y_upper = airfoil.y_upper(x_test)
    y_lower = airfoil.y_lower(x_test)
    thickness = airfoil.thickness(x_test)
    camber = airfoil.camber_line(x_test)

    print(f"✓ y_upper(0.5): {y_upper}")
    print(f"✓ y_lower(0.5): {y_lower}")
    print(f"✓ thickness(0.5): {thickness}")
    print(f"✓ camber_line(0.5): {camber}")

    # Test that results are reasonable
    assert isinstance(y_upper, (float, jnp.ndarray, np.ndarray)), (
        "y_upper should return numeric type"
    )
    assert isinstance(y_lower, (float, jnp.ndarray, np.ndarray)), (
        "y_lower should return numeric type"
    )
    assert isinstance(thickness, (float, jnp.ndarray, np.ndarray)), (
        "thickness should return numeric type"
    )
    assert isinstance(camber, (float, jnp.ndarray, np.ndarray)), (
        "camber should return numeric type"
    )

    # Test array inputs
    x_array = np.array([0.25, 0.5, 0.75])
    y_upper_array = airfoil.y_upper(x_array)
    assert len(y_upper_array) == 3, "Array input should return array output"

    print(f"✓ Array input works: {y_upper_array}")

    # Test properties
    max_thickness = airfoil.max_thickness
    max_thickness_location = airfoil.max_thickness_location

    assert isinstance(max_thickness, float), "max_thickness should be float"
    assert isinstance(max_thickness_location, float), (
        "max_thickness_location should be float"
    )
    assert 0 < max_thickness < 1, "max_thickness should be reasonable"
    assert 0 < max_thickness_location < 1, "max_thickness_location should be reasonable"

    print(f"✓ max_thickness: {max_thickness}")
    print(f"✓ max_thickness_location: {max_thickness_location}")

    print("✓ All delegation tests passed!")


def test_jax_transformations():
    """Test that JAX transformations work on the geometry backend."""
    print("\n=== Testing JAX Transformations ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    geometry = airfoil.geometry

    # Test JIT compilation
    @jax.jit
    def compute_thickness_jit(geom, x):
        return geom.thickness(x)

    x_test = jnp.array([0.5])
    thickness_jit = compute_thickness_jit(geometry, x_test)
    print(f"✓ JIT compilation works: {thickness_jit}")

    # Test gradient computation
    def thickness_at_point(x):
        return geometry.thickness(x).sum()

    grad_fn = jax.grad(thickness_at_point)
    gradient = grad_fn(jnp.array([0.5]))
    print(f"✓ Gradient computation works: {gradient}")

    # Test vmap for batch processing
    def eval_thickness(x):
        return geometry.thickness(x)

    x_batch = jnp.array([[0.25], [0.5], [0.75]])
    thickness_batch = jax.vmap(eval_thickness)(x_batch)
    print(f"✓ Batch processing (vmap) works: {thickness_batch}")

    print("✓ All JAX transformation tests passed!")


def test_api_compatibility():
    """Test that API remains compatible with different input types."""
    print("\n=== Testing API Compatibility ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test different input types
    test_inputs = [
        0.5,  # scalar
        [0.25, 0.5, 0.75],  # list
        np.array([0.25, 0.5, 0.75]),  # numpy array
        jnp.array([0.25, 0.5, 0.75]),  # JAX array
    ]

    for i, x_input in enumerate(test_inputs):
        try:
            y_upper = airfoil.y_upper(x_input)
            y_lower = airfoil.y_lower(x_input)
            thickness = airfoil.thickness(x_input)

            print(f"✓ Input type {i + 1} ({type(x_input).__name__}) works")

        except Exception as e:
            print(f"✗ Input type {i + 1} ({type(x_input).__name__}) failed: {e}")
            raise

    print("✓ All API compatibility tests passed!")


if __name__ == "__main__":
    test_jax_backend_delegation()
    test_jax_transformations()
    test_api_compatibility()
    print("\n🎉 All integration tests passed!")
