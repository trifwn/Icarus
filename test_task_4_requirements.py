#!/usr/bin/env python3
"""
Test Task 4 Requirements: Enhance main Airfoil class with JAX geometry backend

This test verifies the specific requirements for Task 4:
- Replace internal coordinate storage with JaxAirfoil-based geometry
- Add jax_geometry property for direct access to JAX operations
- Implement seamless conversion between NumPy and JAX arrays in public API
- Ensure JIT compilation works for all geometric operations

Requirements: 2.1, 2.2, 2.3, 8.1
"""

import jax
import jax.numpy as jnp
import numpy as np
from ICARUS.airfoils.naca4 import NACA4
from ICARUS.airfoils.airfoil import Airfoil


def test_jax_geometry_backend_replacement():
    """Test that internal coordinate storage is replaced with JaxAirfoil-based geometry."""
    print("=== Testing JAX Geometry Backend Replacement ===")

    # Create NACA4 airfoil and extract coordinates
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface

    # Create Airfoil from coordinates
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Verify that internal geometry is JaxAirfoil-based
    assert hasattr(airfoil, "_jax_geometry"), (
        "Airfoil should have _jax_geometry attribute"
    )
    assert airfoil._geometry is not None, "JAX geometry should not be None"

    # Verify that geometry is JAX-compatible
    geometry = airfoil._geometry
    assert hasattr(geometry, "tree_flatten"), "Geometry should be JAX pytree compatible"
    assert hasattr(geometry, "tree_unflatten"), (
        "Geometry should be JAX pytree compatible"
    )

    print(f"✓ Internal geometry type: {type(geometry)}")
    print(f"✓ JAX pytree compatible: {hasattr(geometry, 'tree_flatten')}")
    print("✓ JAX geometry backend replacement verified!")


def test_jax_geometry_property_access():
    """Test jax_geometry property for direct access to JAX operations."""
    print("\n=== Testing JAX Geometry Property Access ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test both geometry properties exist
    assert hasattr(airfoil, "geometry"), "Airfoil should have geometry property"
    assert hasattr(airfoil, "jax_geometry"), "Airfoil should have jax_geometry property"

    # Test they return the same object
    geometry1 = airfoil.geometry
    geometry2 = airfoil.jax_geometry
    assert geometry1 is geometry2, "geometry and jax_geometry should return same object"

    # Test direct JAX operations access
    x_test = jnp.array([0.5])

    # Test direct method calls on geometry
    y_upper_direct = geometry1.y_upper(x_test)
    y_lower_direct = geometry1.y_lower(x_test)
    thickness_direct = geometry1.thickness(x_test)
    camber_direct = geometry1.camber_line(x_test)

    assert isinstance(y_upper_direct, jnp.ndarray), (
        "Direct geometry calls should return JAX arrays"
    )
    assert isinstance(y_lower_direct, jnp.ndarray), (
        "Direct geometry calls should return JAX arrays"
    )
    assert isinstance(thickness_direct, jnp.ndarray), (
        "Direct geometry calls should return JAX arrays"
    )
    assert isinstance(camber_direct, jnp.ndarray), (
        "Direct geometry calls should return JAX arrays"
    )

    print(f"✓ geometry property: {type(geometry1)}")
    print(f"✓ jax_geometry property: {type(geometry2)}")
    print(f"✓ Same object: {geometry1 is geometry2}")
    print(f"✓ Direct JAX operations work")
    print("✓ JAX geometry property access verified!")


def test_numpy_jax_array_conversion():
    """Test seamless conversion between NumPy and JAX arrays in public API."""
    print("\n=== Testing NumPy/JAX Array Conversion ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    # Test different input types
    test_cases = [
        ("Python float", 0.5, float),
        ("Python list", [0.25, 0.5, 0.75], jnp.ndarray),
        ("NumPy array", np.array([0.25, 0.5, 0.75]), jnp.ndarray),
        ("JAX array", jnp.array([0.25, 0.5, 0.75]), jnp.ndarray),
    ]

    methods_to_test = [
        ("y_upper", airfoil.y_upper),
        ("y_lower", airfoil.y_lower),
        ("thickness", airfoil.thickness),
        ("camber_line", airfoil.camber_line),
    ]

    for input_name, test_input, expected_output_type in test_cases:
        print(f"\n  Testing {input_name}:")

        for method_name, method in methods_to_test:
            result = method(test_input)

            # For scalar inputs, expect scalar outputs
            if input_name == "Python float":
                assert isinstance(result, float), (
                    f"{method_name} should return float for scalar input"
                )
                print(f"    ✓ {method_name}: {type(result).__name__} (scalar)")
            else:
                # For array inputs, expect JAX array outputs
                assert isinstance(result, jnp.ndarray), (
                    f"{method_name} should return JAX array for array input"
                )
                assert result.shape == (3,), (
                    f"{method_name} should preserve input shape"
                )
                print(
                    f"    ✓ {method_name}: {type(result).__name__} shape {result.shape}"
                )

    print("\n✓ NumPy/JAX array conversion verified!")


def test_jit_compilation_geometric_operations():
    """Test that JIT compilation works for all geometric operations."""
    print("\n=== Testing JIT Compilation of Geometric Operations ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    geometry = airfoil.geometry
    test_x = jnp.array([0.5])

    # Test JIT compilation of all geometric operations
    jit_tests = []

    # Test y_upper JIT
    @jax.jit
    def jit_y_upper(geom, x):
        return geom.y_upper(x)

    result = jit_y_upper(geometry, test_x)
    jit_tests.append(("y_upper", result))

    # Test y_lower JIT
    @jax.jit
    def jit_y_lower(geom, x):
        return geom.y_lower(x)

    result = jit_y_lower(geometry, test_x)
    jit_tests.append(("y_lower", result))

    # Test thickness JIT
    @jax.jit
    def jit_thickness(geom, x):
        return geom.thickness(x)

    result = jit_thickness(geometry, test_x)
    jit_tests.append(("thickness", result))

    # Test camber_line JIT
    @jax.jit
    def jit_camber_line(geom, x):
        return geom.camber_line(x)

    result = jit_camber_line(geometry, test_x)
    jit_tests.append(("camber_line", result))

    # Test properties in JIT context
    @jax.jit
    def jit_properties(geom):
        return geom.max_thickness, geom.max_thickness_location

    max_thick, max_thick_loc = jit_properties(geometry)
    jit_tests.append(("properties", (max_thick, max_thick_loc)))

    # Verify all JIT compilations worked
    for method_name, result in jit_tests:
        if method_name == "properties":
            max_thick, max_thick_loc = result
            assert jnp.isfinite(max_thick), (
                f"JIT {method_name} max_thickness should be finite"
            )
            assert jnp.isfinite(max_thick_loc), (
                f"JIT {method_name} max_thickness_location should be finite"
            )
            print(
                f"    ✓ {method_name} JIT: max_thickness={max_thick:.6f}, max_thickness_location={max_thick_loc:.6f}"
            )
        else:
            assert isinstance(result, jnp.ndarray), (
                f"JIT {method_name} should return JAX array"
            )
            assert jnp.all(jnp.isfinite(result)), (
                f"JIT {method_name} should return finite values"
            )
            print(f"    ✓ {method_name} JIT: {result}")

    print("\n✓ JIT compilation of all geometric operations verified!")


def test_jax_transformations():
    """Test advanced JAX transformations (grad, vmap) work correctly."""
    print("\n=== Testing Advanced JAX Transformations ===")

    # Create airfoil
    naca4_instance = NACA4(M=0.02, P=0.4, XX=0.12, n_points=200)
    upper = naca4_instance.upper_surface
    lower = naca4_instance.lower_surface
    airfoil = Airfoil(upper, lower, name="naca2412_test")

    geometry = airfoil.geometry

    # Test gradient computation
    def thickness_at_point(x):
        return geometry.thickness(x).sum()

    grad_fn = jax.grad(thickness_at_point)
    gradient = grad_fn(jnp.array([0.5]))

    assert isinstance(gradient, jnp.ndarray), "Gradient should be JAX array"
    assert jnp.isfinite(gradient), "Gradient should be finite"
    print(f"    ✓ Gradient computation: {gradient}")

    # Test vmap for batch processing
    def eval_thickness(x):
        return geometry.thickness(x)

    x_batch = jnp.array([[0.25], [0.5], [0.75]])
    thickness_batch = jax.vmap(eval_thickness)(x_batch)

    assert isinstance(thickness_batch, jnp.ndarray), "Batch result should be JAX array"
    assert thickness_batch.shape == (3, 1), "Batch result should have correct shape"
    assert jnp.all(jnp.isfinite(thickness_batch)), "Batch result should be finite"
    print(f"    ✓ Batch processing (vmap): shape {thickness_batch.shape}")

    # Test combined JIT + grad
    @jax.jit
    def jit_grad_thickness(x):
        return jax.grad(lambda x: geometry.thickness(x).sum())(x)

    jit_gradient = jit_grad_thickness(jnp.array([0.5]))
    assert isinstance(jit_gradient, jnp.ndarray), "JIT gradient should be JAX array"
    assert jnp.isfinite(jit_gradient), "JIT gradient should be finite"
    print(f"    ✓ JIT + grad combination: {jit_gradient}")

    print("\n✓ Advanced JAX transformations verified!")


def main():
    """Run all Task 4 requirement tests."""
    print("Task 4: Enhance main Airfoil class with JAX geometry backend")
    print("=" * 70)

    try:
        test_jax_geometry_backend_replacement()
        test_jax_geometry_property_access()
        test_numpy_jax_array_conversion()
        test_jit_compilation_geometric_operations()
        test_jax_transformations()

        print("\n" + "=" * 70)
        print("🎉 ALL TASK 4 REQUIREMENTS VERIFIED!")
        print("✓ JAX geometry backend replacement complete")
        print("✓ jax_geometry property implemented")
        print("✓ Seamless NumPy/JAX array conversion working")
        print("✓ JIT compilation works for all geometric operations")
        print("✓ Advanced JAX transformations (grad, vmap) working")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Task 4 requirement failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
