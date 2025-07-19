"""
Unit tests for the JaxAirfoil class.

This module contains tests for the JaxAirfoil class, verifying that it correctly
handles variable-sized input data, supports JAX transformations, and provides
proper property accessors for airfoil surfaces.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


def test_jax_airfoil_initialization():
    """Test basic initialization of JaxAirfoil."""
    # Create an empty airfoil
    airfoil = JaxAirfoil()
    assert airfoil.n_points == 0
    assert airfoil.buffer_size >= 32  # Minimum buffer size

    # Create from coordinates
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],  # y-coordinates
        ],
    )
    airfoil = JaxAirfoil(coords, name="TestAirfoil")

    assert airfoil.name == "TestAirfoil"
    assert airfoil.n_points > 0
    assert airfoil.buffer_size >= airfoil.n_points


def test_from_upper_lower():
    """Test creating JaxAirfoil from separate upper and lower surfaces."""
    # Create upper and lower surfaces
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, 0.05, 0.08, 0.05, 0.0],  # y-coordinates
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, -0.05, -0.08, -0.05, 0.0],  # y-coordinates
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="FromSurfaces")

    assert airfoil.name == "FromSurfaces"
    assert airfoil.n_points > 0

    # Check that we can retrieve the surfaces
    x_upper, y_upper = airfoil.upper_surface_points
    x_lower, y_lower = airfoil.lower_surface_points

    assert len(x_upper) > 0
    assert len(x_lower) > 0


def test_jax_pytree_compatibility():
    """Test that JaxAirfoil works with JAX pytree operations."""
    # Create a simple airfoil
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords)

    # Test tree_flatten and tree_unflatten
    children, aux_data = jax.tree_util.tree_flatten(airfoil)
    reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

    assert reconstructed.n_points == airfoil.n_points
    assert reconstructed.name == airfoil.name


def test_surface_accessors():
    """Test property accessors for upper/lower surfaces."""
    # Create a simple airfoil with known coordinates
    upper = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0],  # x-coordinates (TE to LE)
            [0.0, 0.05, 0.08, 0.05, 0.0],  # y-coordinates
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates (LE to TE)
            [0.0, -0.05, -0.08, -0.05, 0.0],  # y-coordinates
        ],
    )

    # Create JaxAirfoil
    airfoil = JaxAirfoil.from_upper_lower(upper, lower)

    # Get surfaces
    x_upper, y_upper = airfoil.upper_surface_points
    x_lower, y_lower = airfoil.lower_surface_points

    # Check that we get the expected number of points
    assert len(x_upper) > 0
    assert len(x_lower) > 0

    # Check that the surfaces are correctly ordered
    assert jnp.all(jnp.diff(x_upper) <= 0)  # Upper surface x decreases (LE to TE)
    assert jnp.all(jnp.diff(x_lower) >= 0)  # Lower surface x increases (LE to TE)

    # Test the get_*_surface_points methods
    x_upper2, y_upper2 = airfoil.get_upper_surface_points()
    x_lower2, y_lower2 = airfoil.get_lower_surface_points()

    assert jnp.allclose(x_upper, x_upper2)
    assert jnp.allclose(y_upper, y_upper2)
    assert jnp.allclose(x_lower, x_lower2)
    assert jnp.allclose(y_lower, y_lower2)


def test_get_coordinates():
    """Test getting all coordinates in selig format."""
    # Create a simple airfoil
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords)

    # Get all coordinates
    x_coords, y_coords = airfoil.get_coordinates()

    # Check that we get the expected number of points
    assert len(x_coords) == airfoil.n_points
    assert len(y_coords) == airfoil.n_points

    # Check that the coordinates form a closed loop
    assert jnp.isclose(x_coords[0], x_coords[-1])
    assert jnp.isclose(y_coords[0], y_coords[-1])


def test_jit_compatibility():
    """Test that JaxAirfoil works with jax.jit."""

    # Create a simple function that uses JaxAirfoil
    @jax.jit
    def get_upper_y(airfoil):
        x_upper, y_upper = airfoil.upper_surface_points
        return y_upper

    # Create a simple airfoil
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords)

    # Call the jitted function
    y_upper = get_upper_y(airfoil)

    # Check that we get a result
    assert len(y_upper) > 0


def test_variable_sized_input_handling():
    """Test that JaxAirfoil handles variable-sized input data correctly."""
    # Test with different sized inputs
    small_coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, 0.02, 0.03, 0.02, 0.0, -0.02, -0.03, -0.02, 0.0],  # y-coordinates
        ],
    )

    large_coords = jnp.array(
        [
            [
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],  # x-coordinates
            [
                0.0,
                0.02,
                0.04,
                0.06,
                0.07,
                0.08,
                0.07,
                0.06,
                0.04,
                0.02,
                0.0,
                -0.02,
                -0.04,
                -0.06,
                -0.07,
                -0.08,
                -0.07,
                -0.06,
                -0.04,
                -0.02,
                0.0,
            ],  # y-coordinates
        ],
    )

    # Create airfoils with different sizes
    small_airfoil = JaxAirfoil(small_coords, name="Small")
    large_airfoil = JaxAirfoil(large_coords, name="Large")

    # Both should work and have appropriate buffer sizes
    # Note: The actual number of points may be different due to preprocessing
    # (closure, duplicate removal, etc.)
    assert small_airfoil.n_points >= 5  # May have added closure points
    assert large_airfoil.n_points >= 21  # May have added closure points
    assert small_airfoil.buffer_size >= small_airfoil.n_points
    assert large_airfoil.buffer_size >= large_airfoil.n_points

    # Buffer sizes should be different due to different input sizes
    assert large_airfoil.buffer_size >= small_airfoil.buffer_size


def test_jax_array_internal_storage():
    """Test that internal data structures use JAX arrays (Requirement 1.1)."""
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords)

    # Check that internal arrays are JAX arrays
    assert isinstance(airfoil._coordinates, jax.Array)
    assert isinstance(airfoil._validity_mask, jax.Array)

    # Check that property accessors return JAX arrays
    x_upper, y_upper = airfoil.upper_surface_points
    x_lower, y_lower = airfoil.lower_surface_points

    assert isinstance(x_upper, jax.Array)
    assert isinstance(y_upper, jax.Array)
    assert isinstance(x_lower, jax.Array)
    assert isinstance(y_lower, jax.Array)


def test_pytree_registration_completeness():
    """Test comprehensive pytree functionality for JAX transformations."""
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords, name="TestPytree")

    # Test that the airfoil can be used in JAX tree operations
    def tree_sum(airfoil):
        return jnp.sum(airfoil._coordinates)

    # Test tree_map
    mapped_result = jax.tree_util.tree_map(lambda x: x * 2, airfoil)
    assert isinstance(mapped_result, JaxAirfoil)
    assert mapped_result.name == airfoil.name  # Metadata should be preserved

    # Test that coordinates were scaled (only check valid coordinates)
    original_valid = airfoil._coordinates[:, airfoil._validity_mask]
    mapped_valid = mapped_result._coordinates[:, mapped_result._validity_mask]
    original_sum = jnp.sum(original_valid)
    mapped_sum = jnp.sum(mapped_valid)
    assert jnp.isclose(mapped_sum, 2 * original_sum)


def test_constructor_input_formats():
    """Test that constructor accepts various input formats (Requirement 3.2)."""
    # Test with JAX array
    jax_coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil_jax = JaxAirfoil(jax_coords)
    assert airfoil_jax.n_points > 0

    # Test with NumPy array
    numpy_coords = np.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil_numpy = JaxAirfoil(numpy_coords)
    assert airfoil_numpy.n_points > 0

    # Test with None (empty airfoil)
    empty_airfoil = JaxAirfoil(None)
    assert empty_airfoil.n_points == 0

    # Test with metadata
    airfoil_with_meta = JaxAirfoil(
        jax_coords,
        name="WithMetadata",
        metadata={"source": "test", "version": 1.0},
    )
    assert airfoil_with_meta.name == "WithMetadata"
    assert airfoil_with_meta._metadata["source"] == "test"


def test_api_compatibility_methods():
    """Test that public methods have expected signatures (Requirement 3.1)."""
    coords = jnp.array(
        [
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )
    airfoil = JaxAirfoil(coords)

    # Test that expected properties exist and work
    assert hasattr(airfoil, "name")
    assert hasattr(airfoil, "n_points")
    assert hasattr(airfoil, "buffer_size")
    assert hasattr(airfoil, "upper_surface_points")
    assert hasattr(airfoil, "lower_surface_points")

    # Test that methods return expected types
    assert isinstance(airfoil.name, str)
    assert isinstance(airfoil.n_points, int)
    assert isinstance(airfoil.buffer_size, int)

    # Test surface point methods
    x_upper, y_upper = airfoil.upper_surface_points
    x_lower, y_lower = airfoil.lower_surface_points

    assert len(x_upper) > 0
    assert len(y_upper) > 0
    assert len(x_lower) > 0
    assert len(y_lower) > 0

    # Test get_coordinates method
    x_all, y_all = airfoil.get_coordinates()
    assert len(x_all) == airfoil.n_points
    assert len(y_all) == airfoil.n_points


def test_geometric_operations():
    """Test geometric operation methods (Requirements 2.1, 2.2, 7.3, 8.2)."""
    # Create a simple symmetric airfoil
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, 0.05, 0.08, 0.05, 0.0],  # y-coordinates
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, -0.05, -0.08, -0.05, 0.0],  # y-coordinates
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="TestGeometry")

    # Test thickness computation
    query_x = jnp.array([0.5])
    thickness = airfoil.thickness(query_x)
    assert len(thickness) == 1
    assert thickness[0] > 0  # Thickness should be positive

    # Test camber line computation
    camber = airfoil.camber_line(query_x)
    assert len(camber) == 1
    # For symmetric airfoil, camber should be close to zero
    assert jnp.abs(camber[0]) < 0.01

    # Test upper surface query
    y_upper = airfoil.y_upper(query_x)
    assert len(y_upper) == 1
    assert y_upper[0] > 0  # Upper surface should be above x-axis at midpoint

    # Test lower surface query
    y_lower = airfoil.y_lower(query_x)
    assert len(y_lower) == 1
    assert y_lower[0] < 0  # Lower surface should be below x-axis at midpoint

    # Test scalar input
    thickness_scalar = airfoil.thickness(0.5)
    assert jnp.isclose(thickness_scalar[0], thickness[0])

    # Test numpy array input
    thickness_numpy = airfoil.thickness(np.array([0.5]))
    assert jnp.isclose(thickness_numpy[0], thickness[0])


def test_geometric_properties():
    """Test geometric property calculations."""
    # Create a simple symmetric airfoil
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, 0.05, 0.08, 0.05, 0.0],  # y-coordinates
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],  # x-coordinates
            [0.0, -0.05, -0.08, -0.05, 0.0],  # y-coordinates
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="TestProperties")

    # Test maximum thickness
    max_thickness = airfoil.max_thickness
    assert isinstance(max_thickness, float)
    assert max_thickness > 0

    # Test maximum thickness location
    max_thickness_location = airfoil.max_thickness_location
    assert isinstance(max_thickness_location, float)
    assert 0 <= max_thickness_location <= 1

    # Test maximum camber
    max_camber = airfoil.max_camber
    assert isinstance(max_camber, float)
    # For symmetric airfoil, max camber should be small
    assert abs(max_camber) < 0.1

    # Test maximum camber location
    max_camber_location = airfoil.max_camber_location
    assert isinstance(max_camber_location, float)
    assert 0 <= max_camber_location <= 1

    # Test chord length
    chord_length = airfoil.chord_length
    assert isinstance(chord_length, float)
    assert jnp.isclose(
        chord_length,
        1.0,
        atol=1e-6,
    )  # Should be 1.0 for our test airfoil


def test_geometric_operations_jit_compatibility():
    """Test that geometric operations work with JAX JIT compilation."""
    # Create a simple airfoil
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0],
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower)

    # Test JIT compilation of thickness computation
    @jax.jit
    def compute_thickness_at_midpoint(airfoil):
        return airfoil.thickness(jnp.array([0.5]))[0]

    thickness = compute_thickness_at_midpoint(airfoil)
    assert jnp.isfinite(thickness)
    assert thickness > 0

    # Test JIT compilation of surface queries
    @jax.jit
    def compute_surface_values(airfoil):
        query_x = jnp.array([0.5])
        y_upper = airfoil.y_upper(query_x)[0]
        y_lower = airfoil.y_lower(query_x)[0]
        return y_upper, y_lower

    y_upper, y_lower = compute_surface_values(airfoil)
    assert jnp.isfinite(y_upper)
    assert jnp.isfinite(y_lower)
    assert y_upper > y_lower  # Upper should be above lower


def test_geometric_operations_gradients():
    """Test that geometric operations support automatic differentiation (Requirements 2.1, 2.2)."""
    # Create a simple airfoil
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.05, 0.08, 0.05, 0.0],
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.05, -0.08, -0.05, 0.0],
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower)

    # Test gradient of thickness with respect to airfoil coordinates
    def thickness_at_midpoint(airfoil):
        return airfoil.thickness(jnp.array([0.5]))[0]

    # Compute gradient using JAX
    grad_fn = jax.grad(thickness_at_midpoint)
    gradients = grad_fn(airfoil)

    # Gradients should exist and be finite
    assert isinstance(gradients, JaxAirfoil)
    # The gradient airfoil should have the same structure
    assert gradients.n_points == airfoil.n_points
    assert gradients.buffer_size == airfoil.buffer_size


def test_asymmetric_airfoil_operations():
    """Test geometric operations on an asymmetric (cambered) airfoil."""
    # Create an asymmetric airfoil
    upper = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.08, 0.10, 0.06, 0.0],  # More cambered
        ],
    )
    lower = jnp.array(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.02, -0.04, -0.02, 0.0],  # Less cambered
        ],
    )

    airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="AsymmetricTest")

    # Test camber line - should be positive for this airfoil
    camber = airfoil.camber_line(jnp.array([0.5]))
    assert camber[0] > 0  # Should have positive camber

    # Test maximum camber
    max_camber = airfoil.max_camber
    assert max_camber > 0  # Should have positive maximum camber

    # Test that thickness is still positive
    thickness = airfoil.thickness(jnp.array([0.5]))
    assert thickness[0] > 0
