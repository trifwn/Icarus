"""
Integration tests for JAX airfoil API compatibility layer.

This module contains tests to verify that the JaxAirfoil class maintains
API compatibility with the original Airfoil class and can be used as a
drop-in replacement in existing ICARUS code.
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ICARUS.airfoils import NACA4
from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestJaxAirfoilAPICompatibility:
    """Test class for JAX airfoil API compatibility."""

    @pytest.fixture
    def sample_airfoil_data(self):
        """Create sample airfoil data for testing."""
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
        return upper, lower

    @pytest.fixture
    def jax_airfoil(self, sample_airfoil_data):
        """Create a JaxAirfoil instance for testing."""
        upper, lower = sample_airfoil_data
        return JaxAirfoil.from_upper_lower(upper, lower, name="TestAirfoil")

    def test_property_compatibility(self, jax_airfoil):
        """Test that properties match the original API."""
        # Test name property
        assert hasattr(jax_airfoil, "name")
        assert isinstance(jax_airfoil.name, str)
        assert jax_airfoil.name == "TestAirfoil"

        # Test name setter
        jax_airfoil.name = "NewName"
        assert jax_airfoil.name == "NewName"

        # Test file_name property
        assert hasattr(jax_airfoil, "file_name")
        assert jax_airfoil.file_name == "NewName.dat"

        # Test surface properties
        assert hasattr(jax_airfoil, "upper_surface")
        assert hasattr(jax_airfoil, "lower_surface")

        upper_surface = jax_airfoil.upper_surface
        lower_surface = jax_airfoil.lower_surface

        assert upper_surface.shape[0] == 2  # [x, y] coordinates
        assert lower_surface.shape[0] == 2  # [x, y] coordinates
        assert upper_surface.shape[1] > 0  # Has points
        assert lower_surface.shape[1] > 0  # Has points

    def test_geometric_method_compatibility(self, jax_airfoil):
        """Test that geometric methods match the original API."""
        # Test thickness method
        thickness = jax_airfoil.thickness(0.5)
        assert isinstance(thickness, jax.Array)
        assert len(thickness) == 1
        assert thickness[0] > 0

        # Test with array input
        thickness_array = jax_airfoil.thickness(jnp.array([0.25, 0.5, 0.75]))
        assert len(thickness_array) == 3
        assert jnp.all(thickness_array > 0)

        # Test camber_line method
        camber = jax_airfoil.camber_line(0.5)
        assert isinstance(camber, jax.Array)
        assert len(camber) == 1

        # Test y_upper and y_lower methods
        y_upper = jax_airfoil.y_upper(0.5)
        y_lower = jax_airfoil.y_lower(0.5)
        assert isinstance(y_upper, jax.Array)
        assert isinstance(y_lower, jax.Array)
        assert y_upper[0] > y_lower[0]  # Upper should be above lower

    def test_geometric_property_compatibility(self, jax_airfoil):
        """Test that geometric properties match the original API."""
        # Test maximum thickness
        max_thickness = jax_airfoil.max_thickness
        assert isinstance(max_thickness, float)
        assert max_thickness > 0

        # Test maximum thickness location
        max_thickness_location = jax_airfoil.max_thickness_location
        assert isinstance(max_thickness_location, float)
        assert 0 <= max_thickness_location <= 1

        # Test maximum camber
        max_camber = jax_airfoil.max_camber
        assert isinstance(max_camber, float)

        # Test maximum camber location
        max_camber_location = jax_airfoil.max_camber_location
        assert isinstance(max_camber_location, float)
        assert 0 <= max_camber_location <= 1

        # Test chord length
        chord_length = jax_airfoil.chord_length
        assert isinstance(chord_length, float)
        assert chord_length > 0

    def test_to_selig_compatibility(self, jax_airfoil):
        """Test that to_selig method works like the original."""
        selig_coords = jax_airfoil.to_selig()

        assert isinstance(selig_coords, jax.Array)
        assert selig_coords.shape[0] == 2  # [x, y] coordinates
        assert selig_coords.shape[1] == jax_airfoil.n_points

        # Check that it forms a closed loop (first and last points should be close)
        x_coords, y_coords = selig_coords[0, :], selig_coords[1, :]
        assert jnp.isclose(x_coords[0], x_coords[-1], atol=1e-6)
        assert jnp.isclose(y_coords[0], y_coords[-1], atol=1e-6)

    def test_repanel_from_internal_compatibility(self, jax_airfoil):
        """Test that repanel_from_internal method works like the original."""
        original_n_points = jax_airfoil.n_points

        # Test cosine distribution
        jax_airfoil.repanel_from_internal(50, distribution="cosine")
        assert jax_airfoil.n_points != original_n_points  # Should have changed

        # Test uniform distribution
        jax_airfoil.repanel_from_internal(100, distribution="uniform")
        # Note: Actual point count may differ due to preprocessing

    def test_naca_generation_compatibility(self):
        """Test that NACA generation methods work like the original."""
        # Test NACA 4-digit
        naca4_airfoil = JaxAirfoil.naca("2412", n_points=100)
        assert naca4_airfoil.name == "NACA2412"
        assert naca4_airfoil.n_points > 0

        # Test NACA 5-digit
        naca5_airfoil = JaxAirfoil.naca("23012", n_points=100)
        assert naca5_airfoil.name == "NACA23012"
        assert naca5_airfoil.n_points > 0

        # Test with NACA prefix
        naca_with_prefix = JaxAirfoil.naca("NACA0012", n_points=100)
        assert naca_with_prefix.name == "NACA0012"

    def test_morphing_compatibility(self, sample_airfoil_data):
        """Test that morphing works like the original."""
        upper, lower = sample_airfoil_data

        # Create two different airfoils
        airfoil1 = JaxAirfoil.from_upper_lower(upper, lower, name="Airfoil1")

        # Create a slightly different airfoil
        upper2 = upper.at[1, :].multiply(1.2)  # Make it thicker
        airfoil2 = JaxAirfoil.from_upper_lower(upper2, lower, name="Airfoil2")

        # Test morphing
        morphed = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 0.5, 100)

        assert isinstance(morphed, JaxAirfoil)
        assert "morphed" in morphed.name.lower()
        assert morphed.n_points > 0

        # Test edge cases
        morphed_0 = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 0.0, 100)
        assert morphed_0.name == airfoil1.name

        morphed_1 = JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 1.0, 100)
        assert morphed_1.name == airfoil2.name

    def test_flap_operation_compatibility(self, jax_airfoil):
        """Test that flap operations work like the original."""
        # Test basic flap operation
        flapped = jax_airfoil.flap(
            flap_hinge_chord_percentage=0.75,
            flap_angle=10.0,
            flap_hinge_thickness_percentage=0.5,
            chord_extension=1.0,
        )

        assert isinstance(flapped, JaxAirfoil)
        assert "flapped" in flapped.name.lower()
        assert flapped.n_points > 0

        # Test no-op cases
        no_flap = jax_airfoil.flap(0.75, 0.0)  # Zero angle
        assert no_flap is jax_airfoil  # Should return same instance

        no_flap2 = jax_airfoil.flap(1.0, 10.0)  # Hinge at trailing edge
        assert no_flap2 is jax_airfoil  # Should return same instance

    def test_file_io_compatibility(self, jax_airfoil):
        """Test that file I/O methods work like the original."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test save_selig
            jax_airfoil.save_selig(directory=temp_dir, header=True)

            saved_file = os.path.join(temp_dir, jax_airfoil.file_name)
            assert os.path.exists(saved_file)

            # Test loading the saved file
            loaded_airfoil = JaxAirfoil.from_file(saved_file)
            assert isinstance(loaded_airfoil, JaxAirfoil)
            assert loaded_airfoil.n_points > 0

            # Test save_le
            jax_airfoil.save_le(directory=temp_dir, header=True)
            assert os.path.exists(saved_file)  # Should overwrite

    def test_string_representation_compatibility(self, jax_airfoil):
        """Test that string representations work like the original."""
        str_repr = str(jax_airfoil)
        assert isinstance(str_repr, str)
        assert jax_airfoil.name in str_repr
        assert str(jax_airfoil.n_points) in str_repr

        repr_str = repr(jax_airfoil)
        assert isinstance(repr_str, str)
        # Both should contain the airfoil name and point count
        assert jax_airfoil.name in repr_str
        assert str(jax_airfoil.n_points) in repr_str

    def test_legacy_conversion_utilities(self, sample_airfoil_data):
        """Test conversion utilities between old and new formats."""
        upper, lower = sample_airfoil_data
        jax_airfoil = JaxAirfoil.from_upper_lower(upper, lower, name="TestConversion")

        # Test to_legacy_format
        legacy_data = jax_airfoil.to_legacy_format()

        assert isinstance(legacy_data, dict)
        assert "upper" in legacy_data
        assert "lower" in legacy_data
        assert "name" in legacy_data
        assert "n_points" in legacy_data

        assert legacy_data["name"] == "TestConversion"
        assert legacy_data["n_points"] == jax_airfoil.n_points
        assert isinstance(legacy_data["upper"], jax.Array)
        assert isinstance(legacy_data["lower"], jax.Array)

    def test_plotting_compatibility(self, jax_airfoil):
        """Test that plotting methods work (without actually displaying plots)."""
        # Import matplotlib only if available
        pytest.importorskip("matplotlib")

        import matplotlib.pyplot as plt

        # Test basic plot
        fig, ax = plt.subplots()
        jax_airfoil.plot(ax=ax)

        # Test plot with options
        jax_airfoil.plot(ax=ax, camber=True, scatter=True, max_thickness=True)

        # Test plot with color override
        jax_airfoil.plot(ax=ax, overide_color="green")

        plt.close(fig)

    def test_error_handling_compatibility(self):
        """Test that error handling works like the original."""
        # Test invalid NACA designation
        with pytest.raises(ValueError):
            JaxAirfoil.naca("invalid")

        # Test invalid morphing parameter
        upper = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        lower = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        airfoil1 = JaxAirfoil.from_upper_lower(upper, lower)
        airfoil2 = JaxAirfoil.from_upper_lower(upper, lower)

        with pytest.raises(ValueError):
            JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, -0.1, 100)

        with pytest.raises(ValueError):
            JaxAirfoil.morph_new_from_two_foils(airfoil1, airfoil2, 1.1, 100)

        # Test invalid flap parameters
        with pytest.raises(ValueError):
            airfoil1.flap(-0.1, 10.0)  # Invalid hinge position

        with pytest.raises(ValueError):
            airfoil1.flap(0.5, 10.0, -0.1)  # Invalid thickness percentage

        with pytest.raises(ValueError):
            airfoil1.flap(0.5, 10.0, chord_extension=-1.0)  # Invalid extension

    def test_jit_compatibility_with_api_methods(self, jax_airfoil):
        """Test that API methods work with JAX JIT compilation."""

        @jax.jit
        def compute_properties(airfoil):
            thickness = airfoil.thickness(jnp.array([0.5]))[0]
            camber = airfoil.camber_line(jnp.array([0.5]))[0]
            y_upper = airfoil.y_upper(jnp.array([0.5]))[0]
            y_lower = airfoil.y_lower(jnp.array([0.5]))[0]
            return thickness, camber, y_upper, y_lower

        thickness, camber, y_upper, y_lower = compute_properties(jax_airfoil)

        assert jnp.isfinite(thickness)
        assert jnp.isfinite(camber)
        assert jnp.isfinite(y_upper)
        assert jnp.isfinite(y_lower)
        assert thickness > 0
        assert y_upper > y_lower

    def test_gradient_compatibility_with_api_methods(self, jax_airfoil):
        """Test that API methods support automatic differentiation."""

        def thickness_at_midpoint(airfoil):
            return airfoil.thickness(jnp.array([0.5]))[0]

        def camber_at_midpoint(airfoil):
            return airfoil.camber_line(jnp.array([0.5]))[0]

        # Test gradients
        grad_thickness = jax.grad(thickness_at_midpoint)(jax_airfoil)
        grad_camber = jax.grad(camber_at_midpoint)(jax_airfoil)

        assert isinstance(grad_thickness, JaxAirfoil)
        assert isinstance(grad_camber, JaxAirfoil)

        # Gradients should have the same structure
        assert grad_thickness.n_points == jax_airfoil.n_points
        assert grad_camber.n_points == jax_airfoil.n_points

    def test_batch_compatibility(self, sample_airfoil_data):
        """Test that the API works with batched operations."""
        upper, lower = sample_airfoil_data

        # Create multiple airfoils
        airfoils = [
            JaxAirfoil.from_upper_lower(upper, lower, name=f"Airfoil{i}")
            for i in range(3)
        ]

        # Test that we can work with multiple airfoils
        for airfoil in airfoils:
            assert airfoil.n_points > 0
            assert airfoil.max_thickness > 0

            # Test geometric operations
            thickness = airfoil.thickness(0.5)
            assert len(thickness) == 1
            assert thickness[0] > 0

    def test_numpy_array_input_compatibility(self):
        """Test that the API accepts NumPy arrays like the original."""
        # Create NumPy arrays
        upper_np = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.05, 0.08, 0.05, 0.0],
            ],
        )
        lower_np = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, -0.05, -0.08, -0.05, 0.0],
            ],
        )

        # Should work with NumPy arrays
        airfoil = JaxAirfoil.from_upper_lower(upper_np, lower_np, name="NumPyTest")
        assert airfoil.n_points > 0

        # Test geometric operations with NumPy input
        thickness_np = airfoil.thickness(np.array([0.5]))
        assert len(thickness_np) == 1
        assert thickness_np[0] > 0

        # Test with scalar input
        thickness_scalar = airfoil.thickness(0.5)
        assert jnp.isclose(thickness_scalar[0], thickness_np[0])

    def test_metadata_preservation(self, sample_airfoil_data):
        """Test that metadata is preserved through operations."""
        upper, lower = sample_airfoil_data
        metadata = {"source": "test", "version": 1.0, "author": "pytest"}

        airfoil = JaxAirfoil.from_upper_lower(
            upper,
            lower,
            name="MetadataTest",
            metadata=metadata,
        )

        # Check that metadata is preserved
        assert airfoil._metadata["source"] == "test"
        assert airfoil._metadata["version"] == 1.0
        assert airfoil._metadata["author"] == "pytest"

        # Test that metadata is preserved through operations
        flapped = airfoil.flap(0.75, 10.0)
        assert flapped._metadata["source"] == "test"  # Should be copied

        repaneled = airfoil.repanel(50)
        assert repaneled._metadata["source"] == "test"  # Should be copied

    def test_edge_case_compatibility(self):
        """Test edge cases that should work like the original."""
        # Test with minimal airfoil
        minimal_upper = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        minimal_lower = jnp.array([[0.0, 1.0], [0.0, 0.0]])

        minimal_airfoil = JaxAirfoil.from_upper_lower(minimal_upper, minimal_lower)
        assert (
            minimal_airfoil.n_points >= 2
        )  # Should have at least the points we gave it

        # Test geometric operations on minimal airfoil
        thickness = minimal_airfoil.thickness(0.5)
        assert len(thickness) == 1
        assert thickness[0] >= 0  # Should be zero or positive

        # Test with symmetric airfoil (zero camber)
        symmetric_upper = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.1, 0.0]])
        symmetric_lower = jnp.array([[0.0, 0.5, 1.0], [0.0, -0.1, 0.0]])

        symmetric_airfoil = JaxAirfoil.from_upper_lower(
            symmetric_upper,
            symmetric_lower,
        )
        camber = symmetric_airfoil.camber_line(0.5)
        assert abs(camber[0]) < 1e-6  # Should be approximately zero


class TestFileIOCompatibility:
    """Test file I/O compatibility with various formats."""

    def test_selig_format_roundtrip(self):
        """Test that we can save and load in Selig format."""
        # Create a test airfoil
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

        original = JaxAirfoil.from_upper_lower(upper, lower, name="RoundtripTest")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the airfoil
            original.save_selig(directory=temp_dir, header=True)

            # Load it back
            file_path = os.path.join(temp_dir, original.file_name)
            loaded = JaxAirfoil.from_file(file_path)

            # Check that the loaded airfoil is similar
            assert loaded.n_points > 0
            assert "RoundtripTest" in loaded.name

            # Check that geometric properties are similar
            original_thickness = original.thickness(0.5)[0]
            loaded_thickness = loaded.thickness(0.5)[0]
            assert jnp.isclose(original_thickness, loaded_thickness, rtol=1e-2)

    def test_file_loading_error_handling(self):
        """Test error handling in file loading."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            JaxAirfoil.from_file("nonexistent_file.dat")

        # Test empty file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            with pytest.raises(ValueError):
                JaxAirfoil.from_file(temp_file)
        finally:
            os.unlink(temp_file)

        # Test file with invalid data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write("invalid data\n")
            f.write("not numbers\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError):
                JaxAirfoil.from_file(temp_file)
        finally:
            os.unlink(temp_file)


class TestIntegrationWithExistingCode:
    """Test integration with existing ICARUS workflows."""

    def test_naca_comparison_with_original(self):
        """Compare NACA generation with original implementation."""
        # Create NACA airfoil with both implementations
        naca_designation = "2412"
        n_points = 100

        # JAX implementation
        jax_naca = JaxAirfoil.naca(naca_designation, n_points=n_points)

        # Original implementation (if available)
        try:
            original_naca = NACA4(M=0.02, P=0.4, XX=0.12, n_points=n_points)

            # Compare geometric properties
            jax_max_thickness = jax_naca.max_thickness
            original_max_thickness = original_naca.max_thickness

            # Should be close (within 5% relative error)
            assert jnp.isclose(jax_max_thickness, original_max_thickness, rtol=0.05)

        except ImportError:
            # Original NACA4 not available, just check that JAX version works
            assert jax_naca.n_points > 0
            assert jax_naca.max_thickness > 0

    def test_workflow_compatibility(self):
        """Test that JaxAirfoil can be used in typical workflows."""
        # Create an airfoil
        airfoil = JaxAirfoil.naca("0012", n_points=100)

        # Typical workflow: analyze, modify, analyze again
        original_thickness = airfoil.max_thickness

        # Apply flap
        flapped = airfoil.flap(0.8, 15.0)
        flapped_thickness = flapped.max_thickness

        # Repanel
        flapped.repanel_from_internal(150, distribution="cosine")
        repaneled_thickness = flapped.max_thickness

        # All operations should produce valid results
        assert original_thickness > 0
        assert flapped_thickness > 0
        assert repaneled_thickness > 0

        # Save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            flapped.save_selig(directory=temp_dir)
            loaded = JaxAirfoil.from_file(os.path.join(temp_dir, flapped.file_name))
            assert loaded.n_points > 0

    def test_optimization_workflow_compatibility(self):
        """Test compatibility with optimization workflows using gradients."""

        def objective_function(airfoil_coords):
            """Example objective function for optimization."""
            # Create airfoil from coordinates
            airfoil = JaxAirfoil(airfoil_coords, name="OptimizationTest")

            # Compute some objective (e.g., minimize drag proxy)
            thickness = airfoil.thickness(jnp.array([0.3, 0.5, 0.7]))
            max_thickness = jnp.max(thickness)

            # Simple objective: minimize maximum thickness while maintaining some constraints
            return max_thickness

        # Create initial airfoil coordinates
        initial_coords = jnp.array(
            [
                [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.05, 0.08, 0.05, 0.0, -0.05, -0.08, -0.05, 0.0],
            ],
        )

        # Test that we can compute gradients
        grad_fn = jax.grad(objective_function)
        gradients = grad_fn(initial_coords)

        # Gradients should be finite and have the right shape
        assert gradients.shape == initial_coords.shape
        assert jnp.all(jnp.isfinite(gradients))

        # Test that we can use the gradients (simple gradient descent step)
        learning_rate = 0.01
        updated_coords = initial_coords - learning_rate * gradients

        # Updated coordinates should still be valid
        updated_airfoil = JaxAirfoil(updated_coords, name="Updated")
        assert updated_airfoil.n_points > 0
        assert updated_airfoil.max_thickness > 0
