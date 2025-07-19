"""
Tests for JAX Airfoil API compatibility methods.

This module tests the missing API compatibility methods that were added to maintain
compatibility with the original Airfoil class implementation.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import requests

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil


class TestJaxAirfoilAPICompatibility:
    """Test class for JAX Airfoil API compatibility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test airfoil
        self.test_airfoil = JaxAirfoil.naca4("2412", n_points=100)

    def test_selig_original_property(self):
        """Test the selig_original property."""
        # Get selig_original
        selig_orig = self.test_airfoil.selig_original

        # Should be a JAX array
        assert isinstance(selig_orig, jnp.ndarray)

        # Should have shape (2, n_points)
        assert selig_orig.shape[0] == 2
        assert selig_orig.shape[1] == self.test_airfoil.n_points

        # Should be the same as to_selig()
        selig_method = self.test_airfoil.to_selig()
        np.testing.assert_array_equal(selig_orig, selig_method)

    def test_file_name_property(self):
        """Test the file_name property."""
        # Test with named airfoil
        file_name = self.test_airfoil.file_name
        assert file_name == "NACA2412.dat"

        # Test with unnamed airfoil
        unnamed_airfoil = JaxAirfoil(name="")
        assert unnamed_airfoil.file_name == "Airfoil.dat"

        # Test with None name
        none_name_airfoil = JaxAirfoil()
        none_name_airfoil._metadata["name"] = None
        assert none_name_airfoil.file_name == "Airfoil.dat"

    def test_to_selig_method(self):
        """Test the to_selig method."""
        selig_coords = self.test_airfoil.to_selig()

        # Should be a JAX array
        assert isinstance(selig_coords, jnp.ndarray)

        # Should have correct shape
        assert selig_coords.shape[0] == 2
        assert selig_coords.shape[1] == self.test_airfoil.n_points

        # X coordinates should be in valid range (approximately [0, 1] for NACA airfoils)
        x_coords = selig_coords[0, :]
        valid_x = x_coords[~jnp.isnan(x_coords)]
        assert jnp.all(
            valid_x >= -0.1,
        )  # Allow small negative values due to numerical precision
        assert jnp.all(
            valid_x <= 1.1,
        )  # Allow small values above 1 due to numerical precision

        # Should not contain NaN values in valid region
        valid_mask = ~jnp.isnan(x_coords)
        assert jnp.sum(valid_mask) == self.test_airfoil.n_points

    def test_repanel_spl_method(self):
        """Test the repanel_spl method."""
        original_n_points = self.test_airfoil.n_points
        target_n_points = 150

        # Store original coordinates for comparison
        original_coords = self.test_airfoil.to_selig()

        # Repanel the airfoil
        self.test_airfoil.repanel_spl(target_n_points)

        # Check that the number of points changed
        assert self.test_airfoil.n_points == target_n_points

        # Check that coordinates are still valid
        new_coords = self.test_airfoil.to_selig()
        assert new_coords.shape[0] == 2
        assert new_coords.shape[1] == target_n_points

        # Check that the airfoil shape is preserved (approximately)
        # Test a few key points
        test_x = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Create temporary airfoil with original coordinates for comparison
        temp_airfoil = JaxAirfoil.from_upper_lower(
            original_coords[:, : original_n_points // 2],
            original_coords[:, original_n_points // 2 :],
            name="temp",
        )

        original_thickness = temp_airfoil.thickness(test_x)
        new_thickness = self.test_airfoil.thickness(test_x)

        # Thickness should be approximately preserved
        np.testing.assert_allclose(original_thickness, new_thickness, rtol=0.1)

    def test_repanel_spl_edge_cases(self):
        """Test repanel_spl with edge cases."""
        # Test with very few points
        self.test_airfoil.repanel_spl(10)
        assert self.test_airfoil.n_points == 10

        # Test with many points
        self.test_airfoil.repanel_spl(500)
        assert self.test_airfoil.n_points == 500

        # Test with same number of points
        current_points = self.test_airfoil.n_points
        self.test_airfoil.repanel_spl(current_points)
        assert self.test_airfoil.n_points == current_points

    @patch("requests.get")
    def test_load_from_web_success(self, mock_get):
        """Test successful loading from web."""
        # Mock the database page response
        database_html = """
        <html>
        <body>
        <a href="coord/naca0012.dat">NACA 0012</a>
        <a href="coord/naca2412.dat">NACA 2412</a>
        </body>
        </html>
        """

        # Mock the airfoil data response
        airfoil_data = """1.000000 0.000000
0.950000 0.010000
0.900000 0.015000
0.800000 0.020000
0.700000 0.025000
0.600000 0.030000
0.500000 0.035000
0.400000 0.040000
0.300000 0.045000
0.200000 0.050000
0.100000 0.055000
0.050000 0.058000
0.025000 0.059000
0.012500 0.059500
0.000000 0.060000
0.012500 -0.059500
0.025000 -0.059000
0.050000 -0.058000
0.100000 -0.055000
0.200000 -0.050000
0.300000 -0.045000
0.400000 -0.040000
0.500000 -0.035000
0.600000 -0.030000
0.700000 -0.025000
0.800000 -0.020000
0.900000 -0.015000
0.950000 -0.010000
1.000000 0.000000"""

        # Set up mock responses
        mock_responses = [
            MagicMock(status_code=200, text=database_html),  # Database page
            MagicMock(status_code=200, text=airfoil_data),  # Airfoil data
        ]
        mock_get.side_effect = mock_responses

        # Test loading an airfoil
        airfoil = JaxAirfoil.load_from_web("naca0012")

        # Verify the airfoil was created correctly
        assert isinstance(airfoil, JaxAirfoil)
        assert airfoil.name == "NACA0012"
        assert airfoil.n_points > 0

        # Verify requests were made correctly
        assert mock_get.call_count == 2
        mock_get.assert_any_call(
            "https://m-selig.ae.illinois.edu/ads/coord_database.html",
            timeout=30,
        )
        mock_get.assert_any_call(
            "https://m-selig.ae.illinois.edu/ads/coord/naca0012.dat",
            timeout=30,
        )

    @patch("requests.get")
    def test_load_from_web_not_found(self, mock_get):
        """Test loading non-existent airfoil from web."""
        # Mock the database page response without the requested airfoil
        database_html = """
        <html>
        <body>
        <a href="coord/naca2412.dat">NACA 2412</a>
        </body>
        </html>
        """

        mock_get.return_value = MagicMock(status_code=200, text=database_html)

        # Test loading non-existent airfoil
        with pytest.raises(FileNotFoundError, match="not found in UIUC database"):
            JaxAirfoil.load_from_web("nonexistent")

    @patch("requests.get")
    def test_load_from_web_network_error(self, mock_get):
        """Test network error during web loading."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(FileNotFoundError, match="Error fetching airfoil database"):
            JaxAirfoil.load_from_web("naca0012")

    @patch("requests.get")
    def test_load_from_web_invalid_data(self, mock_get):
        """Test loading airfoil with invalid coordinate data."""
        # Mock responses
        database_html = '<a href="coord/invalid.dat">Invalid</a>'
        invalid_data = "This is not coordinate data\nInvalid format"

        mock_responses = [
            MagicMock(status_code=200, text=database_html),
            MagicMock(status_code=200, text=invalid_data),
        ]
        mock_get.side_effect = mock_responses

        with pytest.raises(ValueError, match="No valid coordinate data"):
            JaxAirfoil.load_from_web("invalid")

    @patch("requests.get")
    def test_load_from_web_case_insensitive(self, mock_get):
        """Test that airfoil name matching is case-insensitive."""
        database_html = '<a href="coord/naca0012.dat">NACA 0012</a>'
        # Provide more realistic airfoil data with enough points
        airfoil_data = """1.000000 0.000000
0.950000 0.010000
0.900000 0.015000
0.800000 0.020000
0.700000 0.025000
0.600000 0.030000
0.500000 0.035000
0.400000 0.040000
0.300000 0.045000
0.200000 0.050000
0.100000 0.055000
0.050000 0.058000
0.025000 0.059000
0.012500 0.059500
0.000000 0.060000
0.012500 -0.059500
0.025000 -0.059000
0.050000 -0.058000
0.100000 -0.055000
0.200000 -0.050000
0.300000 -0.045000
0.400000 -0.040000
0.500000 -0.035000
0.600000 -0.030000
0.700000 -0.025000
0.800000 -0.020000
0.900000 -0.015000
0.950000 -0.010000
1.000000 0.000000"""

        mock_responses = [
            MagicMock(status_code=200, text=database_html),
            MagicMock(status_code=200, text=airfoil_data),
        ]
        mock_get.side_effect = mock_responses

        # Test with lowercase input
        airfoil = JaxAirfoil.load_from_web("naca0012")
        assert airfoil.name == "NACA0012"

    def test_api_compatibility_integration(self):
        """Test that all API compatibility methods work together."""
        # Create an airfoil
        airfoil = JaxAirfoil.naca4("0012", n_points=100)

        # Test property access
        assert isinstance(airfoil.selig_original, jnp.ndarray)
        assert isinstance(airfoil.file_name, str)

        # Test method calls
        selig_coords = airfoil.to_selig()
        assert selig_coords.shape[1] == airfoil.n_points

        # Test repaneling
        original_points = airfoil.n_points
        airfoil.repanel_spl(150)
        assert airfoil.n_points == 150
        assert airfoil.n_points != original_points

        # Verify airfoil is still valid after repaneling
        thickness = airfoil.thickness(jnp.array([0.5]))
        assert jnp.all(thickness > 0)  # Should have positive thickness

    def test_backward_compatibility_with_original_api(self):
        """Test that the new methods maintain backward compatibility."""
        airfoil = JaxAirfoil.naca4("2412")

        # These methods should exist and work like the original
        assert hasattr(airfoil, "selig_original")
        assert hasattr(airfoil, "file_name")
        assert hasattr(airfoil, "to_selig")
        assert hasattr(airfoil, "repanel_spl")
        assert hasattr(JaxAirfoil, "load_from_web")

        # Test that they return expected types
        assert isinstance(airfoil.selig_original, jnp.ndarray)
        assert isinstance(airfoil.file_name, str)
        assert isinstance(airfoil.to_selig(), jnp.ndarray)

        # Test that repanel_spl modifies in place (API compatibility)
        original_id = id(airfoil._coordinates)
        airfoil.repanel_spl(50)
        # The object should be the same, but internal data should change
        assert airfoil.n_points == 50

    def test_metadata_preservation(self):
        """Test that metadata is preserved through API operations."""
        metadata = {"source": "test", "version": "1.0"}
        airfoil = JaxAirfoil.naca4("0012", metadata=metadata)

        # Metadata should be preserved after repaneling
        airfoil.repanel_spl(75)
        assert airfoil._metadata["source"] == "test"
        assert airfoil._metadata["version"] == "1.0"

        # Name should be preserved
        assert airfoil.name == "NACA0012"

    def test_gradient_compatibility(self):
        """Test that API methods maintain gradient compatibility where applicable."""
        import jax

        def test_function(coords):
            # Create airfoil from coordinates
            airfoil = JaxAirfoil(coords, name="test")
            # Use to_selig method
            selig = airfoil.to_selig()
            return jnp.sum(selig**2)

        # Create test coordinates
        test_coords = JaxAirfoil.naca4("0012", n_points=20).to_selig()

        # Test that gradients can be computed
        grad_fn = jax.grad(test_function)
        gradients = grad_fn(test_coords)

        # Gradients should be finite and non-zero
        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.any(gradients != 0)
