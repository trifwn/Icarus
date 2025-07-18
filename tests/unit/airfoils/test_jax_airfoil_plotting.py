"""
Unit tests for JAX airfoil plotting functionality.

This module tests the plotting capabilities of the JaxAirfoil class and the
AirfoilPlotter utility class, including basic plotting, batch plotting,
camber line visualization, thickness distribution plots, and debugging features.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ICARUS.airfoils.jax_implementation.jax_airfoil import JaxAirfoil
from ICARUS.airfoils.jax_implementation.plotting_utils import AirfoilPlotter


class TestJaxAirfoilPlotting:
    """Test basic plotting functionality of JaxAirfoil class."""

    @pytest.fixture
    def naca0012(self):
        """Create a NACA 0012 airfoil for testing."""
        return JaxAirfoil.naca("0012", n_points=200)

    @pytest.fixture
    def naca2412(self):
        """Create a NACA 2412 airfoil for testing."""
        return JaxAirfoil.naca("2412", n_points=200)

    @pytest.fixture
    def empty_airfoil(self):
        """Create an empty airfoil for edge case testing."""
        return JaxAirfoil()

    def test_basic_plot(self, naca0012):
        """Test basic airfoil plotting functionality."""
        # Test plotting without axes (should create new figure)
        ax = naca0012.plot()
        assert isinstance(ax, Axes)
        assert "Airfoil: NACA0012" in ax.get_title()
        assert "points)" in ax.get_title()
        plt.close("all")

    def test_plot_with_axes(self, naca0012):
        """Test plotting with provided axes."""
        fig, ax = plt.subplots()
        result = naca0012.plot(ax=ax)
        assert result is None  # Should return None when axes provided
        assert "Airfoil: NACA0012" in ax.get_title()
        assert "points)" in ax.get_title()
        plt.close(fig)

    def test_plot_scatter_mode(self, naca0012):
        """Test scatter plot mode."""
        ax = naca0012.plot(scatter=True, markersize=5.0)
        assert isinstance(ax, Axes)

        # Check that scatter plots were created
        collections = ax.collections
        assert len(collections) >= 2  # Upper and lower surface scatter plots
        plt.close("all")

    def test_plot_with_camber(self, naca2412):
        """Test plotting with camber line."""
        ax = naca2412.plot(camber=True)
        assert isinstance(ax, Axes)

        # Check that lines were plotted (airfoil + camber line)
        lines = ax.get_lines()
        assert len(lines) >= 3  # Upper, lower, and camber line
        plt.close("all")

    def test_plot_with_max_thickness(self, naca0012):
        """Test plotting with maximum thickness indicator."""
        ax = naca0012.plot(max_thickness=True)
        assert isinstance(ax, Axes)

        # Check that thickness line was added
        lines = ax.get_lines()
        assert len(lines) >= 3  # Upper, lower, and thickness line

        # Check that text annotation was added
        texts = ax.texts
        assert len(texts) >= 1
        plt.close("all")

    def test_plot_with_all_options(self, naca2412):
        """Test plotting with all options enabled."""
        ax = naca2412.plot(
            camber=True,
            scatter=True,
            max_thickness=True,
            show_legend=True,
            alpha=0.7,
            linewidth=2.0,
            markersize=3.0,
        )
        assert isinstance(ax, Axes)

        # Check that legend was created
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")

    def test_plot_with_custom_color(self, naca0012):
        """Test plotting with custom color override."""
        ax = naca0012.plot(overide_color="green", linewidth=3.0)
        assert isinstance(ax, Axes)

        # Check that lines exist
        lines = ax.get_lines()
        assert len(lines) >= 2  # Upper and lower surface
        plt.close("all")

    def test_plot_empty_airfoil_error(self, empty_airfoil):
        """Test that plotting empty airfoil raises appropriate error."""
        with pytest.raises(
            ValueError,
            match="Cannot plot airfoil with no valid points",
        ):
            empty_airfoil.plot()

    def test_plot_missing_matplotlib_error(self, naca0012, monkeypatch):
        """Test error handling when matplotlib is not available."""

        # Mock matplotlib import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No module named 'matplotlib'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(
            ImportError,
            match="matplotlib is required for plotting functionality",
        ):
            naca0012.plot()


class TestAirfoilPlotter:
    """Test the AirfoilPlotter utility class."""

    @pytest.fixture
    def airfoils(self):
        """Create multiple airfoils for batch testing."""
        return [
            JaxAirfoil.naca("0012", n_points=50),
            JaxAirfoil.naca("2412", n_points=50),
            JaxAirfoil.naca("4412", n_points=50),
        ]

    @pytest.fixture
    def single_airfoil(self):
        """Create a single airfoil for testing."""
        return JaxAirfoil.naca("0012", n_points=100)

    def test_plot_batch_basic(self, airfoils):
        """Test basic batch plotting functionality."""
        ax = AirfoilPlotter.plot_batch(airfoils)
        assert isinstance(ax, Axes)
        assert "Batch Airfoil Plot (3 airfoils)" in ax.get_title()

        # Check that multiple lines were plotted
        lines = ax.get_lines()
        assert len(lines) >= 6  # 3 airfoils × 2 surfaces each
        plt.close("all")

    def test_plot_batch_with_custom_names(self, airfoils):
        """Test batch plotting with custom names."""
        names = ["Symmetric", "Cambered", "High Camber"]
        ax = AirfoilPlotter.plot_batch(airfoils, names=names)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_batch_with_custom_colors(self, airfoils):
        """Test batch plotting with custom colors."""
        colors = ["red", "blue", "green"]
        ax = AirfoilPlotter.plot_batch(airfoils, colors=colors)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_batch_scatter_mode(self, airfoils):
        """Test batch plotting in scatter mode."""
        ax = AirfoilPlotter.plot_batch(airfoils, scatter=True, alpha=0.6)
        assert isinstance(ax, Axes)

        # Check that scatter plots were created
        collections = ax.collections
        assert len(collections) >= 6  # 3 airfoils × 2 surfaces each
        plt.close("all")

    def test_plot_batch_with_axes(self, airfoils):
        """Test batch plotting with provided axes."""
        fig, ax = plt.subplots(figsize=(12, 8))
        result_ax = AirfoilPlotter.plot_batch(airfoils, ax=ax)
        assert result_ax is ax
        plt.close(fig)

    def test_plot_batch_empty_list_error(self):
        """Test that empty airfoil list raises error."""
        with pytest.raises(ValueError, match="At least one airfoil must be provided"):
            AirfoilPlotter.plot_batch([])

    def test_plot_batch_mismatched_names_error(self, airfoils):
        """Test error for mismatched names length."""
        with pytest.raises(
            ValueError,
            match="Length of names .* must match number of airfoils",
        ):
            AirfoilPlotter.plot_batch(airfoils, names=["Only", "Two"])

    def test_plot_batch_mismatched_colors_error(self, airfoils):
        """Test error for mismatched colors length."""
        with pytest.raises(
            ValueError,
            match="Length of colors .* must match number of airfoils",
        ):
            AirfoilPlotter.plot_batch(airfoils, colors=["red", "blue"])

    def test_plot_camber_comparison(self, airfoils):
        """Test camber line comparison plotting."""
        ax = AirfoilPlotter.plot_camber_comparison(airfoils)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Camber Line Comparison"
        assert ax.get_xlabel() == "x/c"
        assert ax.get_ylabel() == "Camber"

        # Check that camber lines were plotted
        lines = ax.get_lines()
        assert len(lines) == 3  # One line per airfoil
        plt.close("all")

    def test_plot_thickness_comparison(self, airfoils):
        """Test thickness distribution comparison plotting."""
        ax = AirfoilPlotter.plot_thickness_comparison(airfoils)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Thickness Distribution Comparison"
        assert ax.get_xlabel() == "x/c"
        assert ax.get_ylabel() == "Thickness"

        # Check that thickness lines were plotted
        lines = ax.get_lines()
        assert len(lines) == 3  # One line per airfoil
        plt.close("all")

    def test_plot_airfoil_analysis(self, single_airfoil):
        """Test comprehensive airfoil analysis plot."""
        ax = AirfoilPlotter.plot_airfoil_analysis(
            single_airfoil,
            show_camber=True,
            show_thickness=True,
            show_max_thickness=True,
            show_max_camber=True,
        )
        assert isinstance(ax, Axes)
        assert "Airfoil Analysis: NACA0012" in ax.get_title()
        plt.close("all")

    def test_plot_airfoil_analysis_scatter(self, single_airfoil):
        """Test airfoil analysis with scatter points."""
        ax = AirfoilPlotter.plot_airfoil_analysis(
            single_airfoil,
            scatter_points=True,
            show_camber=False,
            show_thickness=False,
        )
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_create_subplot_grid(self, airfoils):
        """Test creating subplot grid for multiple airfoils."""
        fig, axes = AirfoilPlotter.create_subplot_grid(airfoils, ncols=2)
        assert isinstance(fig, Figure)
        assert len(axes) == 3
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_create_subplot_grid_with_names(self, airfoils):
        """Test subplot grid with custom names."""
        names = ["Symmetric", "Cambered", "High Camber"]
        fig, axes = AirfoilPlotter.create_subplot_grid(
            airfoils,
            names=names,
            ncols=3,
            show_camber=True,
        )
        assert isinstance(fig, Figure)
        assert len(axes) == 3

        # Check that titles were set
        for ax, name in zip(axes, names):
            assert ax.get_title() == name
        plt.close(fig)

    def test_create_subplot_grid_empty_error(self):
        """Test error for empty airfoil list in subplot grid."""
        with pytest.raises(ValueError, match="At least one airfoil must be provided"):
            AirfoilPlotter.create_subplot_grid([])

    def test_plot_morphing_sequence(self):
        """Test morphing sequence visualization."""
        airfoil1 = JaxAirfoil.naca("0012", n_points=50)
        airfoil2 = JaxAirfoil.naca("2412", n_points=50)

        fig, ax = AirfoilPlotter.plot_morphing_sequence(
            airfoil1,
            airfoil2,
            n_steps=5,
            alpha=0.8,
        )
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert "Morphing Sequence: NACA0012 → NACA2412" in ax.get_title()

        # Check that multiple lines were plotted (5 steps × 2 surfaces each)
        lines = ax.get_lines()
        assert len(lines) >= 10
        plt.close(fig)

    def test_debug_coordinate_plot(self, single_airfoil):
        """Test debug coordinate plotting."""
        ax = AirfoilPlotter.debug_coordinate_plot(
            single_airfoil,
            show_indices=True,
            show_buffer=True,
        )
        assert isinstance(ax, Axes)
        assert "Debug Plot: NACA0012" in ax.get_title()

        # Check that scatter plots were created
        collections = ax.collections
        assert len(collections) >= 2  # Upper and lower surface points

        # Check that annotations were added
        texts = ax.texts
        assert len(texts) > 0  # Should have index annotations
        plt.close("all")

    def test_debug_coordinate_plot_no_indices(self, single_airfoil):
        """Test debug plotting without indices."""
        ax = AirfoilPlotter.debug_coordinate_plot(
            single_airfoil,
            show_indices=False,
            show_buffer=False,
        )
        assert isinstance(ax, Axes)
        plt.close("all")


class TestPlottingIntegration:
    """Integration tests for plotting functionality."""

    def test_plot_after_morphing(self):
        """Test plotting after airfoil morphing."""
        airfoil1 = JaxAirfoil.naca("0012", n_points=50)
        airfoil2 = JaxAirfoil.naca("2412", n_points=50)

        morphed = JaxAirfoil.morph_new_from_two_foils(
            airfoil1,
            airfoil2,
            eta=0.5,
            n_points=100,
        )

        ax = morphed.plot(camber=True, max_thickness=True)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_plot_after_flap_operation(self):
        """Test plotting after flap operation."""
        airfoil = JaxAirfoil.naca("2412", n_points=100)
        flapped = airfoil.flap(flap_hinge_chord_percentage=0.7, flap_angle=15.0)

        ax = flapped.plot(scatter=True)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_batch_plot_different_types(self):
        """Test batch plotting with different airfoil types."""
        airfoils = [
            JaxAirfoil.naca("0012", n_points=50),
            JaxAirfoil.naca("23012", n_points=50),  # 5-digit NACA
        ]

        ax = AirfoilPlotter.plot_batch(airfoils, legend=True)
        assert isinstance(ax, Axes)
        plt.close("all")

    def test_comprehensive_analysis_plot(self):
        """Test comprehensive analysis with all features."""
        airfoil = JaxAirfoil.naca("4412", n_points=100)

        # Create a comprehensive analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Basic airfoil plot
        airfoil.plot(ax=ax1, camber=True, max_thickness=True)
        ax1.set_title("Basic Plot with Camber and Max Thickness")

        # Scatter plot
        airfoil.plot(ax=ax2, scatter=True, show_legend=True)
        ax2.set_title("Scatter Plot")

        # Camber comparison (single airfoil)
        AirfoilPlotter.plot_camber_comparison([airfoil], ax=ax3)
        ax3.set_title("Camber Line")

        # Thickness comparison (single airfoil)
        AirfoilPlotter.plot_thickness_comparison([airfoil], ax=ax4)
        ax4.set_title("Thickness Distribution")

        plt.tight_layout()
        plt.close(fig)

    def test_plotting_with_custom_coordinates(self):
        """Test plotting with custom coordinate airfoil."""
        # Create custom airfoil coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 0.5 * (1 + np.cos(theta))
        y = 0.1 * np.sin(theta)
        coords = jnp.array([x, y])

        airfoil = JaxAirfoil(coords, name="Custom Airfoil")
        ax = airfoil.plot(camber=True)
        assert isinstance(ax, Axes)
        assert "Custom Airfoil" in ax.get_title()
        plt.close("all")


if __name__ == "__main__":
    # Run a simple test to verify functionality
    print("Testing JAX Airfoil plotting functionality...")

    # Create test airfoils
    naca0012 = JaxAirfoil.naca("0012", n_points=100)
    naca2412 = JaxAirfoil.naca("2412", n_points=100)

    # Test basic plotting
    print("Testing basic plot...")
    ax = naca0012.plot()
    plt.savefig("test_basic_plot.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Test batch plotting
    print("Testing batch plot...")
    airfoils = [naca0012, naca2412]
    ax = AirfoilPlotter.plot_batch(airfoils, legend=True)
    plt.savefig("test_batch_plot.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Test comprehensive analysis
    print("Testing analysis plot...")
    ax = AirfoilPlotter.plot_airfoil_analysis(
        naca2412,
        show_camber=True,
        show_thickness=True,
        show_max_thickness=True,
        show_max_camber=True,
    )
    plt.savefig("test_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("All tests completed successfully!")
