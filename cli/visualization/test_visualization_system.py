"""Test suite for the ICARUS CLI Visualization System

This module provides comprehensive tests for all visualization components,
ensuring they work correctly and integrate properly.
"""

import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rich.console import Console

from .chart_generator import ChartGenerator
from .export_manager import ExportManager
from .interactive_plotter import InteractivePlotter
from .plot_customizer import PlotCustomizer
from .real_time_updater import RealTimeUpdater
from .visualization_manager import VisualizationManager


class TestVisualizationSystem:
    """Test suite for the visualization system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = Console()
        self.viz_manager = VisualizationManager(console=self.console)

        # Sample data for testing
        self.sample_data = {
            "x": list(range(10)),
            "y": [x**2 for x in range(10)],
            "xlabel": "X Values",
            "ylabel": "Y Values",
        }

        self.airfoil_data = {
            "alpha": [-5, 0, 5, 10, 15],
            "cl": [-0.2, 0.0, 0.5, 1.0, 1.2],
            "cd": [0.008, 0.006, 0.008, 0.015, 0.025],
            "cm": [0.02, 0.0, -0.02, -0.05, -0.08],
        }

    def test_interactive_plotter_creation(self):
        """Test interactive plotter creation."""
        plotter = InteractivePlotter(console=self.console)

        # Test line plot
        fig = plotter.create_plot(
            data=self.sample_data,
            plot_type="line",
            title="Test Line Plot",
        )

        assert fig is not None
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == "X Values"
        assert fig.axes[0].get_ylabel() == "Y Values"

        plt.close(fig)

    def test_interactive_plotter_plot_types(self):
        """Test different plot types."""
        plotter = InteractivePlotter(console=self.console)

        plot_types = ["line", "scatter", "bar"]

        for plot_type in plot_types:
            fig = plotter.create_plot(
                data=self.sample_data,
                plot_type=plot_type,
                title=f"Test {plot_type.title()} Plot",
            )

            assert fig is not None
            assert len(fig.axes) == 1
            plt.close(fig)

    def test_chart_generator_airfoil_polar(self):
        """Test airfoil polar chart generation."""
        generator = ChartGenerator(console=self.console)

        fig = generator.generate_chart(
            results=self.airfoil_data,
            chart_type="airfoil_polar",
        )

        assert fig is not None
        assert len(fig.axes) == 4  # CL vs Alpha, CD vs Alpha, CM vs Alpha, CL vs CD

        plt.close(fig)

    def test_plot_customizer(self):
        """Test plot customization."""
        plotter = InteractivePlotter(console=self.console)
        customizer = PlotCustomizer(console=self.console)

        # Create a basic plot
        fig = plotter.create_plot(data=self.sample_data, plot_type="line")

        # Apply customizations
        customizations = {
            "color_palette": "aerospace",
            "line_widths": 3,
            "grid": {"visible": True, "alpha": 0.5},
            "xlabel": {"text": "Custom X Label", "fontsize": 12},
            "ylabel": {"text": "Custom Y Label", "fontsize": 12},
        }

        customized_fig = customizer.apply_customizations(fig, customizations)

        assert customized_fig is not None
        assert customized_fig.axes[0].get_xlabel() == "Custom X Label"
        assert customized_fig.axes[0].get_ylabel() == "Custom Y Label"

        plt.close(customized_fig)

    def test_export_manager(self):
        """Test plot export functionality."""
        plotter = InteractivePlotter(console=self.console)
        exporter = ExportManager(console=self.console)

        # Create a plot
        fig = plotter.create_plot(
            data=self.sample_data,
            plot_type="line",
            title="Export Test Plot",
        )

        # Test export to different formats
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test PNG export
            png_path = temp_path / "test_plot.png"
            success = exporter.export_plot(fig, png_path, format="png")
            assert success
            assert png_path.exists()

            # Test JSON export
            json_path = temp_path / "test_plot.json"
            success = exporter.export_plot(fig, json_path, format="json")
            assert success
            assert json_path.exists()

            # Verify JSON content
            with open(json_path) as f:
                plot_data = json.load(f)
            assert "figure_info" in plot_data
            assert "axes" in plot_data

        plt.close(fig)

    def test_visualization_manager_integration(self):
        """Test the main visualization manager."""
        # Create interactive plot
        plot_id = self.viz_manager.create_interactive_plot(
            data=self.sample_data,
            plot_type="line",
            title="Manager Test Plot",
        )

        assert plot_id is not None
        assert plot_id in self.viz_manager.active_plots

        # Test customization
        customizations = {"color_palette": "publication"}
        success = self.viz_manager.customize_plot(plot_id, customizations)
        assert success

        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "manager_test.png"
            success = self.viz_manager.export_plot(plot_id, output_path)
            assert success
            assert output_path.exists()

        # Test plot listing
        active_plots = self.viz_manager.list_active_plots()
        assert len(active_plots) == 1
        assert active_plots[0]["id"] == plot_id

        # Clean up
        success = self.viz_manager.close_plot(plot_id)
        assert success
        assert plot_id not in self.viz_manager.active_plots

    def test_real_time_updater_basic(self):
        """Test basic real-time update functionality."""
        plotter = InteractivePlotter(console=self.console)
        updater = RealTimeUpdater(console=self.console)

        # Create a plot
        fig = plotter.create_plot(data={"x": [0], "y": [0]}, plot_type="line")

        # Create a simple data source
        counter = {"value": 0}

        def data_source():
            counter["value"] += 1
            return {"x": counter["value"], "y": counter["value"] ** 2}

        # Start updates (very short test)
        update_id = updater.start_updates(
            plot=fig,
            data_source=data_source,
            update_interval=0.1,
        )

        assert update_id is not None
        assert update_id in updater.active_updates

        # Let it run briefly
        time.sleep(0.3)

        # Check status
        status = updater.get_update_status(update_id)
        assert status is not None
        assert status["is_running"]
        assert status["update_count"] > 0

        # Stop updates
        success = updater.stop_updates(update_id)
        assert success
        assert update_id not in updater.active_updates

        plt.close(fig)

    def test_chart_generator_types(self):
        """Test different chart generation types."""
        generator = ChartGenerator(console=self.console)

        # Test pressure distribution chart
        pressure_data = {
            "x": np.linspace(0, 1, 50),
            "cp_upper": -np.sin(np.pi * np.linspace(0, 1, 50)),
            "cp_lower": np.sin(np.pi * np.linspace(0, 1, 50)) * 0.5,
        }

        fig = generator.generate_chart(
            results=pressure_data,
            chart_type="pressure_distribution",
        )

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_export_quality_presets(self):
        """Test export quality presets."""
        plotter = InteractivePlotter(console=self.console)
        exporter = ExportManager(console=self.console)

        fig = plotter.create_plot(data=self.sample_data, plot_type="line")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test different quality presets
            presets = ["draft", "standard", "high", "publication"]

            for preset in presets:
                output_path = temp_path / f"test_{preset}.png"
                success = exporter.export_plot(
                    fig,
                    output_path,
                    format="png",
                    quality_preset=preset,
                )
                assert success
                assert output_path.exists()

        plt.close(fig)

    def test_customization_presets(self):
        """Test customization presets."""
        customizer = PlotCustomizer(console=self.console)

        # Create presets
        customizer.create_customization_preset(
            "test_preset",
            {"color_palette": "aerospace", "line_widths": 2, "grid": {"visible": True}},
        )

        presets = customizer.get_available_presets()
        assert "test_preset" in presets

        # Test template generation
        template = customizer.generate_customization_template("aerospace")
        assert "color_palette" in template
        assert template["color_palette"] == "aerospace"

    def test_batch_operations(self):
        """Test batch export and operations."""
        # Create multiple plots
        plot_ids = []
        for i in range(3):
            plot_id = self.viz_manager.create_interactive_plot(
                data={"x": list(range(10)), "y": [x**2 + i for x in range(10)]},
                plot_type="line",
                title=f"Batch Test Plot {i+1}",
            )
            plot_ids.append(plot_id)

        # Test batch export
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.viz_manager.batch_export(
                plot_ids=plot_ids,
                output_directory=temp_dir,
                format="png",
            )

            assert len(results) == 3
            assert all(results.values())  # All exports successful

        # Clean up
        for plot_id in plot_ids:
            self.viz_manager.close_plot(plot_id)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid plot type
        plotter = InteractivePlotter(console=self.console)

        with pytest.raises(ValueError):
            plotter.create_plot(data=self.sample_data, plot_type="invalid_type")

        # Test invalid chart type
        generator = ChartGenerator(console=self.console)

        with pytest.raises(ValueError):
            generator.generate_chart(
                results=self.airfoil_data,
                chart_type="invalid_chart",
            )

        # Test operations on non-existent plots
        success = self.viz_manager.customize_plot("non_existent", {})
        assert not success

        success = self.viz_manager.export_plot("non_existent", "test.png")
        assert not success


def run_demo():
    """Run a demonstration of the visualization system."""
    console = Console()
    console.print("[bold blue]ICARUS CLI Visualization System Demo[/bold blue]")

    # Initialize the visualization manager
    viz_manager = VisualizationManager(console=console)

    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    sample_data = {
        "x": x.tolist(),
        "y": {"sin(x)": y1.tolist(), "cos(x)": y2.tolist()},
        "xlabel": "X Values",
        "ylabel": "Y Values",
    }

    # Create interactive plot
    console.print("\n[cyan]Creating interactive plot...[/cyan]")
    plot_id = viz_manager.create_interactive_plot(
        data=sample_data,
        plot_type="line",
        title="Trigonometric Functions Demo",
    )

    # Apply customizations
    console.print("[cyan]Applying customizations...[/cyan]")
    customizations = {
        "color_palette": "aerospace",
        "line_widths": 2,
        "grid": {"visible": True, "alpha": 0.3},
        "legend": {"show": True, "loc": "upper right"},
    }
    viz_manager.customize_plot(plot_id, customizations)

    # Export plot
    console.print("[cyan]Exporting plot...[/cyan]")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "demo_plot.png"
        viz_manager.export_plot(plot_id, output_path, format="png")
        console.print(f"[green]✓[/green] Plot exported to {output_path}")

    # Generate airfoil polar chart
    console.print("\n[cyan]Generating airfoil polar chart...[/cyan]")
    airfoil_data = {
        "alpha": np.linspace(-5, 15, 21),
        "cl": 0.1 * np.linspace(-5, 15, 21),
        "cd": 0.006 + 0.0001 * np.linspace(-5, 15, 21) ** 2,
        "cm": -0.002 * np.linspace(-5, 15, 21),
    }

    chart_id = viz_manager.generate_chart(
        analysis_results=airfoil_data,
        chart_type="airfoil_polar",
    )

    # List active plots
    console.print("\n[cyan]Active plots:[/cyan]")
    active_plots = viz_manager.list_active_plots()
    for plot_info in active_plots:
        console.print(
            f"  - {plot_info['id']}: {plot_info['title']} ({plot_info['type']})",
        )

    # Clean up
    console.print("\n[cyan]Cleaning up...[/cyan]")
    viz_manager.close_all_plots()

    console.print("[bold green]Demo completed successfully![/bold green]")


if __name__ == "__main__":
    # Run tests
    test_suite = TestVisualizationSystem()
    test_suite.setup_method()

    try:
        test_suite.test_interactive_plotter_creation()
        test_suite.test_chart_generator_airfoil_polar()
        test_suite.test_visualization_manager_integration()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")

    # Run demo
    print("\nRunning demo...")
    run_demo()
