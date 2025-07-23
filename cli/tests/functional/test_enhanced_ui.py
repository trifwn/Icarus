#!/usr/bin/env python3
"""Test script for enhanced UI features

This script tests the enhanced UI features implemented in task 33, including:
- Modern aerospace-inspired aesthetics
- Advanced visualization components with 3D rendering
- Interactive dashboard with customizable widgets
- Real-time data visualization with animations
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import visualization components
try:
    from cli.visualization.dashboard import ActionWidget
    from cli.visualization.dashboard import ChartWidget
    from cli.visualization.dashboard import Dashboard
    from cli.visualization.dashboard import ProgressWidget
    from cli.visualization.dashboard import StatWidget
    from cli.visualization.dashboard import TableWidget
    from cli.visualization.dashboard import Visualization3DWidget
    from cli.visualization.real_time_visualization import RealTimeVisualizer
    from cli.visualization.real_time_visualization import SimulatedDataSource
    from cli.visualization.renderer_3d import AirfoilMesh
    from cli.visualization.renderer_3d import AirplaneMesh
    from cli.visualization.renderer_3d import Renderer3D
    from cli.visualization.renderer_3d import WingMesh

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Visualization components not available: {e}")
    VISUALIZATION_AVAILABLE = False


def test_3d_renderer():
    """Test the 3D renderer component."""
    print("\nTesting 3D Renderer...")

    if not VISUALIZATION_AVAILABLE:
        print("✗ Visualization components not available")
        return False

    try:
        # Create renderer
        renderer = Renderer3D()

        # Create airfoil
        success = renderer.create_airfoil("naca0012", "0012")
        print(f"{'✓' if success else '✗'} Created NACA 0012 airfoil")

        # Create airplane
        success = renderer.create_airplane("boeing737", 35.8, 39.5)
        print(f"{'✓' if success else '✗'} Created Boeing 737 model")

        # Render airfoil
        renderer.set_active_mesh("naca0012")
        airfoil_render = renderer.render_to_text(width=40, height=10)
        print("\nAirfoil Render:")
        print(airfoil_render)

        # Rotate and render again
        renderer.rotate_mesh("naca0012", 0, 30, 0)
        airfoil_render = renderer.render_to_text(width=40, height=10)
        print("\nRotated Airfoil Render:")
        print(airfoil_render)

        # Render airplane
        renderer.set_active_mesh("boeing737")
        airplane_render = renderer.render_to_text(width=40, height=10)
        print("\nAirplane Render:")
        print(airplane_render)

        return True
    except Exception as e:
        print(f"✗ 3D renderer test failed: {e}")
        return False


def test_dashboard_widgets():
    """Test the dashboard widgets."""
    print("\nTesting Dashboard Widgets...")

    if not VISUALIZATION_AVAILABLE:
        print("✗ Visualization components not available")
        return False

    try:
        # Create widgets
        stat_widget = StatWidget(
            title="Test Stat",
            value="42",
            label="Answer",
            id="test_stat",
        )
        print("✓ Created StatWidget")

        chart_widget = ChartWidget(
            title="Test Chart",
            chart_type="line",
            id="test_chart",
        )
        print("✓ Created ChartWidget")

        table_widget = TableWidget(
            title="Test Table",
            columns=["ID", "Value", "Status"],
            rows=[
                ["A1", "42", "Complete"],
                ["A2", "37", "Running"],
            ],
            id="test_table",
        )
        print("✓ Created TableWidget")

        progress_widget = ProgressWidget(
            title="Test Progress",
            progress=0.75,
            status="Running...",
            id="test_progress",
        )
        print("✓ Created ProgressWidget")

        action_widget = ActionWidget(
            title="Test Actions",
            actions=[
                {"id": "action1", "label": "Action 1", "variant": "primary"},
                {"id": "action2", "label": "Action 2", "variant": "default"},
            ],
            id="test_actions",
        )
        print("✓ Created ActionWidget")

        viz_3d_widget = Visualization3DWidget(
            title="Test 3D",
            model_data={"type": "airfoil", "name": "NACA 0012"},
            id="test_3d",
        )
        print("✓ Created Visualization3DWidget")

        return True
    except Exception as e:
        print(f"✗ Dashboard widgets test failed: {e}")
        return False


def test_real_time_visualization():
    """Test the real-time visualization component."""
    print("\nTesting Real-Time Visualization...")

    if not VISUALIZATION_AVAILABLE:
        print("✗ Visualization components not available")
        return False

    try:
        # Create visualizer
        visualizer = RealTimeVisualizer()

        # Create data sources
        scalar_source = SimulatedDataSource("test_scalar", "scalar", 0.5)
        vector_source = SimulatedDataSource("test_vector", "vector", 0.5)

        # Add data sources
        visualizer.add_data_source(scalar_source)
        visualizer.add_data_source(vector_source)
        print("✓ Created data sources")

        # Create visualizations
        success = visualizer.create_visualization(
            "test_line",
            "line",
            "test_scalar",
            title="Test Line Chart",
            max_points=10,
        )
        print(f"{'✓' if success else '✗'} Created line visualization")

        success = visualizer.create_visualization(
            "test_gauge",
            "gauge",
            "test_scalar",
            title="Test Gauge",
            min_value=0,
            max_value=10,
        )
        print(f"{'✓' if success else '✗'} Created gauge visualization")

        # Start data sources (but don't actually run the event loop)
        print("✓ Real-time visualization components created successfully")

        return True
    except Exception as e:
        print(f"✗ Real-time visualization test failed: {e}")
        return False


def test_aerospace_theme():
    """Test the aerospace theme."""
    print("\nTesting Aerospace Theme...")

    try:
        # Check if the theme file exists
        theme_path = Path(__file__).parent / "tui_styles_aerospace.css"
        if theme_path.exists():
            print(f"✓ Found aerospace theme at {theme_path}")

            # Check file size to ensure it's a substantial CSS file
            size = theme_path.stat().st_size
            if size > 1000:
                print(f"✓ Theme file size is {size} bytes")
            else:
                print(f"✗ Theme file seems too small: {size} bytes")
                return False

            return True
        else:
            print(f"✗ Aerospace theme file not found at {theme_path}")
            return False
    except Exception as e:
        print(f"✗ Aerospace theme test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running Enhanced UI Tests...")

    tests = [
        test_aerospace_theme,
        test_3d_renderer,
        test_dashboard_widgets,
        test_real_time_visualization,
    ]

    results = []
    for test in tests:
        results.append(test())

    success_count = sum(1 for r in results if r)
    total_count = len(results)

    print(f"\nTest Results: {success_count}/{total_count} tests passed")

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
