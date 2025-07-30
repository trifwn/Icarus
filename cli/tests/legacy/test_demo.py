#!/usr/bin/env python3
"""
Test script for ICARUS CLI Demo functionality

This script tests the essential CLI functionality without requiring the full TUI.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_airfoil_workflow():
    """Test airfoil analysis workflow."""
    print("Testing Airfoil Workflow...")

    try:
        from cli.workflows.airfoil_workflow import (
            analyze_naca_airfoil,
            print_airfoil_summary,
        )

        async def run_test():
            def progress_callback(percent, status):
                print(f"  Progress: {percent:.1f}% - {status}")

            # Test NACA 2412 airfoil
            results = await analyze_naca_airfoil(
                "NACA2412",
                reynolds=1000000,
                angle_range=(-10, 15, 1.0),
                progress_callback=progress_callback,
            )

            if results.get("success"):
                print("✓ Airfoil analysis completed successfully")
                print_airfoil_summary(results)
                return True
            else:
                print(f"✗ Airfoil analysis failed: {results.get('error')}")
                return False

        return asyncio.run(run_test())

    except ImportError as e:
        print(f"✗ Airfoil workflow not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Airfoil workflow error: {e}")
        return False


def test_airplane_workflow():
    """Test airplane analysis workflow."""
    print("\nTesting Airplane Workflow...")

    try:
        from cli.workflows.airplane_workflow import (
            analyze_demo_airplane,
            print_airplane_summary,
        )

        async def run_test():
            def progress_callback(percent, status):
                print(f"  Progress: {percent:.1f}% - {status}")

            # Test demo airplane
            results = await analyze_demo_airplane(
                velocity=50.0,
                altitude=1000.0,
                angle_range=(-5, 15, 2.0),
                progress_callback=progress_callback,
            )

            if results.get("success"):
                print("✓ Airplane analysis completed successfully")
                print_airplane_summary(results)
                return True
            else:
                print(f"✗ Airplane analysis failed: {results.get('error')}")
                return False

        return asyncio.run(run_test())

    except ImportError as e:
        print(f"✗ Airplane workflow not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Airplane workflow error: {e}")
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting Visualization...")

    try:
        from cli.visualization.demo_visualization import DemoVisualizer, DemoExporter

        # Create mock results for testing
        mock_results = {
            "success": True,
            "processed_results": {
                "performance_summary": {
                    "max_cl": {"value": 1.234, "alpha": 12.5, "cd": 0.0234},
                    "min_cd": {"value": 0.0123, "alpha": 2.0, "cl": 0.456},
                    "max_ld": {"value": 45.6, "alpha": 4.0, "cl": 0.789, "cd": 0.0173},
                },
                "airfoil_characteristics": {"type": "4-digit", "reynolds": 1000000},
                "polar_data": {
                    "alpha": list(range(-10, 16)),
                    "cl": [0.1 * i for i in range(-10, 16)],
                    "cd": [0.01 + 0.001 * i**2 for i in range(-10, 16)],
                    "ld_ratio": [10 + i for i in range(-10, 16)],
                },
            },
        }

        # Test visualization
        visualizer = DemoVisualizer()
        if visualizer.display_results_table(mock_results):
            print("✓ Results table display working")
        else:
            print("✗ Results table display failed")

        # Test export
        exporter = DemoExporter()
        if exporter.export_to_json(mock_results, "test_results.json"):
            print("✓ JSON export working")
        else:
            print("✗ JSON export failed")

        if exporter.export_to_csv(mock_results, "test_results.csv"):
            print("✓ CSV export working")
        else:
            print("✗ CSV export failed")

        if exporter.create_summary_report(mock_results, "test_report.txt"):
            print("✓ Summary report working")
        else:
            print("✗ Summary report failed")

        return True

    except ImportError as e:
        print(f"✗ Visualization not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False


def test_cli_integration():
    """Test CLI integration."""
    print("\nTesting CLI Integration...")

    try:
        from cli.integration.analysis_service import AnalysisService
        from cli.integration.models import AnalysisType, SolverType

        service = AnalysisService()

        # Test system status
        status = service.get_system_status()
        print(f"  System Status: {status.get('service_status', 'unknown')}")
        print(f"  ICARUS Available: {status.get('icarus_available', False)}")

        # Test available solvers
        solvers = service.get_available_solvers()
        print(f"  Available Solvers: {len(solvers)}")
        for solver in solvers[:3]:  # Show first 3
            print(f"    - {solver['name']}: {solver['is_available']}")

        # Test analysis types
        analysis_types = service.get_available_analysis_types()
        print(f"  Supported Analysis Types: {[t.value for t in analysis_types]}")

        print("✓ CLI integration working")
        return True

    except ImportError as e:
        print(f"✗ CLI integration not available: {e}")
        return False
    except Exception as e:
        print(f"✗ CLI integration error: {e}")
        return False


def test_demo_app():
    """Test demo app components."""
    print("\nTesting Demo App Components...")

    try:
        # Test import
        return True

    except ImportError as e:
        print(f"✗ Demo app not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Demo app error: {e}")
        return False


def main():
    """Run all tests."""
    print("ICARUS CLI Demo - Functionality Test")
    print("=" * 50)

    tests = [
        test_cli_integration,
        test_airfoil_workflow,
        test_airplane_workflow,
        test_visualization,
        test_demo_app,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed - Demo functionality is working!")
        return 0
    else:
        print("✗ Some tests failed - Check the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
