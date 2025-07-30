#!/usr/bin/env python3
"""
Test script for ICARUS integration layer.

This script tests the core functionality of the integration layer
including solver discovery, parameter validation, and analysis execution.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add CLI to path
cli_path = Path(__file__).parent.parent
if str(cli_path) not in sys.path:
    sys.path.insert(0, str(cli_path))

from integration import AnalysisConfig
from integration import AnalysisService
from integration import AnalysisType
from integration import SolverType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_solver_discovery():
    """Test solver discovery functionality."""
    print("\n=== Testing Solver Discovery ===")

    service = AnalysisService()

    # Test getting all solvers
    all_solvers = service.get_available_solvers()
    print(f"Found {len(all_solvers)} available solvers:")
    for solver in all_solvers:
        print(
            f"  - {solver['name']} ({solver['type']}) - Fidelity: {solver['fidelity']}",
        )

    # Test getting solvers for specific analysis
    airfoil_solvers = service.get_solvers_for_analysis(AnalysisType.AIRFOIL_POLAR)
    print(f"\nSolvers for airfoil polar analysis: {len(airfoil_solvers)}")
    for solver in airfoil_solvers:
        print(f"  - {solver['name']} (Available: {solver['is_available']})")

    # Test recommended solver
    recommended = service.get_recommended_solver(AnalysisType.AIRFOIL_POLAR)
    if recommended:
        print(f"\nRecommended solver for airfoil polar: {recommended['name']}")
    else:
        print("\nNo recommended solver available for airfoil polar")

    return len(all_solvers) > 0


def test_parameter_validation():
    """Test parameter validation functionality."""
    print("\n=== Testing Parameter Validation ===")

    service = AnalysisService()

    # Test valid configuration
    valid_config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL_POLAR,
        solver_type=SolverType.XFLR5,
        target="NACA0012",
        parameters={
            "reynolds": 1e6,
            "mach": 0.1,
            "min_aoa": -10,
            "max_aoa": 15,
            "aoa_step": 0.5,
        },
    )

    validation_result = service.validate_analysis_config(valid_config)
    print(
        f"Valid config validation: {'PASS' if validation_result.is_valid else 'FAIL'}",
    )
    if not validation_result.is_valid:
        for error in validation_result.errors:
            print(f"  Error: {error.field} - {error.message}")

    # Test invalid configuration
    invalid_config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL_POLAR,
        solver_type=SolverType.XFOIL,
        target="",  # Empty target
        parameters={
            "reynolds": -1000,  # Invalid Reynolds number
            "mach": 1.5,  # Invalid Mach number
        },
    )

    validation_result = service.validate_analysis_config(invalid_config)
    print(
        f"Invalid config validation: {'FAIL' if not validation_result.is_valid else 'UNEXPECTED PASS'}",
    )
    print(f"Found {len(validation_result.errors)} validation errors:")
    for error in validation_result.errors:
        print(f"  - {error.field}: {error.message}")

    # Test parameter suggestions
    suggestions = service.get_parameter_suggestions(AnalysisType.AIRFOIL_POLAR)
    print(f"\nParameter suggestions for airfoil polar: {len(suggestions)} items")
    for key, value in suggestions.items():
        if not key.endswith("_description"):
            print(f"  - {key}: {value}")

    return True


async def test_analysis_execution():
    """Test analysis execution functionality."""
    print("\n=== Testing Analysis Execution ===")

    service = AnalysisService()

    # Create a simple analysis configuration
    config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL_POLAR,
        solver_type=SolverType.XFLR5,
        target="NACA0012",
        parameters={
            "reynolds": 1e6,
            "mach": 0.0,
            "min_aoa": -5,
            "max_aoa": 10,
            "aoa_step": 1.0,
        },
    )

    # Progress callback
    def progress_callback(progress):
        print(f"  Progress: {progress.progress_percent:.1f}% - {progress.current_step}")

    try:
        print("Starting analysis...")
        result = await service.run_analysis(config, progress_callback)

        print(f"Analysis completed with status: {result.status}")
        if result.is_successful:
            print(f"Analysis duration: {result.duration:.2f} seconds")
            print(
                "Raw data keys:",
                list(result.raw_data.keys())
                if isinstance(result.raw_data, dict)
                else "Non-dict data",
            )
        else:
            print(f"Analysis failed: {result.error_message}")

        return result.is_successful

    except Exception as e:
        print(f"Analysis execution failed: {e}")
        return False


def test_result_processing():
    """Test result processing functionality."""
    print("\n=== Testing Result Processing ===")

    service = AnalysisService()

    # Create a mock analysis result
    from datetime import datetime

    from integration.models import AnalysisResult

    config = AnalysisConfig(
        analysis_type=AnalysisType.AIRFOIL_POLAR,
        solver_type=SolverType.XFOIL,
        target="NACA0012",
        parameters={"reynolds": 1e6, "mach": 0.0},
    )

    # Mock result data
    import numpy as np

    alpha = np.linspace(-5, 10, 16)
    cl = 2 * np.pi * np.sin(np.radians(alpha)) * np.cos(np.radians(alpha))
    cd = 0.01 + 0.02 * (np.radians(alpha)) ** 2
    cm = -0.1 * np.ones_like(alpha)

    mock_result = AnalysisResult(
        analysis_id="test-123",
        config=config,
        status="success",
        start_time=datetime.now(),
        end_time=datetime.now(),
        raw_data={
            "polars": {
                "alpha": alpha,
                "cl": cl,
                "cd": cd,
                "cm": cm,
            },
        },
    )

    try:
        processed = service.process_result(mock_result)
        print("Result processing: SUCCESS")
        print(f"Formatted data keys: {list(processed.formatted_data.keys())}")
        print(f"Number of plots: {len(processed.plots)}")
        print(f"Number of tables: {len(processed.tables)}")
        print(f"Export formats: {processed.export_formats}")

        # Test summary
        if processed.summary:
            print(f"Summary: {processed.summary.get('analysis_type', 'Unknown')}")

        return True

    except Exception as e:
        print(f"Result processing failed: {e}")
        return False


def test_system_status():
    """Test system status reporting."""
    print("\n=== Testing System Status ===")

    service = AnalysisService()
    status = service.get_system_status()

    print(f"ICARUS Available: {status['icarus_available']}")
    print(f"Service Status: {status['service_status']}")
    print(f"Running Analyses: {status['running_analyses']}")
    print(f"Supported Analysis Types: {len(status['supported_analyses'])}")

    solver_status = status["solver_status"]
    print(f"Total Solvers: {solver_status['total_solvers']}")
    print(f"Available Solvers: {solver_status['available_solvers']}")

    return True


async def main():
    """Run all tests."""
    print("ICARUS Integration Layer Test Suite")
    print("=" * 50)

    tests = [
        ("Solver Discovery", test_solver_discovery),
        ("Parameter Validation", test_parameter_validation),
        ("Analysis Execution", test_analysis_execution),
        ("Result Processing", test_result_processing),
        ("System Status", test_system_status),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
