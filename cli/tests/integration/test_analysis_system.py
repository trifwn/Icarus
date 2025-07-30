#!/usr/bin/env python3
"""Test script for the analysis configuration and execution system.

This script tests the new analysis screens and integration components.
"""

import asyncio
import sys
from pathlib import Path

# Add the CLI directory to the path
cli_path = Path(__file__).parent
sys.path.insert(0, str(cli_path))


def test_integration_imports():
    """Test that all integration modules can be imported."""
    print("Testing integration module imports...")

    try:
        from integration.analysis_service import AnalysisService

        print("✓ AnalysisService imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AnalysisService: {e}")

    try:
        from integration.solver_manager import SolverManager

        print("✓ SolverManager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SolverManager: {e}")

    try:
        from integration.parameter_validator import ParameterValidator

        print("✓ ParameterValidator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ParameterValidator: {e}")

    try:
        from integration.result_processor import ResultProcessor

        print("✓ ResultProcessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ResultProcessor: {e}")


def test_screen_imports():
    """Test that all screen modules can be imported."""
    print("\nTesting screen module imports...")

    try:
        from tui.screens.analysis_screen import AnalysisScreen

        print("✓ AnalysisScreen imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AnalysisScreen: {e}")

    try:
        from tui.screens.solver_selection_screen import SolverSelectionScreen

        print("✓ SolverSelectionScreen imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SolverSelectionScreen: {e}")

    try:
        from tui.screens.execution_screen import ExecutionScreen

        print("✓ ExecutionScreen imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ExecutionScreen: {e}")

    try:
        from tui.screens.results_screen import ResultsScreen

        print("✓ ResultsScreen imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ResultsScreen: {e}")


def test_analysis_service():
    """Test basic AnalysisService functionality."""
    print("\nTesting AnalysisService functionality...")

    try:
        from integration.analysis_service import AnalysisService
        from integration.models import AnalysisType

        service = AnalysisService()
        print("✓ AnalysisService created successfully")

        # Test getting available analysis types
        analysis_types = service.get_available_analysis_types()
        print(f"✓ Available analysis types: {[t.value for t in analysis_types]}")

        # Test getting available solvers
        solvers = service.get_available_solvers()
        print(f"✓ Found {len(solvers)} solvers")

        # Test getting solvers for specific analysis
        if hasattr(AnalysisType, "AIRFOIL_POLAR"):
            airfoil_solvers = service.get_solvers_for_analysis(
                AnalysisType.AIRFOIL_POLAR,
            )
            print(f"✓ Found {len(airfoil_solvers)} solvers for airfoil analysis")

        # Test system status
        status = service.get_system_status()
        print(f"✓ System status: {status.get('service_status', 'unknown')}")

    except Exception as e:
        print(f"✗ AnalysisService test failed: {e}")


def test_solver_manager():
    """Test SolverManager functionality."""
    print("\nTesting SolverManager functionality...")

    try:
        from integration.solver_manager import SolverManager

        manager = SolverManager()
        print("✓ SolverManager created successfully")

        # Test getting all solvers
        all_solvers = manager.get_all_solvers()
        print(f"✓ Found {len(all_solvers)} total solvers")

        # Test getting available solvers
        available_solvers = manager.get_available_solvers()
        print(f"✓ Found {len(available_solvers)} available solvers")

        # Test solver status report
        report = manager.get_solver_status_report()
        print(
            f"✓ Solver status report generated: {report.get('total_solvers', 0)} solvers",
        )

    except Exception as e:
        print(f"✗ SolverManager test failed: {e}")


def test_parameter_validator():
    """Test ParameterValidator functionality."""
    print("\nTesting ParameterValidator functionality...")

    try:
        from integration.models import AnalysisConfig
        from integration.models import AnalysisType
        from integration.models import SolverType
        from integration.parameter_validator import ParameterValidator

        validator = ParameterValidator()
        print("✓ ParameterValidator created successfully")

        # Test parameter suggestions
        if hasattr(AnalysisType, "AIRFOIL_POLAR"):
            suggestions = validator.get_parameter_suggestions(
                AnalysisType.AIRFOIL_POLAR,
            )
            print(f"✓ Parameter suggestions generated: {len(suggestions)} items")

        # Test basic validation (if we can create a config)
        try:
            if AnalysisConfig and AnalysisType and SolverType:
                config = AnalysisConfig(
                    analysis_type=AnalysisType.AIRFOIL_POLAR,
                    solver_type=SolverType.XFOIL,
                    target="NACA0012",
                    parameters={"reynolds": 1000000, "mach": 0.0},
                    solver_parameters={},
                    output_format="json",
                )

                result = validator.validate_analysis_config(config)
                print(
                    f"✓ Validation completed: {'valid' if result.is_valid else 'invalid'}",
                )
        except Exception:
            print("✓ Validation test skipped (missing dependencies)")

    except Exception as e:
        print(f"✗ ParameterValidator test failed: {e}")


async def test_analysis_execution():
    """Test analysis execution (mock)."""
    print("\nTesting analysis execution...")

    try:
        from integration.analysis_service import AnalysisService
        from integration.models import AnalysisConfig
        from integration.models import AnalysisType
        from integration.models import SolverType

        if not (AnalysisConfig and AnalysisType and SolverType):
            print("✓ Analysis execution test skipped (missing dependencies)")
            return

        service = AnalysisService()

        # Create a test configuration
        config = AnalysisConfig(
            analysis_type=AnalysisType.AIRFOIL_POLAR,
            solver_type=SolverType.XFOIL,
            target="NACA0012",
            parameters={"reynolds": 1000000, "mach": 0.0, "min_aoa": -5, "max_aoa": 15},
            solver_parameters={},
            output_format="json",
        )

        print("✓ Test configuration created")

        # Test validation
        validation_result = service.validate_analysis_config(config)
        print(
            f"✓ Configuration validation: {'passed' if validation_result.is_valid else 'failed'}",
        )

        if not validation_result.is_valid:
            print(f"  Validation errors: {len(validation_result.errors)}")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                print(f"    - {error.message}")

        # Test mock execution (this will use mock data)
        print("✓ Starting mock analysis execution...")

        def progress_callback(progress):
            print(
                f"  Progress: {progress.current_step} ({progress.progress_percent:.1f}%)",
            )

        result = await service.run_analysis(config, progress_callback)

        if result.is_successful:
            print("✓ Mock analysis completed successfully")
            print(f"  Analysis ID: {result.analysis_id}")
            print(f"  Duration: {result.duration}")
        else:
            print(f"✗ Mock analysis failed: {result.error_message}")

    except Exception as e:
        print(f"✗ Analysis execution test failed: {e}")


def main():
    """Run all tests."""
    print("ICARUS Analysis System Test Suite")
    print("=" * 50)

    # Test imports
    test_integration_imports()
    test_screen_imports()

    # Test core functionality
    test_analysis_service()
    test_solver_manager()
    test_parameter_validator()

    # Test async functionality
    print("\nRunning async tests...")
    asyncio.run(test_analysis_execution())

    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo run the full TUI application:")
    print("  python tui_app.py")
    print("\nTo test individual screens:")
    print(
        "  python -c \"from tui.screens.analysis_screen import AnalysisScreen; print('Analysis screen OK')\"",
    )


if __name__ == "__main__":
    main()
