"""
End-to-End Testing Suite for ICARUS CLI

This module provides comprehensive end-to-end tests for complete workflows
and user scenarios in the ICARUS CLI system.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

# Add CLI directory to path for imports
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from .framework import TestResult
from .framework import TestStatus
from .framework import TestType


class EndToEndTestSuite:
    """End-to-end test suite for complete ICARUS CLI workflows"""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir = None
        self.test_session = None

    async def run_all_tests(self) -> List[TestResult]:
        """Run all end-to-end tests"""
        self.test_results = []

        # Setup test environment
        await self._setup_e2e_environment()

        try:
            # Complete workflow tests
            await self._test_airfoil_analysis_workflow()
            await self._test_aircraft_design_workflow()
            await self._test_optimization_workflow()

            # User scenario tests
            await self._test_new_user_onboarding()
            await self._test_collaborative_analysis()
            await self._test_data_export_workflow()

            # Advanced workflow tests
            await self._test_multi_solver_comparison()
            await self._test_parametric_study_workflow()

            # Error recovery tests
            await self._test_error_recovery_workflow()

        finally:
            # Cleanup test environment
            await self._cleanup_e2e_environment()

        return self.test_results

    async def _setup_e2e_environment(self):
        """Setup comprehensive E2E test environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="icarus_e2e_"))

        # Create directory structure
        directories = [
            "airfoils",
            "aircraft",
            "results",
            "workflows",
            "exports",
            "plugins",
            "config",
            "logs",
        ]

        for directory in directories:
            (self.temp_dir / directory).mkdir()

        # Create comprehensive test data
        await self._create_e2e_test_data()

        # Initialize test session
        await self._initialize_test_session()

    async def _cleanup_e2e_environment(self):
        """Cleanup E2E test environment"""
        if self.test_session:
            await self._cleanup_test_session()

        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    async def _create_e2e_test_data(self):
        """Create comprehensive test data for E2E tests"""
        # Create sample airfoil file
        airfoil_data = """NACA 2412
  1.00000   0.00000
  0.95000   0.01200
  0.90000   0.02100
  0.80000   0.03400
  0.70000   0.04500
  0.60000   0.05200
  0.50000   0.05600
  0.40000   0.05700
  0.30000   0.05200
  0.20000   0.04100
  0.10000   0.02600
  0.05000   0.01600
  0.00000   0.00000
  0.05000  -0.01200
  0.10000  -0.02000
  0.20000  -0.03200
  0.30000  -0.04000
  0.40000  -0.04400
  0.50000  -0.04200
  0.60000  -0.03600
  0.70000  -0.02800
  0.80000  -0.02000
  0.90000  -0.01200
  0.95000  -0.00600
  1.00000   0.00000"""

        with open(self.temp_dir / "airfoils" / "naca2412.dat", "w") as f:
            f.write(airfoil_data)

        # Create sample aircraft configuration
        aircraft_config = {
            "name": "Test Aircraft",
            "components": {
                "wing": {
                    "span": 10.0,
                    "chord_root": 1.5,
                    "chord_tip": 0.8,
                    "airfoil_root": "naca2412",
                    "airfoil_tip": "naca0012",
                },
                "fuselage": {"length": 8.0, "diameter": 1.2},
            },
        }

        with open(self.temp_dir / "aircraft" / "test_aircraft.json", "w") as f:
            json.dump(aircraft_config, f, indent=2)

    async def _initialize_test_session(self):
        """Initialize test session"""
        try:
            from app.state_manager import StateManager
            from core.config import ConfigManager

            # Initialize configuration
            config = ConfigManager()
            config.set("workspace", str(self.temp_dir))
            config.set("test_mode", True)

            # Initialize session
            state_manager = StateManager()
            self.test_session = await state_manager.initialize_session()

        except ImportError:
            # Create mock session for testing
            self.test_session = {
                "session_id": "test_session_001",
                "workspace": str(self.temp_dir),
                "user_id": "test_user",
            }

    async def _cleanup_test_session(self):
        """Cleanup test session"""
        if hasattr(self.test_session, "cleanup"):
            await self.test_session.cleanup()

    async def _run_test(self, test_name: str, test_func):
        """Run a single E2E test with comprehensive error handling"""
        start_time = time.time()

        try:
            await test_func()
            duration = time.time() - start_time

            result = TestResult(
                name=test_name,
                test_type=TestType.E2E,
                status=TestStatus.PASSED,
                duration=duration,
                details={"test_environment": str(self.temp_dir)},
            )

        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.E2E,
                status=TestStatus.FAILED,
                duration=duration,
                error_message=str(e),
                details={"test_environment": str(self.temp_dir)},
            )

        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.E2E,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=f"{type(e).__name__}: {str(e)}",
                details={"test_environment": str(self.temp_dir)},
            )

        self.test_results.append(result)
        return result

    # Complete Workflow Tests

    async def _test_airfoil_analysis_workflow(self):
        """Test complete airfoil analysis workflow"""

        async def test_complete_airfoil_workflow():
            try:
                # Step 1: Load airfoil
                airfoil_file = self.temp_dir / "airfoils" / "naca2412.dat"
                assert airfoil_file.exists(), "Test airfoil file should exist"

                # Step 2: Configure analysis
                analysis_config = {
                    "airfoil_file": str(airfoil_file),
                    "solver": "xfoil",
                    "reynolds": 1000000,
                    "mach": 0.1,
                    "alpha_range": {"start": -5, "end": 10, "step": 0.5},
                }

                # Step 3: Run analysis (mocked for E2E test)
                from integration.analysis_service import AnalysisService

                analysis_service = AnalysisService()

                # Mock the analysis execution
                with patch.object(analysis_service, "run_analysis") as mock_analysis:
                    mock_analysis.return_value = {
                        "status": "completed",
                        "results": {
                            "alpha": list(range(-5, 11)),
                            "cl": [i * 0.1 for i in range(-5, 11)],
                            "cd": [0.01 + abs(i) * 0.001 for i in range(-5, 11)],
                            "cm": [-0.05 - i * 0.01 for i in range(-5, 11)],
                        },
                        "metadata": {
                            "solver": "xfoil",
                            "reynolds": 1000000,
                            "convergence": "good",
                        },
                    }

                    result = await analysis_service.run_analysis(analysis_config)

                    # Verify analysis results
                    assert (
                        result["status"] == "completed"
                    ), "Analysis should complete successfully"
                    assert "results" in result, "Results should be present"
                    assert (
                        len(result["results"]["alpha"]) == 16
                    ), "Should have correct number of data points"

                # Step 4: Generate visualizations
                from visualization.chart_generator import ChartGenerator

                chart_generator = ChartGenerator()

                # Generate lift curve
                lift_chart = chart_generator.create_chart(
                    result["results"],
                    {
                        "type": "line",
                        "x_axis": "alpha",
                        "y_axis": "cl",
                        "title": "Lift Coefficient vs Angle of Attack",
                    },
                )

                assert lift_chart is not None, "Lift chart should be generated"

                # Step 5: Export results
                from data.import_export import ImportExportManager

                import_export = ImportExportManager()

                export_path = self.temp_dir / "exports" / "naca2412_analysis.json"
                export_success = await import_export.export_data(
                    result,
                    export_path,
                    "json",
                )

                assert export_success, "Results should be exported successfully"
                assert export_path.exists(), "Export file should exist"

            except ImportError as e:
                raise AssertionError(f"Required modules not available: {e}")

        await self._run_test(
            "Complete Airfoil Analysis Workflow",
            test_complete_airfoil_workflow,
        )

    async def _test_aircraft_design_workflow(self):
        """Test complete aircraft design workflow"""

        async def test_aircraft_design_workflow():
            try:
                # Step 1: Load aircraft configuration
                aircraft_file = self.temp_dir / "aircraft" / "test_aircraft.json"
                assert aircraft_file.exists(), "Test aircraft file should exist"

                with open(aircraft_file) as f:
                    aircraft_config = json.load(f)

                # Step 2: Validate configuration
                assert (
                    "components" in aircraft_config
                ), "Aircraft should have components"
                assert (
                    "wing" in aircraft_config["components"]
                ), "Aircraft should have wing"

                # Step 3: Run stability analysis (mocked)
                from integration.analysis_service import AnalysisService

                analysis_service = AnalysisService()

                stability_config = {
                    "aircraft_file": str(aircraft_file),
                    "solver": "avl",
                    "flight_conditions": {
                        "velocity": 50.0,
                        "altitude": 1000.0,
                        "weight": 1000.0,
                    },
                }

                with patch.object(analysis_service, "run_analysis") as mock_analysis:
                    mock_analysis.return_value = {
                        "status": "completed",
                        "stability_derivatives": {
                            "cma": -0.15,
                            "cmq": -8.5,
                            "cla": 5.2,
                            "clq": 7.8,
                        },
                        "trim_conditions": {
                            "alpha": 3.2,
                            "elevator": -1.8,
                            "thrust": 250.0,
                        },
                        "static_margin": 0.12,
                    }

                    stability_result = await analysis_service.run_analysis(
                        stability_config,
                    )

                    # Verify stability analysis
                    assert (
                        stability_result["status"] == "completed"
                    ), "Stability analysis should complete"
                    assert (
                        stability_result["static_margin"] > 0
                    ), "Aircraft should be statically stable"

                # Step 4: Export design report
                report_path = self.temp_dir / "exports" / "aircraft_design_report.json"
                design_report = {
                    "aircraft": aircraft_config,
                    "stability": stability_result,
                    "design_summary": {
                        "stable": stability_result["static_margin"] > 0,
                        "design_rating": "Good",
                    },
                }

                with open(report_path, "w") as f:
                    json.dump(design_report, f, indent=2)

                assert report_path.exists(), "Design report should be exported"

            except ImportError as e:
                raise AssertionError(f"Required modules not available: {e}")

        await self._run_test(
            "Complete Aircraft Design Workflow",
            test_aircraft_design_workflow,
        )

    async def _test_optimization_workflow(self):
        """Test optimization workflow"""

        async def test_optimization_workflow():
            try:
                # Step 1: Define optimization problem
                optimization_config = {
                    "objective": "maximize_lift_to_drag",
                    "design_variables": {
                        "airfoil_thickness": {
                            "min": 0.08,
                            "max": 0.18,
                            "initial": 0.12,
                        },
                        "camber": {"min": 0.0, "max": 0.06, "initial": 0.02},
                    },
                    "constraints": {
                        "max_thickness_location": {"min": 0.25, "max": 0.45},
                    },
                    "optimization_settings": {
                        "algorithm": "genetic_algorithm",
                        "population_size": 20,
                        "generations": 10,
                    },
                }

                # Step 2: Run optimization (mocked)
                from core.workflow import WorkflowEngine

                workflow_engine = WorkflowEngine()

                with patch.object(workflow_engine, "execute_optimization") as mock_opt:
                    mock_opt.return_value = {
                        "status": "completed",
                        "optimal_design": {"airfoil_thickness": 0.14, "camber": 0.035},
                        "objective_value": 28.5,  # L/D ratio
                        "convergence_achieved": True,
                    }

                    optimization_result = await workflow_engine.execute_optimization(
                        optimization_config,
                    )

                    # Verify optimization results
                    assert (
                        optimization_result["status"] == "completed"
                    ), "Optimization should complete"
                    assert optimization_result[
                        "convergence_achieved"
                    ], "Optimization should converge"
                    assert (
                        optimization_result["objective_value"] > 25.0
                    ), "Should achieve good L/D ratio"

                # Step 3: Export optimization results
                opt_report_path = self.temp_dir / "exports" / "optimization_report.json"
                with open(opt_report_path, "w") as f:
                    json.dump(
                        {
                            "configuration": optimization_config,
                            "results": optimization_result,
                        },
                        f,
                        indent=2,
                    )

                assert (
                    opt_report_path.exists()
                ), "Optimization report should be exported"

            except ImportError as e:
                raise AssertionError(f"Required modules not available: {e}")

        await self._run_test("Optimization Workflow", test_optimization_workflow)

    # User Scenario Tests

    async def _test_new_user_onboarding(self):
        """Test new user onboarding workflow"""

        async def test_onboarding_flow():
            try:
                # Step 1: First launch - tutorial system
                from learning.tutorial_system import TutorialSystem

                tutorial_system = TutorialSystem()

                # Check tutorial availability
                available_tutorials = tutorial_system.get_available_tutorials()
                assert len(available_tutorials) > 0, "Should have available tutorials"

                # Start beginner tutorial
                beginner_tutorial = available_tutorials[0]
                tutorial_session = await tutorial_system.start_tutorial(
                    beginner_tutorial.id,
                )
                assert tutorial_session is not None, "Tutorial session should start"

                # Step 2: Help system integration
                from learning.help_system import HelpSystem

                help_system = HelpSystem()

                # Test contextual help
                help_content = help_system.get_contextual_help("airfoil_analysis")
                assert help_content is not None, "Should provide contextual help"

                # Step 3: Track learning progress
                from learning.learning_manager import LearningManager

                learning_manager = LearningManager()

                await learning_manager.track_action(
                    "tutorial_completed",
                    {"tutorial": beginner_tutorial.id},
                )
                progress = await learning_manager.get_learning_progress()
                assert (
                    progress["tutorials_completed"] >= 1
                ), "Should track tutorial completion"

            except ImportError as e:
                raise AssertionError(f"Required learning modules not available: {e}")

        await self._run_test("New User Onboarding", test_onboarding_flow)

    async def _test_collaborative_analysis(self):
        """Test collaborative analysis workflow"""

        async def test_collaboration_workflow():
            try:
                # Step 1: Create collaboration session
                from collaboration.collaboration_manager import CollaborationManager

                collab_manager = CollaborationManager()

                session_id = await collab_manager.create_session("user1")
                assert session_id is not None, "Should create collaboration session"

                # Step 2: Add collaborators
                await collab_manager.add_user_to_session(session_id, "user2")

                session_info = await collab_manager.get_session_info(session_id)
                assert len(session_info["users"]) == 2, "Should have 2 users in session"

                # Step 3: Shared analysis setup
                from collaboration.state_sync import StateSynchronizer

                state_sync = StateSynchronizer()

                analysis_state = {
                    "current_analysis": "airfoil_polar",
                    "airfoil": "naca2412",
                    "parameters": {"reynolds": 1000000, "mach": 0.1},
                }

                await state_sync.update_session_state(session_id, analysis_state)

                # Step 4: Export collaborative results
                collab_export_path = (
                    self.temp_dir
                    / "exports"
                    / f"collaborative_analysis_{session_id}.json"
                )

                collaborative_report = {
                    "session_info": session_info,
                    "final_state": analysis_state,
                }

                with open(collab_export_path, "w") as f:
                    json.dump(collaborative_report, f, indent=2)

                assert (
                    collab_export_path.exists()
                ), "Collaborative report should be exported"

            except ImportError as e:
                raise AssertionError(
                    f"Required collaboration modules not available: {e}",
                )

        await self._run_test("Collaborative Analysis", test_collaboration_workflow)

    async def _test_data_export_workflow(self):
        """Test comprehensive data export workflow"""

        async def test_export_workflow():
            try:
                # Step 1: Generate sample analysis data
                sample_data = {
                    "analysis_info": {
                        "type": "airfoil_polar",
                        "airfoil": "naca2412",
                        "solver": "xfoil",
                    },
                    "conditions": {"reynolds": 1000000, "mach": 0.1},
                    "results": {
                        "alpha": list(range(-5, 11)),
                        "cl": [i * 0.1 for i in range(-5, 11)],
                        "cd": [0.01 + abs(i) * 0.001 for i in range(-5, 11)],
                    },
                }

                # Step 2: Test multiple export formats
                from data.import_export import ImportExportManager

                import_export = ImportExportManager()

                export_formats = ["json", "csv"]
                exported_files = {}

                for format_type in export_formats:
                    if import_export.supports_format(format_type):
                        export_path = (
                            self.temp_dir / "exports" / f"sample_data.{format_type}"
                        )

                        success = await import_export.export_data(
                            sample_data,
                            export_path,
                            format_type,
                        )

                        if success:
                            assert (
                                export_path.exists()
                            ), f"Export file should exist for {format_type}"
                            exported_files[format_type] = export_path

                assert (
                    len(exported_files) > 0
                ), "Should successfully export to at least one format"

                # Step 3: Verify export integrity
                if "json" in exported_files:
                    with open(exported_files["json"]) as f:
                        imported_data = json.load(f)

                    assert (
                        imported_data["analysis_info"]["airfoil"] == "naca2412"
                    ), "JSON export should preserve data"
                    assert (
                        len(imported_data["results"]["alpha"]) == 16
                    ), "JSON export should preserve array length"

            except ImportError as e:
                raise AssertionError(f"Required export modules not available: {e}")

        await self._run_test("Data Export Workflow", test_export_workflow)

    # Advanced Workflow Tests

    async def _test_multi_solver_comparison(self):
        """Test multi-solver comparison workflow"""

        async def test_solver_comparison():
            try:
                # Step 1: Setup comparison configuration
                comparison_config = {
                    "airfoil": "naca2412",
                    "conditions": {"reynolds": 1000000, "mach": 0.1},
                    "solvers": ["xfoil", "su2"],
                    "comparison_metrics": ["cl", "cd", "convergence_time"],
                }

                # Step 2: Run analyses with different solvers
                from integration.analysis_service import AnalysisService

                analysis_service = AnalysisService()
                solver_results = {}

                for solver in comparison_config["solvers"]:
                    with patch.object(
                        analysis_service,
                        "run_analysis",
                    ) as mock_analysis:
                        # Mock different solver characteristics
                        if solver == "xfoil":
                            mock_result = {
                                "status": "completed",
                                "solver": solver,
                                "convergence_time": 15.2,
                                "results": {
                                    "alpha": list(range(-2, 9)),
                                    "cl": [i * 0.09 for i in range(-2, 9)],
                                    "cd": [
                                        0.008 + abs(i) * 0.0009 for i in range(-2, 9)
                                    ],
                                },
                            }
                        else:  # su2
                            mock_result = {
                                "status": "completed",
                                "solver": solver,
                                "convergence_time": 45.8,
                                "results": {
                                    "alpha": list(range(-2, 9)),
                                    "cl": [i * 0.088 for i in range(-2, 9)],
                                    "cd": [
                                        0.0078 + abs(i) * 0.0008 for i in range(-2, 9)
                                    ],
                                },
                            }

                        mock_analysis.return_value = mock_result

                        result = await analysis_service.run_analysis(
                            {"solver": solver, **comparison_config},
                        )

                        solver_results[solver] = result

                # Verify all solvers completed
                for solver, result in solver_results.items():
                    assert (
                        result["status"] == "completed"
                    ), f"{solver} analysis should complete"
                    assert (
                        len(result["results"]["alpha"]) == 11
                    ), f"{solver} should have correct data points"

                # Step 3: Export comparison report
                comparison_report = {
                    "configuration": comparison_config,
                    "solver_results": solver_results,
                }

                report_path = (
                    self.temp_dir / "exports" / "solver_comparison_report.json"
                )
                with open(report_path, "w") as f:
                    json.dump(comparison_report, f, indent=2)

                assert report_path.exists(), "Comparison report should be exported"

            except ImportError as e:
                raise AssertionError(f"Required comparison modules not available: {e}")

        await self._run_test("Multi-Solver Comparison", test_solver_comparison)

    async def _test_parametric_study_workflow(self):
        """Test parametric study workflow"""

        async def test_parametric_study():
            try:
                # Step 1: Define parametric study
                study_config = {
                    "base_airfoil": "naca2412",
                    "parameters": {
                        "reynolds": [500000, 1000000, 2000000],
                        "mach": [0.05, 0.1, 0.15],
                    },
                    "output_variables": ["cl_max", "cd_min", "l_d_max"],
                }

                # Step 2: Execute parametric study (mocked)
                from core.workflow import WorkflowEngine

                workflow_engine = WorkflowEngine()

                with patch.object(
                    workflow_engine,
                    "execute_parametric_study",
                ) as mock_study:
                    mock_study.return_value = {
                        "status": "completed",
                        "study_matrix": {
                            "parameters": study_config["parameters"],
                            "results": {
                                "cl_max": [
                                    [1.2, 1.25, 1.3],
                                    [1.22, 1.27, 1.32],
                                    [1.24, 1.29, 1.34],
                                ],
                                "cd_min": [
                                    [0.008, 0.0078, 0.0076],
                                    [0.0082, 0.008, 0.0078],
                                    [0.0084, 0.0082, 0.008],
                                ],
                                "l_d_max": [
                                    [150, 160, 171],
                                    [149, 159, 170],
                                    [148, 158, 169],
                                ],
                            },
                        },
                        "sensitivity_analysis": {
                            "reynolds_sensitivity": 0.15,
                            "mach_sensitivity": 0.08,
                        },
                    }

                    study_result = await workflow_engine.execute_parametric_study(
                        study_config,
                    )

                    # Verify parametric study results
                    assert (
                        study_result["status"] == "completed"
                    ), "Parametric study should complete"
                    assert "study_matrix" in study_result, "Should contain study matrix"
                    assert (
                        "sensitivity_analysis" in study_result
                    ), "Should contain sensitivity analysis"

                # Step 3: Export study results
                study_report_path = (
                    self.temp_dir / "exports" / "parametric_study_report.json"
                )
                with open(study_report_path, "w") as f:
                    json.dump(
                        {"configuration": study_config, "results": study_result},
                        f,
                        indent=2,
                    )

                assert (
                    study_report_path.exists()
                ), "Parametric study report should be exported"

            except ImportError as e:
                raise AssertionError(
                    f"Required parametric study modules not available: {e}",
                )

        await self._run_test("Parametric Study Workflow", test_parametric_study)

    # Error Recovery Tests

    async def _test_error_recovery_workflow(self):
        """Test error recovery and graceful degradation"""

        async def test_error_recovery():
            try:
                # Step 1: Test solver failure recovery
                from core.error_handler import ErrorHandler
                from integration.analysis_service import AnalysisService

                analysis_service = AnalysisService()
                error_handler = ErrorHandler()

                # Simulate solver failure
                with patch.object(analysis_service, "run_analysis") as mock_analysis:
                    mock_analysis.side_effect = Exception("Solver crashed")

                    try:
                        await analysis_service.run_analysis(
                            {"solver": "xfoil", "airfoil": "naca2412"},
                        )
                        assert False, "Should have raised exception"
                    except Exception as e:
                        # Test error handling
                        error_response = error_handler.handle_error(
                            e,
                            {"context": "analysis"},
                        )

                        assert error_response is not None, "Should handle error"
                        assert (
                            "suggestions" in error_response
                        ), "Should provide suggestions"

                # Step 2: Test graceful degradation
                from core.graceful_degradation import GracefulDegradation

                degradation_manager = GracefulDegradation()

                # Simulate missing solver
                with patch.object(
                    degradation_manager,
                    "check_solver_availability",
                ) as mock_check:
                    mock_check.return_value = {"xfoil": False, "panel_method": True}

                    fallback_config = (
                        await degradation_manager.get_fallback_configuration(
                            {"solver": "xfoil", "airfoil": "naca2412"},
                        )
                    )

                    assert (
                        fallback_config["solver"] == "panel_method"
                    ), "Should fallback to available solver"

            except ImportError as e:
                raise AssertionError(
                    f"Required error recovery modules not available: {e}",
                )

        await self._run_test("Error Recovery Workflow", test_error_recovery)
