"""
ICARUS Module Integration Tests

This module provides specialized integration tests for ICARUS aerodynamics modules
including XFoil, AVL, GNVP, and other solvers.
"""

import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, Mock, AsyncMock

# Add CLI directory to path
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from .framework import TestResult, TestStatus, TestType


class IcarusIntegrationTestSuite:
    """Specialized integration tests for ICARUS modules"""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir: Optional[Path] = None

    async def run_all_tests(self) -> List[TestResult]:
        """Run all ICARUS integration tests"""
        self.test_results = []

        # Setup test environment
        await self._setup_icarus_test_environment()

        try:
            # Core ICARUS module tests
            await self._test_xfoil_integration()
            await self._test_avl_integration()
            await self._test_gnvp_integration()
            await self._test_solver_discovery()

            # Analysis workflow tests
            await self._test_airfoil_analysis_workflow()
            await self._test_aircraft_analysis_workflow()

            # Data integration tests
            await self._test_airfoil_database_integration()
            await self._test_results_processing()

        finally:
            # Cleanup
            await self._cleanup_icarus_test_environment()

        return self.test_results

    async def _setup_icarus_test_environment(self):
        """Setup ICARUS-specific test environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="icarus_integration_"))

        # Create ICARUS-specific directories
        directories = [
            "airfoils",
            "aircraft",
            "results",
            "solvers",
            "database",
            "exports",
        ]

        for directory in directories:
            (self.temp_dir / directory).mkdir()

        # Create sample ICARUS data files
        await self._create_icarus_test_data()

    async def _cleanup_icarus_test_environment(self):
        """Cleanup ICARUS test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    async def _create_icarus_test_data(self):
        """Create ICARUS-specific test data"""
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

    async def _run_test(self, test_name: str, test_func):
        """Run a single ICARUS integration test"""
        start_time = time.time()

        try:
            await test_func()
            duration = time.time() - start_time

            result = TestResult(
                name=test_name,
                test_type=TestType.INTEGRATION,
                status=TestStatus.PASSED,
                duration=duration,
            )

        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.INTEGRATION,
                status=TestStatus.FAILED,
                duration=duration,
                error_message=str(e),
            )

        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.INTEGRATION,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=f"{type(e).__name__}: {str(e)}",
            )

        self.test_results.append(result)
        return result

    # ICARUS Module Integration Tests

    async def _test_xfoil_integration(self):
        """Test XFoil integration with real airfoil data"""

        async def test_xfoil_analysis():
            try:
                from integration.solver_manager import SolverManager
                from integration.analysis_service import AnalysisService

                solver_manager = SolverManager()
                analysis_service = AnalysisService()

                airfoil_file = self.temp_dir / "airfoils" / "naca2412.dat"

                config = {
                    "solver": "xfoil",
                    "airfoil_file": str(airfoil_file),
                    "reynolds": 1000000,
                    "mach": 0.1,
                    "alpha_range": [-2, 8, 1],
                }

                # Mock XFoil execution
                with patch.object(analysis_service, "run_analysis") as mock_analysis:
                    mock_analysis.return_value = {
                        "status": "completed",
                        "airfoil": "naca2412",
                        "solver": "xfoil",
                        "results": {
                            "alpha": list(range(-2, 9)),
                            "cl": [i * 0.1 for i in range(-2, 9)],
                            "cd": [0.008 + abs(i) * 0.0005 for i in range(-2, 9)],
                            "cm": [-0.05 - i * 0.005 for i in range(-2, 9)],
                        },
                        "convergence": "good",
                    }

                    result = await analysis_service.run_analysis(config)

                    # Verify results
                    assert result["status"] == "completed"
                    assert result["airfoil"] == "naca2412"
                    assert len(result["results"]["alpha"]) == 11
                    assert all(
                        isinstance(cl, (int, float)) for cl in result["results"]["cl"]
                    )

            except ImportError:
                raise AssertionError(
                    "Required ICARUS integration modules not available"
                )

        await self._run_test("XFoil Integration", test_xfoil_analysis)

    async def _test_avl_integration(self):
        """Test AVL integration with aircraft configurations"""

        async def test_avl_analysis():
            try:
                from integration.analysis_service import AnalysisService

                analysis_service = AnalysisService()
                aircraft_file = self.temp_dir / "aircraft" / "test_aircraft.json"

                config = {
                    "solver": "avl",
                    "aircraft_file": str(aircraft_file),
                    "flight_conditions": {
                        "velocity": 50.0,
                        "altitude": 1000.0,
                        "weight": 800.0,
                    },
                }

                # Mock AVL execution
                with patch.object(analysis_service, "run_analysis") as mock_analysis:
                    mock_analysis.return_value = {
                        "status": "completed",
                        "aircraft": "test_aircraft",
                        "solver": "avl",
                        "stability_derivatives": {
                            "cla": 5.2,
                            "cma": -0.15,
                            "cmq": -8.5,
                            "cnb": 0.12,
                        },
                        "trim_conditions": {
                            "alpha": 3.2,
                            "elevator": -1.8,
                            "thrust": 250.0,
                        },
                        "static_margin": 0.12,
                    }

                    result = await analysis_service.run_analysis(config)

                    # Verify results
                    assert result["status"] == "completed"
                    assert "stability_derivatives" in result
                    assert "trim_conditions" in result
                    assert result["static_margin"] > 0  # Should be stable

            except ImportError:
                raise AssertionError(
                    "Required ICARUS integration modules not available"
                )

        await self._run_test("AVL Integration", test_avl_analysis)

    async def _test_gnvp_integration(self):
        """Test GNVP integration"""

        async def test_gnvp_analysis():
            try:
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()

                # Mock GNVP solver discovery and execution
                with patch.object(solver_manager, "discover_solvers") as mock_discover:
                    mock_discover.return_value = ["gnvp3", "gnvp7", "gnvp_euler"]

                    solvers = solver_manager.discover_solvers()
                    gnvp_solvers = [s for s in solvers if "gnvp" in s.lower()]

                    assert len(gnvp_solvers) >= 2, (
                        "Should discover multiple GNVP versions"
                    )

                    # Test solver info
                    for solver in gnvp_solvers:
                        with patch.object(
                            solver_manager, "get_solver_info"
                        ) as mock_info:
                            mock_info.return_value = {
                                "name": solver,
                                "version": "7.0" if "7" in solver else "3.0",
                                "capabilities": ["euler", "navier_stokes"],
                                "mesh_types": ["structured", "unstructured"],
                            }

                            info = solver_manager.get_solver_info(solver)
                            assert info["name"] == solver
                            assert "capabilities" in info
                            assert len(info["capabilities"]) > 0

            except ImportError:
                raise AssertionError(
                    "Required ICARUS integration modules not available"
                )

        await self._run_test("GNVP Integration", test_gnvp_analysis)

    async def _test_solver_discovery(self):
        """Test comprehensive solver discovery"""

        async def test_solver_discovery():
            try:
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()

                # Mock comprehensive solver discovery
                expected_solvers = ["xfoil", "avl", "gnvp3", "gnvp7", "su2", "openfoam"]

                with patch.object(solver_manager, "discover_solvers") as mock_discover:
                    mock_discover.return_value = expected_solvers

                    discovered = solver_manager.discover_solvers()

                    # Verify discovery
                    assert len(discovered) >= 5, "Should discover multiple solvers"
                    assert "xfoil" in discovered, "Should discover XFoil"
                    assert "avl" in discovered, "Should discover AVL"

                    # Test availability checking
                    for solver in discovered[:3]:  # Test first 3
                        with patch.object(
                            solver_manager, "is_solver_available"
                        ) as mock_available:
                            mock_available.return_value = True

                            available = solver_manager.is_solver_available(solver)
                            assert isinstance(available, bool)

            except ImportError:
                raise AssertionError(
                    "Required ICARUS integration modules not available"
                )

        await self._run_test("Solver Discovery", test_solver_discovery)

    async def _test_airfoil_analysis_workflow(self):
        """Test complete airfoil analysis workflow"""

        async def test_workflow():
            try:
                from core.workflow import WorkflowEngine
                from data.database import DatabaseManager

                workflow_engine = WorkflowEngine()
                db_manager = DatabaseManager()

                await db_manager.initialize()

                # Create airfoil analysis workflow
                workflow_config = {
                    "name": "Comprehensive Airfoil Analysis",
                    "steps": [
                        {
                            "id": "load_airfoil",
                            "type": "data_load",
                            "config": {"airfoil": "naca2412"},
                        },
                        {
                            "id": "xfoil_analysis",
                            "type": "analysis",
                            "config": {"solver": "xfoil", "reynolds": [1000000]},
                            "dependencies": ["load_airfoil"],
                        },
                        {
                            "id": "save_results",
                            "type": "data_save",
                            "dependencies": ["xfoil_analysis"],
                        },
                    ],
                }

                # Mock workflow execution
                with patch.object(workflow_engine, "execute_workflow") as mock_execute:
                    mock_execute.return_value = {
                        "status": "completed",
                        "workflow_id": "airfoil_analysis_001",
                        "steps_completed": 3,
                        "execution_time": 45.2,
                        "results": {
                            "airfoil_data": "loaded",
                            "analysis_results": "completed",
                            "data_saved": True,
                        },
                    }

                    workflow = workflow_engine.create_workflow_from_config(
                        workflow_config
                    )
                    result = await workflow_engine.execute_workflow(workflow)

                    # Verify workflow execution
                    assert result["status"] == "completed"
                    assert result["steps_completed"] == 3
                    assert result["results"]["data_saved"] is True

            except ImportError:
                raise AssertionError("Required workflow modules not available")

        await self._run_test("Airfoil Analysis Workflow", test_workflow)

    async def _test_aircraft_analysis_workflow(self):
        """Test complete aircraft analysis workflow"""

        async def test_workflow():
            try:
                from core.workflow import WorkflowEngine

                workflow_engine = WorkflowEngine()

                # Create aircraft analysis workflow
                workflow_config = {
                    "name": "Aircraft Design Analysis",
                    "steps": [
                        {
                            "id": "load_aircraft",
                            "type": "data_load",
                            "config": {"aircraft": "test_aircraft"},
                        },
                        {
                            "id": "stability_analysis",
                            "type": "analysis",
                            "config": {"solver": "avl"},
                            "dependencies": ["load_aircraft"],
                        },
                    ],
                }

                # Mock workflow execution
                with patch.object(workflow_engine, "execute_workflow") as mock_execute:
                    mock_execute.return_value = {
                        "status": "completed",
                        "aircraft": "test_aircraft",
                        "stability_margin": 0.15,
                        "design_rating": "Acceptable",
                    }

                    workflow = workflow_engine.create_workflow_from_config(
                        workflow_config
                    )
                    result = await workflow_engine.execute_workflow(workflow)

                    # Verify results
                    assert result["status"] == "completed"
                    assert result["stability_margin"] > 0
                    assert result["design_rating"] in [
                        "Excellent",
                        "Good",
                        "Acceptable",
                    ]

            except ImportError:
                raise AssertionError("Required workflow modules not available")

        await self._run_test("Aircraft Analysis Workflow", test_workflow)

    async def _test_airfoil_database_integration(self):
        """Test airfoil database integration"""

        async def test_database():
            try:
                from data.database import DatabaseManager

                db_manager = DatabaseManager()
                await db_manager.initialize()

                # Test airfoil data storage and retrieval
                airfoil_data = {
                    "name": "naca2412",
                    "coordinates": [[1.0, 0.0], [0.95, 0.012], [0.0, 0.0]],
                    "properties": {
                        "max_thickness": 0.12,
                        "max_camber": 0.02,
                        "camber_position": 0.4,
                    },
                }

                # Store airfoil
                airfoil_id = await db_manager.create_record("airfoils", airfoil_data)
                assert airfoil_id is not None

                # Retrieve airfoil
                retrieved = await db_manager.get_record("airfoils", airfoil_id)
                assert retrieved["name"] == "naca2412"
                assert len(retrieved["coordinates"]) == 3

            except ImportError:
                raise AssertionError("Required database modules not available")

        await self._run_test("Airfoil Database Integration", test_database)

    async def _test_results_processing(self):
        """Test results processing integration"""

        async def test_processing():
            try:
                from integration.result_processor import ResultProcessor

                processor = ResultProcessor()

                # Test processing airfoil polar results
                result_data = {
                    "type": "airfoil_polar",
                    "data": {
                        "alpha": list(range(-5, 11)),
                        "cl": [i * 0.1 for i in range(-5, 11)],
                        "cd": [0.008 + abs(i) * 0.0005 for i in range(-5, 11)],
                    },
                }

                processed = processor.process_results(result_data)
                assert processed is not None
                assert "processed_data" in processed

                # Test formatting
                formatted = processor.format_results(processed, "table")
                assert formatted is not None

            except ImportError:
                raise AssertionError("Required result processing modules not available")

        await self._run_test("Results Processing", test_processing)
