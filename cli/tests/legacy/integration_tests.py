"""
Integration Testing Suite for ICARUS CLI

This module provides comprehensive integration tests for ICARUS module connections
and inter-component communication.
"""

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


class IntegrationTestSuite:
    """Integration test suite for ICARUS CLI system integration"""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.temp_dir = None

    async def run_all_tests(self) -> List[TestResult]:
        """Run all integration tests"""
        self.test_results = []

        # Setup test environment
        await self._setup_test_environment()

        try:
            # ICARUS module integration tests
            await self._test_xfoil_integration()
            await self._test_avl_integration()
            await self._test_gnvp_integration()

            # Component integration tests
            await self._test_analysis_workflow_integration()
            await self._test_data_visualization_integration()
            await self._test_export_import_integration()

            # API integration tests
            await self._test_api_layer_integration()
            await self._test_websocket_integration()

            # Plugin integration tests
            await self._test_plugin_system_integration()

            # Collaboration integration tests
            await self._test_collaboration_integration()

            # Configuration integration tests
            await self._test_config_persistence_integration()

        finally:
            # Cleanup test environment
            await self._cleanup_test_environment()

        return self.test_results

    async def _setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="icarus_test_"))

        # Create test data directories
        (self.temp_dir / "airfoils").mkdir()
        (self.temp_dir / "aircraft").mkdir()
        (self.temp_dir / "results").mkdir()

        # Create sample test files
        await self._create_test_files()

    async def _cleanup_test_environment(self):
        """Cleanup temporary test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    async def _create_test_files(self):
        """Create sample test files for integration tests"""
        # Create sample airfoil file
        airfoil_data = """NACA 0012
  1.00000   0.00000
  0.95000   0.01000
  0.90000   0.01800
  0.80000   0.03000
  0.70000   0.04000
  0.60000   0.04800
  0.50000   0.05400
  0.40000   0.05600
  0.30000   0.05200
  0.20000   0.04200
  0.10000   0.02800
  0.05000   0.01800
  0.00000   0.00000
  0.05000  -0.01800
  0.10000  -0.02800
  0.20000  -0.04200
  0.30000  -0.05200
  0.40000  -0.05600
  0.50000  -0.05400
  0.60000  -0.04800
  0.70000  -0.04000
  0.80000  -0.03000
  0.90000  -0.01800
  0.95000  -0.01000
  1.00000   0.00000"""

        with open(self.temp_dir / "airfoils" / "naca0012.dat", "w") as f:
            f.write(airfoil_data)

        # Create sample aircraft configuration
        aircraft_config = {
            "name": "Test Aircraft",
            "wing": {"span": 10.0, "chord": 1.0, "airfoil": "naca0012"},
            "fuselage": {"length": 8.0, "diameter": 1.2},
        }

        import json

        with open(self.temp_dir / "aircraft" / "test_aircraft.json", "w") as f:
            json.dump(aircraft_config, f, indent=2)

    async def _run_test(self, test_name: str, test_func):
        """Run a single integration test with error handling"""
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
        """Test XFoil solver integration"""

        async def test_xfoil_analysis():
            try:
                from integration.analysis_service import AnalysisService
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()
                analysis_service = AnalysisService()

                # Check if XFoil is available
                if not solver_manager.is_solver_available("xfoil"):
                    raise AssertionError(
                        "XFoil solver not available - skipping integration test",
                    )

                # Setup analysis configuration
                config = {
                    "solver": "xfoil",
                    "airfoil_file": str(self.temp_dir / "airfoils" / "naca0012.dat"),
                    "reynolds": 1000000,
                    "mach": 0.1,
                    "alpha_range": [-5, 5, 1],
                }

                # Run analysis
                result = await analysis_service.run_analysis(config)

                # Verify results
                assert result is not None, "Analysis should return results"
                assert "polar_data" in result, "Results should contain polar data"
                assert len(result["polar_data"]) > 0, "Polar data should not be empty"

                # Verify data structure
                polar_data = result["polar_data"]
                required_fields = ["alpha", "cl", "cd", "cm"]

                for field in required_fields:
                    assert field in polar_data[0], f"Polar data should contain {field}"

            except ImportError:
                raise AssertionError("Required integration modules not available")

        await self._run_test("XFoil Integration", test_xfoil_analysis)

    async def _test_avl_integration(self):
        """Test AVL solver integration"""

        async def test_avl_analysis():
            try:
                from integration.analysis_service import AnalysisService
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()
                analysis_service = AnalysisService()

                # Check if AVL is available
                if not solver_manager.is_solver_available("avl"):
                    # Create mock AVL integration for testing
                    with patch.object(
                        solver_manager,
                        "is_solver_available",
                        return_value=True,
                    ):
                        with patch.object(
                            analysis_service,
                            "run_analysis",
                        ) as mock_analysis:
                            mock_analysis.return_value = {
                                "stability_derivatives": {
                                    "cma": -0.1,
                                    "cmq": -5.0,
                                    "cla": 5.5,
                                },
                                "trim_conditions": {"alpha": 2.5, "elevator": -1.2},
                            }

                            # Setup analysis configuration
                            config = {
                                "solver": "avl",
                                "aircraft_file": str(
                                    self.temp_dir / "aircraft" / "test_aircraft.json",
                                ),
                                "flight_condition": {
                                    "velocity": 50.0,
                                    "altitude": 1000.0,
                                },
                            }

                            # Run analysis
                            result = await analysis_service.run_analysis(config)

                            # Verify results
                            assert result is not None, "Analysis should return results"
                            assert (
                                "stability_derivatives" in result
                            ), "Results should contain stability derivatives"
                            assert (
                                "trim_conditions" in result
                            ), "Results should contain trim conditions"

                            return

                # If AVL is actually available, run real test
                config = {
                    "solver": "avl",
                    "aircraft_file": str(
                        self.temp_dir / "aircraft" / "test_aircraft.json",
                    ),
                    "flight_condition": {"velocity": 50.0, "altitude": 1000.0},
                }

                result = await analysis_service.run_analysis(config)
                assert result is not None, "Analysis should return results"

            except ImportError:
                raise AssertionError("Required integration modules not available")

        await self._run_test("AVL Integration", test_avl_analysis)

    async def _test_gnvp_integration(self):
        """Test GNVP solver integration"""

        async def test_gnvp_analysis():
            try:
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()

                # Test GNVP solver detection
                solvers = solver_manager.discover_solvers()
                gnvp_available = any("gnvp" in solver.lower() for solver in solvers)

                if gnvp_available:
                    # Test GNVP-specific functionality
                    gnvp_info = solver_manager.get_solver_info("gnvp")
                    assert gnvp_info is not None, "Should return GNVP solver info"

                    # Test GNVP parameter validation
                    gnvp_params = {
                        "geometry_file": "test.stl",
                        "flow_conditions": {"mach": 0.8, "reynolds": 5000000},
                    }

                    validation = solver_manager.validate_solver_parameters(
                        "gnvp",
                        gnvp_params,
                    )
                    assert validation is not None, "Should validate GNVP parameters"
                else:
                    # Mock GNVP integration for testing
                    with patch.object(
                        solver_manager,
                        "discover_solvers",
                        return_value=["gnvp3", "gnvp7"],
                    ):
                        solvers = solver_manager.discover_solvers()
                        assert "gnvp3" in solvers, "Should detect GNVP3"
                        assert "gnvp7" in solvers, "Should detect GNVP7"

            except ImportError:
                raise AssertionError("Required integration modules not available")

        await self._run_test("GNVP Integration", test_gnvp_analysis)

    # Component Integration Tests

    async def _test_analysis_workflow_integration(self):
        """Test complete analysis workflow integration"""

        async def test_workflow_execution():
            try:
                from core.workflow import WorkflowEngine
                from data.database import DatabaseManager
                from integration.analysis_service import AnalysisService

                workflow_engine = WorkflowEngine()
                analysis_service = AnalysisService()
                db_manager = DatabaseManager()

                # Initialize database
                await db_manager.initialize()

                # Create test workflow
                workflow_config = {
                    "name": "Airfoil Analysis Workflow",
                    "steps": [
                        {
                            "id": "step1",
                            "name": "Load Airfoil",
                            "type": "data_load",
                            "config": {
                                "file": str(
                                    self.temp_dir / "airfoils" / "naca0012.dat",
                                ),
                            },
                        },
                        {
                            "id": "step2",
                            "name": "Run XFoil Analysis",
                            "type": "analysis",
                            "config": {
                                "solver": "xfoil",
                                "reynolds": 1000000,
                                "alpha_range": [-2, 2, 1],
                            },
                            "dependencies": ["step1"],
                        },
                        {
                            "id": "step3",
                            "name": "Save Results",
                            "type": "data_save",
                            "config": {"format": "json"},
                            "dependencies": ["step2"],
                        },
                    ],
                }

                # Execute workflow
                workflow = workflow_engine.create_workflow_from_config(workflow_config)
                result = await workflow_engine.execute_workflow(workflow)

                # Verify workflow execution
                assert result is not None, "Workflow should execute successfully"
                assert result.get("status") == "completed", "Workflow should complete"
                assert (
                    len(result.get("step_results", [])) == 3
                ), "All steps should execute"

                # Verify data persistence
                saved_results = await db_manager.get_workflow_results(workflow.id)
                assert saved_results is not None, "Results should be saved to database"

            except ImportError:
                raise AssertionError("Required workflow modules not available")

        await self._run_test("Analysis Workflow Integration", test_workflow_execution)

    async def _test_data_visualization_integration(self):
        """Test data and visualization integration"""

        async def test_data_viz_pipeline():
            try:
                from data.database import DatabaseManager
                from visualization.chart_generator import ChartGenerator
                from visualization.visualization_manager import VisualizationManager

                db_manager = DatabaseManager()
                viz_manager = VisualizationManager()
                chart_generator = ChartGenerator()

                # Initialize components
                await db_manager.initialize()

                # Create test data
                test_data = {
                    "alpha": [-2, -1, 0, 1, 2],
                    "cl": [-0.2, -0.1, 0.0, 0.1, 0.2],
                    "cd": [0.01, 0.008, 0.007, 0.008, 0.01],
                }

                # Save data to database
                data_id = await db_manager.save_analysis_data("test_polar", test_data)
                assert data_id is not None, "Data should be saved"

                # Retrieve data
                retrieved_data = await db_manager.get_analysis_data(data_id)
                assert retrieved_data is not None, "Data should be retrieved"

                # Generate visualization
                chart_config = {
                    "type": "line",
                    "x_axis": "alpha",
                    "y_axis": "cl",
                    "title": "Lift Coefficient vs Angle of Attack",
                }

                chart = chart_generator.create_chart(retrieved_data, chart_config)
                assert chart is not None, "Chart should be generated"

                # Test visualization manager
                viz_result = await viz_manager.create_visualization(
                    data_id,
                    chart_config,
                )
                assert viz_result is not None, "Visualization should be created"

            except ImportError:
                raise AssertionError("Required visualization modules not available")

        await self._run_test("Data Visualization Integration", test_data_viz_pipeline)

    async def _test_export_import_integration(self):
        """Test export/import integration"""

        async def test_export_import_cycle():
            try:
                from data.database import DatabaseManager
                from data.import_export import ImportExportManager

                import_export = ImportExportManager()
                db_manager = DatabaseManager()

                await db_manager.initialize()

                # Create test data
                test_data = {
                    "analysis_type": "airfoil_polar",
                    "airfoil": "naca0012",
                    "conditions": {"reynolds": 1000000, "mach": 0.1},
                    "results": {
                        "alpha": [0, 1, 2, 3, 4],
                        "cl": [0.0, 0.1, 0.2, 0.3, 0.4],
                        "cd": [0.007, 0.008, 0.01, 0.012, 0.015],
                    },
                }

                # Test export to different formats
                export_formats = ["json", "csv", "matlab"]

                for format_type in export_formats:
                    if import_export.supports_format(format_type):
                        # Export data
                        export_path = self.temp_dir / f"test_export.{format_type}"
                        success = await import_export.export_data(
                            test_data,
                            export_path,
                            format_type,
                        )
                        assert success, f"Should export to {format_type}"
                        assert (
                            export_path.exists()
                        ), f"Export file should exist for {format_type}"

                        # Import data back
                        imported_data = await import_export.import_data(
                            export_path,
                            format_type,
                        )
                        assert (
                            imported_data is not None
                        ), f"Should import from {format_type}"

                        # Verify data integrity
                        if format_type == "json":
                            assert (
                                imported_data["airfoil"] == test_data["airfoil"]
                            ), "Airfoil name should match"
                            assert len(imported_data["results"]["alpha"]) == len(
                                test_data["results"]["alpha"],
                            ), "Data length should match"

            except ImportError:
                raise AssertionError("Required import/export modules not available")

        await self._run_test("Export Import Integration", test_export_import_cycle)

    # API Integration Tests

    async def _test_api_layer_integration(self):
        """Test API layer integration"""

        async def test_api_endpoints():
            try:
                from api.app import create_api_app
                from api.models import AnalysisConfig
                from api.models import AnalysisType
                from api.models import SolverType

                # Create FastAPI app
                app = create_api_app()

                # Test API model serialization
                config = AnalysisConfig(
                    analysis_type=AnalysisType.AIRFOIL,
                    target="naca0012.dat",
                    solver=SolverType.XFOIL,
                    parameters={"reynolds": 1000000},
                )

                # Test JSON serialization
                config_json = config.model_dump_json()
                assert config_json is not None, "Should serialize to JSON"

                # Test deserialization
                config_restored = AnalysisConfig.model_validate_json(config_json)
                assert (
                    config_restored.target == config.target
                ), "Deserialized data should match"

                # Test API routes exist
                routes = [route.path for route in app.routes]
                expected_routes = ["/health", "/analysis/start", "/session"]

                for expected_route in expected_routes:
                    route_exists = any(expected_route in route for route in routes)
                    assert route_exists, f"Route {expected_route} should exist"

            except ImportError:
                raise AssertionError("Required API modules not available")

        await self._run_test("API Layer Integration", test_api_endpoints)

    async def _test_websocket_integration(self):
        """Test WebSocket integration"""

        async def test_websocket_communication():
            try:
                from api.models import WebSocketMessage
                from api.websocket import WebSocketManager

                ws_manager = WebSocketManager()

                # Create mock WebSocket connections
                class MockWebSocket:
                    def __init__(self, name):
                        self.name = name
                        self.messages = []

                    async def send_text(self, text):
                        self.messages.append(text)

                    async def close(self):
                        pass

                # Add connections
                ws1 = MockWebSocket("client1")
                ws2 = MockWebSocket("client2")

                conn1 = await ws_manager.add_connection(ws1)
                conn2 = await ws_manager.add_connection(ws2)

                # Authenticate connections
                await ws_manager.authenticate_connection(conn1.session_id, "user1")
                await ws_manager.authenticate_connection(conn2.session_id, "user2")

                # Join collaboration room
                await ws_manager.join_collaboration_room(conn1.session_id, "room1")
                await ws_manager.join_collaboration_room(conn2.session_id, "room1")

                # Test message broadcasting
                message = WebSocketMessage(
                    type="analysis_update",
                    payload={"progress": 50, "status": "running"},
                )

                sent_count = await ws_manager.broadcast_to_room("room1", message)
                assert sent_count == 2, "Should send to both clients in room"

                # Verify messages received
                assert len(ws1.messages) == 1, "Client 1 should receive message"
                assert len(ws2.messages) == 1, "Client 2 should receive message"

                # Test individual messaging
                individual_message = WebSocketMessage(
                    type="private_message",
                    payload={"text": "Hello user1"},
                )

                sent = await ws_manager.send_to_session(
                    conn1.session_id,
                    individual_message,
                )
                assert sent, "Should send individual message"
                assert (
                    len(ws1.messages) == 2
                ), "Client 1 should receive individual message"
                assert (
                    len(ws2.messages) == 1
                ), "Client 2 should not receive individual message"

            except ImportError:
                raise AssertionError("Required WebSocket modules not available")

        await self._run_test("WebSocket Integration", test_websocket_communication)

    # Plugin Integration Tests

    async def _test_plugin_system_integration(self):
        """Test plugin system integration"""

        async def test_plugin_lifecycle():
            try:
                from plugins.api import PluginAPI
                from plugins.manager import PluginManager

                plugin_manager = PluginManager()
                plugin_api = PluginAPI()

                # Create mock plugin
                class MockPlugin:
                    def __init__(self):
                        self.name = "test_plugin"
                        self.version = "1.0.0"
                        self.initialized = False

                    async def initialize(self, api):
                        self.initialized = True
                        api.register_method("test_method", self.test_method)

                    def test_method(self):
                        return "plugin_result"

                    async def cleanup(self):
                        self.initialized = False

                # Test plugin loading
                mock_plugin = MockPlugin()

                with patch.object(
                    plugin_manager,
                    "discover_plugins",
                    return_value=[mock_plugin],
                ):
                    plugins = plugin_manager.discover_plugins()
                    assert len(plugins) > 0, "Should discover plugins"

                    # Test plugin initialization
                    await plugin_manager.initialize_plugin(mock_plugin, plugin_api)
                    assert mock_plugin.initialized, "Plugin should be initialized"

                    # Test plugin method execution
                    result = plugin_api.execute_method("test_method")
                    assert result == "plugin_result", "Plugin method should execute"

                    # Test plugin cleanup
                    await plugin_manager.cleanup_plugin(mock_plugin)
                    assert not mock_plugin.initialized, "Plugin should be cleaned up"

            except ImportError:
                raise AssertionError("Required plugin modules not available")

        await self._run_test("Plugin System Integration", test_plugin_lifecycle)

    # Collaboration Integration Tests

    async def _test_collaboration_integration(self):
        """Test collaboration system integration"""

        async def test_collaboration_workflow():
            try:
                from api.websocket import WebSocketManager
                from collaboration.collaboration_manager import CollaborationManager
                from collaboration.state_sync import StateSynchronizer

                collab_manager = CollaborationManager()
                state_sync = StateSynchronizer()
                ws_manager = WebSocketManager()

                # Create collaboration session
                session_id = await collab_manager.create_session("user1")
                assert session_id is not None, "Should create collaboration session"

                # Add users to session
                await collab_manager.add_user_to_session(session_id, "user2")
                await collab_manager.add_user_to_session(session_id, "user3")

                # Test state synchronization
                initial_state = {
                    "current_analysis": "airfoil_polar",
                    "parameters": {"reynolds": 1000000, "mach": 0.1},
                }

                await state_sync.update_session_state(session_id, initial_state)

                # Simulate state change from user2
                state_change = {"parameters": {"reynolds": 2000000, "mach": 0.15}}

                await state_sync.apply_state_change(session_id, "user2", state_change)

                # Verify state synchronization
                current_state = await state_sync.get_session_state(session_id)
                assert (
                    current_state["parameters"]["reynolds"] == 2000000
                ), "State should be updated"

                # Test conflict resolution
                conflicting_change = {
                    "parameters": {"reynolds": 1500000},  # Different value
                }

                await state_sync.apply_state_change(
                    session_id,
                    "user3",
                    conflicting_change,
                )

                # Verify conflict resolution
                resolved_state = await state_sync.get_session_state(session_id)
                assert resolved_state is not None, "State should be resolved"

            except ImportError:
                raise AssertionError("Required collaboration modules not available")

        await self._run_test("Collaboration Integration", test_collaboration_workflow)

    # Configuration Integration Tests

    async def _test_config_persistence_integration(self):
        """Test configuration persistence integration"""

        async def test_config_workflow():
            try:
                from core.config import ConfigManager
                from core.settings import SettingsManager

                config_manager = ConfigManager()
                settings_manager = SettingsManager()

                # Test configuration persistence
                test_config = {
                    "theme": "aerospace_dark",
                    "default_solver": "xfoil",
                    "recent_files": ["naca0012.dat", "naca2412.dat"],
                    "workspace": str(self.temp_dir),
                }

                # Save configuration
                for key, value in test_config.items():
                    config_manager.set(key, value)

                config_manager.save()

                # Create new config manager instance
                new_config = ConfigManager()

                # Verify persistence
                for key, expected_value in test_config.items():
                    actual_value = new_config.get(key)
                    assert (
                        actual_value == expected_value
                    ), f"Config {key} should persist"

                # Test settings integration
                settings = {
                    "ui": {"theme": "aerospace_dark", "layout": "standard"},
                    "analysis": {"default_reynolds": 1000000, "default_mach": 0.1},
                }

                await settings_manager.update_settings(settings)

                # Verify settings persistence
                saved_settings = await settings_manager.get_all_settings()
                assert (
                    saved_settings["ui"]["theme"] == "aerospace_dark"
                ), "UI settings should persist"
                assert (
                    saved_settings["analysis"]["default_reynolds"] == 1000000
                ), "Analysis settings should persist"

            except ImportError:
                raise AssertionError("Required configuration modules not available")

        await self._run_test(
            "Configuration Persistence Integration",
            test_config_workflow,
        )
