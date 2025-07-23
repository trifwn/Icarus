"""
Unit Testing Suite for ICARUS CLI

This module provides comprehensive unit tests for all core components
of the ICARUS CLI system.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock
from unittest.mock import Mock

# Add CLI directory to path for imports
cli_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cli_dir))

from .framework import TestResult
from .framework import TestStatus
from .framework import TestType


class UnitTestSuite:
    """Comprehensive unit test suite for ICARUS CLI components"""

    def __init__(self):
        self.test_results: List[TestResult] = []

    async def run_all_tests(self) -> List[TestResult]:
        """Run all unit tests"""
        self.test_results = []

        # Core component tests
        await self._test_config_manager()
        await self._test_event_system()
        await self._test_state_manager()
        await self._test_workflow_engine()

        # UI component tests
        await self._test_theme_system()
        await self._test_screen_manager()
        await self._test_responsive_layout()

        # Service component tests
        await self._test_analysis_service()
        await self._test_data_management()
        await self._test_export_service()

        # Integration component tests
        await self._test_solver_manager()
        await self._test_parameter_validator()
        await self._test_result_processor()

        # Plugin system tests
        await self._test_plugin_manager()
        await self._test_plugin_api()

        # Collaboration system tests
        await self._test_collaboration_manager()
        await self._test_websocket_manager()

        return self.test_results

    async def _run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        start_time = time.time()

        try:
            await test_func()
            duration = time.time() - start_time

            result = TestResult(
                name=test_name,
                test_type=TestType.UNIT,
                status=TestStatus.PASSED,
                duration=duration,
            )

        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.UNIT,
                status=TestStatus.FAILED,
                duration=duration,
                error_message=str(e),
            )

        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                name=test_name,
                test_type=TestType.UNIT,
                status=TestStatus.ERROR,
                duration=duration,
                error_message=f"{type(e).__name__}: {str(e)}",
            )

        self.test_results.append(result)
        return result

    # Core Component Tests

    async def _test_config_manager(self):
        """Test configuration manager functionality"""

        async def test_basic_operations():
            try:
                from core.config import ConfigManager

                config = ConfigManager()

                # Test setting and getting values
                config.set("test_key", "test_value")
                value = config.get("test_key")
                assert value == "test_value", f"Expected 'test_value', got {value}"

                # Test default values
                default_value = config.get("nonexistent_key", "default")
                assert default_value == "default", (
                    f"Expected 'default', got {default_value}"
                )

                # Test UI config
                ui_config = config.get_ui_config()
                assert hasattr(
                    ui_config,
                    "theme",
                ), "UI config should have theme attribute"

            except ImportError:
                # Skip if module not available
                raise AssertionError("ConfigManager module not available")

        await self._run_test("ConfigManager Basic Operations", test_basic_operations)

        async def test_config_persistence():
            try:
                from core.config import ConfigManager

                config = ConfigManager()

                # Test saving configuration
                config.set("persistent_key", "persistent_value")
                config.save()

                # Create new instance and verify persistence
                new_config = ConfigManager()
                value = new_config.get("persistent_key")
                assert value == "persistent_value", "Configuration should persist"

            except ImportError:
                raise AssertionError("ConfigManager module not available")
            except Exception as e:
                # Persistence might not be implemented yet
                raise AssertionError(f"Config persistence test failed: {e}")

        await self._run_test("ConfigManager Persistence", test_config_persistence)

    async def _test_event_system(self):
        """Test event system functionality"""

        async def test_event_subscription():
            try:
                from app.event_system import EventSystem

                event_system = EventSystem()
                test_data = {}

                def test_callback(data):
                    test_data.update(data)

                # Test subscription
                event_system.subscribe("test_event", test_callback)

                # Test synchronous emit
                event_system.emit_sync("test_event", {"message": "test"})

                # Give time for processing
                await asyncio.sleep(0.1)

                assert "message" in test_data, "Event callback should have been called"
                assert test_data["message"] == "test", "Event data should match"

            except ImportError:
                raise AssertionError("EventSystem module not available")

        await self._run_test("EventSystem Subscription", test_event_subscription)

        async def test_async_events():
            try:
                from app.event_system import EventSystem

                event_system = EventSystem()
                test_data = {"called": False}

                async def async_callback(data):
                    test_data["called"] = True
                    test_data.update(data)

                event_system.subscribe("async_test", async_callback)
                await event_system.emit("async_test", {"async_message": "test"})

                assert test_data["called"], "Async callback should have been called"
                assert test_data.get("async_message") == "test", (
                    "Async event data should match"
                )

            except ImportError:
                raise AssertionError("EventSystem module not available")

        await self._run_test("EventSystem Async Events", test_async_events)

    async def _test_state_manager(self):
        """Test state manager functionality"""

        async def test_session_management():
            try:
                from app.state_manager import StateManager

                state_manager = StateManager()

                # Test session initialization
                session = await state_manager.initialize_session()
                assert session is not None, "Session should be initialized"

                # Test session info
                info = state_manager.get_session_info()
                assert "session_id" in info, "Session info should contain session_id"

                # Test state updates
                await state_manager.update_state("test_key", "test_value")
                state = state_manager.get_current_state()
                assert state.get("test_key") == "test_value", "State should be updated"

            except ImportError:
                raise AssertionError("StateManager module not available")

        await self._run_test("StateManager Session Management", test_session_management)

    async def _test_workflow_engine(self):
        """Test workflow engine functionality"""

        async def test_workflow_operations():
            try:
                from core.workflow import WorkflowEngine

                engine = WorkflowEngine()

                # Test getting workflows
                workflows = engine.get_workflows()
                assert len(workflows) > 0, "Should have built-in workflows"

                # Test workflow info
                workflow_info = engine.get_available_workflows()
                assert isinstance(
                    workflow_info,
                    list,
                ), "Should return list of workflow info"

                # Test workflow creation
                template = (
                    engine.get_workflow_templates()[0]
                    if engine.get_workflow_templates()
                    else None
                )
                if template:
                    workflow = engine.create_workflow(template)
                    assert workflow is not None, "Should create workflow from template"

            except ImportError:
                raise AssertionError("WorkflowEngine module not available")

        await self._run_test("WorkflowEngine Operations", test_workflow_operations)

    # UI Component Tests

    async def _test_theme_system(self):
        """Test theme system functionality"""

        async def test_theme_manager():
            try:
                from tui.themes.theme_manager import ThemeManager

                theme_manager = ThemeManager()

                # Test available themes
                themes = theme_manager.get_available_themes()
                assert len(themes) > 0, "Should have available themes"

                # Test theme switching
                original_theme = theme_manager.get_current_theme_id()

                for theme_id in themes[:2]:  # Test first 2 themes
                    success = theme_manager.set_theme(theme_id)
                    assert success, f"Should successfully switch to {theme_id}"

                    current_id = theme_manager.get_current_theme_id()
                    assert current_id == theme_id, f"Current theme should be {theme_id}"

                # Test CSS generation
                css = theme_manager.get_current_css()
                assert css and len(css) > 100, "Should generate valid CSS"

            except ImportError:
                raise AssertionError("ThemeManager module not available")

        await self._run_test("Theme Manager", test_theme_manager)

    async def _test_screen_manager(self):
        """Test screen manager functionality"""

        async def test_screen_operations():
            try:
                # Mock app for testing
                class MockApp:
                    def __init__(self):
                        self.screens = {}
                        self.event_system = MockEventSystem()
                        self.log = MockLogger()

                    def install_screen(self, screen, name):
                        self.screens[name] = screen

                    async def push_screen(self, name):
                        pass

                class MockEventSystem:
                    async def emit(self, event, data):
                        pass

                class MockLogger:
                    def error(self, message):
                        pass

                from app.screen_manager import ScreenManager

                app = MockApp()
                screen_manager = ScreenManager(app)

                # Test screen manager initialization
                await screen_manager.initialize()

                # Test screen switching
                success = await screen_manager.switch_to("dashboard")
                assert success, "Should successfully switch to dashboard"

                current_screen = screen_manager.get_current_screen()
                assert current_screen is not None, "Should have current screen"
                assert current_screen.screen_name == "dashboard", (
                    "Should be dashboard screen"
                )

                # Test screen history
                await screen_manager.switch_to("analysis")
                history = screen_manager.get_screen_history()
                assert len(history) == 1, "Should have one item in history"
                assert history[0] == "dashboard", "History should contain dashboard"

                # Test go back functionality
                success = await screen_manager.go_back()
                assert success, "Should successfully go back"
                assert screen_manager.current_screen == "dashboard", (
                    "Should be back to dashboard"
                )

                # Test refresh current screen
                await screen_manager.refresh_current()

                # Test cleanup
                await screen_manager.cleanup_screen("dashboard")

            except ImportError:
                raise AssertionError("ScreenManager module not available")

        await self._run_test("Screen Manager", test_screen_operations)

    async def _test_responsive_layout(self):
        """Test responsive layout functionality"""

        async def test_layout_calculations():
            try:
                from tui.themes.responsive_layout import LayoutBreakpoints
                from tui.themes.responsive_layout import ResponsiveLayout

                layout = ResponsiveLayout()
                breakpoints = LayoutBreakpoints()

                # Test layout mode detection
                test_cases = [(30, "MINIMAL"), (80, "STANDARD"), (150, "WIDE")]

                for width, expected_mode in test_cases:
                    mode = breakpoints.get_layout_mode(width)
                    assert mode.value == expected_mode, (
                        f"Width {width} should be {expected_mode}"
                    )

                # Test layout updates
                layout.update_dimensions(100, 30)
                info = layout.get_layout_info()

                required_keys = ["mode", "orientation", "dimensions"]
                for key in required_keys:
                    assert key in info, f"Layout info should contain {key}"

            except ImportError:
                raise AssertionError("ResponsiveLayout module not available")

        await self._run_test("Responsive Layout", test_layout_calculations)

    # Service Component Tests

    async def _test_analysis_service(self):
        """Test analysis service functionality"""

        async def test_analysis_operations():
            try:
                from integration.analysis_service import AnalysisService

                service = AnalysisService()

                # Test available modules
                modules = service.get_available_modules()
                assert isinstance(modules, list), "Should return list of modules"

                # Test solver info
                if modules:
                    solver_info = service.get_solver_info("xfoil")
                    assert solver_info is not None, "Should return solver info"

                # Test parameter validation
                test_params = {"reynolds": 1000000, "mach": 0.1}
                validation = service.validate_parameters(test_params)
                assert validation is not None, "Should validate parameters"

            except ImportError:
                raise AssertionError("AnalysisService module not available")

        await self._run_test("Analysis Service", test_analysis_operations)

    async def _test_data_management(self):
        """Test data management functionality"""

        async def test_database_operations():
            try:
                from data.database import DatabaseManager

                db_manager = DatabaseManager()

                # Test database initialization
                await db_manager.initialize()

                # Test basic CRUD operations
                test_data = {"name": "test_analysis", "type": "airfoil"}

                # Create
                record_id = await db_manager.create_record("analyses", test_data)
                assert record_id is not None, "Should create record"

                # Read
                record = await db_manager.get_record("analyses", record_id)
                assert record is not None, "Should retrieve record"
                assert record["name"] == "test_analysis", "Record data should match"

                # Update
                updated_data = {"name": "updated_analysis"}
                success = await db_manager.update_record(
                    "analyses",
                    record_id,
                    updated_data,
                )
                assert success, "Should update record"

                # Delete
                success = await db_manager.delete_record("analyses", record_id)
                assert success, "Should delete record"

            except ImportError:
                raise AssertionError("DatabaseManager module not available")

        await self._run_test("Data Management", test_database_operations)

    async def _test_export_service(self):
        """Test export service functionality"""

        async def test_export_operations():
            try:
                from core.services import ExportService

                export_service = ExportService()

                # Test supported formats
                formats = export_service.get_supported_formats()
                assert len(formats) > 0, "Should support export formats"

                # Test data export
                test_data = {"results": [1, 2, 3], "metadata": {"type": "test"}}

                for format_type in formats[:2]:  # Test first 2 formats
                    exported = export_service.export_data(test_data, format_type)
                    assert exported is not None, f"Should export to {format_type}"

            except ImportError:
                raise AssertionError("ExportService module not available")

        await self._run_test("Export Service", test_export_operations)

    # Integration Component Tests

    async def _test_solver_manager(self):
        """Test solver manager functionality"""

        async def test_solver_operations():
            try:
                from integration.solver_manager import SolverManager

                solver_manager = SolverManager()

                # Test solver discovery
                solvers = solver_manager.discover_solvers()
                assert isinstance(solvers, list), "Should return list of solvers"

                # Test solver validation
                for solver in solvers[:2]:  # Test first 2 solvers
                    is_available = solver_manager.is_solver_available(solver)
                    assert isinstance(
                        is_available,
                        bool,
                    ), "Should return boolean availability"

                # Test solver info
                if solvers:
                    info = solver_manager.get_solver_info(solvers[0])
                    assert info is not None, "Should return solver info"

            except ImportError:
                raise AssertionError("SolverManager module not available")

        await self._run_test("Solver Manager", test_solver_operations)

    async def _test_parameter_validator(self):
        """Test parameter validator functionality"""

        async def test_validation_rules():
            try:
                from integration.parameter_validator import ParameterValidator

                validator = ParameterValidator()

                # Test validation rules
                test_cases = [
                    ({"reynolds": 1000000}, True),  # Valid
                    ({"reynolds": -1000}, False),  # Invalid (negative)
                    ({"mach": 0.5}, True),  # Valid
                    ({"mach": 2.0}, False),  # Invalid (too high for subsonic)
                ]

                for params, expected_valid in test_cases:
                    result = validator.validate_parameters(params)
                    is_valid = result.get("valid", False)
                    assert is_valid == expected_valid, (
                        f"Validation of {params} should be {expected_valid}"
                    )

            except ImportError:
                raise AssertionError("ParameterValidator module not available")

        await self._run_test("Parameter Validator", test_validation_rules)

    async def _test_result_processor(self):
        """Test result processor functionality"""

        async def test_result_processing():
            try:
                from integration.result_processor import ResultProcessor

                processor = ResultProcessor()

                # Test result processing
                raw_results = {
                    "data": [1, 2, 3, 4, 5],
                    "metadata": {"solver": "xfoil", "type": "polar"},
                }

                processed = processor.process_results(raw_results)
                assert processed is not None, "Should process results"
                assert "processed_data" in processed, "Should contain processed data"

                # Test result formatting
                formatted = processor.format_results(processed, "table")
                assert formatted is not None, "Should format results"

            except ImportError:
                raise AssertionError("ResultProcessor module not available")

        await self._run_test("Result Processor", test_result_processing)

    # Plugin System Tests

    async def _test_plugin_manager(self):
        """Test plugin manager functionality"""

        async def test_plugin_operations():
            try:
                from plugins.manager import PluginManager

                plugin_manager = PluginManager()

                # Test plugin discovery
                plugins = plugin_manager.discover_plugins()
                assert isinstance(plugins, list), "Should return list of plugins"

                # Test plugin validation
                for plugin in plugins[:2]:  # Test first 2 plugins
                    is_valid = plugin_manager.validate_plugin(plugin)
                    assert isinstance(is_valid, bool), "Should return boolean validity"

                # Test plugin API
                api = plugin_manager.get_plugin_api()
                assert api is not None, "Should return plugin API"

            except ImportError:
                raise AssertionError("PluginManager module not available")

        await self._run_test("Plugin Manager", test_plugin_operations)

    async def _test_plugin_api(self):
        """Test plugin API functionality"""

        async def test_api_operations():
            try:
                from plugins.api import PluginAPI

                api = PluginAPI()

                # Test API methods
                methods = api.get_available_methods()
                assert isinstance(methods, list), "Should return list of methods"

                # Test method registration
                def test_method():
                    return "test_result"

                api.register_method("test_method", test_method)

                # Test method execution
                result = api.execute_method("test_method")
                assert result == "test_result", "Should execute registered method"

            except ImportError:
                raise AssertionError("PluginAPI module not available")

        await self._run_test("Plugin API", test_api_operations)

    # Collaboration System Tests

    async def _test_collaboration_manager(self):
        """Test collaboration manager functionality"""

        async def test_collaboration_operations():
            try:
                from collaboration.collaboration_manager import CollaborationManager

                collab_manager = CollaborationManager()

                # Test session creation
                session_id = await collab_manager.create_session("test_user")
                assert session_id is not None, "Should create collaboration session"

                # Test user management
                success = await collab_manager.add_user_to_session(session_id, "user2")
                assert success, "Should add user to session"

                # Test session info
                info = await collab_manager.get_session_info(session_id)
                assert info is not None, "Should return session info"
                assert len(info.get("users", [])) >= 2, "Should have multiple users"

            except ImportError:
                raise AssertionError("CollaborationManager module not available")

        await self._run_test("Collaboration Manager", test_collaboration_operations)

    async def _test_websocket_manager(self):
        """Test WebSocket manager functionality"""

        async def test_websocket_operations():
            try:
                from api.websocket import WebSocketManager

                ws_manager = WebSocketManager()

                # Create mock WebSocket
                mock_ws = Mock()
                mock_ws.send_text = AsyncMock()
                mock_ws.close = AsyncMock()

                # Test connection management
                connection = await ws_manager.add_connection(mock_ws)
                assert connection is not None, "Should add connection"

                # Test authentication
                await ws_manager.authenticate_connection(
                    connection.session_id,
                    "test_user",
                )

                # Test message sending
                from api.models import WebSocketMessage

                message = WebSocketMessage(type="test", payload={"data": "test"})

                sent = await ws_manager.send_to_session(connection.session_id, message)
                assert sent, "Should send message to session"

                # Test connection removal
                await ws_manager.remove_connection(connection.session_id)

            except ImportError:
                raise AssertionError("WebSocketManager module not available")

        await self._run_test("WebSocket Manager", test_websocket_operations)
