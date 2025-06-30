#!/usr/bin/env python3
"""Comprehensive Test Script for Restructured ICARUS CLI

This script tests all aspects of the restructured CLI including:
- Core framework functionality
- CLI and TUI integration
- State management
- Workflow system
- Export/import services
- Widget functionality
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add cli directory to path
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir))


def test_core_framework():
    """Test core framework components."""
    print("Testing Core Framework...")

    try:
        # Test imports
        from core.state import session_manager, config_manager, history_manager
        from core.ui import theme_manager, notification_system, ui_components
        from core.workflow import workflow_engine, template_manager
        from core.services import validation_service, export_service

        print("✓ Core imports successful")

        # Test state management
        session_info = session_manager.get_session_info()
        assert isinstance(session_info, dict)
        assert "session_id" in session_info
        print("✓ State management working")

        # Test UI components
        themes = list(theme_manager.themes.keys())
        assert len(themes) > 0
        print("✓ UI components working")

        # Test workflow engine
        workflows = workflow_engine.get_workflows()
        assert len(workflows) > 0
        print("✓ Workflow engine working")

        # Test services
        test_data = {"test": "value"}
        errors = validation_service.validate_data(test_data, "airfoil")
        assert isinstance(errors, dict)
        print("✓ Services working")

        return True

    except Exception as e:
        print(f"✗ Core framework test failed: {e}")
        return False


def test_tui_integration():
    """Test TUI integration components."""
    print("Testing TUI Integration...")

    try:
        # Test TUI integration imports
        from core.tui_integration import TUIEventManager, TUIEvent, TUIEventType, TUISessionManager, TUIWorkflowManager

        print("✓ TUI integration imports successful")

        # Test event system
        event_manager = TUIEventManager()
        callback_called = False

        def test_callback(event):
            nonlocal callback_called
            callback_called = True

        event_manager.subscribe(TUIEventType.SESSION_UPDATED, test_callback)

        event = TUIEvent(type=TUIEventType.SESSION_UPDATED, data={"test": "data"}, timestamp=0.0, source="test")
        event_manager.emit(event)

        assert callback_called
        print("✓ Event system working")

        # Test TUI managers
        session_manager = TUISessionManager(event_manager)
        workflow_manager = TUIWorkflowManager(event_manager)

        session_info = session_manager.update_session_info()
        assert isinstance(session_info, dict)
        print("✓ TUI managers working")

        return True

    except Exception as e:
        print(f"✗ TUI integration test failed: {e}")
        return False


def test_tui_widgets():
    """Test TUI widgets."""
    print("Testing TUI Widgets...")

    try:
        # Test widget imports
        from tui.widgets.session_widget import SessionWidget
        from tui.widgets.workflow_widget import WorkflowWidget
        from tui.widgets.analysis_widget import AnalysisWidget
        from tui.widgets.results_widget import ResultsWidget
        from tui.widgets.progress_widget import ProgressWidget
        from tui.widgets.notification_widget import NotificationWidget

        print("✓ Widget imports successful")

        # Test widget instantiation
        session_widget = SessionWidget()
        workflow_widget = WorkflowWidget()
        analysis_widget = AnalysisWidget()
        results_widget = ResultsWidget()
        progress_widget = ProgressWidget()
        notification_widget = NotificationWidget()
        print("✓ Widget instantiation successful")

        # Test widget methods
        session_widget.update_session_info()
        workflow_widget.load_workflows()
        analysis_widget.update_analysis_config()
        results_widget.clear_results()
        progress_widget.update_progress(50.0, "Test")
        notification_widget.add_notification("Test", "info")
        print("✓ Widget methods working")

        return True

    except Exception as e:
        print(f"✗ TUI widgets test failed: {e}")
        return False


def test_tui_utilities():
    """Test TUI utilities."""
    print("Testing TUI Utilities...")

    try:
        # Test utility imports
        from tui.utils.event_helpers import EventHelper
        from tui.utils.theme_helpers import ThemeHelper
        from tui.utils.data_helpers import DataHelper
        from tui.utils.validation_helpers import ValidationHelper

        print("✓ Utility imports successful")

        # Test utility instantiation
        from core.tui_integration import TUIEventManager

        event_manager = TUIEventManager()

        event_helper = EventHelper(event_manager)
        theme_helper = ThemeHelper(Mock())
        data_helper = DataHelper()
        validation_helper = ValidationHelper()
        print("✓ Utility instantiation successful")

        # Test utility methods
        subscription_id = event_helper.subscribe(TUIEventType.SESSION_UPDATED, lambda e: None, "test")
        assert subscription_id is not None

        theme_info = theme_helper.get_current_theme_info()
        assert isinstance(theme_info, dict)

        session_data = data_helper.get_session_data()
        assert isinstance(session_data, dict)

        validation_result = validation_helper.validate_airfoil_data({"name": "test"})
        assert hasattr(validation_result, "is_valid")
        print("✓ Utility methods working")

        return True

    except Exception as e:
        print(f"✗ TUI utilities test failed: {e}")
        return False


def test_cli_commands():
    """Test CLI command structure."""
    print("Testing CLI Commands...")

    try:
        # Test enhanced CLI import
        from enhanced_main import app, cli

        print("✓ Enhanced CLI import successful")

        # Test CLI initialization
        assert cli is not None
        assert hasattr(cli, "db")
        print("✓ CLI initialization successful")

        # Test CLI methods
        cli.show_banner()
        cli.show_session_info()
        print("✓ CLI methods working")

        return True

    except Exception as e:
        print(f"✗ CLI commands test failed: {e}")
        return False


def test_state_persistence():
    """Test state persistence across sessions."""
    print("Testing State Persistence...")

    try:
        # Create temporary config directory
        temp_dir = tempfile.mkdtemp()

        from core.state import SessionManager, ConfigManager

        # Create session manager
        session_manager = SessionManager(config_dir=temp_dir)

        # Add some data
        session_manager.add_airfoil("naca2412")
        session_manager.set_result("test_key", "test_value")

        # Create new session manager (should load existing data)
        new_session_manager = SessionManager(config_dir=temp_dir)

        # Check data persistence
        assert "naca2412" in new_session_manager.current_session.airfoils
        assert new_session_manager.get_result("test_key") == "test_value"
        print("✓ State persistence working")

        # Clean up
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ State persistence test failed: {e}")
        return False


def test_workflow_system():
    """Test workflow system."""
    print("Testing Workflow System...")

    try:
        from core.workflow import workflow_engine, template_manager

        # Test workflow availability
        workflows = workflow_engine.get_workflows()
        assert len(workflows) > 0

        # Test workflow types
        workflow_types = [w.type for w in workflows]
        from core.workflow import WorkflowType

        assert WorkflowType.AIRFOIL_ANALYSIS in workflow_types
        print("✓ Workflow system working")

        return True

    except Exception as e:
        print(f"✗ Workflow system test failed: {e}")
        return False


def test_export_services():
    """Test export and import services."""
    print("Testing Export Services...")

    try:
        from core.services import export_service

        # Test data export/import
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            # Export
            success = export_service.export_data(test_data, f.name, "json")
            assert success

            # Import
            imported_data = export_service.import_data(f.name, "json")
            assert imported_data == test_data

            # Clean up
            os.unlink(f.name)

        print("✓ Export services working")

        return True

    except Exception as e:
        print(f"✗ Export services test failed: {e}")
        return False


def test_validation_system():
    """Test validation system."""
    print("Testing Validation System...")

    try:
        from core.services import validation_service

        # Test airfoil validation
        valid_data = {"name": "naca2412", "reynolds": 1e6, "angles": "0:15:16"}
        errors = validation_service.validate_data(valid_data, "airfoil")
        assert len(errors) == 0

        # Test invalid data
        invalid_data = {"name": "", "reynolds": -1, "angles": "invalid"}
        errors = validation_service.validate_data(invalid_data, "airfoil")
        assert len(errors) > 0

        print("✓ Validation system working")

        return True

    except Exception as e:
        print(f"✗ Validation system test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("ICARUS CLI v2.0 - Comprehensive Test Suite")
    print("=" * 50)

    tests = [
        ("Core Framework", test_core_framework),
        ("TUI Integration", test_tui_integration),
        ("TUI Widgets", test_tui_widgets),
        ("TUI Utilities", test_tui_utilities),
        ("CLI Commands", test_cli_commands),
        ("State Persistence", test_state_persistence),
        ("Workflow System", test_workflow_system),
        ("Export Services", test_export_services),
        ("Validation System", test_validation_system),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The restructured CLI is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
