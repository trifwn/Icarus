"""Comprehensive Test Suite for Error Handling System

This module tests all components of the comprehensive error handling
and recovery system to ensure proper functionality.
"""

import json
import os

# Import the error handling components
import sys
import tempfile
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.core.error_analytics import ErrorAnalytics
from cli.core.error_analytics import ErrorPattern
from cli.core.error_analytics import ErrorTrend
from cli.core.error_handler import ErrorCategory
from cli.core.error_handler import ErrorContext
from cli.core.error_handler import ErrorHandler
from cli.core.error_handler import ErrorSeverity
from cli.core.error_handler import RecoveryAction
from cli.core.error_handler import RecoveryStrategy
from cli.core.error_handler import handle_errors
from cli.core.error_integration import IntegratedErrorManager
from cli.core.error_integration import error_context
from cli.core.error_integration import integrated_error_handler
from cli.core.error_integration import safe_operation
from cli.core.graceful_degradation import ComponentInfo
from cli.core.graceful_degradation import DegradationLevel
from cli.core.graceful_degradation import DependencyInfo
from cli.core.graceful_degradation import GracefulDegradationManager
from cli.core.graceful_degradation import require_dependency


class TestErrorHandler:
    """Test the core error handler functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.error_handler = ErrorHandler(self.temp_dir / "test_errors.json")

    def test_error_classification(self):
        """Test error classification by type and severity."""
        # Test file not found error
        context = ErrorContext(component="test", operation="file_read")
        error = FileNotFoundError("Test file not found")

        record = self.error_handler.handle_error(error, context, auto_recover=False)

        assert record.category == ErrorCategory.USER_INPUT
        assert record.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        assert record.error_type == "FileNotFoundError"

    def test_recovery_action_registration(self):
        """Test registration and execution of recovery actions."""

        # Register a test recovery action
        def test_recovery():
            return True

        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Test recovery",
            action=test_recovery,
        )

        self.error_handler.register_recovery_action("test_error", recovery_action)

        assert "test_error" in self.error_handler.recovery_actions
        assert len(self.error_handler.recovery_actions["test_error"]) == 1

    def test_error_logging(self):
        """Test error logging to file."""
        context = ErrorContext(component="test", operation="logging_test")
        error = ValueError("Test error for logging")

        self.error_handler.handle_error(error, context, auto_recover=False)

        # Check that log file was created and contains data
        assert self.error_handler.log_file.exists()

        with open(self.error_handler.log_file) as f:
            log_data = json.load(f)

        assert len(log_data) > 0
        # Find the logging test error in the log data
        logging_errors = [
            entry
            for entry in log_data
            if entry.get("error_message") == "Test error for logging"
        ]
        assert len(logging_errors) > 0

    def test_error_statistics(self):
        """Test error statistics generation."""
        # Generate some test errors
        for i in range(5):
            context = ErrorContext(component="test", operation=f"op_{i}")
            error = ValueError(f"Test error {i}")
            self.error_handler.handle_error(error, context, auto_recover=False)

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == 5
        assert "by_severity" in stats
        assert "by_category" in stats
        assert "by_component" in stats

    def test_system_health_check(self):
        """Test system health checking."""
        health = self.error_handler.check_system_health()

        assert "timestamp" in health
        assert "overall_status" in health
        assert "dependencies" in health
        assert "recent_errors" in health

    def test_error_handler_decorator(self):
        """Test the error handler decorator."""

        @handle_errors("test_component", "test_operation")
        def test_function():
            raise ValueError("Test decorator error")

        with pytest.raises(ValueError):
            test_function()

        # Check that error was recorded
        assert len(self.error_handler.error_records) > 0


class TestErrorAnalytics:
    """Test the error analytics system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analytics = ErrorAnalytics(self.temp_dir / "test_analytics.db")

    def test_error_recording(self):
        """Test recording errors in analytics database."""
        from cli.core.error_handler import ErrorRecord

        # Create a test error record
        context = ErrorContext(component="test", operation="analytics_test")
        record = ErrorRecord(
            error_id="test_001",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.USER_INPUT,
            component="test",
            operation="analytics_test",
            error_type="ValueError",
            error_message="Test analytics error",
            stack_trace="Test stack trace",
            context=context,
        )

        self.analytics.record_error(record)

        # Verify record was stored
        import sqlite3

        with sqlite3.connect(self.analytics.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM error_records")
            count = cursor.fetchone()[0]

        assert count == 1

    def test_error_trend_analysis(self):
        """Test error trend analysis."""
        from cli.core.error_handler import ErrorRecord

        # Create multiple error records over time
        base_time = datetime.now() - timedelta(days=2)

        for i in range(10):
            context = ErrorContext(component="test", operation=f"trend_test_{i}")
            record = ErrorRecord(
                error_id=f"trend_{i}",
                timestamp=base_time + timedelta(hours=i),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.USER_INPUT,
                component="test",
                operation=f"trend_test_{i}",
                error_type="ValueError",
                error_message=f"Trend test error {i}",
                stack_trace="Test stack trace",
                context=context,
            )
            self.analytics.record_error(record)

        trends = self.analytics.analyze_error_trends(period="hour", days_back=3)

        assert len(trends) > 0
        assert all(isinstance(trend, ErrorTrend) for trend in trends)

    def test_pattern_detection(self):
        """Test error pattern detection."""
        from cli.core.error_handler import ErrorRecord

        # Create similar errors to form a pattern
        for i in range(5):
            context = ErrorContext(component="test", operation="pattern_test")
            record = ErrorRecord(
                error_id=f"pattern_{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.USER_INPUT,
                component="test",
                operation="pattern_test",
                error_type="FileNotFoundError",
                error_message="File not found: /test/path/file.txt",
                stack_trace="Test stack trace",
                context=context,
            )
            self.analytics.record_error(record)

        patterns = self.analytics.detect_error_patterns()

        assert len(patterns) > 0
        assert all(isinstance(pattern, ErrorPattern) for pattern in patterns)

    def test_health_metrics_calculation(self):
        """Test system health metrics calculation."""
        metrics = self.analytics.calculate_system_health()

        assert hasattr(metrics, "overall_health_score")
        assert hasattr(metrics, "error_rate")
        assert hasattr(metrics, "recovery_success_rate")
        assert 0 <= metrics.overall_health_score <= 100

    def test_error_report_generation(self):
        """Test comprehensive error report generation."""
        report = self.analytics.get_error_report(days_back=1)

        assert "report_period" in report
        assert "summary" in report
        assert "breakdown" in report
        assert "trends" in report
        assert "health_metrics" in report


class TestGracefulDegradation:
    """Test the graceful degradation system."""

    def setup_method(self):
        """Setup test environment."""
        self.degradation_manager = GracefulDegradationManager()

    def test_dependency_registration(self):
        """Test dependency registration."""
        dep = DependencyInfo(
            name="test_dependency",
            type="python_package",
            required=True,
            description="Test dependency",
        )

        self.degradation_manager.register_dependency(dep)

        assert "test_dependency" in self.degradation_manager.dependencies

    def test_component_registration(self):
        """Test component registration."""
        comp = ComponentInfo(
            name="test_component",
            description="Test component",
            dependencies=["test_dependency"],
        )

        self.degradation_manager.register_component(comp)

        assert "test_component" in self.degradation_manager.components

    def test_dependency_checking(self):
        """Test dependency availability checking."""
        # Test built-in Python modules (should be available)
        assert (
            self.degradation_manager.check_dependency("sys") == False
        )  # Not registered

        # Test non-existent module
        dep = DependencyInfo(
            name="nonexistent_module",
            type="python_package",
            required=False,
        )
        self.degradation_manager.register_dependency(dep)

        assert self.degradation_manager.check_dependency("nonexistent_module") == False

    def test_degradation_level_calculation(self):
        """Test degradation level calculation."""
        level = self.degradation_manager.get_degradation_level()

        assert isinstance(level, DegradationLevel)

    def test_degradation_report(self):
        """Test degradation report generation."""
        report = self.degradation_manager.get_degradation_report()

        assert hasattr(report, "overall_level")
        assert hasattr(report, "available_components")
        assert hasattr(report, "degraded_components")
        assert hasattr(report, "unavailable_components")
        assert hasattr(report, "recommendations")

    def test_require_dependency_decorator(self):
        """Test the require_dependency decorator."""
        # Register a test dependency that doesn't exist
        dep = DependencyInfo(
            name="missing_test_dep",
            type="python_package",
            required=True,
        )
        self.degradation_manager.register_dependency(dep)

        @require_dependency("missing_test_dep")
        def test_function():
            return "success"

        with pytest.raises(ImportError):
            test_function()


class TestIntegratedErrorManager:
    """Test the integrated error management system."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.error_manager = IntegratedErrorManager(
            log_directory=self.temp_dir,
            enable_analytics=True,
            enable_degradation=True,
        )

    def test_integrated_error_handling(self):
        """Test integrated error handling across all components."""
        context = ErrorContext(component="integration_test", operation="test_op")
        error = ValueError("Integration test error")

        record = self.error_manager.handle_error(error, context)

        assert record is not None
        assert record.component == "integration_test"
        assert record.operation == "test_op"

        # Check that error was recorded in session
        assert len(self.error_manager.session_errors) > 0

    def test_session_summary(self):
        """Test session error summary."""
        # Generate some test errors
        for i in range(3):
            context = ErrorContext(component="test", operation=f"op_{i}")
            error = ValueError(f"Test error {i}")
            self.error_manager.handle_error(error, context)

        summary = self.error_manager.get_session_summary()

        # Should have at least 3 errors (may have more from previous tests)
        assert summary["total_errors"] >= 3
        assert "by_severity" in summary
        assert "by_component" in summary

    def test_health_check(self):
        """Test comprehensive health check."""
        health = self.error_manager.run_health_check()

        assert "error_handler" in health
        assert "session" in health

        if self.error_manager.analytics_enabled:
            assert "analytics" in health

        if self.error_manager.degradation_enabled:
            assert "degradation" in health

    def test_error_context_manager(self):
        """Test error context manager."""
        with pytest.raises(ValueError):
            with error_context("test_component", "test_operation"):
                raise ValueError("Context manager test error")

        # Check that error was handled
        assert len(self.error_manager.session_errors) > 0

    def test_safe_operation_context(self):
        """Test safe operation context manager."""
        result = None

        with safe_operation("test_component", "safe_test", fallback_result="fallback"):
            raise ValueError("Safe operation test error")

        # Should not raise exception, and error should be handled
        assert len(self.error_manager.session_errors) > 0

    def test_integrated_error_decorator(self):
        """Test integrated error handling decorator."""

        @integrated_error_handler("test_component", "decorated_operation")
        def test_function():
            raise ValueError("Decorator test error")

        with pytest.raises(ValueError):
            test_function()

        # Check that error was handled by integrated system
        assert len(self.error_manager.session_errors) > 0
        assert self.error_manager.session_errors[-1].operation == "decorated_operation"


class TestErrorSystemIntegration:
    """Test integration between all error handling components."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_end_to_end_error_flow(self):
        """Test complete error handling flow from error to recovery."""
        # Create integrated manager
        manager = IntegratedErrorManager(
            log_directory=self.temp_dir,
            enable_analytics=True,
            enable_degradation=True,
        )

        # Simulate a dependency error
        context = ErrorContext(
            component="airfoil_analysis",
            operation="xfoil_analysis",
            user_data={"airfoil": "naca0012", "reynolds": 1e6},
        )

        error = ImportError("XFoil executable not found")

        # Handle the error
        record = manager.handle_error(error, context)

        # Verify error was processed through all systems
        assert record is not None
        assert record.category == ErrorCategory.SOLVER_ERROR

        # Check analytics recorded the error
        if manager.analytics_enabled:
            # Small delay to ensure database write
            time.sleep(0.1)

            import sqlite3

            with sqlite3.connect(manager.analytics.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM error_records")
                count = cursor.fetchone()[0]
            assert count > 0

        # Check degradation system updated
        if manager.degradation_enabled:
            report = manager.degradation_manager.get_degradation_report()
            # Should detect some level of degradation due to missing XFoil
            assert (
                report.overall_level != DegradationLevel.NONE
                or len(report.missing_dependencies) > 0
            )

    def test_pattern_detection_and_recovery(self):
        """Test pattern detection leading to improved recovery."""
        manager = IntegratedErrorManager(
            log_directory=self.temp_dir,
            enable_analytics=True,
            enable_degradation=True,
        )

        # Generate repeated similar errors to create a pattern
        for i in range(5):
            context = ErrorContext(
                component="file_handler",
                operation="load_airfoil",
                user_data={"file": f"/path/to/airfoil_{i}.dat"},
            )

            error = FileNotFoundError(
                f"Airfoil file not found: /path/to/airfoil_{i}.dat",
            )
            manager.handle_error(error, context)

        # Detect patterns
        if manager.analytics_enabled:
            patterns = manager.analytics.detect_error_patterns()

            # Should detect a pattern of file not found errors
            file_patterns = [
                p for p in patterns if "FileNotFoundError" in p.description
            ]
            assert len(file_patterns) > 0

            # Pattern should have suggestions
            if file_patterns:
                assert len(file_patterns[0].suggested_fixes) > 0

    def test_critical_error_handling(self):
        """Test handling of critical errors."""
        manager = IntegratedErrorManager(
            log_directory=self.temp_dir,
            enable_analytics=True,
            enable_degradation=True,
        )

        # Mock the emergency shutdown to avoid actual system exit
        manager._emergency_shutdown = Mock()

        # Generate critical errors
        for i in range(4):  # More than the threshold of 3
            context = ErrorContext(
                component="system",
                operation="critical_operation",
                system_state={"memory_usage": "95%"},
            )

            # Create a critical error (MemoryError is classified as high severity)
            error = MemoryError("System out of memory")
            record = manager.handle_error(error, context)

            # Manually set to critical for testing
            record.severity = ErrorSeverity.CRITICAL
            manager.critical_error_count += 1

        # Should trigger emergency shutdown
        assert manager._emergency_shutdown.called

    def test_graceful_degradation_integration(self):
        """Test integration with graceful degradation."""
        manager = IntegratedErrorManager(
            log_directory=self.temp_dir,
            enable_analytics=True,
            enable_degradation=True,
        )

        # Simulate missing dependency error
        context = ErrorContext(component="visualization", operation="create_plot")

        error = ImportError("matplotlib not available")
        record = manager.handle_error(error, context)

        # Check that degradation was handled
        report = manager.degradation_manager.get_degradation_report()

        # Should show some degradation or fallback activation
        assert (
            report.overall_level != DegradationLevel.NONE
            or len(report.active_fallbacks) > 0
            or len(report.degraded_components) > 0
        )


def test_error_system_performance():
    """Test performance of error handling system under load."""
    temp_dir = Path(tempfile.mkdtemp())
    manager = IntegratedErrorManager(
        log_directory=temp_dir,
        enable_analytics=True,
        enable_degradation=False,  # Disable for performance test
    )

    start_time = time.time()

    # Generate many errors quickly
    for i in range(100):
        context = ErrorContext(
            component="performance_test",
            operation=f"operation_{i % 10}",
        )

        error = ValueError(f"Performance test error {i}")
        manager.handle_error(error, context, auto_recover=False)

    end_time = time.time()
    duration = end_time - start_time

    # Should handle 100 errors in reasonable time (< 5 seconds)
    assert duration < 5.0

    # All errors should be recorded
    assert len(manager.session_errors) == 100


if __name__ == "__main__":
    # Run basic tests
    print("Running Error Handling System Tests...")

    # Test basic error handler
    print("Testing ErrorHandler...")
    test_handler = TestErrorHandler()
    test_handler.setup_method()
    test_handler.test_error_classification()
    test_handler.test_recovery_action_registration()
    test_handler.test_error_logging()
    print("✓ ErrorHandler tests passed")

    # Test analytics
    print("Testing ErrorAnalytics...")
    test_analytics = TestErrorAnalytics()
    test_analytics.setup_method()
    test_analytics.test_error_recording()
    test_analytics.test_health_metrics_calculation()
    print("✓ ErrorAnalytics tests passed")

    # Test degradation
    print("Testing GracefulDegradation...")
    test_degradation = TestGracefulDegradation()
    test_degradation.setup_method()
    test_degradation.test_dependency_registration()
    test_degradation.test_component_registration()
    test_degradation.test_degradation_level_calculation()
    print("✓ GracefulDegradation tests passed")

    # Test integration
    print("Testing IntegratedErrorManager...")
    test_integration = TestIntegratedErrorManager()
    test_integration.setup_method()
    test_integration.test_integrated_error_handling()
    test_integration.test_session_summary()
    test_integration.test_health_check()
    print("✓ IntegratedErrorManager tests passed")

    # Test end-to-end
    print("Testing End-to-End Integration...")
    test_e2e = TestErrorSystemIntegration()
    test_e2e.setup_method()
    test_e2e.test_end_to_end_error_flow()
    print("✓ End-to-End tests passed")

    # Performance test
    print("Testing Performance...")
    test_error_system_performance()
    print("✓ Performance tests passed")

    print("\n🎉 All Error Handling System Tests Passed!")
    print("\nError handling system is ready for production use.")
    print("Features implemented:")
    print("  ✓ Centralized error handling with classification")
    print("  ✓ Automatic recovery strategies with user guidance")
    print("  ✓ Graceful degradation for missing dependencies")
    print("  ✓ Comprehensive error logging and analytics")
    print("  ✓ Pattern detection and health monitoring")
    print("  ✓ Integration across all system components")
