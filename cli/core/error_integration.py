"""Error Handling Integration Module

This module integrates all error handling components and provides
a unified interface for comprehensive error management throughout
the ICARUS CLI system.
"""

import atexit
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from ..learning.error_system import ErrorExplanationSystem
from .error_analytics import error_analytics
from .error_handler import ErrorContext
from .error_handler import ErrorHandler
from .error_handler import ErrorRecord
from .error_handler import ErrorSeverity
from .graceful_degradation import degradation_manager


class IntegratedErrorManager:
    """Integrated error management system that coordinates all error handling components."""

    def __init__(
        self,
        log_directory: Optional[Path] = None,
        enable_analytics: bool = True,
        enable_degradation: bool = True,
    ):
        self.log_directory = log_directory or Path("cli/logs")
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self.error_handler = ErrorHandler(self.log_directory / "errors.json")
        self.analytics = error_analytics if enable_analytics else None
        self.degradation_manager = degradation_manager if enable_degradation else None
        self.explanation_system = ErrorExplanationSystem()

        # Configuration
        self.auto_recovery_enabled = True
        self.user_notification_enabled = True
        self.analytics_enabled = enable_analytics
        self.degradation_enabled = enable_degradation

        # State tracking
        self.session_errors: List[ErrorRecord] = []
        self.critical_error_count = 0
        self.shutdown_handlers: List[Callable] = []

        # Setup system-level error handling
        self._setup_system_error_handling()

        # Register cleanup
        atexit.register(self._cleanup)

    def _setup_system_error_handling(self):
        """Setup system-level error handling."""

        # Handle uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Handle graceful shutdown
                self._handle_shutdown()
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Handle other uncaught exceptions
            context = ErrorContext(
                component="system",
                operation="uncaught_exception",
                system_state={"exc_type": str(exc_type)},
            )

            error_record = self.handle_error(exc_value, context, auto_recover=False)

            if error_record.severity == ErrorSeverity.CRITICAL:
                self._handle_critical_error(error_record)

            # Call original exception handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception

        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            self._handle_shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        auto_recover: Optional[bool] = None,
    ) -> ErrorRecord:
        """Handle an error with full integration of all error management components."""

        if auto_recover is None:
            auto_recover = self.auto_recovery_enabled

        # Step 1: Basic error handling and recovery
        error_record = self.error_handler.handle_error(error, context, auto_recover)
        self.session_errors.append(error_record)

        # Step 2: Record in analytics if enabled
        if self.analytics_enabled and self.analytics:
            self.analytics.record_error(error_record)

        # Step 3: Check for degradation needs
        if self.degradation_enabled and self.degradation_manager:
            self._handle_degradation(error_record)

        # Step 4: Update system health
        self._update_system_health(error_record)

        # Step 5: Handle critical errors
        if error_record.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_record)

        return error_record

    def _handle_degradation(self, error_record: ErrorRecord):
        """Handle system degradation based on error."""

        # Check if error affects system dependencies
        component = error_record.component
        error_message = error_record.error_message.lower()

        # Update component status if dependency-related
        if any(
            dep in error_message
            for dep in ["import", "module", "package", "executable"]
        ):
            self.degradation_manager.update_component_status()

            # Get degradation report
            report = self.degradation_manager.get_degradation_report()

            # Log degradation changes
            if report.overall_level.value != "none":
                self.error_handler.logger.warning(
                    f"System degradation detected: {report.overall_level.value}",
                )

                if self.user_notification_enabled:
                    self._notify_degradation(report)

    def _update_system_health(self, error_record: ErrorRecord):
        """Update system health metrics."""
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.critical_error_count += 1

        # Trigger health check if too many errors
        if len(self.session_errors) % 10 == 0:  # Every 10 errors
            health_report = self.error_handler.check_system_health()

            if health_report["overall_status"] != "healthy":
                self.error_handler.logger.warning(
                    f"System health degraded: {health_report['overall_status']}",
                )

    def _handle_critical_error(self, error_record: ErrorRecord):
        """Handle critical errors that may require system shutdown."""
        self.error_handler.logger.critical(
            f"Critical error in {error_record.component}: {error_record.error_message}",
        )

        # Notify all registered shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler(error_record)
            except Exception as e:
                self.error_handler.logger.error(f"Shutdown handler failed: {e}")

        # If too many critical errors, consider emergency shutdown
        if self.critical_error_count >= 3:
            self.error_handler.logger.critical(
                "Too many critical errors - initiating emergency shutdown",
            )
            self._emergency_shutdown()

    def _notify_degradation(self, report):
        """Notify user about system degradation."""
        print(f"\n⚠️  System Degradation Detected: {report.overall_level.value.title()}")
        print(f"Impact: {report.user_impact}")

        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations[:3]:  # Show top 3
                print(f"  • {rec}")

    def _handle_shutdown(self):
        """Handle graceful shutdown."""
        self.error_handler.logger.info("Initiating graceful shutdown")

        # Generate final reports
        if self.analytics_enabled and self.analytics:
            try:
                report = self.analytics.get_error_report(days_back=1)
                report_file = (
                    self.log_directory
                    / f"session_report_{report['report_period']['end'][:10]}.json"
                )

                import json

                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)

                self.error_handler.logger.info(f"Session report saved to {report_file}")
            except Exception as e:
                self.error_handler.logger.error(f"Failed to save session report: {e}")

        # Save degradation status
        if self.degradation_enabled and self.degradation_manager:
            try:
                report = self.degradation_manager.get_degradation_report()
                if report.overall_level.value != "none":
                    self.error_handler.logger.info(
                        f"Final degradation status: {report.overall_level.value}",
                    )
            except Exception as e:
                self.error_handler.logger.error(
                    f"Failed to get degradation status: {e}",
                )

    def _emergency_shutdown(self):
        """Emergency shutdown procedure."""
        print("\n🚨 EMERGENCY SHUTDOWN - Critical system errors detected")
        print("Saving critical data and shutting down...")

        # Save emergency data
        try:
            emergency_data = {
                "timestamp": self.session_errors[-1].timestamp.isoformat()
                if self.session_errors
                else None,
                "critical_errors": self.critical_error_count,
                "last_errors": [
                    {
                        "component": err.component,
                        "operation": err.operation,
                        "message": err.error_message,
                        "severity": err.severity.value,
                    }
                    for err in self.session_errors[-5:]  # Last 5 errors
                ],
            }

            emergency_file = self.log_directory / "emergency_shutdown.json"
            import json

            with open(emergency_file, "w") as f:
                json.dump(emergency_data, f, indent=2)

            print(f"Emergency data saved to {emergency_file}")
        except Exception as e:
            print(f"Failed to save emergency data: {e}")

        sys.exit(1)

    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            self._handle_shutdown()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def register_shutdown_handler(self, handler: Callable[[ErrorRecord], None]):
        """Register a handler to be called on critical errors."""
        self.shutdown_handlers.append(handler)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of errors in current session."""
        if not self.session_errors:
            return {"total_errors": 0, "status": "clean"}

        summary = {
            "total_errors": len(self.session_errors),
            "critical_errors": self.critical_error_count,
            "by_severity": {},
            "by_component": {},
            "recovery_rate": 0,
            "most_recent": None,
        }

        # Count by severity
        for severity in ErrorSeverity:
            count = sum(1 for err in self.session_errors if err.severity == severity)
            summary["by_severity"][severity.value] = count

        # Count by component
        components = {}
        for err in self.session_errors:
            components[err.component] = components.get(err.component, 0) + 1
        summary["by_component"] = components

        # Calculate recovery rate
        recovery_attempts = sum(
            1 for err in self.session_errors if err.recovery_attempted
        )
        if recovery_attempts > 0:
            successful = sum(
                1
                for err in self.session_errors
                if err.recovery_attempted and err.recovery_successful
            )
            summary["recovery_rate"] = successful / recovery_attempts

        # Most recent error
        if self.session_errors:
            recent = self.session_errors[-1]
            summary["most_recent"] = {
                "component": recent.component,
                "operation": recent.operation,
                "message": recent.error_message,
                "severity": recent.severity.value,
                "timestamp": recent.timestamp.isoformat(),
            }

        return summary

    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check."""
        health_data = {}

        # Basic error handler health
        health_data["error_handler"] = self.error_handler.check_system_health()

        # Analytics health
        if self.analytics_enabled and self.analytics:
            health_data["analytics"] = self.analytics.calculate_system_health()

        # Degradation status
        if self.degradation_enabled and self.degradation_manager:
            health_data["degradation"] = (
                self.degradation_manager.get_degradation_report()
            )

        # Session summary
        health_data["session"] = self.get_session_summary()

        return health_data

    def export_error_data(self, filepath: Path, format: str = "json"):
        """Export comprehensive error data."""
        if self.analytics_enabled and self.analytics:
            self.analytics.export_analytics_data(filepath, format)
        else:
            # Export basic error data
            data = {
                "session_errors": [
                    {
                        "error_id": err.error_id,
                        "timestamp": err.timestamp.isoformat(),
                        "severity": err.severity.value,
                        "component": err.component,
                        "operation": err.operation,
                        "error_message": err.error_message,
                        "recovery_attempted": err.recovery_attempted,
                        "recovery_successful": err.recovery_successful,
                    }
                    for err in self.session_errors
                ],
                "summary": self.get_session_summary(),
            }

            import json

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)


# Context managers for error handling
@contextmanager
def error_context(component: str, operation: str, **context_data):
    """Context manager for handling errors in a specific context."""
    context = ErrorContext(
        component=component,
        operation=operation,
        user_data=context_data,
    )

    try:
        yield context
    except Exception as e:
        integrated_error_manager.handle_error(e, context)
        raise


@contextmanager
def safe_operation(
    component: str,
    operation: str,
    fallback_result=None,
    **context_data,
):
    """Context manager for safe operations that don't re-raise exceptions."""
    context = ErrorContext(
        component=component,
        operation=operation,
        user_data=context_data,
    )

    try:
        yield context
    except Exception as e:
        integrated_error_manager.handle_error(e, context)
        return fallback_result


# Async context manager
class AsyncErrorContext:
    """Async context manager for error handling."""

    def __init__(self, component: str, operation: str, **context_data):
        self.context = ErrorContext(
            component=component,
            operation=operation,
            user_data=context_data,
        )

    async def __aenter__(self):
        return self.context

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            integrated_error_manager.handle_error(exc_val, self.context)
        return False  # Don't suppress exceptions


# Global integrated error manager
integrated_error_manager = IntegratedErrorManager()


# Convenience functions
def handle_error(
    error: Exception,
    component: str,
    operation: str,
    **context_data,
) -> ErrorRecord:
    """Convenience function for handling errors."""
    context = ErrorContext(
        component=component,
        operation=operation,
        user_data=context_data,
    )
    return integrated_error_manager.handle_error(error, context)


def get_error_summary() -> Dict[str, Any]:
    """Get current session error summary."""
    return integrated_error_manager.get_session_summary()


def run_health_check() -> Dict[str, Any]:
    """Run system health check."""
    return integrated_error_manager.run_health_check()


def check_system_dependencies() -> Dict[str, Any]:
    """Check system dependencies and degradation status."""
    if integrated_error_manager.degradation_enabled:
        return integrated_error_manager.degradation_manager.get_degradation_report()
    return {"status": "degradation_checking_disabled"}


# Decorator for automatic error handling with integration
def integrated_error_handler(component: str, operation: str = None):
    """Decorator for integrated error handling."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            context = ErrorContext(
                component=component,
                operation=op_name,
                user_data={"function": func.__name__},
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                integrated_error_manager.handle_error(e, context)
                raise

        return wrapper

    return decorator
