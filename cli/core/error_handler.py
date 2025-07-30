"""Comprehensive Error Handling and Recovery System

This module provides centralized error handling, classification, recovery strategies,
graceful degradation, and error logging/analytics for the ICARUS CLI.
"""

import json
import logging
import traceback
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from ..learning.error_system import ErrorCategory
from ..learning.error_system import ErrorExplanationSystem


class ErrorSeverity(Enum):
    """Error severity levels."""

    CRITICAL = "critical"  # System cannot continue
    HIGH = "high"  # Major functionality affected
    MEDIUM = "medium"  # Some functionality affected
    LOW = "low"  # Minor issues, warnings
    INFO = "info"  # Informational messages


class RecoveryStrategy(Enum):
    """Error recovery strategies."""

    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use alternative approach
    GRACEFUL_DEGRADATION = "degrade"  # Reduce functionality
    USER_INTERVENTION = "user"  # Require user action
    RESTART_COMPONENT = "restart"  # Restart affected component
    IGNORE = "ignore"  # Continue with warning


@dataclass
class ErrorContext:
    """Context information for error handling."""

    component: str
    operation: str
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class RecoveryAction:
    """Represents a recovery action."""

    strategy: RecoveryStrategy
    description: str
    action: Callable[[], Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_message: str = "Recovery successful"
    failure_message: str = "Recovery failed"
    max_attempts: int = 3
    delay_seconds: float = 1.0


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    user_notified: bool = False
    resolved: bool = False


class DependencyChecker:
    """Checks for missing dependencies and provides graceful degradation."""

    def __init__(self):
        self.dependency_cache: Dict[str, bool] = {}
        self.fallback_options: Dict[str, List[str]] = {
            "xfoil": ["foil2wake", "openfoam"],
            "avl": ["gnvp3", "lspt"],
            "matplotlib": ["rich_plotting", "ascii_plots"],
            "numpy": ["python_math"],
            "pandas": ["csv_processing"],
        }

    def check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        if dependency in self.dependency_cache:
            return self.dependency_cache[dependency]

        import subprocess

        try:
            if dependency == "xfoil":
                # Check if XFoil executable is available
                result = subprocess.run(["xfoil", "-h"], capture_output=True, timeout=5)
                available = result.returncode == 0
            elif dependency == "avl":
                # Check if AVL executable is available
                result = subprocess.run(["avl"], capture_output=True, timeout=5)
                available = result.returncode == 0
            else:
                # Check Python package
                __import__(dependency)
                available = True
        except (
            ImportError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            available = False

        self.dependency_cache[dependency] = available
        return available

    def get_fallback_options(self, dependency: str) -> List[str]:
        """Get fallback options for a missing dependency."""
        return self.fallback_options.get(dependency, [])

    def get_available_fallbacks(self, dependency: str) -> List[str]:
        """Get available fallback options for a missing dependency."""
        fallbacks = self.get_fallback_options(dependency)
        return [fb for fb in fallbacks if self.check_dependency(fb)]


class ErrorHandler:
    """Centralized error handling and recovery system."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("cli/logs/error_log.json")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.error_explanation_system = ErrorExplanationSystem()
        self.dependency_checker = DependencyChecker()

        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_handlers: Dict[Type[Exception], List[Callable]] = {}
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}

        # Configuration
        self.max_retry_attempts = 3
        self.enable_auto_recovery = True
        self.enable_graceful_degradation = True
        self.log_level = logging.INFO

        # Setup logging
        self._setup_logging()

        # Register default recovery actions
        self._register_default_recovery_actions()

    def _setup_logging(self):
        """Setup error logging."""
        self.logger = logging.getLogger("icarus_error_handler")
        self.logger.setLevel(self.log_level)

        # File handler
        file_handler = logging.FileHandler(self.log_file.with_suffix(".log"))
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _register_default_recovery_actions(self):
        """Register default recovery actions for common error types."""

        # File not found recovery
        self.register_recovery_action(
            "file_not_found",
            RecoveryAction(
                strategy=RecoveryStrategy.USER_INTERVENTION,
                description="Prompt user to locate missing file",
                action=self._prompt_for_file_location,
                success_message="File located successfully",
                failure_message="Could not locate file",
            ),
        )

        # Memory error recovery
        self.register_recovery_action(
            "memory_error",
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                description="Reduce problem size to fit available memory",
                action=self._reduce_problem_size,
                success_message="Problem size reduced, continuing with limited scope",
                failure_message="Could not reduce problem size sufficiently",
            ),
        )

        # Solver not found recovery
        self.register_recovery_action(
            "solver_not_found",
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Use alternative solver",
                action=self._use_fallback_solver,
                success_message="Using alternative solver",
                failure_message="No alternative solver available",
            ),
        )

        # Network error recovery
        self.register_recovery_action(
            "network_error",
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry network operation",
                action=self._retry_network_operation,
                max_attempts=3,
                delay_seconds=2.0,
                success_message="Network operation successful",
                failure_message="Network operation failed after retries",
            ),
        )

        # Configuration error recovery
        self.register_recovery_action(
            "config_error",
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Reset to default configuration",
                action=self._reset_to_default_config,
                success_message="Configuration reset to defaults",
                failure_message="Could not reset configuration",
            ),
        )

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        auto_recover: bool = None,
    ) -> ErrorRecord:
        """Handle an error with comprehensive recovery strategies."""

        if auto_recover is None:
            auto_recover = self.enable_auto_recovery

        # Create error record
        error_record = self._create_error_record(error, context)
        self.error_records.append(error_record)

        # Log the error
        self._log_error(error_record)

        # Get error explanation
        explanation = self.error_explanation_system.explain_error(str(error))

        # Classify error severity
        severity = self._classify_error_severity(error, context)
        error_record.severity = severity

        # Attempt recovery if enabled
        if auto_recover and severity != ErrorSeverity.CRITICAL:
            recovery_success = self._attempt_recovery(error_record, explanation)
            error_record.recovery_attempted = True
            error_record.recovery_successful = recovery_success

        # Notify user if necessary
        if (
            severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
            or not error_record.recovery_successful
        ):
            self._notify_user(error_record, explanation)
            error_record.user_notified = True

        return error_record

    def _create_error_record(
        self,
        error: Exception,
        context: ErrorContext,
    ) -> ErrorRecord:
        """Create an error record from an exception and context."""
        error_id = (
            f"{context.component}_{context.operation}_{datetime.now().timestamp()}"
        )

        return ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,  # Will be updated by classification
            category=self._categorize_error(error),
            component=context.component,
            operation=context.operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
        )

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        if isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorCategory.USER_INPUT
        elif isinstance(error, MemoryError):
            return ErrorCategory.SYSTEM_ERROR
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK_ERROR
        elif "config" in error_message or "setting" in error_message:
            return ErrorCategory.CONFIGURATION
        elif (
            "xfoil" in error_message
            or "avl" in error_message
            or "solver" in error_message
        ):
            return ErrorCategory.SOLVER_ERROR
        elif "data" in error_message or "format" in error_message:
            return ErrorCategory.DATA_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR

    def _classify_error_severity(
        self,
        error: Exception,
        context: ErrorContext,
    ) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        error_type = type(error).__name__

        # Critical errors that prevent system operation
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM

        # Low severity errors
        if isinstance(error, (UserWarning, DeprecationWarning)):
            return ErrorSeverity.LOW

        # Default to medium
        return ErrorSeverity.MEDIUM

    def _attempt_recovery(self, error_record: ErrorRecord, explanation=None) -> bool:
        """Attempt to recover from an error."""
        error_key = self._get_error_key(error_record)

        # Get recovery actions for this error type
        recovery_actions = self.recovery_actions.get(error_key, [])

        # Try each recovery action
        for action in recovery_actions:
            try:
                self.logger.info(f"Attempting recovery: {action.description}")

                # Execute recovery action with retries
                success = self._execute_recovery_action(action)

                if success:
                    error_record.recovery_strategy = action.strategy
                    self.logger.info(f"Recovery successful: {action.success_message}")
                    return True
                else:
                    self.logger.warning(f"Recovery failed: {action.failure_message}")

            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery action failed with error: {recovery_error}",
                )

        return False

    def _execute_recovery_action(self, action: RecoveryAction) -> bool:
        """Execute a recovery action with retry logic."""
        for attempt in range(action.max_attempts):
            try:
                if attempt > 0:
                    import time

                    time.sleep(action.delay_seconds)

                result = action.action()

                # Consider action successful if it doesn't raise an exception
                # and returns a truthy value (or None)
                return result is not False

            except Exception as e:
                if attempt == action.max_attempts - 1:
                    raise e
                self.logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")

        return False

    def _get_error_key(self, error_record: ErrorRecord) -> str:
        """Get a key for looking up recovery actions."""
        error_message = error_record.error_message.lower()

        if "file not found" in error_message or "no such file" in error_message:
            return "file_not_found"
        elif "memory" in error_message:
            return "memory_error"
        elif "solver" in error_message or "xfoil" in error_message:
            return "solver_not_found"
        elif "network" in error_message or "connection" in error_message:
            return "network_error"
        elif "config" in error_message or "setting" in error_message:
            return "config_error"
        else:
            return "generic_error"

    def _log_error(self, error_record: ErrorRecord):
        """Log an error record."""
        log_data = {
            "error_id": error_record.error_id,
            "timestamp": error_record.timestamp.isoformat(),
            "severity": error_record.severity.value,
            "category": error_record.category.value,
            "component": error_record.component,
            "operation": error_record.operation,
            "error_type": error_record.error_type,
            "error_message": error_record.error_message,
            "context": {
                "user_data": error_record.context.user_data,
                "system_state": error_record.context.system_state,
                "session_id": error_record.context.session_id,
                "user_id": error_record.context.user_id,
            },
        }

        # Log to file
        self._append_to_log_file(log_data)

        # Log to Python logger
        log_message = f"{error_record.component}.{error_record.operation}: {error_record.error_message}"

        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _append_to_log_file(self, log_data: Dict[str, Any]):
        """Append error data to JSON log file."""
        try:
            # Read existing data
            if self.log_file.exists():
                with open(self.log_file) as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Append new data
            existing_data.append(log_data)

            # Write back to file
            with open(self.log_file, "w") as f:
                json.dump(existing_data, f, indent=2, default=str)

        except Exception as e:
            # Fallback to basic logging if JSON logging fails
            self.logger.error(f"Failed to write to JSON log: {e}")

    def _notify_user(self, error_record: ErrorRecord, explanation=None):
        """Notify user about an error."""
        # This would integrate with the UI notification system
        # For now, we'll use console output

        print(f"\n🚨 Error in {error_record.component}")
        print(f"Operation: {error_record.operation}")
        print(f"Error: {error_record.error_message}")

        if explanation:
            print(f"\nExplanation: {explanation.explanation}")

            if explanation.solutions:
                print("\nSuggested solutions:")
                for i, solution in enumerate(explanation.solutions[:3], 1):
                    print(f"{i}. {solution.title}")
                    print(f"   {solution.description}")

        if error_record.recovery_attempted and error_record.recovery_successful:
            print("✅ Automatic recovery was successful")
        elif error_record.recovery_attempted and not error_record.recovery_successful:
            print("❌ Automatic recovery failed")

    def register_recovery_action(self, error_key: str, action: RecoveryAction):
        """Register a recovery action for a specific error type."""
        if error_key not in self.recovery_actions:
            self.recovery_actions[error_key] = []
        self.recovery_actions[error_key].append(action)

    def register_error_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable,
    ):
        """Register a custom error handler for a specific exception type."""
        if exception_type not in self.error_handlers:
            self.error_handlers[exception_type] = []
        self.error_handlers[exception_type].append(handler)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and analytics."""
        if not self.error_records:
            return {"total_errors": 0}

        stats = {
            "total_errors": len(self.error_records),
            "by_severity": {},
            "by_category": {},
            "by_component": {},
            "recovery_rate": 0,
            "most_common_errors": [],
        }

        # Count by severity
        for severity in ErrorSeverity:
            count = sum(1 for r in self.error_records if r.severity == severity)
            stats["by_severity"][severity.value] = count

        # Count by category
        for category in ErrorCategory:
            count = sum(1 for r in self.error_records if r.category == category)
            stats["by_category"][category.value] = count

        # Count by component
        components = {}
        for record in self.error_records:
            components[record.component] = components.get(record.component, 0) + 1
        stats["by_component"] = components

        # Calculate recovery rate
        recovery_attempts = sum(1 for r in self.error_records if r.recovery_attempted)
        if recovery_attempts > 0:
            successful_recoveries = sum(
                1
                for r in self.error_records
                if r.recovery_attempted and r.recovery_successful
            )
            stats["recovery_rate"] = successful_recoveries / recovery_attempts

        # Most common errors
        error_counts = {}
        for record in self.error_records:
            key = f"{record.error_type}: {record.error_message[:50]}"
            error_counts[key] = error_counts.get(key, 0) + 1

        stats["most_common_errors"] = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return stats

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health and dependency status."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "dependencies": {},
            "recent_errors": len(
                [
                    r
                    for r in self.error_records
                    if (datetime.now() - r.timestamp).seconds < 3600
                ],
            ),
            "critical_errors": len(
                [r for r in self.error_records if r.severity == ErrorSeverity.CRITICAL],
            ),
            "recommendations": [],
        }

        # Check key dependencies
        key_dependencies = ["xfoil", "avl", "matplotlib", "numpy", "pandas"]
        missing_deps = []

        for dep in key_dependencies:
            available = self.dependency_checker.check_dependency(dep)
            health_report["dependencies"][dep] = {
                "available": available,
                "fallbacks": self.dependency_checker.get_available_fallbacks(dep)
                if not available
                else [],
            }

            if not available:
                missing_deps.append(dep)

        # Determine overall status
        if health_report["critical_errors"] > 0:
            health_report["overall_status"] = "critical"
        elif missing_deps or health_report["recent_errors"] > 10:
            health_report["overall_status"] = "degraded"

        # Generate recommendations
        if missing_deps:
            health_report["recommendations"].append(
                f"Install missing dependencies: {', '.join(missing_deps)}",
            )

        if health_report["recent_errors"] > 5:
            health_report["recommendations"].append(
                "High error rate detected - consider reviewing recent operations",
            )

        return health_report

    # Recovery action implementations
    def _prompt_for_file_location(self) -> bool:
        """Prompt user to locate a missing file."""
        # This would integrate with the UI to show a file picker
        # For now, return False to indicate manual intervention needed
        return False

    def _reduce_problem_size(self) -> bool:
        """Reduce problem size to fit available memory."""
        # This would implement logic to reduce analysis parameters
        # For now, return True to indicate successful degradation
        return True

    def _use_fallback_solver(self) -> bool:
        """Use an alternative solver when primary is unavailable."""
        # This would implement solver fallback logic
        return True

    def _retry_network_operation(self) -> bool:
        """Retry a failed network operation."""
        # This would implement network retry logic
        return True

    def _reset_to_default_config(self) -> bool:
        """Reset configuration to default values."""
        # This would implement configuration reset logic
        return True


# Context manager for error handling
class ErrorHandlingContext:
    """Context manager for handling errors in specific operations."""

    def __init__(self, error_handler: ErrorHandler, context: ErrorContext):
        self.error_handler = error_handler
        self.context = context
        self.error_record: Optional[ErrorRecord] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_record = self.error_handler.handle_error(exc_val, self.context)
            # Return False to re-raise the exception after handling
            return False
        return True


# Global error handler instance
error_handler = ErrorHandler()


# Decorator for automatic error handling
def handle_errors(component: str, operation: str = None):
    """Decorator for automatic error handling."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            context = ErrorContext(
                component=component,
                operation=op_name,
                user_data={"args": str(args), "kwargs": str(kwargs)},
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context)
                raise  # Re-raise after handling

        return wrapper

    return decorator
