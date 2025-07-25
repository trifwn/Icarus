"""
Custom exception hierarchy for the simulation framework.

This module defines specific exceptions for different failure scenarios
in the simulation framework, providing better error handling and debugging.
"""

from typing import Optional

from .types import TaskId


class SimulationFrameworkError(Exception):
    """Base exception for all simulation framework errors."""

    pass


class TaskExecutionError(SimulationFrameworkError):
    """Raised when a task fails during execution."""

    def __init__(
        self,
        task_id: TaskId,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Task {task_id} failed: {message}")


class ResourceManagerError(SimulationFrameworkError):
    """Raised when resource management operations fail."""

    pass


class DependencyResolutionError(SimulationFrameworkError):
    """Raised when task dependencies cannot be resolved."""

    def __init__(self, task_id: TaskId, missing_dependencies: list[TaskId]):
        self.task_id = task_id
        self.missing_dependencies = missing_dependencies
        super().__init__(
            f"Task {task_id} has unresolved dependencies: {missing_dependencies}",
        )


class TaskTimeoutError(SimulationFrameworkError):
    """Raised when a task exceeds its timeout duration."""

    def __init__(self, task_id: TaskId, timeout_seconds: float):
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Task {task_id} timed out after {timeout_seconds} seconds")


class ConfigurationError(SimulationFrameworkError):
    """Raised when there are configuration-related errors."""

    pass


class SerializationError(SimulationFrameworkError):
    """Raised when serialization or deserialization fails."""

    pass
