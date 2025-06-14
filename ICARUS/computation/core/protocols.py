"""
Protocol definitions for the simulation framework.

This module contains all the protocol definitions (interfaces) that define
the contracts for various components in the simulation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Generic
from typing import Protocol
from typing import runtime_checkable

from .types import TaskInput
from .types import TaskOutput

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data_structures import ProgressUpdate
    from .data_structures import TaskResult

# ===== MIXINS =====


class SerializableMixin:
    """
    Mixin class for objects that can be serialized.

    Provides a standard interface for converting objects to and from
    dictionary representations for serialization purposes.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns:
            Dictionary containing the object's serializable data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerializableMixin:
        """
        Create an object instance from a dictionary representation.

        Args:
            data: Dictionary containing the object's data

        Returns:
            New instance of the class

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


# ===== PROTOCOLS =====


@runtime_checkable
class TaskExecutor(Protocol, Generic[TaskInput, TaskOutput]):
    """
    Protocol defining task execution interface.

    This protocol defines the contract that all task executors must implement
    to be compatible with the simulation framework.
    """

    async def execute(self, task_input: TaskInput, context: ExecutionContext) -> TaskOutput:
        """
        Execute the task with the given input and context.

        Args:
            task_input: The input data for the task
            context: Execution context containing resources and utilities

        Returns:
            The result of the task execution
        """
        ...

    async def validate_input(self, task_input: TaskInput) -> bool:
        """
        Validate that the task input is acceptable.

        Args:
            task_input: The input data to validate

        Returns:
            True if the input is valid, False otherwise
        """
        ...

    async def cleanup(self) -> None:
        """
        Perform any necessary cleanup after task execution.

        This method is called after task execution completes,
        regardless of success or failure.
        """
        ...


@runtime_checkable
class ProgressReporter(Protocol):
    """
    Protocol for progress reporting.

    Defines the interface for components that can receive and handle
    progress updates from running tasks.
    """

    async def report_progress(self, progress: ProgressUpdate) -> None:
        """
        Report progress update for a task.

        Args:
            progress: Progress update containing current status
        """
        ...

    async def report_completion(self, result: TaskResult) -> None:
        """
        Report completion of a task.

        Args:
            result: Final result of the task execution
        """
        ...


@runtime_checkable
class ResourceManager(Protocol):
    """
    Protocol for resource management.

    Defines the interface for managing resources that tasks may require,
    such as database connections, file handles, or external services.
    """

    async def acquire_resources(self, requirements: dict[str, Any]) -> dict[str, Any]:
        """
        Acquire resources based on requirements.

        Args:
            requirements: Dictionary describing required resources

        Returns:
            Dictionary of acquired resources
        """
        ...

    async def release_resources(self, resources: dict[str, Any]) -> None:
        """
        Release previously acquired resources.

        Args:
            resources: Dictionary of resources to release
        """
        ...
