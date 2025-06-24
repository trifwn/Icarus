"""
Protocol definitions for the simulation framework.

This module contains all the protocol definitions (interfaces) that define
the contracts for various components in the simulation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Protocol
from typing import Sequence
from typing import runtime_checkable

from ICARUS.computation.core.utils.concurrency import EventLike

from .types import TaskInput
from .types import TaskOutput

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data_structures import ProgressEvent
    from .data_structures import TaskResult
    from .task import Task


class SerializableMixin:
    """
    Mixin class for objects that can be serialized.

    Provides a standard interface for converting objects to and from
    dictionary representations. For robust serialization, implementations
    should handle nested serializable objects and complex data types.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a serializable dictionary.

        Returns:
            A dictionary containing the object's data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerializableMixin:
        """
        Create an object instance from a dictionary representation.

        Args:
            data: A dictionary containing the object's data.

        Returns:
            A new instance of the class.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


@runtime_checkable
class TaskExecutor(Protocol, Generic[TaskInput, TaskOutput]):
    """
    Protocol defining the task execution interface.

    This protocol defines the contract that all task executors must implement
    to be compatible with the simulation framework.
    """

    async def execute(self, task_input: TaskInput, context: ExecutionContext) -> TaskOutput:
        """
        Execute the task with the given input and context.

        Args:
            task_input: The input data for the task.
            context: The execution context, providing resources and utilities.

        Returns:
            The result of the task execution.
        """
        ...

    async def validate_input(self, task_input: TaskInput) -> bool:
        """
        Validate that the task input is acceptable.

        Args:
            task_input: The input data to validate.

        Returns:
            True if the input is valid, False otherwise.
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

    async def report_progress(self, progress: ProgressEvent) -> None:
        """
        Report a progress update for a task.

        Args:
            progress: A ProgressUpdate object with the current status.
        """
        ...

    async def report_completion(self, result: TaskResult) -> None:
        """
        Report the completion of a task.

        Args:
            result: The final result of the task execution.
        """
        ...


class ProgressObserver(Protocol):
    """
    Protocol for observing progress updates.

    Defines the interface for components that can observe and react to
    progress updates from tasks.
    """

    async def on_progress_update(self, progress: ProgressEvent) -> None:
        """
        Handle a progress update event.

        Args:
            progress: The progress update event to handle.
        """
        ...

    async def on_task_completion(self, result: TaskResult) -> None:
        """
        Handle task completion event.

        Args:
            result: The result of the completed task.
        """
        ...


@runtime_checkable
class ProgressMonitor(ProgressObserver, Protocol):
    """
    Protocol for progress monitoring implementations.

    Defines the interface for components that create, update,
    and finalize progress visualization or reporting.
    """

    def __enter__(self) -> ProgressMonitor:
        """Enter the monitoring context (e.g., initialize bars)."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the monitoring context (e.g., clean up bars)."""
        ...

    async def monitor_loop(self) -> None:
        """Run the monitoring loop to consume and display updates."""
        ...

    def set_tasks(self, tasks: Sequence[Task[Any, Any]]) -> None:
        """
        Set the tasks to monitor.

        Args:
            tasks: A list of tasks to monitor.
        """
        ...

    def add_cancellation_event(self, event: EventLike) -> None:
        """
        Add cancellation event for graceful shutdown.

        Args:

        """
        ...

    def set_event_queue(self, queue: Any) -> None:
        """
        Set the event queue for inter-process communication.

        Args:
            queue: The queue to use for event communication.
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
            requirements: A dictionary describing the required resources.

        Returns:
            A dictionary of acquired resources.
        """
        ...

    async def release_resources(self, resources: dict[str, Any]) -> None:
        """
        Release previously acquired resources.

        Args:
            resources: The dictionary of resources to release.
        """
        ...
