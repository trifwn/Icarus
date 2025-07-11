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

from .types import TaskInput
from .types import TaskOutput
from .utils.concurrency import ConcurrencyFeature
from .utils.concurrency import ConcurrentVariable
from .utils.concurrency import EventLike
from .utils.concurrency import QueueLike

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .data_structures import ProgressEvent
    from .data_structures import TaskResult
    from .task import Task


@runtime_checkable
class ConcurrentMixin(Protocol):
    """
    Mixin class for concurrent objects.

    Provides a standard interface for objects that can share computer resources
    and run concurrently.
    """

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """
        Request the concurrent variables needed by the object.

        This method should be called before using any concurrent features
        to ensure that the necessary resources are available.
        """
        ...

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """
        Set the concurrent variables for the object.

        Args:
            vars: A dictionary containing the concurrent variables (e.g., locks, events, queues)
        """
        ...


@runtime_checkable
class TaskExecutorProtocol(Protocol, Generic[TaskInput, TaskOutput]):
    """
    Protocol defining the task execution interface.

    This protocol defines the contract that all task executors must implement
    to be compatible with the simulation framework.
    """

    async def execute(
        self,
        task_input: TaskInput,
        context: ExecutionContext,
    ) -> TaskOutput:
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

    async def cancel(self) -> None:
        """
        Handle cancellation of the task execution.

        This method can be used to clean up resources or state if needed.
        It should be idempotent and safe to call multiple times.
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
class ProgressReporter(ConcurrentMixin, Protocol):
    event_queue: QueueLike | None
    """
    Protocol for progress reporting.

    Defines the interface for components that can receive and handle
    progress updates from running tasks.
    """

    def report_progress(self, event: ProgressEvent) -> None:
        """
        Report a progress update for a task.

        Args:
            event: A ProgressEvent object with the current status.
        """
        ...

    def report_completion(self, result: TaskResult[Any]) -> None:
        """
        Report the completion of a task.

        Args:
            result: The final result of the task execution.
        """
        ...


@runtime_checkable
class ProgressObserver(ConcurrentMixin, Protocol):
    """
    Protocol for observing progress updates.

    Defines the interface for components that can observe and react to
    progress updates from tasks.
    """

    def on_progress_update(self, progress: ProgressEvent) -> None:
        """
        Handle a progress update event.

        Args:
            progress: The progress update event to handle.
        """
        ...

    def on_task_completion(self, result: TaskResult[Any]) -> None:
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

    event_queue: QueueLike | None
    termination_event: EventLike

    def __enter__(self) -> ProgressMonitor:
        """Enter the monitoring context (e.g., initialize bars)."""
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the monitoring context (e.g., clean up bars)."""
        ...

    async def monitor_loop(self) -> None:
        """Run the monitoring loop to consume and display updates."""
        ...

    def set_job(self, job_name: str, tasks: Sequence[Task[Any, Any]]) -> None:
        """
        Set the tasks to monitor.

        Args:
            tasks: A list of tasks to monitor.
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
