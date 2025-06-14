"""
Execution context for task execution.

This module provides the ExecutionContext class which serves as a rich
execution environment for tasks, providing access to resources, progress
reporting, logging, and cancellation mechanisms.
"""

import logging
from datetime import datetime
from datetime import timedelta
from threading import Event
from typing import Any
from typing import Dict
from typing import Optional

from .data_structures import ProgressUpdate
from .protocols import ProgressReporter
from .protocols import ResourceManager
from .types import TaskConfiguration
from .types import TaskId


class ExecutionContext:
    """
    Rich execution context for tasks.

    Provides a comprehensive execution environment that tasks can use to:
    - Report progress updates
    - Acquire and release resources
    - Access logging facilities
    - Check for cancellation requests
    - Store and retrieve execution metadata

    Attributes:
        task_id: Unique identifier of the task being executed
        config: Configuration for the task
        progress_reporter: Optional progress reporter for updates
        resource_manager: Optional resource manager for resource handling
        logger: Logger instance for this task
        start_time: When the task execution started
        cancellation_token: Event for checking cancellation status
    """

    def __init__(
        self,
        task_id: TaskId,
        config: TaskConfiguration,
        progress_reporter: Optional[ProgressReporter] = None,
        resource_manager: Optional[ResourceManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the execution context.

        Args:
            task_id: Unique identifier for the task
            config: Task configuration
            progress_reporter: Optional progress reporter
            resource_manager: Optional resource manager
            logger: Optional logger (will create one if not provided)
        """
        self.task_id = task_id
        self.config = config
        self.progress_reporter = progress_reporter
        self.resource_manager = resource_manager
        self.logger = logger or logging.getLogger(f"task.{task_id}")
        self.start_time = datetime.now()
        self.cancellation_token = Event()
        self._resources: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}

    async def report_progress(self, current: int, total: int, message: str = "", name: str = "") -> None:
        """
        Report progress if reporter is available.

        Args:
            current: Current step number
            total: Total number of steps
            message: Optional progress message
            name: Optional task name for the progress update
        """
        if self.progress_reporter:
            progress = ProgressUpdate(self.task_id, name, current, total, message)
            await self.progress_reporter.report_progress(progress)

    async def acquire_resources(self) -> None:
        """
        Acquire required resources.

        Uses the resource manager to acquire any resources specified
        in the task configuration.
        """
        if self.resource_manager and self.config.resources:
            self._resources = await self.resource_manager.acquire_resources(self.config.resources)

    async def release_resources(self) -> None:
        """
        Release acquired resources.

        Releases all resources that were previously acquired and
        clears the internal resource cache.
        """
        if self.resource_manager and self._resources:
            await self.resource_manager.release_resources(self._resources)
            self._resources.clear()

    def get_resource(self, name: str) -> Any:
        """
        Get an acquired resource by name.

        Args:
            name: Name of the resource to retrieve

        Returns:
            The requested resource, or None if not found
        """
        return self._resources.get(name)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the task.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata for the task.

        Args:
            key: Metadata key to retrieve
            default: Default value if key not found

        Returns:
            The metadata value or default
        """
        return self._metadata.get(key, default)

    @property
    def elapsed_time(self) -> timedelta:
        """
        Get elapsed execution time.

        Returns:
            Time elapsed since the context was created
        """
        return datetime.now() - self.start_time

    @property
    def is_cancelled(self) -> bool:
        """
        Check if task was cancelled.

        Returns:
            True if the task has been cancelled
        """
        return self.cancellation_token.is_set()

    def cancel(self) -> None:
        """
        Cancel the task.

        Sets the cancellation token to signal that the task
        should stop execution as soon as possible.
        """
        self.cancellation_token.set()
