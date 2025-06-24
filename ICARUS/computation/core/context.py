"""
Execution context for task execution.

This module provides the ExecutionContext class which serves as a rich
execution environment for tasks, providing access to resources, progress
reporting, logging, and cancellation mechanisms.
"""

import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional

from .data_structures import ProgressEvent
from .protocols import ProgressReporter
from .protocols import ResourceManager
from .types import ExecutionMode
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
    - Obtain concurrency primitives (locks, events) suitable for the execution mode
    - Store and retrieve execution metadata

    Attributes:
        task_id: Unique identifier of the task being executed.
        config: Configuration for the task.
        execution_mode: The concurrency model being used (e.g., ASYNC, THREADING).
        progress_reporter: Optional progress reporter for updates.
        resource_manager: Optional resource manager for resource handling.
        logger: Logger instance for this task.
        start_time: When the task execution started.
        cancellation_token: Event for checking cancellation status.
    """

    def __init__(
        self,
        task_id: TaskId,
        config: TaskConfiguration,
        execution_mode: ExecutionMode,
        progress_reporter: Optional[ProgressReporter] = None,
        resource_manager: Optional[ResourceManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the execution context.

        Args:
            task_id: Unique identifier for the task.
            config: Task configuration.
            execution_mode: The execution mode (e.g., THREADING, ASYNC).
            progress_reporter: Optional progress reporter.
            resource_manager: Optional resource manager.
            logger: Optional logger (will create one if not provided).
        """
        self.task_id = task_id
        self.config = config
        self.execution_mode = execution_mode
        self.progress_reporter = progress_reporter
        self.resource_manager = resource_manager
        self.logger = logger or logging.getLogger(f"task.{task_id}")
        self.start_time = datetime.now()

        # Get concurrency primitives appropriate for the execution mode
        self.cancellation_token = execution_mode.create_event()

        self._resources: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}

    async def report_progress(self, current: int, total: int, message: str = "", name: str = "") -> None:
        """
        Report progress if a reporter is available.

        Args:
            current: Current step number.
            total: Total number of steps.
            message: Optional progress message.
            name: Optional task name for the progress update.
        """
        if self.progress_reporter:
            progress = ProgressEvent(
                task_id=self.task_id,
                name=name,
                current_step=current,
                total_steps=total,
                message=message,
            )
            await self.progress_reporter.report_progress(progress)

    async def acquire_resources(self) -> None:
        """
        Acquire required resources.

        Uses the resource manager to acquire any resources specified
        in the task configuration.
        """
        if self.resource_manager and self.config.resources:
            self.logger.debug(f"Acquiring resources: {self.config.resources}")
            self._resources = await self.resource_manager.acquire_resources(self.config.resources)
            self.logger.info("Successfully acquired resources.")

    async def release_resources(self) -> None:
        """
        Release acquired resources.

        Releases all resources that were previously acquired and
        clears the internal resource cache.
        """
        if self.resource_manager and self._resources:
            self.logger.debug("Releasing resources.")
            await self.resource_manager.release_resources(self._resources)
            self._resources.clear()
            self.logger.info("Successfully released resources.")

    @property
    def is_cancelled(self) -> bool:
        """
        Check if task was cancelled.

        Returns:
            True if the task has been cancelled.
        """
        return self.cancellation_token.is_set()

    def cancel(self) -> None:
        """
        Cancel the task.

        Sets the cancellation token to signal that the task
        should stop execution as soon as possible.
        """
        self.logger.warning("Cancellation requested for task.")
        self.cancellation_token.set()
