from __future__ import annotations

import logging
import signal
from abc import ABC
from abc import abstractmethod
from types import FrameType
from types import TracebackType
from typing import Any

from ICARUS.computation.core import ConcurrencyFeature
from ICARUS.computation.core import ConcurrentVariable
from ICARUS.computation.core import EventLike
from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core.protocols import ConcurrentMixin
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.protocols import ProgressReporter


class AbstractEngine(ConcurrentMixin, ABC):
    """Abstract base for execution engines"""

    execution_mode: ExecutionMode

    def __init__(
        self,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute_tasks(self) -> list[TaskResult[Any]]:
        """Execute tasks and return results"""
        ...

    @abstractmethod
    async def _start_progress_monitoring(self) -> None: ...

    @abstractmethod
    async def _stop_progress_monitoring(self) -> None: ...

    def __enter__(self) -> AbstractEngine:
        """Context manager entry point to prepare execution context."""
        self.logger.info(f"Entering engine: {self.__class__.__name__}")
        concurrent_vars_req = self.request_concurrent_vars()
        concurent_vars = self.execution_mode.primitives.get_concurrent_variables(
            concurrent_vars_req,
        )
        self.set_concurrent_vars(concurent_vars)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Context manager exit point to clean up execution context."""
        pass

    # Enter Arguments
    def __call__(
        self,
        tasks: list[Task],
        progress_reporter: ProgressReporter | None,
        progress_monitor: ProgressMonitor | None = None,
        resource_manager: ResourceManager | None = None,
        max_workers: int | None = None,
    ) -> AbstractEngine:
        self.logger.info(f"Starting execution with {self.__class__.__name__}")
        self.tasks = tasks
        self.progress_reporter = progress_reporter
        self.progress_monitor = progress_monitor
        self.resource_manager = resource_manager
        self.max_workers = max_workers
        return self

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """Request concurrent variables required by this engine."""
        vars = {
            "TERMINATE": ConcurrencyFeature.EVENT,
        }
        if self.progress_reporter:
            vars.update(self.progress_reporter.request_concurrent_vars())
        if self.progress_monitor:
            vars.update(self.progress_monitor.request_concurrent_vars())
        return vars

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """Set concurrent variables for this engine."""
        if "TERMINATE" in vars:
            terminate_event = vars["TERMINATE"]
        else:
            raise KeyError("TERMINATE event is required for execution engine")
        if not isinstance(terminate_event, EventLike):
            raise TypeError(
                f"Expected EventLike for 'TERMINATE', got {type(terminate_event)}",
            )
        # Set up terminate event for graceful shutdown
        self.terminate_event = terminate_event

        if self.progress_reporter:
            self.progress_reporter.set_concurrent_vars(vars)

        if self.progress_monitor:
            self.progress_monitor.set_concurrent_vars(vars)
        self.concurrent_variables = vars

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: FrameType | None) -> None:
            self.logger.warning("\nShutdown signal received. Cleaning up...")
            self.terminate_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self.logger.info("Signal handlers set up for graceful shutdown.")
