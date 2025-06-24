from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.protocols import ProgressReporter


class BaseExecutionEngine(ABC):
    """Abstract base for execution engines"""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        # store prepared execution resources
        self._exec_context: dict[str, Any] = {}

    @abstractmethod
    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_reporter: ProgressReporter | None,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks and return results"""
        pass

    @abstractmethod
    async def _start_progress_monitoring(
        self,
        progress_monitor: ProgressMonitor,
    ) -> None: ...

    @abstractmethod
    async def _stop_progress_monitoring(self) -> None:
        """Stop the progress monitor if it is running."""
        pass
