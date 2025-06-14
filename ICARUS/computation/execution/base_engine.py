from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.monitoring.progress import TqdmProgressMonitor


class BaseExecutionEngine(ABC):
    """Abstract base for execution engines"""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks and return results"""
        pass
