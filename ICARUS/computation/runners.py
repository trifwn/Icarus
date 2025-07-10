from __future__ import annotations

import logging
import os

from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskId
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressMonitor

from . import Reporter
from . import SimpleResourceManager
from . import create_execution_engine


class SimulationRunner:
    """Enhanced simulation runner with improved multiprocessing support"""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
        max_workers: int | None = None,
        resource_manager: ResourceManager | None = None,
        progress_monitor: ProgressMonitor | None = None,
        simulation_name: str | None = None
    ) -> None:
        self.execution_mode = execution_mode
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Task management
        self._tasks: list[Task] = []
        self._task_graph: dict[TaskId, list[TaskId]] = {}

        # Progress monitoring
        self.progress_monitor: ProgressMonitor | None = progress_monitor
        self.resource_manager = resource_manager or SimpleResourceManager(
            execution_mode.primitives,
        )

        # State management
        self._results: list[TaskResult] = []
        self.simulation_name = simulation_name if simulation_name else "Simulation"

    def add_task(self, task: Task) -> None:
        """
        Add a task to the execution queue.

        Args:
            task: Task to add
        """
        self._tasks.append(task)

        # Update task graph for dependency tracking
        self._task_graph[task.id] = task.config.dependencies

    def add_tasks(self, tasks: list[Task]) -> SimulationRunner:
        """Add multiple tasks"""
        for task in tasks:
            self.add_task(task)
        return self

    async def run(self) -> list[TaskResult]:
        """Execute all tasks with dependency resolution and progress bars"""
        if not self._tasks:
            self.logger.warning("No tasks to execute")
            return []

        self._running = True
        self.logger.info(
            f"Starting execution of {len(self._tasks)} tasks in {self.execution_mode.value} mode",
        )

        try:
            # Sort tasks by dependencies and priority
            sorted_tasks = self._resolve_dependencies()
            # Get the ProgressReporter
            reporter = Reporter(sorted_tasks)
            monitor = self.progress_monitor
            manager = self.resource_manager

            # Get ProgressMonitor if enabled
            if monitor:
                # Add progress monitor as an observer to the reporter
                reporter.add_observer(monitor)
                monitor.set_job(self.simulation_name, sorted_tasks)

            # Get execution engine
            engine = create_execution_engine(self.execution_mode)
            if not engine:
                raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

            with engine(
                tasks=sorted_tasks,
                max_workers=self.max_workers,
                progress_reporter=reporter,
                progress_monitor=monitor,
                resource_manager=manager,
            ) as execution_engine:
                if execution_engine.progress_monitor:
                    await execution_engine._start_progress_monitoring()

                # Execute tasks
                self._results = await execution_engine.execute_tasks()

                if execution_engine.progress_monitor:
                    await execution_engine._stop_progress_monitoring()
                return self._results

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise (e)

        finally:
            self.logger.info("Execution finished.")

    async def run_tasks(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute a list of tasks and return results.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of task results
        """
        self.add_tasks(tasks)
        return await self.run()

    def _resolve_dependencies(self) -> list[Task]:
        """Resolve task dependencies using topological sort"""
        # Simple implementation - in production, use proper topological sort
        tasks_by_priority = sorted(
            self._tasks,
            key=lambda t: t.config.priority.value,
            reverse=True,
        )
        return tasks_by_priority

    def cancel(self) -> None:
        """Cancel all running tasks"""
        for task in self._tasks:
            if task.state == TaskState.RUNNING:
                task.state = TaskState.CANCELLED
