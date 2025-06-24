from __future__ import annotations

import logging
import os
import signal
from typing import Any

from ICARUS.computation.core import ExecutionMode
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskId
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core import ConcurrencyPrimitives
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.utils.concurrency import EventLike
from ICARUS.computation.execution import AdaptiveExecutionEngine
from ICARUS.computation.execution import AsyncExecutionEngine
from ICARUS.computation.execution import BaseExecutionEngine
from ICARUS.computation.execution import MultiprocessingExecutionEngine
from ICARUS.computation.execution import SequentialExecutionEngine
from ICARUS.computation.execution import ThreadingExecutionEngine
from ICARUS.computation.reporters import Reporter
from ICARUS.computation.resources.manager import SimpleResourceManager


class SimulationRunner:
    """Enhanced simulation runner with improved multiprocessing support"""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.ASYNC,
        max_workers: int | None = None,
        resource_manager: ResourceManager | None = None,
        progress_monitor: ProgressMonitor | None = None,
    ):
        self.execution_mode = execution_mode
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)
        self.resource_manager = resource_manager or SimpleResourceManager()
        self.enable_progress_monitoring = progress_monitor is not None
        self.logger = logging.getLogger(self.__class__.__name__)

        # Task management
        self._tasks: list[Task] = []
        self._task_graph: dict[TaskId, list[TaskId]] = {}
        self._execution_engines: dict[ExecutionMode, BaseExecutionEngine] = {
            ExecutionMode.ASYNC: AsyncExecutionEngine(max_workers),
            ExecutionMode.SEQUENTIAL: SequentialExecutionEngine(),
            ExecutionMode.THREADING: ThreadingExecutionEngine(max_workers),
            ExecutionMode.MULTIPROCESSING: MultiprocessingExecutionEngine(max_workers),
            ExecutionMode.ADAPTIVE: AdaptiveExecutionEngine(max_workers),
        }

        # Progress monitoring
        self.progress_monitor: ProgressMonitor | None = progress_monitor
        self.monitor_primatives: ConcurrencyPrimitives | None = None

        # State management
        self._running = False
        self._cancelled = False
        self._results: list[TaskResult] = []

        # Setup logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _setup_signal_handlers(self, event: EventLike) -> None:
        """Setup signal handlers for clean shutdown on CTRL+C."""

        def signal_handler(signum, frame):
            self.logger.warning("\nShutdown signal received. Cleaning up...")
            event.set()
            self.cancel()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
        self.logger.info(f"Starting execution of {len(self._tasks)} tasks in {self.execution_mode.value} mode")

        try:
            # Sort tasks by dependencies and priority
            sorted_tasks = self._resolve_dependencies()

            # Get execution engine
            engine = self._execution_engines.get(self.execution_mode)
            if not engine:
                raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

            if self.enable_progress_monitoring and self.progress_monitor:
                # Setup signal handlers and progress monitoring
                self.progress_monitor.set_tasks(sorted_tasks)

                cancellation_event = self.execution_mode.create_event()

                self._setup_signal_handlers(cancellation_event)
                self.progress_monitor.add_cancellation_event(cancellation_event)

                await engine._start_progress_monitoring(progress_monitor=self.progress_monitor)

            reporter = Reporter(sorted_tasks)
            if self.enable_progress_monitoring and self.progress_monitor:
                # Add progress monitor as an observer to the reporter
                reporter.add_observer(self.progress_monitor)

            # Execute tasks
            self._results = await engine.execute_tasks(sorted_tasks, reporter, self.resource_manager)

            if self.enable_progress_monitoring and self.progress_monitor:
                await engine._stop_progress_monitoring()
            return self._results

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise

        finally:
            self._running = False
            if self.monitor_primatives:
                self.monitor_primatives.event.set()  # Signal monitor to stop
            self.logger.info("Execution finished.")

    async def run_tasks(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute a list of tasks and return results.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of task results
        """
        self._tasks = tasks
        return await self.run()

    def _resolve_dependencies(self) -> list[Task]:
        """Resolve task dependencies using topological sort"""
        # Simple implementation - in production, use proper topological sort
        tasks_by_priority = sorted(self._tasks, key=lambda t: t.config.priority.value, reverse=True)
        return tasks_by_priority

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive execution summary"""
        if not self._results:
            return {
                "total_tasks": len(self._tasks),
                "successful": 0,
                "failed": 0,
                "success_rate": 0,
                "execution_mode": self.execution_mode.value,
            }

        successful = [r for r in self._results if r.success]
        execution_times = [r.execution_time.total_seconds() for r in self._results if r.execution_time]
        return {
            "total_tasks": len(self._results),
            "successful": len(successful),
            "failed": len(self._results) - len(successful),
            "success_rate": len(successful) / len(self._results) * 100 if self._results else 0,
            "total_execution_time": sum(execution_times),
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "execution_mode": self.execution_mode.value,
            "task_results": [r.to_dict() for r in self._results],
            "task_states": {str(task.id): task.state.name for task in self._tasks},
            "task_progress": {
                str(task.id): {
                    "current_step": task._progress,
                    "total_steps": task._total_progress,
                    "message": task._progress_message,
                    "completed": task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED],
                }
                for task in self._tasks
            },
        }

    def cancel(self) -> None:
        """Cancel all running tasks"""
        self._cancelled = True
        for task in self._tasks:
            if task.state == TaskState.RUNNING:
                task.state = TaskState.CANCELLED
