from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ICARUS.computation.core import ExecutionContext
from ICARUS.computation.core import ResourceManager
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.monitoring.progress import TqdmProgressMonitor

from .base_engine import BaseExecutionEngine


class ThreadingExecutionEngine(BaseExecutionEngine):
    """Threading-based execution engine using ThreadPoolExecutor"""

    async def execute_tasks(
        self,
        tasks: list[Task],
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None = None,
    ) -> list[TaskResult]:
        """Execute tasks using thread pool"""
        self.logger.info(f"Starting threading execution of {len(tasks)} tasks with max_workers={self.max_workers}")

        max_workers = self.max_workers or min(32, (len(tasks) or 1) + 4)

        def sync_execute_task(task: Task) -> TaskResult:
            """Synchronous wrapper for task execution"""
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._execute_task_with_context(task, progress_monitor, resource_manager),
                )
            finally:
                loop.close()

        # Execute in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [loop.run_in_executor(executor, sync_execute_task, task) for task in tasks]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                processed_results.append(TaskResult(task_id=task.id, state=TaskState.FAILED, error=result))
            else:
                processed_results.append(result)

        self.logger.info("Threading execution completed")
        return processed_results

    async def _execute_task_with_context(
        self,
        task: Task,
        progress_monitor: TqdmProgressMonitor,
        resource_manager: ResourceManager | None,
    ) -> TaskResult:
        """Execute a single task with full context management"""
        context = ExecutionContext(
            task.id,
            task.config,
            progress_monitor,
            resource_manager,
            logging.getLogger(f"task.{task.name}"),
        )

        start_time = datetime.now()
        task.state = TaskState.RUNNING

        try:
            # Acquire resources
            await context.acquire_resources()

            # Validate input
            if not await task.executor.validate_input(task.input):
                raise ValueError("Task input validation failed")

            # Execute with timeout
            if task.config.timeout:
                result = await asyncio.wait_for(
                    task.executor.execute(task.input, context),
                    timeout=task.config.timeout.total_seconds(),
                )
            else:
                result = await task.executor.execute(task.input, context)

            # Create successful result
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.COMPLETED,
                result=result,
                execution_time=datetime.now() - start_time,
            )

            task.state = TaskState.COMPLETED
            await progress_monitor.report_completion(task_result)
            return task_result

        except asyncio.TimeoutError:
            error = Exception(f"Task timed out after {task.config.timeout}")
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=error,
                execution_time=datetime.now() - start_time,
            )
            task.state = TaskState.FAILED
            await progress_monitor.report_completion(task_result)
            return task_result

        except Exception as e:
            task_result = TaskResult(
                task_id=task.id,
                state=TaskState.FAILED,
                error=e,
                execution_time=datetime.now() - start_time,
            )
            task.state = TaskState.FAILED
            await progress_monitor.report_completion(task_result)
            return task_result

        finally:
            # Always clean up resources
            await context.release_resources()
            await task.executor.cleanup()
