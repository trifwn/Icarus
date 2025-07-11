from __future__ import annotations

import asyncio
import logging
from time import sleep
from typing import Any
from typing import Self
from typing import Sequence

from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskID
from rich.progress import TextColumn
from rich.table import Table

from ICARUS.computation.core import ConcurrencyFeature
from ICARUS.computation.core import ConcurrentVariable
from ICARUS.computation.core import EventLike
from ICARUS.computation.core import ProgressEvent
from ICARUS.computation.core import QueueLike
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core import TaskState
from ICARUS.computation.core.protocols import ProgressMonitor

from .rich_ui_manager import RichUIManager


class RichProgressMonitor(ProgressMonitor):
    """
    Enhanced progress monitoring using rich with full OOP integration.
    """

    progress: Progress
    overall_progress: Progress
    overall_task: TaskID

    def __init__(
        self,
        refresh_rate: float = 0.5,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.refresh_rate = refresh_rate
        self.tasks: Sequence[Task[Any, Any]] = []
        self.job_name: str = ""

        self._event_queue: QueueLike | None = None
        self._termination_event: EventLike | None = None

    def set_job(self, job_name: str, tasks: Sequence[Task[Any, Any]]) -> None:
        """Set the tasks to monitor"""
        self.tasks = tasks
        self.job_name = job_name
        self.task_id_map = {}

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """Request concurrent variables for this monitor"""
        return {
            "TERMINATE": ConcurrencyFeature.EVENT,
            "event_queue": ConcurrencyFeature.QUEUE,
            # "UI_Queue": ConcurrencyFeature.QUEUE,
            # "UI_Lock": ConcurrencyFeature.LOCK,
        }

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """Set concurrent variables for this monitor"""
        stop_event = vars.get("TERMINATE")
        if stop_event is None:
            raise ValueError("stop_event must be provided")
        if not isinstance(stop_event, EventLike):
            raise TypeError("stop_event must be an EventLike type")

        event_queue = vars.get("event_queue")
        if event_queue is None:
            raise ValueError("event_queue must be provided")
        if not isinstance(event_queue, QueueLike):
            raise TypeError("event_queue must be a QueueLike type")

        self.termination_event = stop_event
        self.event_queue = event_queue

        # ui_queue = vars.get("UI_Queue")
        # if ui_queue is None:
        #     raise ValueError("UI_Queue must be provided")
        # if not isinstance(event_queue, QueueLike):
        #     raise TypeError("UI_Queue must be a QueueLike type")

        # ui_lock = vars.get("UI_Lock")
        # if ui_queue is None:
        #     raise ValueError("UI_Lock must be provided")
        # if not isinstance(event_queue, QueueLike):
        #     raise TypeError("UI_Lock must be a QueueLike type")

        # self.ui_queue = ui_queue
        # self.ui_lock = ui_lock

    @property
    def event_queue(self) -> QueueLike | None:
        """Get the event queue used for inter-process communication."""
        return self._event_queue

    @event_queue.setter
    def event_queue(self, queue: QueueLike) -> None:
        """Set the event queue for inter-process communication."""
        self._event_queue = queue

    @property
    def termination_event(self) -> EventLike:
        """Get the event used for stopping the monitor."""
        if self._termination_event is None:
            raise ValueError("Termination event has not been set")
        return self._termination_event

    @termination_event.setter
    def termination_event(self, event: EventLike) -> None:
        """Set the event used for stopping the monitor."""
        self._termination_event = event

    def __enter__(self) -> Self:
        """Context manager entry - create progress bars and register with RichUIManager."""
        self.logger.debug("Entering RichProgressMonitor context")
        self.progress = Progress(
            TextColumn("{task.description}", justify="left", style="cyan"),
            SpinnerColumn(),
            BarColumn(bar_width=None),  # Auto-expand bar
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.overall_progress = Progress(
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )

        total = 0
        for task in self.tasks:
            total_steps = task.total_steps
            tid = self.progress.add_task(f"[cyan]{task.name}", total=total_steps)
            self.task_id_map[task.id_num] = tid
            total += total_steps
        self.overall_task = self.overall_progress.add_task("All Jobs", total=int(total))

        # Compose the jobs row (Panel)
        jobs_table = Table.grid(expand=True, padding=(0, 1))
        jobs_table.add_column(ratio=1)
        jobs_table.add_column(ratio=2)
        jobs_table.add_row(
            Panel(
                self.overall_progress,
                title="Overall Progress",
                border_style="green",
                padding=(1, 2),
                expand=True,
            ),
            Panel(
                self.progress,
                title="[b]Jobs",
                border_style="red",
                padding=(1, 2),
                expand=True,
            ),
        )

        # Register the jobs row with the RichUIManager singleton
        self._ui_manager = RichUIManager.get_instance()
        self._ui_manager.add_row(self.job_name, jobs_table)
        self._ui_manager.__enter__()  # Enter the Live context if not already entered
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - cleanup progress bars and unregister from RichUIManager."""
        self.logger.debug("Exiting RichProgressMonitor context")
        # Remove the jobs row from the UI manager
        if hasattr(self, "_ui_manager"):
            self._ui_manager.remove_row("Jobs")
            self._ui_manager.__exit__(exc_type, exc_val, exc_tb)
        self.termination_event.set()
        sleep(0.1)
        self.progress.stop()

    def on_progress_update(self, progress: ProgressEvent) -> None:
        """Report progress to progress bars and observers"""
        self.handle_progress_event(progress)

    def on_task_completion(self, result: TaskResult) -> None:
        """Report task completion"""
        task = next((t for t in self.tasks if t.id == result.task_id), None)
        if task:
            progress = ProgressEvent(
                task_id=result.task_id,
                name=task.name,
                current_step=task.total_steps,
                total_steps=task.total_steps,
                completed=True,
                error=result.error,
            )
            self.handle_progress_event(progress)

    def handle_progress_event(self, update: ProgressEvent) -> None:
        """Applies a ProgressUpdate to its corresponding progress bar and updates the UI row."""
        if update.task_id is None:
            return
        # Ensure update.task_id is the correct type (TaskId)
        task = next((t for t in self.tasks if t.id == update.task_id), None)
        if not task or task.id_num not in self.task_id_map:
            return
        tid = self.task_id_map[task.id_num]
        completed = update.current_step
        total = update.total_steps
        if update.error:
            self.progress.update(
                tid,
                total=total,
                completed=completed,
                description=f"[red]{update.name} - ERROR",
            )
        elif update.completed:
            self.progress.update(
                tid,
                total=total,
                completed=total,
                description=f"[green]{update.name} - DONE",
            )
        else:
            desc = f"{update.name} - {update.percentage:.2f}%"
            if update.message:
                desc += f" - {update.message}"
            self.progress.update(
                tid,
                total=total,
                completed=completed,
                description=desc,
            )
        # Update overall progress
        total_completed = sum(
            self.progress.tasks[tid].completed for tid in self.task_id_map.values()
        )
        total_steps = sum(
            self.progress.tasks[tid].total for tid in self.task_id_map.values()
        )
        if self.overall_task is not None:
            self.overall_progress.update(
                self.overall_task,
                completed=total_completed,
                total=total_steps,
            )

    async def monitor_loop(self) -> None:
        """Main monitoring loop, polls each task's probe."""
        while not self.termination_event.is_set():
            events: list[ProgressEvent] = []
            if hasattr(self, "event_queue") and self.event_queue:
                try:
                    # Drain all pending events
                    while not self.event_queue.empty():
                        evt = self.event_queue.get_nowait()
                        events.append(evt)
                except Exception as e:
                    self.logger.debug(f"Error reading event queue: {e}")

            # # Otherwise, probe local tasks (threading or async mode)
            if not events:
                for task in self.tasks:
                    if task.progress_probe:
                        try:
                            update = task.progress_probe()
                            events.append(update)
                        except Exception as e:
                            self.logger.info(f"Error probing task {task.id_num}: {e}")

            # Process all collected events
            for evt in events:
                self.handle_progress_event(evt)

            if all(
                task.state
                in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]
                for task in self.tasks
            ):
                break
            await asyncio.sleep(self.refresh_rate)
        self.logger.debug("Progress monitor loop stopped")

    def __str__(self) -> str:
        """String representation of the progress monitor."""
        return f"<RichProgressMonitor tasks={len(self.tasks)} refresh_rate={self.refresh_rate}>"

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the progress monitor for serialization."""
        state = {
            "refresh_rate": self.refresh_rate,
            "tasks": self.tasks,
            "event_queue": self._event_queue,
            "termination_event": self._termination_event,
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the progress monitor from serialized data."""
        self.refresh_rate = state.get("refresh_rate", 0.5)
        self.tasks = state.get("tasks", [])
        self._event_queue = state.get("event_queue")
        self._termination_event = state.get("termination_event")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.task_id_map = {}
