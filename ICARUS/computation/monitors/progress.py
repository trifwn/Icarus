from __future__ import annotations

import asyncio
import logging
from multiprocessing import Queue
from time import sleep

from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.table import Table

from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core.data_structures import ProgressEvent
from ICARUS.computation.core.protocols import ProgressMonitor
from ICARUS.computation.core.utils.concurrency import EventLike


class RichProgressMonitor(ProgressMonitor):
    """
    Enhanced progress monitoring using rich with full OOP integration.
    """

    def __init__(
        self,
        refresh_rate: float = 0.5,
    ):
        self.refresh_rate = refresh_rate
        self.event_queue = None

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

        self.logger = logging.getLogger(__name__)
        self.live = None
        self.overall_progress = Progress()
        self.overall_task = None
        self.progress_table = None
        self.tasks: list[Task] = []

    def set_tasks(self, tasks: list[Task]) -> None:
        """Set the tasks to monitor"""
        self.tasks = tasks
        self.task_id_map = {}

    def set_event_queue(self, queue: Queue) -> None:
        """Set the event queue for multiprocessing mode"""
        self.event_queue = queue

    def add_cancellation_event(self, event: EventLike) -> None:
        """Set concurrency primitives for synchronization"""
        self.stop_event = event
        self.logger.debug("Cancellation event set for progress monitor")

    def __enter__(self):
        """Context manager entry - create progress bars."""
        total = 0
        for task in self.tasks:
            total_steps = task._total_progress if task._total_progress else 100
            tid = self.progress.add_task(f"[cyan]{task.name}", total=total_steps)
            self.task_id_map[task.id_num] = tid
            total += total_steps
        self.overall_task = self.overall_progress.add_task("All Jobs", total=int(total))

        self.progress_table = Table.grid(expand=True)
        self.progress_table.add_row(
            Panel(self.overall_progress, title="Overall Progress", border_style="green", padding=(1, 2), expand=True),
            Panel(self.progress, title="[b]Jobs", border_style="red", padding=(1, 2), expand=True),
        )
        # self.progress_table.add_row(
        #     Panel(
        #         Group(self.overall_progress, self.progress),
        #         title="Overall Progress & Jobs",
        #         border_style="green",
        #         padding=(1, 2),
        #         expand=True
        #     )
        # )

        self.live = Live(self.progress_table, refresh_per_second=10, screen=False, auto_refresh=True, transient=True)
        # self.live = Live(self.progress_table, refresh_per_second=10)
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup progress bars."""
        if self.live is not None:
            self.live.__exit__(exc_type, exc_val, exc_tb)
        self.stop_event.set()
        sleep(0.1)
        self.progress.stop()

    async def on_progress_update(self, progress: ProgressEvent) -> None:
        """Report progress to progress bars and observers"""
        self.handle_progress_event(progress)

    async def on_task_completion(self, result: TaskResult) -> None:
        """Report task completion"""
        task = next((t for t in self.tasks if t.id == result.task_id), None)
        if task:
            progress = ProgressEvent(
                task_id=result.task_id,
                name=task.name,
                current_step=task._total_progress if task._total_progress else 1,
                total_steps=task._total_progress if task._total_progress else 1,
                completed=True,
                error=result.error,
            )
            self.handle_progress_event(progress)

    def handle_progress_event(self, update: ProgressEvent):
        """Applies a ProgressUpdate to its corresponding progress bar."""
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
            self.progress.update(tid, total=total, completed=completed, description=f"[red]{update.name} - ERROR")
        elif update.completed:
            self.progress.update(tid, total=total, completed=total, description=f"[green]{update.name} - DONE")
        else:
            desc = f"{update.name} - {update.percentage:.2f}%"
            if update.message:
                desc += f" - {update.message}"
            self.progress.update(tid, total=total, completed=completed, description=desc)
        # Update overall progress
        total_completed = sum(self.progress.tasks[tid].completed for tid in self.task_id_map.values())
        if self.overall_task is not None:
            self.overall_progress.update(self.overall_task, completed=total_completed)

        # all_completed = all(task.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED] for task in self.tasks)

    async def monitor_loop(self):
        """Main monitoring loop, polls each task's probe."""
        while not self.stop_event.is_set():
            # ("Starting progress monitor loop") Add this message to the rich
            # progress bar
            # print("Starting progress monitor loop")
            events: list[ProgressEvent] = []
            # If an event_queue is provided, only consume events (multiprocessing mode)
            # print(f"Checking for events in the queue {self.event_queue}")
            if hasattr(self, "event_queue") and self.event_queue:
                try:
                    # Drain all pending events
                    while not self.event_queue.empty():
                        evt = self.event_queue.get_nowait()
                        events.append(evt)
                except Exception as e:
                    self.logger.debug(f"Error reading event queue: {e}")
            else:
                # Otherwise, probe local tasks (threading or async mode)
                for task in self.tasks:
                    try:
                        update = task.progress_probe()
                        events.append(update)
                    except Exception as e:
                        self.logger.debug(f"Error probing task {task.id_num}: {e}")

            # Process all collected events
            all_completed = True
            for evt in events:
                self.handle_progress_event(evt)
                if not evt.completed:
                    all_completed = False

            if all_completed:
                break

            await asyncio.sleep(self.refresh_rate)
            continue

    def __str__(self):
        """String representation of the progress monitor."""
        return f"<RichProgressMonitor tasks={len(self.tasks)} refresh_rate={self.refresh_rate}>"
