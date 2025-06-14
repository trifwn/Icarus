from __future__ import annotations

import logging
from threading import Event, Lock
from time import sleep
from typing import TextIO
from weakref import WeakSet

from tqdm import tqdm
import io
import sys
from ICARUS.computation import TQDM_AVAILABLE
from ICARUS.computation.core import ProgressReporter
from ICARUS.computation.core import ProgressUpdate
from ICARUS.computation.core import Task
from ICARUS.computation.core import TaskResult


class TqdmProgressMonitor:
    """
    Enhanced progress monitoring using tqdm with full OOP integration.
    """

    def __init__(
        self,
        tasks: list[Task],
        refresh_rate: float = 0.5,
        output: TextIO | io.TextIOWrapper | io.StringIO | None = None,
    ):
        if output is None:
            self.output = sys.stdout
        elif isinstance(output, (io.TextIOWrapper, io.StringIO, TextIO)):
            self.output = output
        else:
            raise TypeError("Output must be a TextIO, TextIOWrapper, StringIO, or None")

        self.tasks = tasks
        self.refresh_rate = refresh_rate
        self.progress_bars: dict[int, tqdm] = {}
        self.stop_event = Event()
        self._lock = Lock()  # Added lock for thread safety
        self.logger = logging.getLogger(__name__)
        self._observers: WeakSet[ProgressReporter] = WeakSet()

    def add_observer(self, observer: ProgressReporter) -> None:
        """Add progress observer"""
        self._observers.add(observer)

    def __enter__(self):
        """Context manager entry - create progress bars."""
        if not TQDM_AVAILABLE:
            return self

        for i, task in enumerate(self.tasks):
            pbar = tqdm(
                total=100,  # Use percentage as the total
                desc=f"{task.name}",
                position=i,
                leave=True,
                colour="#cc3300",
                bar_format="{l_bar}{bar:30}{r_bar}",
                file=self.output,
            )
            self.progress_bars[task.id_num] = pbar
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup progress bars."""
        self.stop_event.set()
        sleep(0.1)  # Brief pause for final updates

        for pbar in self.progress_bars.values():
            if pbar.n < pbar.total and not pbar.disable:
                pbar.n = pbar.total
                pbar.refresh()
            pbar.close()

        print("Progress monitor closed.")

    async def report_progress(self, progress: ProgressUpdate) -> None:
        """Report progress to progress bars and observers"""

        self._update_pbar(progress)

        # Notify observers
        for observer in self._observers:
            try:
                await observer.report_progress(progress)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    async def report_completion(self, result: TaskResult) -> None:
        """Report task completion"""
        # Update progress bar to completed state
        task = next((t for t in self.tasks if t.id == result.task_id), None)
        if task:
            progress = ProgressUpdate(
                task_id=result.task_id,
                name=task.name,
                current_step=100,
                total_steps=100,
                completed=True,
                error=result.error,
            )
            self._update_pbar(progress)

        # Notify observers
        for observer in self._observers:
            try:
                await observer.report_completion(result)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    def lock(self):
        """Acquire the internal lock."""
        self._lock.acquire()

    def unlock(self):
        """Release the internal lock."""
        self._lock.release()

    def _update_pbar(self, update: ProgressUpdate):
        """Applies a ProgressUpdate to its corresponding tqdm bar."""
        if not TQDM_AVAILABLE:
            return

        with self._lock:  # Ensure only one thread/process updates at a time
            # Find task by ID
            task = next((t for t in self.tasks if t.id == update.task_id), None)
            if not task or task.id_num not in self.progress_bars:
                return

            pbar = self.progress_bars[task.id_num]
            percentage = update.progress_percentage if update.progress_percentage is not None else 0

            if update.error:
                pbar.set_description(f"{update.name} - ERROR")
                pbar.colour = "#ff0000"  # Red
                pbar.n = int(percentage)
            elif update.completed:
                pbar.n = 100
                pbar.set_description(f"{update.name} - DONE")
                pbar.colour = "#00ff00"  # Green
            else:
                pbar.n = int(percentage)
                progress_text = f"{update.name} - {update.current_iteration}/{update.max_iterations}"
                if update.message:
                    progress_text += f" - {update.message}"
                pbar.set_description(progress_text)

            pbar.refresh()

    def monitor_loop(self):
        """Main monitoring loop, polls each task's probe."""
        while not self.stop_event.is_set():
            all_completed = True
            for task in self.tasks:
                if self.stop_event.is_set():
                    break
                try:
                    update = task.progress_probe(task)
                    self._update_pbar(update)
                    if not update.completed:
                        all_completed = False
                except Exception as e:
                    self.logger.debug(f"Error probing task {task.id_num}: {e}")

            if all_completed:
                sleep(self.refresh_rate)  # One last sleep to show 100%
                break
            sleep(self.refresh_rate)
