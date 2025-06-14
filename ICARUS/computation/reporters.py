from datetime import datetime

from ICARUS.computation.core import ProgressReporter
from ICARUS.computation.core import ProgressUpdate
from ICARUS.computation.core import TaskResult


class ConsoleProgressReporter(ProgressReporter):
    """Simple console progress reporter"""

    def __init__(self):
        self._last_update = {}

    async def report_progress(self, progress: ProgressUpdate) -> None:
        """Print progress to console"""
        # Throttle updates
        last = self._last_update.get(progress.task_id, datetime.min)
        if (datetime.now() - last).total_seconds() < 0.5:
            return

        self._last_update[progress.task_id] = datetime.now()
        print(f"Task {progress.task_id}: {progress.percentage:.1f}% - {progress.message}")

    async def report_completion(self, result: TaskResult) -> None:
        """Print completion status"""
        status = "✓" if result.success else "✗"
        print(f"{status} Task {result.task_id} completed in {result.execution_time}")
