from datetime import datetime

from ICARUS.computation.core import ProgressEvent
from ICARUS.computation.core import TaskResult
from ICARUS.computation.core.protocols import ProgressObserver
from ICARUS.computation.core.utils.concurrency import ConcurrencyFeature
from ICARUS.computation.core.utils.concurrency import ConcurrentVariable


class ConsoleProgressObserver(ProgressObserver):
    """Simple console progress reporter"""

    def __init__(self):
        self._last_update = {}

    def request_concurrent_vars(self) -> dict[str, ConcurrencyFeature]:
        """Request concurrent variables for this observer"""
        return {}

    def set_concurrent_vars(self, vars: dict[str, ConcurrentVariable]) -> None:
        """Set concurrent variables for this observer"""
        pass

    def on_progress_update(self, progress: ProgressEvent) -> None:
        """Print progress to console"""
        # Throttle updates
        last = self._last_update.get(progress.task_id, datetime.min)
        print(f"Task {progress.task_id}: {progress.percentage:.1f}% - {progress.message}")
        if (datetime.now() - last).total_seconds() < 0.5:
            return

        self._last_update[progress.task_id] = datetime.now()

    def on_task_completion(self, result: TaskResult) -> None:
        """Print completion status"""
        status = "✓" if result.success else "✗"
        print(f"{status} Task {result.task_id} completed in {result.execution_time}")
